"""Reference implementation of Li & Durbin's (2011) PSMC model."""
import itertools
import os
import tempfile
import textwrap
from dataclasses import dataclass
from typing import List, TextIO, Tuple, Union

import numpy as np
import sh
from scipy.interpolate import PPoly

psmc = sh.Command(os.environ.get("PSMC_PATH", "psmc"))

__version__ = (
    sh.grep(psmc(_err_to_out=True, _ok_code=1), "Version").strip().split(" ")[1]
)


@dataclass(frozen=True)
class PSMCResult:
    theta: float
    rho: float
    Ne: Union[np.ndarray, "SizeHistory"]

    def __call__(self, x):
        "Evaluate size history at point(s) x."
        x = np.atleast_1d(x)
        return self.Ne.Ne[np.searchsorted(self.Ne.t, x, "right") - 1]


@dataclass(frozen=True)
class PSMCPosterior:
    t: np.ndarray
    gamma: np.ndarray


def _gen_psmcfa(
    ts: "tskit.TreeSequence",
    contig: str,
    nodes: Tuple[int, int],
    out: TextIO,
    w: int = 100,
):
    "Generate a PSMCFA file for nodes in tree seqeuence."
    L = int(ts.get_sequence_length() // w)
    outstr = ["T"] * L
    for v in ts.variants(samples=nodes):
        gt = v.genotypes
        if gt[0] != gt[1]:
            outstr[int(v.position // w)] = "K"
    print("> %s" % contig, file=out)
    print("\n".join(textwrap.wrap("".join(outstr), width=79)), file=out)
    print("", file=out)


def _psmciter(out):
    "split psmc output on // and return groups of lines"
    i = 0

    def keyfunc(line):
        nonlocal i
        if line.startswith("//"):
            i += 1
        return i

    # find last estimate
    return [list(lines) for i, lines in itertools.groupby(out, keyfunc)]


def _parse_psmc(out) -> List[PSMCResult]:
    "Parse PSMC output"
    iterations = _psmciter(out)
    ret = []
    for iterate in iterations:
        for line in iterate:
            if line.startswith("TR"):
                theta, rho = list(map(float, line.strip().split("\t")[1:3]))
        t, lam = zip(
            *[
                list(map(float, line.strip().split("\t")[2:4]))
                for line in iterations[-2]
                if line.startswith("RS")
            ]
        )
        ret.append(dict(theta=theta, rho=rho, t=np.array(t), Ne=np.array(lam)))
    return ret


def _parse_posterior(out):
    "Parse PSMC posterior output"
    iterations = _psmciter(out)
    posterior = iterations[-1][1:]  # strip leading //
    groups = itertools.groupby(posterior, lambda line: line[:2])
    _, times = next(groups)
    t = [0.0] + [float(line.strip().split("\t")[4]) for line in times]
    _, pd = next(groups)
    gamma = np.array([list(map(float, line.strip().split("\t")[3:])) for line in pd])
    return PSMCPosterior(np.array(t), gamma)


@dataclass
class SizeHistory:
    t: np.ndarray
    Ne: np.ndarray

    def rescale(self, c: float) -> "SizeHistory":
        return SizeHistory(c * self.t, c * self.Ne)

    def draw(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(self.t, self.Ne, drawstyle="steps-post", **kwargs)

    def to_pp(self):
        return PPoly(x=np.r_[self.t, np.inf], c=[self.Ne])

    def __call__(self, x):
        return self.Ne[np.searchsorted(self.t, np.atleast_1d(x), "right") - 1]


@dataclass
class msPSMC:
    """PSMC model.

    Args:
         data: List of `(tree sequence, (hap1, hap2))` pairs.
    """

    data: List[Tuple["tskit.TreeSequence", Tuple[int, int]]]
    w: int = 100

    def __post_init__(self):
        self._tf = tempfile.NamedTemporaryFile(suffix=".psmcfa")
        self._fa = self._tf.name
        with open(self._fa, "wt") as out:
            for i, (ts, h) in enumerate(self.data):
                _gen_psmcfa(ts, "contig%d" % i, h, out, self.w)

    def estimate(self, *args, **kwargs) -> PSMCResult:
        """Run model for x em iterations."""
        with tempfile.NamedTemporaryFile(suffix=".psmc") as f:
            psmc("-o", f.name, *args, self._fa)
            res = _parse_psmc(open(f.name, "rt"))
        r = res[-1]
        return PSMCResult(
            theta=r["theta"], rho=r["rho"], Ne=SizeHistory(r["t"], r["Ne"])
        )

    def posterior(self, *args) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior decoding"""
        with tempfile.NamedTemporaryFile(suffix=".psmc") as f:
            psmc("-o", f.name, "-d", "-D", *args, self._fa)
            with open(f.name, "rt") as f:
                res = _parse_psmc(f)
                f.seek(0)
                pd = _parse_posterior(f)
        return res[-1].theta, res[-1].rho, pd.t, pd.gamma.T


def test_mspsmc():
    import msprime as msp

    n = 2
    ts = msp.simulate(
        sample_size=2 * n, length=1e6, mutation_rate=1e-4, recombination_rate=1e-4
    )
    p = msPSMC([(ts, (2 * i, 2 * i + 1)) for i in range(n // 2)])
    res = p.estimate()
    theta, rho, times, gamma = p.posterior()
    print(times.shape)
    print(gamma.shape)
    print(gamma.sum(axis=0))


if __name__ == "__main__":
    test_mspsmc()
