"Exact Bayesian and frequentist decoding of the sequentially Markov coalescent"

import logging
from dataclasses import InitVar, dataclass
from typing import List, Union

import numpy as np
from tskit import TableCollection

import xsmc._sampler
import xsmc._xsmc

from . import _viterbi
from .sampler import XSMCSampler
from .segmentation import ArraySegmentation, Segmentation
from .size_history import SizeHistory
from .supporting import watterson

logger = logging.getLogger(__name__)


@dataclass
class XSMC:
    r"""Sample from, or maximize, the posterior distribution :math:`p(X_{1:N} | \mathbf{Y})` of the sequentially Markov
    coalescent.

    Args:
        ts: Tree sequence containing the data.
        theta: Population-scaled mutation rate. If None, Watterson's estimator is used.
        rho_over_theta: Ratio of recombination to mutation rates.
        w: Window size. Observations are binned into windows of this size. Recombinations are assumed to occur between
            adjacent bins, but not within them. If None, try to calculate a sensible default based on `rho`.

    Notes:
        The height of segments returned by this class are expressed in coalescent units. To convert to generations,
        rescale them by `2 * self.theta / (4 * mu)`, where `mu` is the biological mutation rate.
    """
    ts: InitVar["tskit.TreeSequence"]
    pi: np.ndarray = None
    theta: float = None
    rho_over_theta: float = 1.0
    w: int = None
    eps: float = 1e-4

    def __post_init__(self, ts):
        # This handles the task of converting from the passed-in tree sequence to our internal version.
        # (These could be based off of different libraries).
        tables = ts.dump_tables()
        lwtc = xsmc._tskit.LightweightTableCollection()
        lwtc.fromdict(tables.asdict())
        self._ts = xsmc._xsmc.TreeSequence(lwtc)
        if self._ts.num_sites == 0:
            raise ValueError("There aren't any mutations in the tree sequence")
        if self.theta is None:
            self.theta = watterson(ts)
            logger.debug("Estimated θ=%f", self.theta)
        if self.theta == 0.0:
            raise ValueError("theta must be positive")
        if self.rho_over_theta <= 0.0:
            raise ValueError("rho_over_theta must be positive")
        self.rho = self.theta * self.rho_over_theta
        if self.w is None:
            self.w = 1 + int(1 / (10 * self.rho))
            logger.debug("Setting window size w=%f", self.w)
        if self.pi is None:
            self.pi = np.array([1., 1., 0.])[:, None]  # a, b, log_c
        assert self.pi.shape[0] == 3
        a, b, log_c = self.pi
        assert np.all(a >= 0.), "pi should be a gamma mixture with positive shape parameters"
        assert np.all(b >= 0.), "pi should be a gamma mixture with positive scale parameters"
        assert np.isclose(np.exp(log_c).sum(), 1.), "pi should be a mixture distribution with exp(log_c).sum() ~= 1."

    @property
    def sequence_length(self):
        return self._ts.sequence_length

    def _sampler(self, focal, panel):
        X = xsmc._sampler.get_mismatches(self._ts, focal, panel, self.w)
        deltas = np.ones_like(X[0])
        # Perform sampling
        assert deltas.shape[0] == X.shape[1]
        return XSMCSampler(
            X=X,
            deltas=deltas,
            theta=self.w * self.theta,
            rho=self.w * self.rho,
            pi=self.pi,
            eps=self.eps,
        )

    def sample(self, focal: int, panel: List[int], k: int = 1, seed: int = None) -> List[Segmentation]:
        r"""Sample path(s) from the posterior distribution.

        Args:
            k: Number of posterior path samples to draw, default 1.
            seed: Random seed used for sampling.

        Returns:
            A list of `k` posterior samples for the positional min-TMRCA of `focal` with `panel`.

        Notes:
            If sampling many paths at once, it is more efficient to set `k > 1` than to call `sample()`
            repeatedly.
        """
        prime = False
        sampler = self._sampler(focal, panel)
        segs = sampler.sample_paths(k, seed, prime)
        ret = [
            ArraySegmentation(segments=s, panel_inds=p, panel=panel)
            for s, p in segs
        ]
        for a in ret:
            a.segments[:2] *= self.w  # expand intervals by w
        return ret

    def sample_heights(self, focal: int, panel: List[int], j: int, k: int, seed: Union[int, None]) -> np.ndarray:
        sampler = self._sampler(focal, panel)
        return sampler.sample_heights(j, k, seed)

    def viterbi(self, focal: int, panel: List[int], beta: float = None) -> Segmentation:
        """Compute the maximum *a posteriori* (a.k.a. Viterbi) path in haplotype copying model.

        Args:
            beta: Penalty parameter for changepoint formation. See manuscript for details about this parameter.

        Returns:
            A segmentation containing the MAP path.
        """
        eta: SizeHistory = SizeHistory(t=np.array([0.0, np.inf]), Ne=np.array([1.0]))
        ret, self._K_n = _viterbi.viterbi_path(
            ts=self._ts,
            focal=focal,
            panel=panel,
            eta=eta,
            theta=self.theta,
            rho=self.rho,
            beta=beta,
            robust=False,
            w=self.w,
        )
        return ret
