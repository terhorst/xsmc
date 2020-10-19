"Exact Bayesian and frequentist decoding of the sequentially Markov coalescent"

import logging
from dataclasses import InitVar, dataclass
from typing import List, Union

import numpy as np

import xsmc._sampler

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
        focal: Leaf node in `ts` corresponding to focal haplotype.
        panel: Leaf node(s) in `ts` corresponding to panel haplotypes.
        theta: Population-scaled mutation rate. If None, Watterson's estimator is used.
        rho_over_theta: Ratio of recombination to mutation rates.
        w: Window size. Observations are binned into windows of this size. Recombinations are assumed to occur between
            adjacent bins, but not within them. If None, try to calculate a sensible default based on `rho`.
        robust: If True, use robust model in which bins are classified as either segregating or nonsegregating, as in PSMC.
            Otherwise, allow for any number of mutations per bin.

    Notes:
        The height of segments returned by this class are expressed in coalescent units. To convert to generations,
        rescale them by `2 * self.theta / (4 * mu)`, where `mu` is the biological mutation rate.
    """
    ts: InitVar["tskit.TreeSequence"]
    focal: int
    panel: List[int]
    theta: float = None
    rho_over_theta: float = 1.0
    w: int = None
    robust: bool = False

    def __post_init__(self, ts):
        # This handles the task of converting from the passed-in tree sequence to our internal version. (These could be based off of different libraries).
        self.L = ts.get_sequence_length()
        tables = ts.dump_tables()
        self._lwtc = xsmc._tskit.LightweightTableCollection()
        self._lwtc.fromdict(tables.asdict())
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
        self._sampler = None

    @property
    def H(self):
        return len(self.panel)

    @property
    def sampler(self):
        if self._sampler is None:
            X = xsmc._sampler.get_mismatches(self._lwtc, self.focal, self.panel, self.w)
            deltas = np.ones_like(X[0])
            # Perform sampling
            assert deltas.shape[0] == X.shape[1]
            self._sampler = XSMCSampler(
                X=X,
                deltas=deltas,
                theta=self.w * self.theta,
                rho=self.w * self.rho,
                robust=self.robust,
                eps=1e-4,
            )
        return self._sampler

    def sample(self, k: int = 1, seed: int = None) -> List[Segmentation]:
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
        segs = self.sampler.sample_paths(k, seed, prime)
        ret = [
            ArraySegmentation(segments=s, panel_inds=p, panel=self.panel)
            for s, p in segs
        ]
        for a in ret:
            a.segments[:2] *= self.w  # expand intervals by w
        return ret

    def sample_heights(self, j: int, k: int, seed: Union[int, None]) -> np.ndarray:
        return self.sampler.sample_heights(j, k, seed)

    def viterbi(self, beta: float = None) -> Segmentation:
        """Compute the maximum *a posteriori* (a.k.a. Viterbi) path in haplotype copying model.

        Args:
            beta: Penalty parameter for changepoint formation. See manuscript for details about this parameter.

        Returns:
            A segmentation containing the MAP path.
        """
        eta: SizeHistory = SizeHistory(t=np.array([0.0, np.inf]), Ne=np.array([1.0]))
        return _viterbi.viterbi_path(
            lwtc=self._lwtc,
            focal=self.focal,
            panel=self.panel,
            eta=eta,
            theta=self.theta,
            rho=self.rho,
            w=self.w,
            robust=self.robust,
            beta=beta,
        )
