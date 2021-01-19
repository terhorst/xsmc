"Exact Bayesian and frequentist decoding of the sequentially Markov coalescent"

import logging
from dataclasses import dataclass
from typing import List, Union

import numpy as np

import tskit
import xsmc._sampler
import xsmc.arg

from . import _viterbi
from .sampler import XSMCSampler
from .segmentation import ArraySegmentation, Segmentation
from .size_history import KINGMAN, SizeHistory
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
            adjacent bins, but not within them.
        robust: If True, use robust model in which bins are classified as either segregating or nonsegregating, as in PSMC.
            Otherwise, allow for any number of mutations per bin.

    Notes:
        The height of segments returned by this class are expressed in coalescent units. To convert to generations,
        rescale them by `self.theta / (4 * mu)`, where `mu` is the biological mutation rate.
    """
    ts: tskit.TreeSequence
    theta: float = None
    rho_over_theta: float = 1.0
    w: int = None
    robust: bool = False
    eta: SizeHistory = KINGMAN

    def __post_init__(self):
        if self.theta is None:
            self.theta = watterson(self.ts)
            logger.debug("Estimated θ=%f", self.theta)
        self.rho = self.theta * self.rho_over_theta
        if self.w is None:
            if self.rho == 0.0:
                logger.warning("Got rho=0; is this really what you want?")
                self.w = 100
            else:
                self.w = int(1.0 / (10 * self.rho))
            logger.debug("Setting window size w=%f", self.w)
        self._sampler = None

    @property
    def L(self):
        return self.ts.get_sequence_length()

    @property
    def H(self):
        return len(self.panel)

    def sampler(self, focal, panel):
        X = xsmc._sampler.get_mismatches(self.ts, focal, panel, self.w)
        deltas = np.ones_like(X[0])
        # Perform sampling
        assert deltas.shape[0] == X.shape[1]
        return XSMCSampler(
            X=X,
            deltas=deltas,
            theta=self.w * self.theta,
            rho=self.w * self.rho,
            robust=self.robust,
            eps=1e-4,
        )

    def sample(
        self, focal: int, panel: List[int], k: int = 1, seed: int = None
    ) -> List[Segmentation]:
        r"""Sample path(s) from the posterior distribution.

        Args:
            k: Number of posterior path samples to draw, default 1.
            seed: Random seed used for sampling.

        Returns:
            A list of `k` posterior samples for the positional min-TMRCA of `focal` with `panel`.

        Notes:
            If sampling many paths at once, it is more efficient to set `k > 1` than to call `sample_paths()`
            repeatedly.
        """
        prime = False
        segs = self.sampler(focal, panel).sample_paths(k, seed, prime)
        ret = [
            ArraySegmentation(segments=s, panel_inds=p, focal=focal, panel=panel)
            for s, p in segs
        ]
        for a in ret:
            a.segments[:2] *= self.w  # expand intervals by w
        return ret

    def sample_heights(self, j: int, k: int, seed: Union[int, None]) -> np.ndarray:
        return self.sampler.sample_heights(j, k, seed)

    def viterbi(
        self,
        focal: int,
        panel: List[int],
        beta: float = None,
    ) -> Segmentation:
        """Compute the maximum *a posteriori* (a.k.a. Viterbi) path in haplotype copying model.

        Args:
            beta: Penalty parameter for changepoint formation. See manuscript for details about this parameter.

        Returns:
            A segmentation containing the MAP path.
        """
        trunk = xsmc.arg.make_trunk(panel, 1 + self.L // self.w)
        return _viterbi.viterbi_path(
            self.ts,
            focal,
            panel,
            trunk,
            self.eta,
            self.theta,
            self.rho,
            beta,
            self.robust,
            self.w,
        )

    def arg(
        self,
        haps: List[int],
    ) -> tskit.TreeSequence:
        """
        Construct an ancestral recombination graph by iteratively threading haps onto tree sequence.

        Args:
            ts: The tree sequence containing the variation data.
            haps: A list of samples (leaf nodes) in ts for which to build the ARG.

        Returns:
            A tree sequence containing the estimated ARG.

        Notes:
            The output depends on the order of haps. First, haps[1] will be "threaded" onto haps[0] by estimating
            the local TMRCA at each position. (Equivalent to PSMC). Then, haps[2] will be threaded onto the
            (haps[0], haps[1]) tree sequence. And so forth.

            The length of the returned tree sequence will be ts.get_sequence_length() // self.w.
        """
        scaffold = xsmc.arg.make_trunk(
            [haps[0]], self.ts.get_sequence_length() // self.w
        )
        for i in range(1, len(haps)):
            panel = haps[:i]
            seg = _viterbi.viterbi_path(
                self.ts,
                haps[i],
                panel,
                scaffold,
                self.eta,
                self.theta,
                self.rho,
                beta=None,
                robust=False,
                w=self.w,
            )
            scaffold = xsmc.arg.thread(scaffold, seg)
        return scaffold
