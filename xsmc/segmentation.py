from typing import NamedTuple
import numpy as np
from typing import Tuple, List
import scipy.interpolate
import tskit

from scipy.interpolate import PPoly

import logging

logger = logging.getLogger(__name__)


class Segment(NamedTuple):
    """A segment in a haplotype copying path.

    Args:
        hap: the source haplotype for this segment.
        interval: bounds of this segment.
        height: TMRCA of this segment.
        mutations: Number of pairwise differences between `hap` and focal haplotype on this segment.
    """

    hap: int
    interval: Tuple[float, float]
    height: float
    mutations: float


# FIXME too much duplicated code between Segmentation and ArraySegmentation. Abstract to a base class.


class Segmentation(NamedTuple):
    """A path through a haplotype copying model.

    Args:
        segments: list of segments composing the path.
        panel_size: size of panel used to construct this path.
    """

    segments: List[Segment]
    panel: List[int]

    def rescale(self, x: float) -> "Segmentation":
        "Return a new segmentation where each height is multiplied by `x`."
        return self._replace(
            segments=[s._replace(height=s.height * x) for s in self.segments],
            panel=self.panel,
        )

    def draw(self, axis: "matplotlib.axis.Axis" = None, **kwargs) -> None:
        """Plot this segmentation as a step function.

        Args:
            axis: Axis on which to draw plot. If `None`, current axis is used.
            kwargs: Additional arguments passed to `axis.plot()`.

        Notes:
            Only plots segment heights. Segment haplotype is currently ignored.
        """
        if axis is None:
            import matplotlib.pyplot

            axis = matplotlib.pyplot.gca()
        x = [s.interval[0] for s in self.segments]
        y = [s.height for s in self.segments]
        x.append(self.segments[-1].interval[1])
        y.append(y[-1])
        axis.plot(x, y, drawstyle="steps-post", **kwargs)

    def to_pp(self) -> PPoly:
        """Return a piecewise polynomial representation of this segmentation.

        Notes:
            Only represents segments heights. Identity of haplotype panel is currently ignored.
        """
        x = np.array(
            [self.segments[0].interval[0]] + [s.interval[1] for s in self.segments]
        )
        c = np.array([s.height for s in self.segments])[None]
        return PPoly(x=x, c=c)

    @classmethod
    def from_ts(
        cls, ts: tskit.TreeSequence, focal: int, panel: List[int]
    ) -> "Segmentation":
        """Return a segmentation consisting of the genealogical MRCA of `focal` among `panel`.

        Args:
            ts: Tree sequence containing the data.
            focal: Focal leaf node in `ts`.
            panel: Panel leaf nodes in `ts` over which to compute GMRCA.

        Returns:
            Computed segmentation.

        Notes:
            The returned segmentation is not necessarily unique. If there is >1 GMRCA for a given IBD segment, one is
            chosen arbitrarily.
        """
        full_truth = []
        for t in ts.trees():
            mrcas = [t.get_mrca(focal, j) for j in panel]
            tmrcas = [t.get_time(m) for m in mrcas]
            i: int = np.argmin(tmrcas).item()
            # count all mutations beneath mrca. assumes biallely.
            path = set(list(t._postorder_traversal(mrcas[i]))[:-1])
            num_muts = len([m for m in t.mutations() if m.node in path])
            full_truth.append(
                Segment(
                    hap=panel[i],
                    interval=tuple(t.interval),
                    height=tmrcas[i],  # diploid -> haploid
                    mutations=num_muts,
                )
            )
        # compact adjacent ibd segments that copy from same haplotype
        truth = full_truth[:1]
        for seg in full_truth[1:]:
            if seg.hap == truth[-1].hap and seg.height == truth[-1].height:
                truth[-1] = truth[-1]._replace(
                    interval=(truth[-1].interval[0], seg.interval[1])
                )
            else:
                truth.append(seg)
        truth[-1] = truth[-1]._replace(
            interval=(truth[-1].interval[0], ts.get_sequence_length())
        )
        truth = [s._replace(interval=tuple(s.interval)) for s in truth]
        return cls(segments=truth, panel=panel)


class ArraySegmentation(NamedTuple):
    # This has the same interface as Segmentation, but stores the underlying data in an ndarray instead
    # of a list of tuples. This class is better for manipulating large amounts of long paths.
    segments: np.ndarray
    panel_inds: np.ndarray
    panel: List[int]

    @property
    def intervals(self):
        return self.segments[:2].T

    @property
    def heights(self):
        return self.segments[3]

    def __call__(self, x: np.ndarray):
        """Return the values of this segmentation at a position.

        Args:
            x: Array of one or more positions.

        Returns:
            Array of heights at each position.
        """
        x = np.atleast_1d(x)
        return self.heights[np.searchsorted(self.intervals[:, 0], x, side="right") - 1]

    def rescale(self, x: float) -> "ArraySegmentation":
        "Return a new segmentation where each height is multiplied by x."
        return self._replace(
            segments=np.concatenate([self.segments[:3], self.segments[3:] * x], axis=0)
        )

    def draw(self, axis: "matplotlib.axis.Axis" = None, **kwargs) -> None:
        """Plot this segmentation as a step function.

        Args:
            axis: Axis on which to draw plot. If `None`, current axis is used.
            kwargs: Additional arguments passed to `axis.plot()`.
        """
        if axis is None:
            import matplotlib.pyplot

            axis = matplotlib.pyplot.gca()
        x = self.segments[0]
        y = self.segments[3]
        x = np.append(x, self.segments[1, -1])
        y = np.append(y, self.segments[3, -1])
        axis.plot(x, y, drawstyle="steps-post", **kwargs)

    def to_pp(self) -> PPoly:
        x = np.concatenate([self.intervals[0, :1], self.intervals[:, 1]])
        return PPoly(x=x, c=self.heights[None])

    def to_seg(self) -> Segmentation:
        "Convert to a normal segmentation, for readability"
        segments = []
        for i in range(self.segments.shape[1]):
            row = self.segments[:, i]
            segments.append(
                Segment(
                    interval=tuple(row[:2]),
                    panel_ind=self.panel_inds[i],
                    mutations=row[2],
                    height=row[3],
                )
            )
        return Segmentation(segments=segments, panel=self.panel)


class SizeHistory(NamedTuple):
    r"""A piecewise constant size-history function:

    .. math:: \eta(t) = 1 / (N_e)_i,\quad t_i \le t < t_{i+1}

    Args:
        Ne: vector length `T` of effective population sizes.
        t: vector of length `T + 1` of break points, with `t[0] == 0.` and `t[-1] = inf`.
    """

    Ne: np.ndarray
    t: np.ndarray

    @property
    def _p(self) -> PPoly:
        return PPoly(c=1.0 / self.Ne[None], x=self.t)

    @property
    def R(self) -> PPoly:
        return self._p.antiderivative()

    def __call__(self, x: float) -> float:
        return self._p(x)

    def draw(self, axis=None) -> "matplotlib.axes.Axis":
        """Plot this size history.

        Args:
            axis: Axis on which to draw plot. If None, `matplotlib.pyplot.gca()` is used.

        Returns:
            Axis plot was drawn on.

        Note:
            Plots on a log-log scale.
        """
        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()
        axis.plot(self.t[:-1], self.Ne, drawstyle="steps-post")
        axis.set_xscale("log")
        axis.set_yscale("log")
        return axis

    def rescale(self, N0: float) -> "SizeHistory":
        "Return size history scaled by reference effective population size N0."
        return SizeHistory(t=self.t * N0, Ne=self.Ne * N0)

    def simulate(self, **kwargs) -> tskit.TreeSequence:
        """Simulate (using msprime) under demographic represented by self.

        Args:
            **kwargs: Passed through to msprime.simulate().

        Returns:
            Tree sequence containing simulated data.
        """
        import msprime as msp

        de = [
            msp.PopulationParametersChange(time=tt, initial_size=Ne_t)
            for tt, Ne_t in zip(self.t, self.Ne[:-1])
        ]
        kwargs.update({"demographic_events": de})
        return msp.simulate(**kwargs)

    def smooth(self):
        "Return an C2 interpolant of self, suitable for optimization"
        pc = scipy.interpolate.PchipInterpolator(x=self.t[:-1], y=1.0 / self.Ne)
        # the base class of PchipInterpolator seems to have switched from a BPoly to a PPoly at some point
        assert isinstance(pc, scipy.interpolate.PPoly)
        # add a final piece at the back that is constant
        cinf = np.eye(pc.c.shape[0])[:, -1:] * pc.c[-1, -1]
        ret = PPoly(c=np.append(pc.c, cinf, axis=1), x=np.append(pc.x, np.inf))
        return ret
