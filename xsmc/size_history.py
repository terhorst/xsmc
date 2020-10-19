from typing import NamedTuple

import numpy as np
from scipy.interpolate import PPoly


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

    def simulate(self, **kwargs) -> "tskit.TreeSequence":
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
