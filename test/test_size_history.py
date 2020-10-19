import msprime as msp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PPoly

import xsmc
from xsmc.size_history import SizeHistory


@pytest.fixture
def eta():
    return SizeHistory(t=np.array([np.inf]), Ne=np.array([1.0]))


@pytest.fixture
def rnd_eta():
    T = 10
    t = np.r_[0.0, np.cumsum(np.random.rand(T - 1)), np.inf]
    Ne = np.random.rand(T)
    return SizeHistory(t=t, Ne=Ne)


def test_size_history_call(rnd_eta):
    p = rnd_eta
    q = PPoly(x=p.t, c=[1.0 / p.Ne])
    for t in np.random.rand(100) * len(p.t):
        assert abs(p(t) - q(t)) < 1e-4


def test_size_history_R(rnd_eta):
    p = rnd_eta
    q = PPoly(x=p.t, c=[1.0 / p.Ne]).antiderivative()
    for t in np.random.rand(100) * len(p.t):
        assert abs(p.R(t) - q(t)) < 1e-4
    for t1, t2 in np.random.rand(100, 2) * len(p.t):
        assert abs((p.R(t1 + t2) - p.R(t1)) - (q(t1 + t2) - q(t1))) < 1e-4
