import msprime as msp
import numpy as np
import pytest

import xsmc


@pytest.fixture
def ts():
    return msp.simulate(
        length=1e6, mutation_rate=1e-4, recombination_rate=1e-4, sample_size=10
    )


def test_sample_paths_1(ts):
    xsmc.XSMC(ts, 0, [1]).sample(k=100)


def test_sample_paths_2(ts):
    xsmc.XSMC(ts, 0, [1, 2]).sample(k=100)


def test_viterbi(ts):
    xsmc.XSMC(ts, 0, [1, 2]).viterbi()


def test_viterbi_100():
    data = msp.simulate(
        sample_size=100, length=1e8, mutation_rate=1.4e-8, recombination_rate=1e-9
    )
    xs = xsmc.XSMC(data, 0, list(range(1, 100)))
    N0 = xs.theta / (4 * 1e-8)
    map_path = xs.viterbi()
