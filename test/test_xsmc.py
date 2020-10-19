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
