import msprime as msp
import numpy as np
import pytest

import xsmc

@pytest.fixture
def ts():
    return msp.simulate(
        length=1e6, mutation_rate=1e-4, recombination_rate=1e-4, sample_size=10
    )


@pytest.fixture
def x(ts):
    return xsmc.XSMC(ts)
    

def test_sample_paths_1(x):
    return x.sample(0, [1], k=100)

def test_sample_paths_2(x):
    x.sample(0, [1, 2], k=100)


def test_viterbi_1(x):
    x.viterbi(0, [1])

def test_arg_1(x):
    return x.arg([2, 3, 1])