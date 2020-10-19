import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import msprime as msp
import numpy as np
import pytest
from scipy.special import logsumexp

import xsmc
from xsmc.sampler import XSMCSampler


def prior(L, rho):
    Y = np.zeros(L)
    ell = 0
    Ys = []
    while ell < L:
        a = np.random.exponential()
        t = np.floor(np.random.exponential() / (rho * a)).astype(int)
        Y[ell : min(ell + t, L)] = a
        Ys.append([ell, a])
        ell += t
    Ys.append((L, Ys[-1][1]))
    return np.array(Y), np.array(Ys)


def generate_data(L, theta, rho, seed):
    # draw from prior with seed i
    np.random.seed(seed)
    Y, Ys = prior(L, rho)
    X = np.random.poisson(theta * Y)
    return X, Y


theta = rho = 1.0


def test_Q_1():
    X = np.array([[1]])
    deltas = [1]
    es = XSMCSampler(X, deltas, theta, rho, False)
    p_X = es.log_P(0, 1)
    np.testing.assert_allclose(p_X, es.log_Q[0])


def test_Q_2():
    Xcs = np.cumsum([[2, 1]], 1)
    deltas = [1, 5]
    es = XSMCSampler(Xcs, deltas, theta, rho, True)
    p_X = logsumexp([es.log_P(0, 2), es.log_P(0, 1) + es.log_P(1, 2)])
    np.testing.assert_allclose(p_X, es.log_Q[0])


@pytest.fixture
def big_data():
    return msp.simulate(
        sample_size=50,
        recombination_rate=1e-8,
        mutation_rate=1e-9,
        length=1e5,
        Ne=1e4,
    )


@pytest.fixture
def data():
    import msprime as msp

    import xsmc

    return msp.simulate(
        sample_size=4,
        recombination_rate=1e-9,
        mutation_rate=1.4e-8,
        length=1e6,
        Ne=1e4,
        random_seed=1,
    )


@pytest.fixture
def tiny_data():
    import msprime as msp

    import xsmc

    return msp.simulate(
        sample_size=4,
        recombination_rate=1e-9,
        mutation_rate=1.4e-8,
        length=100 * 10,
        Ne=1e4,
        random_seed=1,
    )


def test_theta_equals_0(big_data):
    with pytest.raises(ValueError):
        p = xsmc.XSMC(big_data, 0, [1, 2, 3], theta=0.0, rho_over_theta=0.0).sample(
            k=5, seed=1
        )


def test_rho_equals_0(big_data):
    with pytest.raises(ValueError):
        p = xsmc.XSMC(big_data, 0, [1, 2, 3], theta=1, rho_over_theta=0.0).sample(
            k=5, seed=1
        )


def test_multiple_panel(data):
    p = xsmc.XSMC(data, 0, [1, 2, 3], theta=1, rho_over_theta=1.0).sample(k=1, seed=1)
    # print(p.segments[1])


def test_multithreaded(big_data):
    from concurrent.futures import ThreadPoolExecutor

    xs = [
        xsmc.XSMC(big_data, 2 * i, [2 * i + 1])
        for i in range(big_data.get_sample_size() // 2)
    ]
    with ThreadPoolExecutor() as p:
        futs = [p.submit(x.sample, k=100, seed=i) for i, x in enumerate(xs)]
        res = [f.result() for f in futs]


def test_bug():
    data = msp.simulate(
        sample_size=40,
        recombination_rate=1.4e-8,
        mutation_rate=1.4e-8,
        length=1e5,
        Ne=1e4,
        demographic_events=[],
        random_seed=1,
    )
    L = data.get_sequence_length()
    focal = 0
    panel = list(range(1, 19))
    xs = [xsmc.XSMC(data, focal, panel, rho_over_theta=1.0, w=100, robust=True)]
    with ThreadPoolExecutor() as p:
        futs = [p.submit(x.sample, k=1, seed=1) for i, x in enumerate(xs)]
        paths = [f.result() for f in futs]


def _compositions(n, k):
    if n < 0 or k < 0:
        return
    elif k == 0:
        # the empty sum, by convention, is zero, so only return something if
        # n is zero
        if n == 0:
            yield []
        return
    elif k == 1:
        yield [n]
        return
    else:
        for i in range(0, n + 1):
            for comp in _compositions(n - i, k - 1):
                yield [i] + comp


def strong_compositions(n, k):
    for c in _compositions(n, k):
        if 0 not in c:
            yield c


@pytest.fixture
def sampler():
    import itertools

    from scipy.special import logsumexp

    from xsmc.sampler import XSMCSampler

    H = 5
    n = 5
    robust = False
    X = np.random.randint(0, 1, size=(H, n))
    deltas = np.ones(n)
    sampler = XSMCSampler(X, deltas, 0.01, 0.01, True, 0.0)
    return sampler


def test_log_Q_last(sampler):
    n = sampler.n
    np.testing.assert_allclose(sampler.log_Q[n - 1], logsumexp(sampler.log_P(n - 1, n)))


def test_log_Q_exact(sampler):
    n = sampler.n
    p = -np.inf
    for k in range(1, n + 1):
        for c in strong_compositions(n, k):
            q = 0.0
            s = 0
            for j in c:
                q += logsumexp(sampler.log_P(s, s + j))
                s += j
            p = np.logaddexp(p, q)
    np.testing.assert_allclose(p, sampler.log_Q[0])


def test_memory_leak(tiny_data):
    import logging

    # import tracemalloc
    # tracemalloc.start()
    # logger = logging.getLogger('xsmc')
    # logger.info('tracing')
    # start = tracemalloc.take_snapshot()
    for _ in range(4):
        focal = 0
        panel = list(range(1, 2))
        x = xsmc.XSMC(tiny_data, focal, panel, rho_over_theta=1.0, w=100, robust=True)
        s = x.sample(k=100, seed=1)
        # current = tracemalloc.take_snapshot()
        # stats = current.compare_to(start, 'filename')
        # for i, stat in enumerate(stats[:5], 1):
        #     logger.info('since_start %d %s', i, str(stat))
        # start = current
