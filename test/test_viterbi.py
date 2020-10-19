import warnings

import msprime as msp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PPoly

import tskit
import xsmc._viterbi
from xsmc.size_history import SizeHistory


def test_root_bug1():
    k = 0
    a = 22.22652704459693495664
    b = -17.00000000000000000000
    c = -17650.27904568414669483900
    x = xsmc._viterbi.test_root(k, a, b, c)
    assert np.isfinite(x)
    np.testing.assert_allclose(x, -6.670774648388571, atol=0.01)
    # np.testing.assert_allclose(a * np.exp(-x) + b * x + c, 0.)


def test_root_bug2():
    a = 0.118924
    b = 1.000000
    c = -992.595175
    x0 = xsmc._viterbi.test_root(0, a, b, c)
    xm1 = xsmc._viterbi.test_root(-1, a, b, c)
    np.testing.assert_allclose(x0, -c)
    np.testing.assert_allclose(xm1, -9.031736)


def test_pmin1():
    # a = 1, b = -5, c = -5
    ret = xsmc._viterbi.test_pmin([1, -5, -5], [0, 0, 0], [-2, 2])
    assert len(ret) == 2
    np.testing.assert_allclose(ret[0]["t"][1], -0.6259832407)
    np.testing.assert_allclose(ret[1]["t"][0], -0.6259832407)
    assert ret[0]["f"]["c"][0] == 0
    assert ret[0]["f"]["c"][1] == 0
    assert ret[0]["f"]["c"][2] == 0
    assert ret[0]["f"]["k"] == 1
    assert ret[1]["f"]["c"][0] == 1
    assert ret[1]["f"]["c"][1] == -5
    assert ret[1]["f"]["c"][2] == -5
    assert ret[1]["f"]["k"] == 0
    assert ret[0]["t"][0] == -2.0
    assert ret[1]["t"][1] == 2.0


def test_pmin2():
    # a = -5, b = -5, c = -5
    # function is strictly negative
    ret = xsmc._viterbi.test_pmin([-5, -5, -5], [0, 0, 0], [-2, 2])
    assert len(ret) == 1
    assert ret[0]["f"]["c"][0] == -5
    assert ret[0]["f"]["c"][1] == -5
    assert ret[0]["f"]["c"][2] == -5
    assert ret[0]["f"]["k"] == 0
    assert ret[0]["t"][0] == -2
    assert ret[0]["t"][1] == 2


def test_pmin_linear():
    ret = xsmc._viterbi.test_pmin([0, -5, -5], [0, 0, 0], [-10, 10])
    assert len(ret) == 2
    assert ret[0]["f"]["c"][0] == 0
    assert ret[0]["f"]["c"][1] == 0
    assert ret[0]["f"]["c"][2] == 0
    assert ret[1]["f"]["c"][0] == 0
    assert ret[1]["f"]["c"][1] == -5
    assert ret[1]["f"]["c"][2] == -5
    np.testing.assert_allclose(ret[0]["t"][1], -1.0)


def test_pmin_rnd():
    ret = xsmc._viterbi.test_pmin([0, -5, -5], [0, 0, 0], [-10, 10])


def test_pmin_inf():
    ret = xsmc._viterbi.test_pmin([-5, -5, np.inf], [0, 0, 0], [-2, -2])
    # assert doesn't raise


def test_pmin_bug1():
    inf = np.inf
    fns = [
        {
            "f": {"c": [2.134610447370598, 2.0, 179.57125436908768], "k": 1},
            "t": [-inf, -2.449939611803787],
        },
        {
            "f": {"c": [8.063373232524182, 19.0, 152.51981781846845], "k": 18},
            "t": [-2.449939611803787, -2.302585092994046],
        },
        {
            "f": {"c": [7.0633732326241825, 19.0, 185.54566873405744], "k": 18},
            "t": [-2.302585092994046, 1.496518285540267],
        },
        {
            "f": {"c": [1.5521831861288637, 3.0, 210.7239629910881], "k": 2},
            "t": [1.496518285540267, 1.8044225753082512],
        },
        {
            "f": {"c": [1.1346104474705976, 2.0, 212.5971052846767], "k": 1},
            "t": [1.8044225753082512, inf],
        },
    ]
    for fn in fns:
        # assert does not fail
        m = xsmc._viterbi.test_min_f(fn["f"]["c"], fn["t"])


def test_pmin_bug2():
    inf = np.inf
    prior = {"f": {"c": [1.0, 1.0, 2001.1123679415307], "k": 0}, "t": [-inf, inf]}
    cost = {
        "f": {"c": [1.118924483700997, 2.0, 1008.5171931914163], "k": 1},
        "t": [-inf, inf],
    }
    ret = xsmc._viterbi.test_pmin(prior["f"]["c"], cost["f"]["c"], prior["t"])
    assert len(ret) > 0


def test_pmin_convex():
    ret = xsmc._viterbi.test_pmin([-5, -5, 6], [0, 0, 0], [-2, 2])
    assert len(ret) == 3
    assert ret[0]["f"] == ret[2]["f"]
    assert ret[0]["f"]["c"][0] == -5
    assert ret[0]["f"]["c"][1] == -5
    assert ret[0]["f"]["c"][2] == 6
    assert ret[1]["f"]["c"][0] == 0
    assert ret[1]["f"]["c"][1] == 0
    assert ret[1]["f"]["c"][2] == 0
    np.testing.assert_allclose(ret[0]["t"][1], -0.5722498296)
    np.testing.assert_allclose(ret[1]["t"][1], 0.7067605762)


def test_bc_equal():
    f = [-4, 3, 4]
    g = [-10, 3, 4]
    rho = 1e-4
    ret = xsmc._viterbi.test_pmin(f, g, [-5, 5])
    assert ret[0]["f"]["c"][0] == -10
    assert ret[0]["f"]["c"][1] == 3
    assert ret[0]["f"]["c"][2] == 4


def test_min_f_bug1():
    f = [12.000000, 13.000000, 6.173786]
    ret = xsmc._viterbi.test_min_f(f, [-1.129216, 1.857802])
    np.testing.assert_allclose([ret["f"], ret["x"]], [18.133231, -0.0800427])


def test_min_f_random():
    from scipy.optimize import minimize_scalar

    for _ in range(20):
        bounds = sorted(np.random.normal(size=2, scale=3))
        a, b, c = np.random.normal(size=3, scale=3)

        def f(x):
            return a * np.exp(-x) + b * x + c

        opt = minimize_scalar(
            f, method="bounded", bounds=bounds, tol=1e-10, options=dict(xatol=1e-10)
        )
        ret = xsmc._viterbi.test_min_f([a, b, c], bounds)
        assert bounds[0] <= ret["x"] <= bounds[1]
        # ret = sorted([dict(f=f(x), x=x) for x in list(bounds) + [opt['x']]], key=lambda d: d['f'])[0]
        try:
            np.testing.assert_allclose(
                [ret["f"], ret["x"]], [opt["fun"], opt["x"]], atol=1e-6, rtol=1e-6
            )
        except AssertionError:
            # minimize_scalar() does not always test the endpoints
            assert ret["f"] < opt["fun"]
            assert np.isclose(ret["x"], bounds[0]) or np.isclose(ret["x"], bounds[1])
            warnings.warn("minimize_scalar() missed the endpoint")


def _pmin_func_test(f, g, t):
    a1, b1, c1 = f
    a2, b2, c2 = g
    rho = 1e-4
    ret = xsmc._viterbi.test_pmin(f, g, t)
    for p in ret:
        x = np.linspace(p["t"][0], p["t"][1], 12)[1:-1]
        s = (-1) ** np.allclose(p["f"]["c"], [a1, b1, c1])
        np.testing.assert_array_less(
            -s * ((rho + (a1 - 1) * (1.0 + rho)) * np.exp(-x) + b1 * x + c1),
            -s * ((rho + (a2 - 1) * (1.0 + rho)) * np.exp(-x) + b2 * x + c2),
        )


def test_pmin_random():
    for _ in range(100):
        f, g = np.random.randint(-10, 10, size=(2, 3))
        _pmin_func_test(f, g, [-5, 5])


def test_pmin_bug_1():
    f = [0, -2, -2]
    g = [-5, -4, 4]
    _pmin_func_test(f, g, [-5, 5])


def test_pwc1():
    a = np.array([1.0])
    t = np.array([np.inf])
    # pi(t) = e^(-t) so -log(pi(t)) = t
    # => -log(pi(z)) = z
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 10
        z = -np.log(t)
        for p in ret:
            assert p["f"]["k"] == 0
            if p["t"][0] <= z < p["t"][1]:
                np.testing.assert_allclose(
                    t,
                    p["f"]["c"][0] * np.exp(-z)
                    + (p["f"]["c"][1] - 2) * z
                    + p["f"]["c"][2],
                )


def test_pwc2():
    a = np.array([1.0, 1.0])
    t = np.array([1.0, np.inf])
    # pi(t) = e^(-t) so -log(pi(t)) = t
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 10
        z = -np.log(t)
        for p in ret:
            assert p["f"]["k"] == 0
            if p["t"][0] <= z < p["t"][1]:
                np.testing.assert_allclose(
                    t,
                    p["f"]["c"][0] * np.exp(-z)
                    + (p["f"]["c"][1] - 2) * z
                    + p["f"]["c"][2],
                )


def test_pwc3():
    a = np.array([1.0, 2.0])
    t = np.array([1.0, np.inf])
    # pi(t) = e^-t, t <= 1
    #       = 2 e^{-(1 + 2 * (t - 1))}, t > 1
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 3
        t0 = t
        if t > 1:
            t0 = -np.log(2 * np.exp(-(1 + (2 * (t - 1)))))
        z = -np.log(t)
        for p in ret:
            assert p["f"]["k"] == 0
            if p["t"][0] <= z < p["t"][1]:
                np.testing.assert_allclose(
                    t0,
                    p["f"]["c"][0] * np.exp(-z)
                    + (p["f"]["c"][1] - 2) * z
                    + p["f"]["c"][2],
                )


def test_pwc4():
    a = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 2.0, np.inf])
    # pi(t) = e^-t, t <= 1
    #       = 2 e^{-[1 + (2 * (t - 1))]}, 1 <= t < 2
    #       = 3 e^{-[3 + (3 * (t - 2))]}, t > 3
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 5
        t0 = t
        if 1 <= t < 2:
            t0 = -np.log(2 * np.exp(-(1 + (2 * (t - 1)))))
        elif t >= 2:
            t0 = -np.log(3 * np.exp(-(3 + 3 * (t - 2))))
        z = -np.log(t)
        for p in ret:
            assert p["f"]["k"] == 0
            if p["t"][0] <= z < p["t"][1]:
                np.testing.assert_allclose(
                    t0,
                    p["f"]["c"][0] * np.exp(-z)
                    + (p["f"]["c"][1] - 2) * z
                    + p["f"]["c"][2],
                )


def test_piecewise_min1():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_piecewise_min(prior, 0.0, cost)
    assert len(ret) == 3
    assert ret[0]["f"]["k"] == 1
    assert ret[0]["f"]["c"] == [0, 0, 0]
    assert ret[0]["t"] == [-np.inf, 0]
    assert ret[1]["f"]["k"] == 3
    assert ret[1]["f"]["c"] == [0, 0, 0.5]
    assert ret[1]["t"] == [0.0, 1.0]
    assert ret[2]["f"]["k"] == 2
    assert ret[2]["f"]["c"] == [0, 0, 1]
    assert ret[2]["t"] == [1.0, np.inf]


def test_piecewise_min2():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_piecewise_min(prior, 3.0, cost)
    assert len(ret) == 2
    assert ret[0]["f"]["k"] == 3
    assert ret[0]["f"]["c"] == [0, 0, 0.5]
    assert ret[0]["t"] == [-np.inf, 1.0]
    assert ret[1]["f"]["k"] == 4
    assert ret[1]["f"]["c"] == [0, 0, 2]
    assert ret[1]["t"] == [1.0, np.inf]


def test_piecewise_min3():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, 1.0], "k": 2},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 3},
    ]
    cost = [
        {"f": [0, 0, 3], "t": [-np.inf, 0.0], "k": 4},
        {"f": [0, 0, 4], "t": [0.0, 1.0], "k": 5},
        {"f": [0, 0, 5], "t": [1.0, np.inf], "k": 6},
    ]
    ret = xsmc._viterbi.test_piecewise_min(prior, 3.0, cost)
    assert len(ret) == 3
    assert ret[0]["f"]["k"] == 1
    assert ret[1]["f"]["k"] == 2
    assert ret[2]["f"]["k"] == 3


def test_min_f_bug_20200808():
    c = [1001.0, 0.0, 0.0]
    t = [3.546621830333237, np.inf]
    m = xsmc._viterbi.test_min_f(c, t)
    assert not np.isnan(m["f"])
    assert m["f"] == 0.0
