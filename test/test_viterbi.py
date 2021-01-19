import warnings

import msprime as msp
import numpy as np
import pytest
import tskit
import xsmc._viterbi
from scipy.integrate import quad
from scipy.interpolate import PPoly
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
    assert len(ret["f"]) == 2
    assert len(ret["t"]) == 3
    assert ret["f"][0]["c"][0] == 0
    assert ret["f"][0]["c"][1] == 0
    assert ret["f"][0]["c"][2] == 0
    assert ret["f"][0]["k"] == 1
    assert ret["f"][1]["c"][0] == 1
    assert ret["f"][1]["c"][1] == -5
    assert ret["f"][1]["c"][2] == -5
    assert ret["f"][1]["k"] == 0
    assert ret["t"][0] == -2.0
    np.testing.assert_allclose(ret["t"], [-2.0, -0.6259832407, 2.0])


def test_pmin2():
    # a = -5, b = -5, c = -5
    # function is strictly negative
    ret = xsmc._viterbi.test_pmin([-5, -5, -5], [0, 0, 0], [-2, 2])
    assert len(ret["f"]) == 1
    assert len(ret["t"]) == 2
    assert ret["f"][0]["c"][0] == -5
    assert ret["f"][0]["c"][1] == -5
    assert ret["f"][0]["c"][2] == -5
    assert ret["f"][0]["k"] == 0
    assert ret["t"][0] == -2
    assert ret["t"][1] == 2


def test_pmin_linear():
    ret = xsmc._viterbi.test_pmin([0, -5, -5], [0, 0, 0], [-10, 10])
    assert len(ret["f"]) == 2
    assert ret["f"][0]["c"][0] == 0
    assert ret["f"][0]["c"][1] == 0
    assert ret["f"][0]["c"][2] == 0
    assert ret["f"][1]["c"][0] == 0
    assert ret["f"][1]["c"][1] == -5
    assert ret["f"][1]["c"][2] == -5
    np.testing.assert_allclose(ret["t"], [-10.0, -1.0, 10.0])


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
    assert len(ret["f"]) == 3
    assert ret["f"][0] == ret["f"][2]
    assert ret["f"][0]["c"][0] == -5
    assert ret["f"][0]["c"][1] == -5
    assert ret["f"][0]["c"][2] == 6
    assert ret["f"][1]["c"][0] == 0
    assert ret["f"][1]["c"][1] == 0
    assert ret["f"][1]["c"][2] == 0
    np.testing.assert_allclose(ret["t"], [-2.0, -0.5722498296, 0.7067605762, 2.0])


def test_bc_equal():
    f = [-4, 3, 4]
    g = [-10, 3, 4]
    rho = 1e-4
    ret = xsmc._viterbi.test_pmin(f, g, [-5, 5])
    assert ret["f"][0]["c"][0] == -10
    assert ret["f"][0]["c"][1] == 3
    assert ret["f"][0]["c"][2] == 4


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


def _pwiter(pwf):
    yield from zip(pwf["f"], zip(pwf["t"][:-1], pwf["t"][1:]))


def _pmin_func_test(f, g, t):
    a1, b1, c1 = f
    a2, b2, c2 = g
    rho = 1e-4
    ret = xsmc._viterbi.test_pmin(f, g, t)
    for p, (t0, t1) in _pwiter(ret):
        x = np.linspace(t0, t1, 50)[1:-1]
        s = (-1) ** np.allclose(p["c"], [a1, b1, c1])
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
        for f, (t0, t1) in _pwiter(ret):
            assert f["k"] == 0
            if t0 <= z < t1:
                np.testing.assert_allclose(
                    t,
                    f["c"][0] * np.exp(-z) + (f["c"][1] - 2) * z + f["c"][2],
                )


def test_pwc2():
    a = np.array([1.0, 1.0])
    t = np.array([1.0, np.inf])
    # pi(t) = e^(-t) so -log(pi(t)) = t
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 10
        z = -np.log(t)
        q = t
        for f, (t0, t1) in _pwiter(ret):
            assert f["k"] == 0
            if t0 <= z < t1:
                np.testing.assert_allclose(
                    q,
                    f["c"][0] * np.exp(-z) + (f["c"][1] - 2) * z + f["c"][2],
                )


def test_pwc3():
    a = np.array([1.0, 2.0])
    t = np.array([1.0, np.inf])
    # pi(t) = e^-t, t <= 1
    #       = 2 e^{-(1 + 2 * (t - 1))}, t > 1
    ret = xsmc._viterbi.test_piecewise_const_log_pi(a, t)
    for _ in range(20):
        t = np.random.rand() * 3
        q = t
        if t > 1:
            q = -np.log(2 * np.exp(-(1 + (2 * (t - 1)))))
        z = -np.log(t)
        for f, (t0, t1) in _pwiter(ret):
            assert f["k"] == 0
            if t0 <= z < t1:
                np.testing.assert_allclose(
                    q,
                    f["c"][0] * np.exp(-z) + (f["c"][1] - 2) * z + f["c"][2],
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
        q = t
        if 1 <= t < 2:
            q = -np.log(2 * np.exp(-(1 + (2 * (t - 1)))))
        elif t >= 2:
            q = -np.log(3 * np.exp(-(3 + 3 * (t - 2))))
        z = -np.log(t)
        for f, (t0, t1) in _pwiter(ret):
            assert f["k"] == 0
            if t0 <= z < t1:
                np.testing.assert_allclose(
                    q,
                    f["c"][0] * np.exp(-z) + (f["c"][1] - 2) * z + f["c"][2],
                )


def test_pointwise_min1():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_pointwise_min(prior, 0.0, cost)
    assert len(ret["f"]) == 3
    assert ret["f"][0]["k"] == 1
    assert ret["f"][0]["c"] == [0, 0, 0]
    assert ret["f"][1]["k"] == 3
    assert ret["f"][1]["c"] == [0, 0, 0.5]
    assert ret["f"][2]["k"] == 2
    assert ret["f"][2]["c"] == [0, 0, 1]
    np.testing.assert_allclose(ret["t"], [-np.inf, 0, 1.0, np.inf])


def test_pointwise_min2():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_pointwise_min(prior, 3.0, cost)
    assert len(ret["f"]) == 2
    assert ret["f"][0]["k"] == 3
    assert ret["f"][0]["c"] == [0, 0, 0.5]
    assert ret["f"][1]["k"] == 4
    assert ret["f"][1]["c"] == [0, 0, 2]
    assert ret["t"] == [-np.inf, 1.0, np.inf]


def test_pointwise_min2bis():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_pointwise_min(prior, -1, cost)
    assert len(ret["f"]) == 2
    assert ret["f"][0]["k"] == 1
    assert ret["f"][0]["c"] == [0, 0, -1]
    assert ret["f"][1]["k"] == 2
    assert ret["f"][1]["c"] == [0, 0, 0]
    assert ret["t"] == [-np.inf, 0.0, np.inf]


def test_pointwise_min2cis():
    prior = [
        {"f": [0, 0, 0], "t": [-np.inf, 0], "k": 1},
        {"f": [0, 0, 1], "t": [0.0, np.inf], "k": 2},
    ]
    cost = [
        {"f": [0, 0, 0.5], "t": [-np.inf, 1.0], "k": 3},
        {"f": [0, 0, 2], "t": [1.0, np.inf], "k": 4},
    ]
    ret = xsmc._viterbi.test_pointwise_min(prior, -1, cost)
    assert len(ret["f"]) == 2
    assert ret["f"][0]["k"] == 1
    assert ret["f"][0]["c"] == [0, 0, -1]
    assert ret["f"][1]["k"] == 2
    assert ret["f"][1]["c"] == [0, 0, 0]
    assert ret["t"] == [-np.inf, 0.0, np.inf]


def test_pointwise_min3():
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
    ret = xsmc._viterbi.test_pointwise_min(prior, 3.0, cost)
    assert len(ret["f"]) == 3
    assert ret["f"][0]["k"] == 1
    assert ret["f"][1]["k"] == 2
    assert ret["f"][2]["k"] == 3


def test_pointwise_min4():
    inf = np.inf
    prior = [
        {"f": [0.0, 0.0, inf], "k": 0, "t": [-inf, 0.43363598507486]},
        {
            "f": [1.0, 2.0, 2.3025850929940455],
            "k": 0,
            "t": [0.43363598507486, inf],
        },
    ]
    cost = [
        {
            "f": [14.999999999999979, 37.0, inf],
            "k": 75,
            "t": [-inf, 0.43363598507486],
        },
        {
            "f": [1.2, 3.0, 60.44518822165908],
            "k": 1,
            "t": [0.43363598507486, inf],
        },
    ]
    ret = xsmc._viterbi.test_pointwise_min(prior, 1.0, cost)
    print(ret)


def test_pointwise_min5():
    inf = np.inf
    d = dict(
        [
            (
                "prior",
                [
                    {"f": [0.0, 0.0, inf], "k": 0, "t": [-inf, -0.3788980807234443]},
                    {
                        "f": [1.0, 2.0, 2.3025850929940455],
                        "k": 0,
                        "t": [-0.3788980807234443, inf],
                    },
                ],
            ),
            (
                "cost",
                [
                    {
                        "f": [14.999999999999979, 24.0, inf],
                        "k": 75,
                        "t": [-inf, -0.579818495252942],
                    },
                    {
                        "f": [12.799999999999986, 18.0, 52.13467387946979],
                        "k": 59,
                        "t": [-0.579818495252942, -0.3788980807234443],
                    },
                    {
                        "f": [1.2, 2.0, 58.14260312866504],
                        "k": 1,
                        "t": [2.1972245773362213, inf],
                    },
                ],
            ),
        ]
    )
    ret = xsmc._viterbi.test_pointwise_min(d["prior"], 1.0, d["cost"])


def test_pointwise_min6():
    inf = np.inf
    kwargs = dict(
        [
            (
                "prior",
                {
                    "f": [
                        {"c": [0.0, 0.0, inf], "k": 0},
                        {"c": [1.0, 2.0, 2.3025850929940455], "k": 0},
                    ],
                    "t": [-inf, 0.43363598507486, inf],
                },
            ),
            ("F_t", 51.34866148694316),
            (
                "cost",
                {
                    "f": [
                        {"c": [13.999999999999982, 33.0, inf], "k": 70},
                        {"c": [1.2, 2.0, 53.62902435771498], "k": 1},
                    ],
                    "t": [-inf, 0.43363598507486, inf],
                },
            )
        ]
    )
    xsmc._viterbi.test_pointwise_min_new_fmt(**kwargs)
    
def test_pointwise_min7():
    inf = np.inf
    kwargs = dict([
    ('prior', {'f': [{'c': [0.0, 0.0, inf], 'k': 0}, {'c': [1.0, 2.0, 2.3025850929940455], 'k'
    : 0}], 't': [-inf, -0.37889808072344427, inf]}),
    ('F_t', 51.34866148694316),
    ('cost', {'f': [{'c': [13.999999999999982, 23.0, inf], 'k': 70},
                    {'c': [3.0000000000000004, 3, 51.231799780378758], 'k': 0},
                    {'c': [1.3999999999999999, 2, 53.606802135492757], 'k': 2},
                    {'c': [2.6000000000000001, 2, 53.473468802159424], 'k': 8},
                    {'c': [2.8000000000000003, 2, 53.451246579937205], 'k': 9}],
                   't': [-inf, -0.57981849525294205, 
    -0.37889808072344427, 2.1972245773362213, 2.1972245773362227, 2.1972245773363825, inf]})
     ])
    xsmc._viterbi.test_pointwise_min_new_fmt(**kwargs)



def test_min_f_bug_20200808():
    c = [1001.0, 0.0, 0.0]
    t = [3.546621830333237, np.inf]
    m = xsmc._viterbi.test_min_f(c, t)
    assert not np.isnan(m["f"])
    assert m["f"] == 0.0


def test_compact_1():
    inf = np.inf
    func = {
        "f": [
            {"c": [0.0, 0.0, inf], "k": 0},
            {"c": [0.0, 1.0, 2.0], "k": 0},
            {"c": [1.0, 2.0, 3.0], "k": 10},
        ],
        "t": [-inf, 0.6931471805599453, 1.0, inf],
    }
    c = xsmc._viterbi.test_compact(func)
    assert c == func


def test_compact_2():
    inf = np.inf
    func = {
        "f": [
            {"c": [0.0, 0.0, inf], "k": 0},
            {"c": [0.0, 1.0, 2.0], "k": 0},
            {"c": [0.0, 1.0, 2.0], "k": 0},
            {"c": [1.0, 2.0, 3.0], "k": 10},
        ],
        "t": [-inf, 0.6931471805599453, 0.75, 1.0, inf],
    }
    c = xsmc._viterbi.test_compact(func)
    assert c["t"] == func["t"][:2] + func["t"][3:]


def test_compact_3():
    inf = np.inf
    func = {
        "f": [
            {"c": [0.0, 0.0, inf], "k": 0},
            {"c": [0.0, 1.0, 2.0], "k": 0},
            {"c": [0.0, 1.0, 2.0], "k": 0},
            {"c": [1.0, 2.0, 3.0], "k": 10},
        ],
        "t": [-inf, 0.75, 0.75, 1.0, inf],
    }
    c = xsmc._viterbi.test_compact(func)
    assert c["t"] == [-inf, 0.75, 1.0, inf]
    assert c["f"][0] == func["f"][0]
    assert c["f"][1] == func["f"][1]
    assert c["f"][2] == func["f"][3]


def test_truncate_prior():
    prior = {
        "f": [{"c": [0.0, 1.0, 2.0], "k": 0}, {"c": [1.0, 2.0, 3.0], "k": 10}],
        "t": [-np.inf, 1.0, np.inf],
    }
    tau = 0.5
    trunc = xsmc._viterbi.test_truncate_prior(prior, tau)
    assert np.isinf(trunc["f"][0]["c"][2])
    assert trunc["t"][:2] == [-np.inf, -np.log(tau)]
    assert trunc["t"][2:] == prior["t"][1:]
    assert trunc["f"][1:] == prior["f"]


def test_truncate_prior_exact_edge():
    prior = {
        "f": [{"c": [0.0, 1.0, 2.0], "k": 0}, {"c": [1.0, 2.0, 3.0], "k": 10}],
        "t": [-np.inf, 1.0, np.inf],
    }
    tau = np.exp(-1.0)
    trunc = xsmc._viterbi.test_truncate_prior(prior, tau)
    assert np.isinf(trunc["f"][0]["c"][2])
    assert trunc["t"][:2] == [-np.inf, -np.log(tau)]
    assert trunc["t"] == prior["t"]
    assert trunc["f"][1] == prior["f"][1]
    assert len(trunc["f"]) == 2
