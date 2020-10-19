# cython: bounll_dscheck=False
# cython: cdivision=True
# cython: language=c++
# distutils: extra_compile_args=['-O3', '-Wno-unused-but-set-variable', '-ffast-math']

from typing import List, NamedTuple

import numpy as np

import xsmc._tskit

from .segmentation import Segment, Segmentation
from .size_history import SizeHistory


cdef struct obs_iter:
    int ell, L, w  # current position, total obs, window size
    tsk_vargen_t* vg
    tsk_variant_t *var
    int err
    int32_t* mismatches

cdef int get_next_obs(obs_iter *state) nogil:
    cdef int h
    cdef int np1 = state.vg.num_samples
    cdef int n = np1 - 1
    cdef int8_t* gt = state.var.genotypes.i8
    memset(state.mismatches, 0, sizeof(int32_t) * n)
    if state.ell > state.L:
        return 0
    while (state.err == 1) and (state.var.site.position / state.w < 1 + state.ell):
        for h in range(1, n + 1):
            state.mismatches[h - 1] += <int32_t>(gt[0] != gt[h])
        state.err = tsk_vargen_next(state.vg, &state.var)
    state.ell += 1
    return 1

def viterbi_path(LightweightTableCollection lwtc,
                 focal: int,
                 panel: List[int],
                 eta: SizeHistory,
                 theta: float,
                 rho: float,
                 beta,
                 robust: bool,
                 w: int,
                 ) -> Segmentation:
    '''Compute optimal segmentation for renewal approximation to sequentially
    Markov coalescent.

    Args:
        ts: Tree sequence containing the data to segment.
        focal: Leaf index of the focal haplotype.
        panel: Leaf indices of the panel haplotypes.
        eta: Size history to use for coalescent prior. If None, Kingman's
            coalescent is used.
        theta: Population-scaled mutation rate per bp per generation.
        rho: Population-scaled recombination rate per bp per generation.
        beta: Penalization parameter. If None, defaults to :math:`-log(\rho)`, resulting in the MAP path.
    Returns:
        The maximum *a posteriori* segmentation under specified parameters.
    '''
    assert theta > 0
    assert rho > 0
    cdef piecewise_func q
    cdef double rho_ = w * rho
    cdef double theta_ = w * theta
    if beta is None:
        beta = -np.log(rho_)
    cdef double beta_ = beta
    cdef double log_theta = log(theta_)
    cdef int i, j, k, y_i

    if focal in panel:
        raise ValueError("focal haplotype cannot be a member of panel")
    if len(panel) == 0:
        raise ValueError("empty panel")

    cdef unordered_map[tsk_id_t, int] panel_
    for i, tid in enumerate(panel):
        panel_[tid] = i

    # Minimal cost over each haplotype
    cdef int n = len(panel)
    cdef minimum F_t_j
    cdef vector[backtrace] F_t
    F_t.resize(n)

    cdef tsk_treeseq_t ts
    cdef int err
    err = tsk_treeseq_init(&ts, lwtc.tables, TSK_BUILD_INDEXES)
    assert err == 0
    cdef double L = tsk_treeseq_get_sequence_length(&ts)
    cdef tsk_size_t S = tsk_treeseq_get_num_sites(&ts);

    # cp is the backtrace list of changepoints: (pos, hap)
    cdef backtrace b
    cdef vector[backtrace] cp
    cdef vector[backtrace] bt
    cp.reserve(S)

    assert eta.t[0] == 0.
    assert np.isinf(eta.t[-1])

    assert np.all(eta.Ne > 0), "Ne cannot be 0."
    assert np.all(np.isfinite(eta.Ne)), "Ne must be finite"
    coal_rate = np.array([eta(tt) for tt in eta.t[:-1]])
    cdef vector[piecewise_func] prior = piecewise_const_log_pi(
        coal_rate.astype(np.float64),
        eta.t[1:].astype(np.float64),
        beta_
    )
    # i_p += 1
    # C are the cost functions for each haplotype
    cdef vector[vector[piecewise_func]] C
    cdef piecewise_func f
    for j in range(n):
        C.push_back(prior)
        # subtract off one x because the first prior is not length-biased
        for k in range(C[j].size()):
            C[j][k].f.c[1] -= 1.

    cdef vector[int] positions
    cdef int pos = 0
    positions.push_back(pos)
    cdef int L_w = <int>(L // w)
    i = -1

    # Initialize the variant generator for our sample. the focal haplotype has
    # genotype index 0, and the panel haps have genotype indices 1, ..., n + 1
    cdef obs_iter state
    state.w = w
    state.L = L_w
    state.ell = 0
    state.err = 1
    assert n + 1 == state.vg.num_samples
    cdef int32_t[:] mismatches = np.zeros(n, dtype=np.int32)
    state.mismatches = &mismatches[0]

    cdef tsk_id_t[:] samples = np.array([focal] + list(panel), dtype=np.int32)
    err = tsk_vargen_init(state.vg, &ts, &samples[0], 1 + len(panel), NULL, 0)
    assert err == 0
    state.err = tsk_vargen_next(state.vg, &state.var)
    assert state.err == 0
    err = get_next_obs(&state)
    with nogil:
        while err == 1:
            # proceed to next variant
            # loop1: increment all the current ibd tracts
            i += 1
            pos = state.ell
            delta = pos - positions.back()  # = 1
            positions.push_back(pos)
            for j in range(n):
                for k in range(C[j].size()):
                    if robust:
                        # bernoulli obs, which adds contribution 
                        # (1-min(y_i, 1)) * (-w theta x) + min(y_i, 1) log(w theta x)
                        y_i = min(1, state.mismatches[j])
                        C[j][k].f.c[0] += ((1 - y_i) * theta_ + rho_)
                        C[j][k].f.c[1] += y_i
                        C[j][k].f.c[2] += -y_i * log_theta
                    else:
                        y_i = state.mismatches[j]
                        C[j][k].f.c[0] += theta_ + rho_
                        C[j][k].f.c[1] += y_i
                        C[j][k].f.c[2] += gsl_sf_lngamma(1 + y_i) - y_i * log_theta
                    C[j][k].f.k += 1

            # loop 2: compute minimal cost function for recombination at this position
            for j in range(n):
                F_t[j].m.f = INFINITY
                for q in C[j]:
                    F_t_j = min_f(q.f, q.t)
                    if F_t_j.f < F_t[j].m.f:
                        F_t[j].m = F_t_j
                        F_t[j].pos = i - q.f.k
                        F_t[j].s = <int>q.f.c[1]
                        F_t[j].hap = j

            b.m.f = INFINITY
            for j in range(n):
                if F_t[j].m.f < b.m.f:
                    b = F_t[j]
            cp.push_back(b)


            # loop 3
            # compute piecewise min and eliminate any duplicate pieces
            for j in range(n):
                 C[j] = compact(piecewise_min(prior, b.m.f, C[j]))

            err = get_next_obs(&state)

        # loop ends; compute backtrace
        if i >= 0:
            bt.push_back(cp.back())
            while bt.back().pos >= 0:
                bt.push_back(cp[bt.back().pos])

    ret = [[panel[b.hap], positions[b.pos + 1], np.exp(-b.m.x), b.s] for b in reversed(bt)]
    # print('cp', cp)
    # print('backtrace', bt)
    # print('postions', positions)
    # print('ret', ret)
    seg_pos = [r[1] for r in ret] + [L_w]
    return Segmentation(
        segments=[
            Segment(hap=h, interval=w * np.array(p), height=x, mutations=s)
            for (h, _, x, s), p in zip(ret, zip(seg_pos[:-1], seg_pos[1:]))
        ],
        panel=panel
    )



#### SUPPORT FUNCTIONS

cimport cython
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from gsl cimport *


# test functions, used for testing only
cdef vector[piecewise_func] _monotone_decreasing_case(
    func f, func g, interval t, double r) nogil:
    '''
    pointwise min of f, g over t under the condition that the function f - g is
    monotone decreasing with root r
    '''
    cdef piecewise_func q1, q2
    cdef vector[piecewise_func] ret
    q1.f = f
    q1.t[0] = t[0]
    q1.t[1] = t[1]
    if r <= t[0]:
        # the function is always -
        ret.push_back(q1)
    elif r >= t[1]:
        # the function is always +
        q1.f = g
        ret.push_back(q1)
    else: # the function is + on [a, r) and and - on (r, b]
        q1.t[0] = t[0]
        q1.t[1] = r
        q1.f = g
        q2.t[0] = r
        q2.t[1] = t[1]
        q2.f = f
        ret.push_back(q1)
        ret.push_back(q2)
    return ret

@cython.cdivision(True)
cdef vector[piecewise_func] pmin(func f, func g, interval t) nogil:
    '''
    pointwise min of f, g on the interval t
    '''
    cdef double a = f.c[0] - g.c[0]
    cdef double b = f.c[1] - g.c[1]
    cdef double c = f.c[2] - g.c[2]

    cdef double r, x_star, h_star, r0, r1
    cdef vector[piecewise_func] ret
    cdef piecewise_func q1, q2, q3

    # strategy: find the roots of h = f - g
    q1.f = f
    q1.t[0] = t[0]
    q1.t[1] = t[1]
    if f.c[2] == INFINITY:
        q1.f = g
        ret.push_back(q1)
        return ret
    if g.c[2] == INFINITY:
        ret.push_back(q1)
        return ret
    if b == 0:
        if a == 0:
            # the function is constant
            if c > 0:  # h > 0 => f is greater
                q1.f = g
            ret.push_back(q1)
            return ret
        else:  # a e^(-x) + c == 0
            if a < 0:
                return pmin(g, f, t)
            # assume a > 0
            if c >= 0:
                # the function is always +, so g is smaller
                q1.f = g
                ret.push_back(q1)
                return ret
            else:
                # as a > 0, c <= 0 the function is decreasing
                x_star = -c / a
                # assert x_star > 0
                r = -log(x_star)  # solution of a e^-x + c = 0
                return _monotone_decreasing_case(f, g, t, r)
    if a == 0:
        # y = b x + c
        if b > 0:
            return pmin(g, f, t)
        else:
            r = -c / b
            return _monotone_decreasing_case(f, g, t, r)
    else:
        if a < 0:
            return pmin(g, f, t)
        # a > 0
        if b < 0:
            # f' = -a exp(-x) + b so a > 0, b < 0 => f is monotone decreasing
            # solve a exp(-x) + b x + c == 0
            r = _root(0, a, b, c)
            return _monotone_decreasing_case(f, g, t, r)
        else:
            # a > 0, b > 0
            # f'' = a exp(-x) > 0 so f is convex
            # solve f'(x*) = -a exp(-x) + b = 0
            x_star = -log(b / a)
            # h_star = a * exp(-x_star) + b * x_star + c
            h_star = b * (1 + log(a) - log(b)) + c
            # printf("a:%f b:%f c:%f h_star:%f\n", a, b, c, h_star)
            if h_star > 0:  # minimum > 0, so f > g
                q1.f = g
                q1.t[0] = t[0]
                q1.t[1] = t[1]
                ret.push_back(q1)
                return ret
            else:
                # minimum < 0, but f is convex and -> oo at +-oo.
                # so it has two real roots.
                r0 = _root(0, a, b, c)
                r1 = _root(-1, a, b, c)
                # printf("r0:%f r1:%f a:%f b:%f c:%f t[0]:%f t[1]:%f\n", r0, r1, a, b, c, t[0], t[1])
                # order the roots r0 < r1
                r = r1
                r1 = max(r0, r)
                r0 = min(r0, r)
                # r0 < r1
                # now we consider how t relates to R
                # case 1: t.b <= r0 implies that the function is
                # positive on all of t, meaning g is smaller
                if t[1] <= r0:
                    q1.f = g
                    q1.t[0] = t[0]
                    q1.t[1] = t[1]
                    ret.push_back(q1)
                    return ret
                # case 2: t.a < r0 < t[1] < r1
                elif t[0] <= r0 < t[1] < r1:
                    q1.f = g
                    q1.t[0] = t[0]
                    q1.t[1] = r0
                    ret.push_back(q1)
                    q2.f = f
                    q2.t[0] = r0
                    q2.t[1] = t[1]
                    ret.push_back(q2)
                    return ret
                # case 3: both roots in interval
                elif t[0] <= r0 < r1 < t[1]:
                    q1.f = g
                    q1.t[0] = t[0]
                    q1.t[1] = r0
                    ret.push_back(q1)
                    q2.f = f
                    q2.t[0] = r0
                    q2.t[1] = r1
                    ret.push_back(q2)
                    q3.f = g
                    q3.t[0] = r1
                    q3.t[1] = t[1]
                    ret.push_back(q3)
                    return ret
                # case 4: r0 <= t.a < t[1] < r1
                # so the function is negative on t => f is minimal
                elif r0 <= t[0] < t[1] < r1:
                    q1.f = f
                    q1.t[0] = t[0]
                    q1.t[1] = t[1]
                    ret.push_back(q1)
                    return ret
                # case 5: r0 <= t.a < r1 < t.b
                elif r0 <= t[0] < r1 <= t[1]:
                    q1.f = f
                    q1.t[0] = t[0]
                    q1.t[1] = r1
                    ret.push_back(q1)
                    q2.f = g
                    q2.t[0]= r1
                    q2.t[1] = t[1]
                    ret.push_back(q2)
                    return ret
                # case 6: r1 < t.a
                elif r1 <= t[0]:
                    q1.f = g
                    q1.t[0] = t[0]
                    q1.t[1] = t[1]
                    ret.push_back(q1)
                    return ret

cdef vector[piecewise_func] piecewise_min(
    const vector[piecewise_func] prior,
    const double F_t,
    const vector[piecewise_func] cost
) nogil:
    '''
    pointwise minimum of vectors of piecewise_funcs. the break points do not
    necessarily align.
    '''
    # printf('in piecewise_min\n')
    cdef vector[piecewise_func] prior_a, cost_a  # aligned functions
    cdef vector[piecewise_func].const_iterator prior_it = prior.const_begin(), cost_it = cost.const_begin()
    cdef piecewise_func prior_i, cost_i, tmp1, tmp2
    cdef double x
    # assert cost.size() > 0
    # assert prior.size() > 0
    prior_i = deref(prior_it)
    cost_i = deref(cost_it)
    prior_i.f.c[2] += F_t
    cdef vector[piecewise_func] ret, tmp
    # print("***** in piecewise_min")
    # print('prior', prior)
    # print('cost', cost)
    while True:
        # prior:
        # -oo ---- 1 ------- +oo
        # cost:
        # -oo ------- 1.5 -- +oo
        if (
            (cost_i.t[0] == prior_i.t[0]) and
            isinf(prior_i.t[1]) and isinf(cost_i.t[1])
        ):
            tmp = pmin(prior_i.f, cost_i.f, cost_i.t)
            ret.insert(ret.end(), tmp.begin(), tmp.end())
            break
        # prior:
        # -oo ---- -5 ---- -1 ---- 0 ---- 1 ------- +oo
        # cost:
        # -oo -------- -2 -------- 0 ------- 1.5 -- +oo
        if prior_i.t[1] <= cost_i.t[1]:
            x = cost_i.t[1]
            cost_i.t[1] = prior_i.t[1]
            tmp = pmin(prior_i.f, cost_i.f, cost_i.t)
            ret.insert(ret.end(), tmp.begin(), tmp.end())
            cost_i.t[0] = prior_i.t[1]
            cost_i.t[1] = x
            inc(prior_it)
            prior_i = deref(prior_it)
            prior_i.f.c[2] += F_t
        else:
            x = prior_i.t[1]
            prior_i.t[1] = cost_i.t[1]
            tmp = pmin(prior_i.f, cost_i.f, cost_i.t)
            ret.insert(ret.end(), tmp.begin(), tmp.end())
            prior_i.t[0] = cost_i.t[1]
            prior_i.t[1] = x
            inc(cost_it)
            cost_i = deref(cost_it)
        # print("\tprior_i", prior_i)
        # print("\tcost_i", cost_i)
    # print("**** end piecewise_min")
    return compact(ret)

@cython.cdivision(True)
cdef double _root(int branch, double a, double b, double c) nogil:
    '''solve a e^(-x) + b x + c = 0'''
    cdef double x, h_star, log_x, w
    w = INFINITY
    cdef int status = -1
    cdef gsl_sf_result result
    # if c/b is huge this can overflow
    log_x = log(-a / b) + c / b
    log_mx = log(a / b) + c / b
    if branch == 0 and -a / b > 0 and log_x > 20:
        # we are calling LambertW(x) for x > exp(20).
        # return the asymptotic approximation
        w = log_x - log(log_x)  # + o(1)
    elif branch == -1 and a / b > 0 and log_mx < -20:
        w = log_mx - log(-log_mx)  # + o(1)
    else:
        x = -a * exp(c / b) / b
        if branch == 0:
            status = gsl_sf_lambert_W0_e(x, &result)
        elif branch == -1:
            status = gsl_sf_lambert_Wm1_e(x, &result)
        if status != 0:
            h_star = b * (1 + log(a) - log(b)) + c
            printf('*** branch=%d a=%.20f b=%.20f c=%.20f x=%.20f\n h_star=%.16f\n',
                   branch, a, b, c, x, h_star)
            printf('*** status=%d\n result.val=%.10f result.err=%.10f\n',
                   status, result.val, result.err)
        w = result.val
    return w - c / b


@cython.cdivision(True)
cdef minimum min_f(const func f, const interval t) nogil:
    '''minimize f over the interval [t[0], t[1])'''
    cdef minimum ret
    cdef double x_star, f_star
    cdef double a = f.c[0]
    cdef double b = f.c[1]
    cdef double c = f.c[2]
    if a == b == 0.:
        ret.x = t[0]
        ret.f = c
        return ret
    cdef double e0 = a * exp(-t[0]) + c, e1 = a * exp(-t[1]) + c
    if b != 0:
        # can have t[i] = inf, which results ei = NaN
        e0 += b * t[0]
        e1 += b * t[1]
    if e0 < e1:
        ret.f = e0
        ret.x = t[0]
    else:
        ret.f = e1
        ret.x = t[1]
    if b == 0 or b / a < 0:  # the function is monotone
        return ret
    if t[0] <= -log(b / a) < t[1]:
        # f' = -a exp(-x) + b == 0 => x* = -log(b / a)
        x_star = -log(b / a)
        # f_star = a * exp(log(b/a)) - b * log(b/a) + c
        #        = a * a / b - b * log(b / a) + c
        f_star = b * (1. + x_star) + c
        if f_star < ret.f:
            ret.f = f_star
            ret.x = x_star
    return ret

# for piecewise constant coalescent function
cdef vector[piecewise_func] piecewise_const_log_pi(double[:] a,
                                                   double[:] t,
                                                   double beta) nogil:
    '''
    piecewise constant prior. if eta(t) = a_k, t_{k-1} <= t < t_k, then
    R(t) = a_1(t_1 - t_0) + ... + a_k(t - t_{k-1}), t_{k-1} <= t < t_k
    pi(t) = t eta(t) exp(-R(t)) = t a_k e^{-c_k} e^{-a_k t} for
    c_k = a_1(t_1 - t_0) + ... + a_{k-1}(t{k-1} - t{k-2}) - a_k t_{k-1}
    thus -log pi(t) = -log(t) - b_k + c_k + a_k t for b_k = log(a_k)
    thus -log pi(z) = z - b_k + c_k + a_k exp(-z) for b_k = log(a_k),
                      t_{k-1} <= exp(-z) < t_k, z = -log(t)
    thus -log pi(z) = z - b_k + c_k + a_k exp(-z) for b_k = log(a_k),
                      -log(t_{k-1}) >= z > -log(t_k)

    :param: t an array of times of length K, t[-1] = inf, sorted; none of this is checked.
    :param: a an array of inverse effective population size.
    '''

    # segment prob: pi(t) t rho exp(-t delta (rho + theta)) + s log(theta t)
    #        => -log pi(z) - t delta (rho + theta)

    cdef int32_t j, k
    cdef int32_t K = t.shape[0]
    cdef double c_k = 0, tk1
    cdef vector[piecewise_func] ret
    ret.resize(K)
    for k in range(K):
        j = K - 1 - k
        ret[j].f.k = 0
        if k == 0:
            tk1 = 0.
        else:
            tk1 = t[k - 1]
        ret[j].f.c[0] = a[k]
        ret[j].f.c[1] = 2.
        c_k -= a[k] * tk1
        ret[j].f.c[2] = c_k - log(a[k]) + beta
        c_k += a[k] * t[k]
        ret[j].t[0] = -log(t[k])
        ret[j].t[1] = -log(tk1)
    return ret


cdef vector[piecewise_func] compact(const vector[piecewise_func] C) nogil:
    cdef piecewise_func q0, q
    cdef vector[piecewise_func] ret
    if C.size() == 0:
        return ret
    q0 = C[0]
    cdef int i
    for i in range(1, C.size()):
        q = C[i]
        if q.t[0] == q.t[1]:
            continue
        if memcmp(&q.f, &q0.f, sizeof(func)) == 0:
            q0.t[1] = q.t[1]
        else:
            ret.push_back(q0)
            q0 = q
    ret.push_back(q0)
    return ret

#### TESTING FUNCTIONS
def test_min_f(f, t):
    cdef func _f
    cdef interval _t
    for i in range(3):
        _f.c[i] = f[i]
    for i in range(2):
        _t[i] = t[i]
    return min_f(_f, _t)

def test_piecewise_const_log_pi(a, t):
    return piecewise_const_log_pi(a, t, 0.)

def test_piecewise_min(prior, double F_t, cost):
    cdef vector[piecewise_func] _prior, _cost
    _prior.resize(len(prior))
    _cost.resize(len(cost))
    for i, p in enumerate(prior):
        for j in range(3):
            _prior[i].f.c[j] = p['f'][j]
        _prior[i].f.k = p['k']
        _prior[i].t[0] = p['t'][0]
        _prior[i].t[1] = p['t'][1]
    for i, p in enumerate(cost):
        for j in range(3):
            _cost[i].f.c[j] = p['f'][j]
        _cost[i].f.k = p['k']
        _cost[i].t[0] = p['t'][0]
        _cost[i].t[1] = p['t'][1]
    return piecewise_min(_prior, F_t, _cost)


def test_pmin(f, g, t, f_k=0, g_k=1):
    cdef interval _t
    cdef func _f, _g
    cdef int i
    for i in range(2):
        _t[i] = t[i]
    for i in range(3):
        _f.c[i] = f[i]
        _g.c[i] = g[i]
    _f.k = f_k
    _g.k = g_k
    return pmin(_f, _g, _t)


def test_root(k, a, b, c):
    return _root(k, a, b, c)
