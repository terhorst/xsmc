# cython: boundscheck=False
# cython: cdivision=True
# cython: language=c++
# distutils: extra_compile_args=['-O2', '-Wno-unused-but-set-variable', '-ffast-math']

DEF DEBUG = 1

import tskit
import _tskit
import numpy as np
from typing import List, NamedTuple, Union

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

def viterbi_path(
    ts: tskit.TreeSequence,
    focal: int,
    panel: List[int],
    scaffold: tskit.TreeSequence,
    eta: SizeHistory,
    theta: float,
    rho: float,
    beta: Union[None, float],
    robust: bool,
    w: int,
    ) -> Segmentation:
    '''Compute optimal segmentation for renewal approximation to sequentially Markov coalescent.

    Args:
        ts: Tree sequence containing the data to segment.
        focal: Leaf index of the focal haplotype.
        panel: Leaf indices of the panel haplotypes.
        scaffold: Tree sequence containing underlying scaffolding on which the focal lineage coalesces. 
        eta: Size history to use for coalescent prior. If None, Kingman's
            coalescent is used.
        theta: Population-scaled mutation rate per bp per generation.
        rho: Population-scaled recombination rate per bp per generation.
        beta: Penalization parameter. If None, defaults to :math:`-log(\rho)`, resulting in the MAP path.
        
    Returns:
        The maximum *a posteriori* segmentation under specified parameters.
        
    Notes:
        The following assumptions are made about the input:
            - panel[i] corresponds to ts.nodes(panel[i]).
            - panel[i] corresponds to scaffold.nodes()[i].
            - both of the above nodes are flagged with tskit.NODE_IS_SAMPLE.
            - scaffold.get_sequence_length() == ts.get_sequence_length() // w
    '''
    # Take the passed-in tree sequence and export/import the tables in order
    # to get a tree sequence that is binary compatble with whatever version of
    # tskit was used to compile the software (which could be different).
    cdef LightweightTableCollection lwt = LightweightTableCollection()
    lwt.fromdict(ts.dump_tables().asdict())
    cdef tsk_treeseq_t _ts
    cdef int err = tsk_treeseq_init(&_ts, lwt.tables, 0)
    check_error(err)

    assert theta > 0
    assert rho > 0
    assert scaffold.get_sequence_length() >= ts.get_sequence_length() // w
    cdef double rho_ = w * rho
    cdef double theta_ = w * theta
    if beta is None:
        beta = -np.log(rho_)
    cdef double beta_ = beta
    cdef double log_theta = log(theta_)
    cdef int i, j, k, y_i
    cdef interval intv

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

    # cp is the backtrace list of changepoints: (pos, hap)
    cdef backtrace b
    cdef vector[backtrace] cp
    cdef vector[backtrace] bt
    cp.reserve(ts.get_num_sites())

    assert eta.t[0] == 0.
    assert np.isinf(eta.t[-1])

    assert np.all(eta.Ne > 0), "Ne cannot be 0."
    assert np.all(np.isfinite(eta.Ne)), "Ne must be finite"
    coal_rate = np.array([eta(tt) for tt in eta.t[:-1]])
    cdef func q
    cdef piecewise_func log_pi = piecewise_const_log_pi(
        coal_rate.astype(np.float64),
        eta.t[1:].astype(np.float64),
        beta_
    )
    cdef vector[int] positions
    cdef int pos = 0
    positions.push_back(pos)
    cdef int L_w = <int>(ts.get_sequence_length() // w)
    i = -1

    # Initialize the variant generator for our sample. the focal haplotype has
    # genotype index 0, and the panel haps have genotype indices 1, ..., n + 1
    cdef tsk_vargen_t _vg
    cdef vector[tsk_id_t] _samples = [focal] + panel
    err = tsk_vargen_init(&_vg, &_ts, _samples.data(), _samples.size(), NULL, 0)

    cdef obs_iter state
    state.w = w
    state.L = L_w
    state.ell = 0
    state.err = 1
    state.vg = &_vg
    assert n + 1 == state.vg.num_samples
    cdef int32_t[:] mismatches = np.zeros(n, dtype=np.int32)
    state.mismatches = &mismatches[0]
    state.err = tsk_vargen_next(state.vg, &state.var)
    
    # Initialize the tree sequence iterator for the arg
    cdef LightweightTableCollection lwt_arg = LightweightTableCollection()
    lwt_arg.fromdict(scaffold.dump_tables().asdict())
    cdef tsk_treeseq_t _arg_ts
    err = tsk_treeseq_init(&_arg_ts, lwt_arg.tables, 0)
    check_error(err)
    cdef tsk_tree_t _arg_tree
    err = tsk_tree_init(&_arg_tree, &_arg_ts, 0);
    check_error(err)
    tsk_tree_first(&_arg_tree)
    cdef vector[piecewise_func] priors = arg_prior(log_pi, &_arg_tree)

    # i_p += 1
    # C are the cost functions for each haplotype
    cdef vector[piecewise_func] C
    for j in range(n):
        C.push_back(priors.at(j))
        # subtract off one x because the first prior is not length-biased
        for k in range(C.at(j).f.size()):
            C.at(j).f.at(k).c[1] -= 1.

    err = get_next_obs(&state)
    with nogil:
        while err == 1:
            # proceed to next variant
            
            # loop1: increment all the current ibd tracts
            i += 1
            # if DEBUG:
            #     with gil:
            #         print("*** i=", i)
            while _arg_tree.right < i:
                tsk_tree_next(&_arg_tree)  # FIXME assert ret == 1
                priors = arg_prior(log_pi, &_arg_tree)
            pos = state.ell
            delta = pos - positions.back()  # = 1
            positions.push_back(pos)
            for j in range(n):
                for k in range(C.at(j).f.size()):
                    if robust:
                        # bernoulli obs, which adds contribution 
                        # (1-min(y_i, 1)) * (-w theta x) + min(y_i, 1) log(w theta x)
                        y_i = min(1, state.mismatches[j])
                        C.at(j).f.at(k).c[0] += ((1 - y_i) * theta_ + rho_)
                        C.at(j).f.at(k).c[1] += y_i
                        C.at(j).f.at(k).c[2] += -y_i * log_theta
                    else:
                        y_i = state.mismatches[j]
                        C.at(j).f.at(k).c[0] += theta_ + rho_
                        C.at(j).f.at(k).c[1] += y_i
                        C.at(j).f.at(k).c[2] += gsl_sf_lngamma(1 + y_i) - y_i * log_theta
                    C.at(j).f.at(k).k += 1
            # if DEBUG:
            #     with gil:
            #         print("Loop 1 C=", C)

            # loop 2: compute minimal cost function for recombination at this position
            for j in range(n):
                F_t.at(j).m.f = INFINITY
                for k in range(C.at(j).f.size()):
                    q = C.at(j).f.at(k)
                    intv[0] = C.at(j).t.at(k)
                    intv[1] = C.at(j).t.at(k + 1)
                    F_t_j = min_f(q, intv)
                    if F_t_j.f < F_t.at(j).m.f:
                        F_t.at(j).m = F_t_j
                        F_t.at(j).pos = i - q.k
                        F_t.at(j).s = <int>q.c[1]
                        F_t.at(j).hap = j
            # if DEBUG:
            #     with gil:
            #         print("Loop 2 F_t=", F_t)

            b.m.f = INFINITY
            for j in range(n):
                if F_t.at(j).m.f < b.m.f:
                    b = F_t.at(j)
            cp.push_back(b)

            # loop 3
            # compute piecewise min and eliminate any duplicate pieces
            for j in range(n):
                 C[j] = compact(pointwise_min(priors.at(j), b.m.f, C.at(j)))
                
            # if DEBUG:
            #     with gil:
            #         print("Loop 3 C=", C)

            err = get_next_obs(&state)

        # loop ends; compute backtrace
        if i >= 0:
            bt.push_back(cp.back())
            while bt.back().pos >= 0:
                bt.push_back(cp[bt.back().pos])

        tsk_tree_free(&_arg_tree)
        tsk_vargen_free(&_vg)
        tsk_treeseq_free(&_ts)
        tsk_treeseq_free(&_arg_ts)

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
from cython.operator cimport dereference as deref, preincrement as inc
cimport cython
from gsl cimport *

# test functions, used for testing only
cdef piecewise_func _monotone_decreasing_case(
    func f, func g, interval t, double r) nogil:
    '''
    pointwise min of f, g over t under the condition that the function f - g is
    monotone decreasing with root r
    '''
    cdef piecewise_func ret
    ret.t.push_back(t[0])
    if r <= t[0]:
        # the function is always -
        ret.f.push_back(f)
        ret.t.push_back(t[1])
        return ret
    elif r >= t[1]:
        # the function is always +
        ret.f.push_back(g)
        ret.t.push_back(t[1])
        return ret
    else: # the function is + on [a, r) and and - on (r, b]
        ret.f.push_back(g)
        ret.f.push_back(f)
        ret.t.push_back(r)
        ret.t.push_back(t[1])
    return ret

@cython.cdivision(True)
cdef piecewise_func pmin(func f, func g, interval t) nogil:
    '''
    pointwise min of f, g on the interval t
    '''
    if DEBUG:
        with gil:
            print("taking the piecewise min of f=%s g=%s t=(%f,%f)" % (f, g, t[0], t[1]))
    cdef double a = f.c[0] - g.c[0]
    cdef double b = f.c[1] - g.c[1]
    cdef double c = f.c[2] - g.c[2]

    cdef double r, x_star, h_star, r0, r1
    cdef piecewise_func ret
    
    cdef piecewise_func f_is_greater
    f_is_greater.f.push_back(g)
    f_is_greater.t.push_back(t[0])
    f_is_greater.t.push_back(t[1])
    
    cdef piecewise_func g_is_greater
    g_is_greater.f.push_back(f)
    g_is_greater.t.push_back(t[0])
    g_is_greater.t.push_back(t[1])

    # strategy: find the roots of h = f - g
    if f.c[2] == INFINITY:
        return f_is_greater
    if g.c[2] == INFINITY:
        return g_is_greater
    if b == 0:
        if a == 0:
            # the function is constant
            if c > 0:  # h > 0 => f is greater
                return f_is_greater
            else:
                return g_is_greater
        else:  # a e^(-x) + c == 0
            if a < 0:
                return pmin(g, f, t)
            # assume a > 0
            if c >= 0:
                # the function is always +, so g is smaller
                return f_is_greater
            else:
                # as a > 0, c <= 0 the function is decreasing
                x_star = -c / a
                # assert x_star > 0
                r = -log(x_star)  # solution of a e^-x + c = 0
                return _monotone_decreasing_case(f, g, t, r)
    elif a == 0:
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
            # if DEBUG:
            #     printf("a:%f b:%f c:%f h_star:%f\n", a, b, c, h_star)
            if h_star > 0:  # minimum > 0, so f > g
                return f_is_greater
            else:
                # minimum < 0, but f is convex and -> oo at +-oo.
                # so it has two real roots.
                r0 = _root(0, a, b, c)
                r1 = _root(-1, a, b, c)
                # if DEBUG:
                #     printf("r0:%f r1:%f a:%f b:%f c:%f t[0]:%f t[1]:%f\n", r0, r1, a, b, c, t[0], t[1])
                # order the roots r0 < r1
                r = r1
                r1 = max(r0, r)
                r0 = min(r0, r)
                # r0 < r1
                # now we consider how t relates to R
                # case 1: t.b <= r0 implies that the function is
                # positive on all of t, meaning g is smaller
                if t[1] <= r0:
                    return f_is_greater
                # case 2: t.a < r0 < t[1] < r1
                elif t[0] <= r0 < t[1] < r1:
                    ret.f.push_back(g)
                    ret.f.push_back(f)
                    ret.t.push_back(t[0])
                    ret.t.push_back(r0)
                    ret.t.push_back(t[1])
                    return ret
                # case 3: both roots in interval
                elif t[0] <= r0 < r1 < t[1]:
                    ret.f.push_back(g)
                    ret.f.push_back(f)
                    ret.f.push_back(g)
                    ret.t.push_back(t[0])
                    ret.t.push_back(r0)
                    ret.t.push_back(r1)
                    ret.t.push_back(t[1])
                    return ret
                # case 4: r0 <= t.a < t[1] < r1
                # so the function is negative on t => f is minimal
                elif r0 <= t[0] < t[1] < r1:
                    return g_is_greater
                # case 5: r0 <= t.a < r1 < t.b
                elif r0 <= t[0] < r1 <= t[1]:
                    ret.f.push_back(f)
                    ret.f.push_back(g)
                    ret.t.push_back(t[0])
                    ret.t.push_back(r1)
                    ret.t.push_back(t[1])
                    return ret
                # case 6: r1 < t.a
                elif r1 <= t[0]:
                    return f_is_greater
    with gil:
        print("fell through -- this should never happen!")
        raise

cdef piecewise_func pointwise_min(
    const piecewise_func prior,
    const double F_t,
    const piecewise_func cost
) nogil:
    '''
    pointwise minimum of vectors of piecewise_funcs. the break points do not necessarily align.
    '''
    if DEBUG:
        with gil:
            check_piecewise(prior)
            check_piecewise(cost)
            print('in pointwise_min')
            print("---> prior", prior)
            print("---> F_t", F_t)
            print("---> cost", cost)
    cdef int i = 0, j = 0
    cdef interval prior_intv, cost_intv, intv
    cdef func prior_f, cost_f
    prior_f = prior.f.at(i)
    prior_f.c[2] += F_t
    cost_f = cost.f.at(j)
    cdef piecewise_func ret, tmp
    prior_intv[0] = prior.t.at(i)
    prior_intv[1] = prior.t.at(i + 1)
    cost_intv[0] = cost.t.at(j)
    cost_intv[1] = cost.t.at(j + 1)
    intv[0] = prior_intv[0]
    while isfinite(prior_intv[1]) or isfinite(cost_intv[1]):
        if DEBUG:
            with gil:
                assert prior_intv[0] == cost_intv[0]
        intv[0] = prior_intv[0]
        intv[1] = min(cost_intv[1], prior_intv[1])
        tmp = pmin(prior_f, cost_f, intv)
        if DEBUG:
            with gil:
                print("got from pmin:", tmp)
        ret.f.insert(ret.f.end(), tmp.f.begin(), tmp.f.end())
        ret.t.insert(ret.t.end(), tmp.t.begin(), tmp.t.end() - 1)
        prior_intv[0] = intv[1]
        cost_intv[0] = intv[1]
        if prior_intv[1] == intv[1]:
            i += 1
            prior_f = prior.f.at(i)
            prior_intv[1] = prior.t.at(i + 1)
            prior_f.c[2] += F_t
        if cost_intv[1] == intv[1]:
            j += 1
            cost_f = cost.f.at(j)
            cost_intv[1] = cost.t.at(j + 1)
        if DEBUG:
            with gil:
                print('pointwise_min', i, j, cost_intv, cost_f, prior_intv, prior_f, tmp, ret)
    intv[0] = prior_intv[0]
    intv[1] = min(cost_intv[1], prior_intv[1])
    tmp = pmin(prior_f, cost_f, intv)
    if DEBUG:
        with gil:
            print("got from final pmin:", tmp)
    ret.f.insert(ret.f.end(), tmp.f.begin(), tmp.f.end())
    ret.t.insert(ret.t.end(), tmp.t.begin(), tmp.t.end())
    if DEBUG:
        with gil:
            print('pointwise_min done', i, j, cost_intv, cost_f, prior_intv, prior_f, tmp, ret)
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
            printf('*** status=%d desc=%s\n*** result.val=%.10f result.err=%.10f\n',
                   status, gsl_strerror(status), result.val, result.err)
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

cdef vector[piecewise_func] arg_prior(const piecewise_func &log_pi, tsk_tree_t* tree) nogil:
    cdef vector[piecewise_func] ret
    cdef int num_samples = tsk_treeseq_get_num_samples(tree.tree_sequence)
    ret.resize(num_samples)
    cdef tsk_size_t i
    cdef tsk_id_t u, p
    cdef double t
    for i in range(num_samples):
        tsk_tree_get_parent(tree, tree.samples[i], &u)
        if u == TSK_NULL:  # root parent means "trunk lineage"
            ret[i] = log_pi
        else:
            tsk_tree_get_time(tree, u, &t)
            ret[i] = truncate_prior(log_pi, t)
    if DEBUG:
        with gil:
            print('---> arg_prior')
            print('----> log_pi=', log_pi)
            print('----> ret=', ret)
    return ret

cdef piecewise_func truncate_prior(const piecewise_func &prior, double t) nogil:
    # given piecewise constant rate prior and time t, define the rate to be 0. more anciently than
    # time t. Thus, focal lineage has infinite cost of coalescing prior to that time (as when
    # threading onto an arg, above the coalescence time of a particular lineage.
    
    #   [-inf, 1] tau=2
    cdef piecewise_func ret
    cdef func f
    cdef interval intv
    cdef double tau = -log(t)
    f.k = 0
    f.c[0] = 0.
    f.c[1] = 0.
    f.c[2] = INFINITY
    ret.f.push_back(f)
    ret.t.push_back(-INFINITY)
    ret.t.push_back(tau)
    cdef int i, j
    for i in range(prior.f.size()):
        intv[0] = prior.t.at(i)
        intv[1] = prior.t.at(i + 1)
        if intv[0] <= tau and tau < intv[1]:
            break
    ret.f.insert(ret.f.end(), prior.f.begin() + i, prior.f.end())
    ret.t.insert(ret.t.end(), prior.t.begin() + i + 1, prior.t.end())
    if DEBUG:
        with gil:
            check_piecewise(ret)
    return compact(ret)

# for piecewise constant coalescent function
cdef piecewise_func piecewise_const_log_pi(double[:] a, double[:] t, double beta) nogil:
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
    cdef piecewise_func ret
    ret.f.resize(K)
    ret.t.resize(K + 1)
    for k in range(K):
        j = K - 1 - k
        ret.f.at(j).k = 0
        if k == 0:
            tk1 = 0.
        else:
            tk1 = t[k - 1]
        ret.f.at(j).c[0] = a[k]
        ret.f.at(j).c[1] = 2.
        c_k -= a[k] * tk1
        ret.f.at(j).c[2] = c_k - log(a[k]) + beta
        ret.f.at(j).k = 0
        c_k += a[k] * t[k]
        ret.t[j] = -log(t[k])
    ret.t[K] = INFINITY
    return ret


cdef piecewise_func compact(const piecewise_func &C) nogil:
    if DEBUG:
        with gil:
            try:
                check_piecewise(C)
            except:
                print("compact/C")
                raise
    cdef func q, q0
    cdef double t, t0
    cdef piecewise_func ret
    if C.f.size() == 0:
        return ret
    q = C.f.at(0)
    t = C.t.at(0)
    for i in range(1, C.f.size()):
        q0 = C.f.at(i)
        t0 = C.t.at(i)
        # if memcmp(&q, &q0, sizeof(func)) != 0:  this approach does not work. or rather, it's overly conservative.
        if (t == t0) or (q.c[0] == q0.c[0] and q.c[1] == q0.c[1] and q.c[2] == q0.c[2] and q.k == q0.k):
            continue
        ret.f.push_back(q)
        ret.t.push_back(t)
        q = q0
        t = t0
    ret.f.push_back(q)
    ret.t.push_back(t)
    ret.t.push_back(C.t.back())
    if DEBUG:
        with gil:
            try:
                check_piecewise(ret)
            except:
                print("compact/ret")
                raise
    return ret

cdef void check_piecewise(const piecewise_func &v):
    assert len(v.f) == len(v.t) - 1
    assert v.t[0] == -INFINITY
    assert v.t.back() == INFINITY
    assert sorted(v.t) == list(v.t)
    return
    # cdef float t1last
    # for i in range(v.t.size()):
    #     if i == 0:
    #         assert v.at(i).t == -INFINITY, "func has wrong left endpoint"
    #     if i == v.size() - 1:
    #         assert v.at(i).t < INFINITY, "func has wrong right endpoint"
    #     if i > 0:
    #         assert v.at(i).t > t1last, ("func not defined everywhere", i, t1last, v)
    #     t1last = v.at(i).t
            

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

def test_pointwise_min(prior, double F_t, cost):
    cdef piecewise_func _prior, _cost
    _prior.f.resize(len(prior))
    _prior.t.resize(len(prior))
    _cost.f.resize(len(cost))
    _cost.t.resize(len(cost))
    for i, p in enumerate(prior):
        for j in range(3):
            _prior.f[i].c[j] = p['f'][j]
        _prior.f[i].k = p['k']
        _prior.t[i] = p['t'][0]
    _prior.t.push_back(np.inf)
    for i, p in enumerate(cost):
        for j in range(3):
            _cost.f[i].c[j] = p['f'][j]
        _cost.f[i].k = p['k']
        _cost.t[i] = p['t'][0]
    _cost.t.push_back(np.inf)
    print(_prior, _cost)
    return pointwise_min(_prior, F_t, _cost)


def test_pointwise_min_new_fmt(prior, double F_t, cost):
    return pointwise_min(prior, F_t, cost)


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


def test_truncate_prior(prior, t):
    cdef piecewise_func p
    p.f = prior['f']
    p.t = prior['t']
    return truncate_prior(prior, t)


def test_compact(func):
    return compact(func)
