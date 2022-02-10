# cython: boundscheck=False
# cython: cdivision=True
# cython: language=c++
# distutils: extra_compile_args=['-O2', '-Wno-unused-but-set-variable']

from logging import getLogger
from typing import List, Tuple

import numpy as np
import scipy.special

from scipy.special.cython_special cimport gammaln, xlogy
from numpy.math cimport logaddexp

import xsmc._tskit
from _xsmc cimport TreeSequence

from cython.operator cimport dereference as deref

DEF DEBUG = 0

logger = getLogger(__name__)


cdef double NINF = float("-inf")


cdef extern from "_cache.h":
    struct log_P_cache_key:
        int Xb, delta, u, v

cdef struct sampler_params:
    float theta, rho, eps
    int cache_hit, cache_miss
    unordered_map[log_P_cache_key, float]* log_P_cache

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass default_random_engine:
        default_random_engine()
        void seed(int)

cdef extern from '_sampler.hpp' nogil:
    cdef double rexp(default_random_engine&)
    cdef void shuffle(vector[int]&, default_random_engine&)

cdef void log_P(double[:] out, int s, int t,
        long[:, :] Xcs, double[:, :] lgammacs,
        double[:, :] pi,
        double[:] positions,
        sampler_params params) nogil:
    cdef int h, k, H, K, n, Xb, delta, u, v
    cdef double lg, e, alpha, beta, log_c
    K = pi.shape[1]
    H = Xcs.shape[0]
    n = Xcs.shape[1] - 1
    delta = int(positions[t] - positions[s])
    u = int(s > 0)
    v = int(t < n)
    cdef double x
    cdef log_P_cache_key key
    key.delta = delta
    key.u = u
    key.v = v
    for h in range(H):
        Xb = Xcs[h, t] - Xcs[h, s]
        key.Xb = Xb
        lg = lgammacs[h, t] - lgammacs[h, s]
        if True:  # deref(params.log_P_cache).count(key) == 0:
            params.cache_miss += 1
            # deref(params.log_P_cache)[key] = NINF
            out[h] = NINF
            for k in range(pi.shape[1]):
                alpha = pi[0, k]
                beta = pi[1, k]
                log_c = pi[2, k]
                e = 1 + u + v + Xb + alpha
                x = (
                    + xlogy(Xb, params.theta)
                    + xlogy(v, params.rho)
                    + xlogy(alpha, beta)
                    - xlogy(e, beta + delta * (params.rho + params.theta))
                    + gammaln(e)
                    - gammaln(alpha)
                )
                # deref(params.log_P_cache)[key] = logaddexp(deref(params.log_P_cache).at(key), log_c + x)
                out[h] = logaddexp(out[h], log_c + x)
        # else:
        #    params.cache_hit += 1
        # out[h] = deref(params.log_P_cache).at(key) - lg
        out[h] -= lg
        if DEBUG:
            with gil:
                print('h', h, 'out[h]', out[h])

def get_mismatches(
    TreeSequence ts,
    focal: int,
    panel: List[int],
    int w
):
    '''Cumulate genotype matrix for use in sampling algorithm.'''
    # cdef int err
    # err = tsk_treeseq_init(&ts, lwtc.tables, TSK_BUILD_INDEXES)
    # assert err == 0
    cdef double L = tsk_treeseq_get_sequence_length(&ts._ts)

    H = len(panel)
    L_w = int(np.floor(1. + L / w))
    X_np = np.zeros((H, L_w), dtype=np.int32)
    cdef int[:, :] X = X_np

    cdef tsk_id_t[:] samples = np.array([focal] + list(panel), dtype=np.int32)
    cdef tsk_vargen_t vg
    err = tsk_vargen_init(&vg, &ts._ts, &samples[0], 1 + len(panel), NULL, 0)
    assert err == 0
    cdef tsk_variant_t *var
    err = tsk_vargen_next(&vg, &var)
    assert err == 1
    cdef tsk_id_t focal_ = focal
    cdef int i = 0, h, y_i
    logger.debug('Counting mismatches for focal=%d panel=%s', focal, panel)
    with nogil:
        while err == 1:
            # proceed to next variant
            i = <int>(var.site.position / w)
            for h in range(H):
                y_i = <int>(var.genotypes.i8[h + 1] != var.genotypes.i8[0])
                X[h, i] += y_i
            err = tsk_vargen_next(&vg, &var)
    tsk_vargen_free(&vg);
    # tsk_treeseq_free(&ts);
    return X_np


cdef double logsumexp(double[:] x) nogil:
    cdef int n
    cdef int N = x.shape[0]
    cdef double m = NINF
    for n in range(N):
        m = max(m, x[n])
    if m == NINF:
        return m
    cdef double tmp = 0.
    for n in range(N):
        tmp += exp(x[n] - m)
    cdef double ret = m + log(tmp)
    return ret

cdef void init_log_Q(double[:] log_Q, double[:] tmp, 
        long[:, :] Xcs, double[:, :] lgammacs,
        double[:, :] pi,
        double[:] positions,
    sampler_params params) nogil:
    """Model evidence for observations X_{t:}."""
    cdef double eps = log(params.eps)
    cdef double p, m

    cdef int n = log_Q.shape[0]
    cdef int s, t
    for t in range(n - 1, -1, -1):
        log_P(tmp, t, n, Xcs, lgammacs, pi, positions, params)
        log_Q[t] = logsumexp(tmp)
        if DEBUG:
            with gil:
                print('t', t, 'log_Q[t]', np.asarray(log_Q[t]))
        for s in range(t + 1, n):
            log_P(tmp, t, s, Xcs, lgammacs, pi, positions, params)
            p = logsumexp(tmp) + log_Q[s]
            m = max(log_Q[t], p)
            log_Q[t] = m + log(exp(log_Q[t] - m) + exp(p - m))
            if p - log_Q[t] < eps:
                break


cdef void P_tau_i(double[:] out, const int tau_jm1, const int k, double[:] log_Q, long[:, :] Xcs, double[:, :] lgammacs,
        double[:, :] pi, double[:] positions,
        sampler_params params) nogil:
    # log(P_tau[i]) where P_tau is defined above
    cdef int H = out.shape[0]
    cdef int tau0 = tau_jm1 + 1
    log_P(out, tau0, k, Xcs, lgammacs, pi, positions, params)
    # with gil:
    #     print("P_tau_i(%d,%d)=%s log_Q[%d]=%f log_Q[%d]=%f" % (tau0, k, np.array(out), k, log_Q[k], tau0, log_Q[tau0]))
    for h in range(H):
        out[h] = exp(out[h] + log_Q[k] - log_Q[tau0])


cdef void uniform_order_statistics(double[:] out, int n, default_random_engine &rng) nogil:
    # lazily sample from log_P_tau
    out[0] = rexp(rng)
    cdef int j
    for j in range(1, n + 1):
        out[j] = out[j - 1] + rexp(rng)
    # with gil:
    #     print('out', n, out[n], np.array(out))
    for j in range(n + 1):
        out[j] /= out[n]
    # with gil:
    #     print('out2', n, out[n], np.array(out))

cdef vector[vector[pair[int, int]]] _sample_paths(int u, seed, double[:] log_Q, long[:, :] Xcs, 
        double[:, :] lgammacs,
        double[:, :] pi,
        double[:] positions, sampler_params params):
    cdef int H, n
    cdef default_random_engine rng
    if seed is not None:
        rng.seed(seed)
    H = Xcs.shape[0]
    n = Xcs.shape[1] - 1
    cdef vector[vector[pair[int, int]]] paths
    paths.resize(u)
    cdef vector[int] a
    cdef int h, i, j, k, m
    cdef double p, q, s
    cdef double[:] cum_prob = np.zeros(H)
    cdef double[:] p_j = np.zeros(H)
    cdef double[:] z = np.zeros(u + 1)
    for i in range(u):
        paths.at(i).push_back((-1, -1))
    params.cache_hit = 0
    params.cache_miss = 0
    with nogil:
        for i in range(n - 1):
            # these need a new changepoint
            a.clear()
            cum_prob[:] = 0.
            for j in range(u):
                if paths.at(j).back().first == i - 1:
                    a.push_back(j)
            if a.size() == 0:
                continue
            shuffle(a, rng)
            uniform_order_statistics(z, a.size(), rng)
            q = 0.0
            j = i + 1   # j <= n - 1 here.
            k = 0
            P_tau_i(p_j, i - 1, j, log_Q, Xcs, lgammacs, pi, positions, params)
            # with gil:
            #     print('i', i, 'j', j, 'p_j', np.array(p_j), 'z', np.array(z))
            h = 0
            while k < a.size() and j < n - 1:
                if z[k] < q + p_j[h]:
                    paths.at(a.at(k)).push_back(pair[int, int](j, h))
                    k += 1
                else:
                    q += p_j[h]
                    cum_prob[h] += p_j[h]
                    # with gil:
                    #     print('i', i, 'j', j, 'q', q, 'h', h, 'p_j[h]', p_j[h])
                    if h < H - 1:
                        h += 1
                    else:
                        j += 1
                        if j == n - 1:
                            break
                        P_tau_i(p_j, i - 1, j, log_Q, Xcs, lgammacs, pi, positions, params)
                        # with gil:
                        #     print(i, j, np.array(p_j))
                        h = 0
            # all remaining changpoints occur at n, i.e. beyond end of sequence
            s = 0.
            # with gil:
            #     print('i', i, 'n', n, 'cum prob', np.array(cum_prob), k, a.size())
            #     print(paths)
            for h in range(H):
                cum_prob[h] = max(cum_prob[h], 1e-8)
                s += cum_prob[h]
            for h in range(H):
                p_j[h] = cum_prob[h] / s
            # sample len(a) - k
            m = a.size() - k
            if m > 0:
                if DEBUG:
                    with gil:
                        print('i', i, 'n', n, 'cum prob', np.array(cum_prob), k, a.size())
                        print(paths)
                uniform_order_statistics(z, m, rng)
                q = 0.
                h = 0
                while k < a.size():
                    if z[m - (a.size() - k)] <= q + p_j[h]:
                        paths.at(a.at(k)).push_back(pair[int, int](n - 1, h))
                        k += 1
                    else:
                        q += p_j[h]
                        h += 1
    # print(f"cache_hit={params.cache_hit} cache_miss={params.cache_miss}")
    return paths

cdef class _SamplerProxy:
    cdef object Xcs, lgammacs, pi, positions, _log_Q
    cdef sampler_params params

    def __cinit__(self):
        self.params.log_P_cache = new unordered_map[log_P_cache_key, float]()

    def __dealloc__(self):
        del self.params.log_P_cache

    def __init__(self, Xcs, pi, positions, theta, rho, eps):
        self.params.theta = theta  # emission probabilities
        self.params.rho = rho
        self.params.eps = eps
        self.Xcs = Xcs
        self.pi = pi
        X = np.diff(Xcs, axis=1)
        self.lgammacs = np.pad(scipy.special.gammaln(X + 1), [[0, 0], [0, 1]]).cumsum(axis=1)
        self.positions = positions
        self._log_Q = None


    @property
    def log_Q(self):
        cdef double[:] v_log_Q, v_tmp
        cdef long[:, :] v_Xcs = self.Xcs
        cdef double[:, :] v_lgamma = self.lgammacs
        cdef double[:, :] v_pi = self.pi
        cdef double[:] v_positions = self.positions
        if self._log_Q is None:
            logger.debug("Computing log Q")
            # logger.debug("Xcs.shape=%s positions=%s params=%s", self.Xcs.shape, self.positions, self.params)
            H, n = self.Xcs.shape
            tmp = np.zeros(H)
            self._log_Q = np.zeros(n - 1)
            v_log_Q = self._log_Q
            v_tmp = tmp
            with nogil:
                init_log_Q(v_log_Q, v_tmp, v_Xcs, v_lgamma, v_pi, v_positions, self.params)
            logger.debug("Done computing log Q")
        return self._log_Q

    # used for testing
    def log_P(self, s, t):
        H, n = self.Xcs.shape
        tmp = np.zeros(H)
        log_P(tmp, s, t, self.Xcs, self.lgammacs, self.pi, self.positions, self.params)
        return tmp

    def sample_paths(self, k: int, seed: int) -> List[np.ndarray]:
        logger.debug("Sampling paths")
        cdef vector[vector[pair[int, int]]] paths
        paths = _sample_paths(k, seed, self.log_Q, self.Xcs, self.lgammacs, self.pi, self.positions, self.params)
        logger.debug("Done sampling paths.")
        cdef vector[pair[int, int]] p
        cdef int i
        cdef pair[int, int] q
        cdef int[:, :] z
        ret = []
        for p in paths:
            ret.append(np.zeros((p.size(), 2), dtype=np.int32))
            z = ret[-1]
            i = 0
            with nogil:
                for i in range(p.size()):
                    z[i, 0] = p.at(i).first
                    z[i, 1] = p.at(i).second
        return [ary[1:] for ary in ret]
        # equivalent to:
        # return [list(x)[1:] for x in paths]
        # but this seems to consume a ton of memory
