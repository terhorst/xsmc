"""Exact samplers for renewal SMC model. Algorithms based on Fearnhead, P. 'Exact and efficient Bayesian inference for multiple changepoint problems'. 2006.
"""
from logging import getLogger
from typing import List, Tuple, Union

import numpy as np

from xsmc._sampler import _SamplerProxy

logger = getLogger(__name__)


class XSMCSampler:
    def __init__(
        self,
        X: np.ndarray,
        deltas: np.ndarray,
        theta: float,
        rho: float,
        robust: bool,
        eps: float = 1e-4,
    ):
        X = np.asarray(X, dtype=np.int32)
        deltas = np.asarray(deltas, dtype=np.float64)
        # restrict to segregating sites
        if robust:
            X = np.minimum(1, X)
        self.X = X
        self.theta = theta
        self.rho = rho
        self.Xcs = np.pad(self.X, [[0, 0], [0, 1]]).cumsum(axis=1)
        self.positions = np.concatenate((np.zeros_like(deltas[:1]), deltas.cumsum()))
        self.robust = robust
        self._proxy = _SamplerProxy(self.Xcs, self.positions, theta, rho, robust, eps)

    @property
    def H(self):
        return self.X.shape[0]

    @property
    def n(self):
        return self.X.shape[1]

    @property
    def log_Q(self):
        return self._proxy.log_Q

    def log_P(self, s, t):
        return self._proxy.log_P(s, t)

    def _sample_path_helper(self, k: int, seed: Union[int, None]):
        paths = self._proxy.sample_paths(k, seed)
        segs = []
        for p in paths:
            pos_inds, hap_inds = p.T
            p0 = np.concatenate([[0], pos_inds[:-1]])
            p1 = pos_inds
            seg = np.array(
                [
                    self.positions[p0],  # intervals
                    self.positions[p1],
                    self.Xcs[hap_inds, p1] - self.Xcs[hap_inds, p0],  # mutations
                    np.zeros_like(p0),  # heights -- initially zero
                ]
            )
            segs.append((seg, hap_inds))
        return segs

    def sample_paths(
        self, k: int, seed: Union[int, None], prime: bool
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if seed is not None:
            np.random.seed(seed)
        segs = self._sample_path_helper(k, seed)
        logger.debug("Sampling heights")
        f = _sample_path_heights_smcp if prime else _sample_heights
        for seg, _ in segs:
            seg[3] = f(
                seg, theta=self.theta, rho=self.rho, robust=self.robust, H=self.H
            )
        logger.debug("Done sampling heights")
        return segs

    def sample_heights(self, j: int, k: int, seed: Union[int, None]) -> np.ndarray:
        "Randomly sample heights at j paths and k random positions from each path."
        if seed is not None:
            np.random.seed(seed)
        segs = self._sample_path_helper(j, seed)
        ret = np.empty((j, k))
        for j, (seg, _) in enumerate(segs):
            p = (seg[1] - seg[0]) / seg[1][-1]
            i = np.random.choice(seg.shape[1], size=k, p=p, replace=True)
            rseg = seg[:, i]
            ret[j] = _sample_heights(
                rseg, theta=self.theta, rho=self.rho, robust=self.robust, H=self.H
            )
        return ret


def _sample_heights(
    segments: np.ndarray, theta: float, rho: float, H: int, robust: bool
) -> np.ndarray:
    """Randomly sample heights from posterior given an array of segments"""
    a = np.ones_like(segments[0])
    a[0] = 0.0
    b = a[::-1]
    delta = segments[1] - segments[0]
    Xb = segments[2]
    alpha = 1 + a + b + Xb
    beta = H + delta * (theta + rho)
    if robust:
        beta -= Xb * theta
    return np.random.gamma(alpha, 1.0 / beta)


def _sample_path_heights_smcp(
    segments: np.ndarray, theta: float, rho: float, H: int, robust: bool
) -> np.ndarray:
    """Randomly sample heights from posterior given an array of segments"""
    a = np.ones_like(segments[0])
    a[0] = 0.0
    b = a[::-1]
    heights = np.zeros_like(a)

    def log_q(t, s):
        return np.log(np.exp(-(max(t, s) - s)) - np.exp(-(min(t, s) + t))) - np.log(
            2 * s
        )

    rej = 0
    for i, (i0, i1, Xb, _) in enumerate(segments.T):
        delta = i1 - i0
        if i == 0:
            # draw from prior
            alpha = 1 + a[i] + b[i] + Xb
            beta = H + delta * (theta + rho)
            heights[i] = np.random.gamma(alpha, 1.0 / beta)
        else:
            # rejection sample
            # target distribution:
            # f(t) = q(t|s) f_gamma(t; alpha, 1. / beta)
            #      = q(t|s) (t rho)^b exp(-rho delta t) pois(Xb; t theta)
            # proposal distribution:
            # g(t) = f_gamma(t; alpha, 1. / beta)
            #      = exp(-t/2) (t rho)^b exp(-rho delta t) pois(Xb; t theta)
            # we have f(t) <= g(t) * exp(s/2) / (2s)
            s = heights[i - 1]
            alpha = 1 + b[i] + Xb
            beta = H / 2.0 + delta * (theta + rho)
            q = 0
            while True:
                t = heights[i] = np.random.gamma(alpha, 1.0 / beta)
                # U <= f(t)/[g(t) * exp(s/2) * (2s)]
                # -log(u) >= log(g * ...) - log(f)
                u = log_q(H * t, s)
                v = -H * t / 2.0 + s / 2 - np.log(2 * s)
                if np.random.exponential() > v - u:
                    break
                rej += 1
                q += 1
                assert q < 10_000
        # logger.debug('rejections:%d', rej)
    return heights
