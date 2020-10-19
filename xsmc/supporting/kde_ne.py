"Estimation of Ne(t) by kernel smoothing."

from collections import Counter

import numpy as np
from scipy.signal import convolve


def nelson_aalen(x):
    "Compute the Nelson-Aalen estimator given a sample of survival times `x`."
    c = Counter(x)
    xu = np.array(sorted(c))
    y = np.array([c[xui] for xui in xu])  # "deaths"
    yc = (y[:-1] / (y.sum() - y.cumsum()[:-1])).cumsum()
    xu = xu[:-1]
    return xu, yc


def kde_ne(heights, b=None):
    "Estimate 1/hazard rate by smoothing increments of the Nelson-Aalen estimator."
    if b is None:
        b = int(0.02 * len(heights))
    x = np.sort(heights.reshape(-1))
    c = Counter(x)
    xu, yc = nelson_aalen(heights)
    # w = np.ones(b)
    w = np.hamming(b)
    # yc[b:] - yc[:-b]  # "local" bandwidth, essentially k-NN
    dy = convolve(np.diff(yc), w, "valid")
    dx = convolve(np.diff(xu), w, "valid")
    return (xu[b // 2 : -b // 2], dx / dy)
