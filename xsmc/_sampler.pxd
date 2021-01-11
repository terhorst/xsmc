from libc.math cimport log, log1p, exp, isinf
from libcpp cimport unordered_map as map
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdio cimport printf
from scipy.special.cython_special cimport gammaln

from ._tskit cimport *

