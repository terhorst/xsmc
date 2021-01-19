from libc.math cimport exp, isinf, log, log1p
from libc.stdio cimport printf
from libcpp cimport bool
from libcpp cimport unordered_map as map
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from scipy.special.cython_special cimport gammaln

from ._tskit cimport *
