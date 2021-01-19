from libcpp.iterator import back_inserter
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libc.string cimport memcmp, memset, memcpy
from libcpp.algorithm cimport lower_bound
from libc.stdio cimport printf
from libcpp cimport bool
from libc.stdint cimport int32_t
from libc.math cimport INFINITY, isinf, isfinite
from gsl cimport *

from _tskit cimport *

cdef extern from "<algorithm>" namespace "std" nogil:
    cdef OutputIter set_union[Iter1, Iter2, OutputIter](
        Iter1 first1,
        Iter1 last1,
        Iter2 first2,
        Iter2 last2,
        OutputIter result)

cdef extern from "<cmath>" namespace "std" nogil:
    bool isinf(double)
    double fabs(double)
    double fmin(double, double)
    double fmax(double, double)
    double exp(double)
    double log(double)

###### Typedefs
# data structures used to represent functions of the form
# c[0] exp(-x) + c[1] x + c[2], t[0] <= x < t[1]
ctypedef double[2] interval

ctypedef struct func:
    double[3] c
    int k
    
ctypedef struct piecewise_func:
    vector[func] f
    vector[double] t

ctypedef struct minimum:
    double f, x

ctypedef struct backtrace:
    int hap, pos, s
    minimum m

##### Public Functions
# cdef vector[piecewise_func] piecewise_min(vector[piecewise_func], double, vector[piecewise_func]) nogil
# cdef vector[piecewise_func] chop(vector[piecewise_func], double) nogil
# cdef vector[piecewise_func] piecewise_const_log_pi(double[:], double[:], double beta) nogil
# cdef vector[piecewise_func] compact(vector[piecewise_func]) nogil
# cdef vector[vector[piecewise_func]] construct_prior(double[:], double[:], double[:], uint[:], double) nogil
# cdef minimum min_f(const func, const interval) nogil
