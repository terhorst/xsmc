from libc.math cimport log, log1p, exp, isinf
from libcpp cimport unordered_map as map
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdio cimport printf
from scipy.special.cython_special cimport gammaln
from xsmc._tskit cimport VariantGenerator, tsk_vargen_t, tsk_variant_t, tsk_vargen_next, tsk_id_t

