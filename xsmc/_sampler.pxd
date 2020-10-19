from libc.math cimport log, log1p, exp, isinf
from libcpp cimport unordered_map as map
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdio cimport printf
from scipy.special.cython_special cimport gammaln
from xsmc._tskit cimport LightweightTableCollection, tsk_vargen_t, tsk_treeseq_t, tsk_variant_t, tsk_vargen_next, tsk_id_t, tsk_vargen_init, tsk_treeseq_init, tsk_flags_t, TSK_BUILD_INDEXES, tsk_treeseq_get_sequence_length