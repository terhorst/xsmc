from libc.math cimport exp, isinf, log, log1p
from libc.stdio cimport printf
from libcpp cimport bool
from libcpp cimport unordered_map as map
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from scipy.special.cython_special cimport gammaln

from xsmc._tskit cimport (TSK_BUILD_INDEXES, LightweightTableCollection,
                          tsk_flags_t, tsk_id_t,
                          tsk_treeseq_get_sequence_length, tsk_treeseq_init,
                          tsk_treeseq_t, tsk_vargen_init, tsk_vargen_next,
                          tsk_vargen_t, tsk_variant_t)
