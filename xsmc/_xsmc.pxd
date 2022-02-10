from _tskit cimport tsk_treeseq_t

cdef class TreeSequence:
    cdef tsk_treeseq_t _ts
