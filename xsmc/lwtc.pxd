from _tskit cimport tsk_table_collection_t, tsk_table_collection_build_index, tsk_treeseq_t, tsk_treeseq_init, check_error

cdef extern:
    ctypedef class xsmc._lwtc.LightweightTableCollection [object LightweightTableCollection]:
        cdef tsk_table_collection_t *tables

cdef inline void from_ts(pyts, tsk_treeseq_t *ts):
    cdef LightweightTableCollection lwtc = LightweightTableCollection()
    lwtc.fromdict(pyts.dump_tables().asdict())
    cdef int err = tsk_table_collection_build_index(lwtc.tables, 0)  # this recently became necessary?
    check_error(err)
    err = tsk_treeseq_init(ts, lwtc.tables, 0)
    check_error(err)
