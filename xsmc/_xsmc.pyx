from _tskit cimport LightweightTableCollection, tsk_treeseq_t, tsk_treeseq_init, tsk_treeseq_free, TSK_BUILD_INDEXES, tsk_treeseq_get_num_sites, tsk_treeseq_get_sequence_length

cdef class TreeSequence:
    def __init__(self, LightweightTableCollection lwtc):
        tsk_treeseq_init(&self._ts, lwtc.tables, TSK_BUILD_INDEXES)

    def __dealloc__(self):
        tsk_treeseq_free(&self._ts)

    @property
    def num_sites(self):
        return tsk_treeseq_get_num_sites(&self._ts)

    @property
    def sequence_length(self):
        return tsk_treeseq_get_sequence_length(&self._ts)
