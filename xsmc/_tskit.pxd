from libc.stdint cimport int8_t, int16_t, int32_t, uint32_t
from libcpp cimport bool


cdef extern from "tskit.h" nogil:
    ctypedef int32_t tsk_id_t
    ctypedef uint32_t tsk_size_t
    ctypedef uint32_t tsk_flags_t
    ctypedef struct tsk_mutation_t:
        tsk_id_t id
        tsk_id_t site
        tsk_id_t node
        tsk_id_t parent
        const char *derived_state
        tsk_size_t derived_state_length
    ctypedef struct tsk_site_t:
        double position;
        const char *ancestral_state
        tsk_size_t ancestral_state_length
        const char *metadata
        tsk_size_t metadata_length
        tsk_mutation_t *mutations
        tsk_size_t mutations_length
    cdef union genotypes:
        int8_t *i8
        int16_t *i16
    ctypedef struct tsk_variant_t:
        tsk_site_t *site
        const char **alleles
        tsk_size_t *allele_lengths
        tsk_size_t num_alleles
        tsk_size_t max_alleles
        bool has_missing_data
        genotypes genotypes
    ctypedef struct tsk_treeseq_t:
        pass
    ctypedef struct tsk_tree_t:
        double left
        double right
        tsk_id_t* left_child
        tsk_id_t* right_child
        tsk_site_t* sites
        tsk_size_t sites_length
    ctypedef struct tsk_vargen_t:
        size_t num_samples
        size_t tree_site_index
        tsk_tree_t tree
        tsk_treeseq_t *tree_sequence
    ctypedef struct tsk_table_collection_t:
        pass
    enum: TSK_NULL
    enum: TSK_BUILD_INDEXES
    bool tsk_tree_is_sample(tsk_tree_t *self, tsk_id_t u)
    int tsk_vargen_init(tsk_vargen_t *self, tsk_treeseq_t *tree_sequence,
            tsk_id_t *samples, size_t num_samples, const char **alleles,
            tsk_flags_t options);
    int tsk_vargen_next(tsk_vargen_t *self, tsk_variant_t **variant);
    int tsk_vargen_prev(tsk_vargen_t *self, tsk_variant_t **variant);
    int tsk_vargen_goto_end(tsk_vargen_t *self)
    int tsk_vargen_free(tsk_vargen_t *self);
    int tsk_tree_get_parent(tsk_tree_t *self, tsk_id_t u, tsk_id_t *parent);
    int tsk_tree_get_time(tsk_tree_t *self, tsk_id_t u, double *t)
    int tsk_treeseq_init(tsk_treeseq_t *self, tsk_table_collection_t *tables, tsk_flags_t options)
    double tsk_treeseq_get_sequence_length(tsk_treeseq_t *self)
    tsk_size_t tsk_treeseq_get_num_sites(tsk_treeseq_t *self)
    # void tsk_vargen_print_state(tsk_vargen_t *self, FILE *out);

cdef extern:
    ctypedef class xsmc._tskit.LightweightTableCollection [object LightweightTableCollection]:
        cdef tsk_table_collection_t *tables
