import msprime as msp
import tskit
import pytest
import numpy as np
import itertools as it

from xsmc.arg import make_trunk, thread
from xsmc._viterbi import viterbi_path
from xsmc.size_history import KINGMAN

@pytest.fixture(params=range(1, 10))
def rng(request):
    return np.random.default_rng(seed=request.param)

@pytest.fixture(params=[25])
def sample_size(request):
    return request.param

def test_matching_arg_ts_lengths():
    ts = msp.simulate(sample_size=2, length=1e5, mutation_rate=1e-3, recombination_rate=1e-3)
    focal = 0
    panel = [1]
    arg = make_trunk(panel, 1e2)
    with pytest.raises(AssertionError):
        viterbi_path(ts, focal, panel, arg, KINGMAN, 1e-3, 1e-3, None, False, w=10)
    with pytest.raises(AssertionError):
        viterbi_path(ts, focal, panel, arg, KINGMAN, 1e-3, 1e-3, None, False, w=100)
    viterbi_path(ts, focal, panel, arg, KINGMAN, 1e-3, 1e-3, None, False, w=1_000)



def test_arg_finite_coalescent(rng, sample_size):
    'test decoding an arg where the trunk ends at a certain finite point'
    L = 1e4
    w = 1e3
    ts = msp.simulate(sample_size=sample_size, length=L, mutation_rate=1e-3, recombination_rate=1e-3)
    focal = 0
    panel = list(range(1, sample_size))
    tc = tskit.TableCollection(int(L // w))
    ref_tc = make_trunk(panel, int(L // w)).dump_tables()
    tc.individuals.metadata_schema = ref_tc.individuals.metadata_schema
    sample_heights = []
    for p in panel:
        sample_heights.append(rng.exponential())
        t = sample_heights[-1]
        r = tc.nodes.add_row(time=t)
        i = tc.individuals.add_row(metadata={"sample_id": p})
        n = tc.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, individual=i, time=0.)
        tc.edges.add_row(0, tc.sequence_length, parent=r, child=n)
    tc.sort()
    arg = tc.tree_sequence()
    seg = viterbi_path(ts, focal, panel, arg, KINGMAN, 1e-3, 1e-3, None, False, w)
    for s in seg.segments:
        t = sample_heights[s.hap - 1]
        assert t - s.height >= -1e-8  # some small numerical error might occur


def test_scaffolding(rng, sample_size):
    'test threading onto a scaffold'
    if sample_size == 2:
        pytest.xfail("need more than two samples to thread")
    ts = msp.simulate(sample_size=sample_size, length=10, mutation_rate=1e-1, recombination_rate=1e-1)
    scaffold = msp.simulate(sample_size=sample_size - 1, length=10)
    focal = 0
    panel = list(range(1, sample_size))
    w = 1
    seg = viterbi_path(ts, focal, panel, scaffold, KINGMAN, 1e-3, 1e-3, None, False, w)
    tree = scaffold.first()
    for s in seg.segments:
        t = tree.get_time(tree.get_parent(s.hap - 1))
        assert t - s.height >= -1e-8  # some small numerical error might occur
        
        
def test_thread_func(rng, sample_size):
    'test conversion of threading into arg'
    # build up an arg using <sample_size> samples
    ts = msp.simulate(sample_size=sample_size, length=100, mutation_rate=1e-1, recombination_rate=1e-1, 
                      random_seed=rng.integers(1, 1_000_000))
    scaffold = make_trunk([0], 100)
    for focal in range(1, sample_size):
        panel = list(range(focal))
        seg = viterbi_path(ts, focal, panel, scaffold, KINGMAN, 1e-1, 1e-1, None, False, 1)
        new_scaffold = thread(scaffold, seg)
        projection = new_scaffold.simplify(panel)
        for attr in ["nodes", "edges"]:
            assert getattr(scaffold.tables, attr) == getattr(projection.tables, attr)
            assert getattr(scaffold.tables, attr) != getattr(new_scaffold.tables, attr)
        new_trees = new_scaffold.trees()
        new_t = next(new_trees)
        for t in scaffold.trees():
            # the trees in new_scaffold should strictly refine the trees in the old scaffold
            while t.interval[0] <= new_t.interval[0] <= new_t.interval[1] <= t.interval[1]:
                for n1, n2 in it.combinations(panel, 2):
                    assert t.get_tmrca(n1, n2) == new_t.get_tmrca(n1, n2)
                try:
                    new_t = next(new_trees)
                except StopIteration:
                    break
        scaffold = new_scaffold
        