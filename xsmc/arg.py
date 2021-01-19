from typing import List, Sequence
import intervals as I

import tskit
from .segmentation import Segmentation


def make_trunk(samples: List[int], sequence_length: int) -> tskit.TreeSequence:
    """Create a "trunk genealogy" tree sequence consisting of forest of unrooted sticks, one per sample.

    Args:
        samples: List of integer sample ids corresponding to each node.
        sequence_length: Length of resulting tree sequence.

    Notes:
        Due to the design of tskit, the nodes of the resulting tree sequence will be numbered
        0, ..., len(samples). Each node can be mapped back to its corresponding sample using the individuals table.
        For example:

            >>> samples = list(range(101, 110))
            >>> ts = make_trunk(samples=samples, sequence_length=1000)
            >>> isinstance(ts, tskit.TreeSequence)
            True
            >>> ts.get_sequence_length()
            1000.0
            >>> ts.get_sample_size() == len(samples)
            True
            >>> ts.samples()[0]
            0
            >>> ts.individual(ts.node(0).individual).metadata['sample_id']
            101

    """
    tc = tskit.TableCollection(sequence_length)
    tc.individuals.metadata_schema = tskit.metadata.MetadataSchema(
        {
            "codec": "struct",
            "type": "object",
            "properties": {
                "sample_id": {"type": "integer", "binaryFormat": "i"},
            },
            "required": ["sample_id", "collection_date"],
            "additionalProperties": False,
        }
    )
    for s in samples:
        i = tc.individuals.add_row(metadata={"sample_id": s})
        n = tc.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, individual=i, time=0.)  # TODO heterochronous?
    assert n == len(samples) - 1
    return tc.tree_sequence()


def thread(
    scaffold: tskit.TableCollection,
    segmentation: Segmentation
) -> tskit.TreeSequence:
    """Thread a new decoding into an existing scaffold.

    Args:
        scaffold: An existing tree sequence to weave in the newly decoded chromosome.
        segmentation: A segmentation representing the local TMRCA at each position.
    Returns:
        A new table collection with the additional chromosome threaded in.

    Note:
        :spans:, :nodes: and :times: should all have the same length.
    """
    # FIXME this algorithm is very inefficient
    ret = scaffold.dump_tables()
    new_edges = ret.edges
    new_edges.reset()
    # the labeled leaf nodes / haplotypes in our sample
    n = ret.nodes.add_row(time=0.0, flags=tskit.NODE_IS_SAMPLE)
    i = 0
    for h, (left, right), t, _ in segmentation.segments:
        # find all of the edges which are ancestral to h and overlap the given
        # time interval. break the edge, insert a new node and three new edges
        # to mark the new coalescent event.
        assert left < right
        sub_tables = scaffold.keep_intervals([(left, right)], simplify=False).dump_tables()
        sub_nodes = sub_tables.nodes
        ancestral_nodes = I.IntervalDict()
        ancestral_nodes[I.closedopen(left, right)] = h
        for edge in sub_tables.edges:

            def add_edge(
                left=edge.left, right=edge.right, parent=edge.parent, child=edge.child
            ):
                new_edges.add_row(left, right, parent, child)

            if edge.child in ancestral_nodes.values():
                edge_interval = I.closedopen(edge.left, edge.right)
                if (
                    sub_nodes[edge.child].time == t
                ):  # the algorithm can produce polytomies
                    add_edge(parent=edge.child, child=n)
                    add_edge()
                    del ancestral_nodes[edge_interval]
                elif sub_nodes[edge.child].time < t < sub_nodes[edge.parent].time:
                    inserted_node = ret.nodes.add_row(time=t)
                    additional_edges = [
                        (inserted_node, edge.child),  # node to old child
                        (inserted_node, n),  # node to leaf
                        (edge.parent, inserted_node),  # node to old parent
                    ]
                    for p, c in additional_edges:
                        add_edge(parent=p, child=c)
                        assert ret.nodes[c].time < ret.nodes[p].time, (
                            c,
                            p,
                            ret.nodes[c],
                            ret.nodes[p],
                            inserted_node,
                        )
                    del ancestral_nodes[edge_interval]
                else:
                    # edge is ancestral but doesn't overlap time interval
                    ancestral_nodes[edge_interval] = edge.parent
                    add_edge()
            else:
                # edge is not ancestral, so add it straightaway
                add_edge()
        if ancestral_nodes:
            # for these intervals, the tmrca <= coalescence time of n
            for intv, c in ancestral_nodes.items():
                # tmrca is above node, so create new edges
                if sub_nodes[c].time < t:
                    p = ret.nodes.add_row(time=t)
                    new_edges.add_row(
                        left=intv.lower, right=intv.upper, parent=p, child=c
                    )
                else:
                    # tmrca is at edge
                    p = c
                new_edges.add_row(left=intv.lower, right=intv.upper, parent=p, child=n)
    ret.sort()
    ret.simplify()
    # print(ret.tree_sequence().draw_text())
    return ret.tree_sequence()