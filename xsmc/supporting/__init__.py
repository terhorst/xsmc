"Supporting classes and functions."

import numpy as np


def watterson(ts: "tskit.TreeSequence"):
    "Returns Watterson's estimate of `4 * N0 * mu` computed from tree sequence."
    K = ts.get_num_sites()
    n = ts.get_sample_size()
    L = ts.get_sequence_length()
    Hn1 = (1.0 / np.arange(1, n)).sum()
    return K / Hn1 / L
