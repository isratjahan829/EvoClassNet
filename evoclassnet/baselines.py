"""Untrained reference baselines.

`mutation_count` ranks variants by how many substitutions they carry (more substitutions, worse
phenotype). It has no parameters, sees no training data and knows nothing about which positions
or residues are involved. Any model evaluated on multi-substitution variants must be compared
against it: a large part of the rank variance in a combinatorial mutagenesis library is carried by
the number of substitutions alone, and a model can appear to "generalise to unseen positions"
while only reproducing that.
"""
import numpy as np


def mutation_count_scores(n_sub):
    """Higher score = predicted fitter. Rank-only; the scale is arbitrary."""
    return -np.asarray(n_sub, dtype=float)
