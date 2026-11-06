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


_CACHE = {}


def add_mutation_count(runs, split_random=None, split_heldout=None):
    """Evaluate the untrained mutation-count baseline on exactly the partitions each run used.

    `runs` maps a job name (``main_random_s3``, ``sweep_f30_s1``, ...) to that run's result dict.
    The baseline is injected under the key ``mutation_count`` in place, and the dict is returned.
    This lives here rather than in a script because both the aggregation step and the manuscript
    build need it, and the job JSON files on disk do not contain it.
    """
    if not runs:
        return runs
    from scipy.stats import spearmanr, pearsonr
    from . import data as _d
    sr = split_random or _d.split_random
    sh = split_heldout or _d.split_heldout_sites
    if "g" not in _CACHE:
        g = _d.load_genotypes()
        _CACHE.update(g=g, Y=g[["bind", "expr", "escape"]].to_numpy(float), n=g.n_sub.to_numpy())
    g, Y, nsub = _CACHE["g"], _CACHE["Y"], _CACHE["n"]
    for name, r in runs.items():
        if "mutation_count" in r:
            continue
        seed = int(name.split("_s")[-1])
        frac = r.get("_test_frac", 0.15)
        _, _, te = sr(g, seed) if "random" in name else sh(g, seed, test_site_frac=frac, val_site_frac=0.10)
        p = mutation_count_scores(nsub[te])
        out = {}
        for i, t in enumerate(["bind", "expr", "escape"]):
            m = ~np.isnan(Y[te, i])
            if m.sum() < 3 or np.ptp(p[m]) < 1e-9:
                out[t] = dict(spearman=0.0, pearson=0.0, rmse=float("nan"), n=int(m.sum()), constant=True)
            else:
                out[t] = dict(spearman=float(spearmanr(Y[te, i][m], p[m])[0]),
                              pearson=float(pearsonr(Y[te, i][m], p[m])[0]),
                              rmse=float("nan"), n=int(m.sum()))
        r["mutation_count"] = out
    return runs
