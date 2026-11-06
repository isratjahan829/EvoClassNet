"""Antibody-escape labels from the aggregated RBD escape maps
(github.com/jbloomlab/SARS2_RBD_Ab_escape_maps; Greaney et al. and 12 further studies).

For every single amino-acid substitution we average the per-antibody escape fraction over all
monoclonal antibodies that were assayed against it. Substitutions measured against fewer than
`min_ab` antibodies are dropped, because a mean over a handful of antibodies is dominated by
which antibodies happened to be chosen. Escape fractions are strongly right-skewed, so the label
is log10(mean escape); this is the quantity the model is asked to predict.

A small fraction of rows (~1.5 %) were measured in a non-Wuhan-Hu-1 background (the listed
wild-type residue is an Omicron residue, e.g. P373, Y501, H505). Those rows are dropped, because
the model and the DMS training data use the Wuhan-Hu-1 background and because the affected sites
are precisely the Variant-of-Concern sites used in the retrospective analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "bloom_escape"


def load_escape(min_ab=20, kind="antibody", background=None, offset=330):
    """-> DataFrame[site_spike, wildtype, mutation, escape, n_ab] with escape = log10(mean fraction).

    `background`: the 201-residue wild-type RBD sequence; rows whose listed wild-type residue
    disagrees with it are dropped. Defaults to the Wuhan-Hu-1 sequence from RBD_sites.csv.
    """
    if background is None:
        from .data import wildtype
        background, offset = wildtype()
    e = pd.read_csv(RAW / "escape_data.csv", low_memory=False,
                    usecols=["condition", "condition_type", "site", "wildtype", "mutation", "mut_escape"])
    e = e[e.condition_type == kind].dropna(subset=["mut_escape"])
    pos = e.site - offset
    ok = (pos >= 1) & (pos <= len(background))
    e, pos = e[ok], pos[ok]
    e = e[e.wildtype.values == np.array([background[p - 1] for p in pos])]
    g = e.groupby(["site", "wildtype", "mutation"]).mut_escape.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_ab]
    g = g.rename(columns={"site": "site_spike", "mean": "escape_raw", "size": "n_ab"})
    g["escape"] = np.log10(g.escape_raw)
    return g[["site_spike", "wildtype", "mutation", "escape", "escape_raw", "n_ab"]]


def escape_lookup(min_ab=20, offset=330):
    """(wt, rbd_position, mutant) -> log10 mean escape."""
    g = load_escape(min_ab)
    return {(r.wildtype, int(r.site_spike) - offset, r.mutation): r.escape for r in g.itertuples()}
