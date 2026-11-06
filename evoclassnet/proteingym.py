"""ProteinGym: 217 deep mutational scanning assays, used to ask whether the evaluation result
found on the SARS-CoV-2 RBD holds across proteins.

The RBD study establishes three things on one protein: that a held-out-position split is
uninterpretable without an untrained mutation-count control, that models which index a
substitution by its position collapse there, and that models whose features remain defined at an
unseen position do not. A finding on one assay is an anecdote. This module runs the same
comparison over every substitution assay in ProteinGym v1.3, so the claim can be stated as a
property of the benchmark rather than of one protein.

Each assay provides a target sequence, a set of variants written as `A123B` (colon-separated for
multiples) and a continuous score. Nothing else is needed: the splits, the untrained baselines
and the transferable features are all computable from those.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "swhitfield/proteingym-dms_substitutions"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "proteingym"
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _dl(filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, filename, repo_type="dataset",
                           cache_dir=str(CACHE) if CACHE.exists() else None)


@lru_cache(maxsize=1)
def reference():
    """One row per assay: dms_id, uniprot_id, coarse_selection_type, target_seq."""
    return pd.read_parquet(_dl("reference.parquet"))


def assay_ids(max_len=None, selection=None):
    ref = reference()
    m = ~ref.is_indel & ~ref.is_clinical
    if max_len:
        m &= ref.target_seq.str.len() <= max_len
    if selection:
        m &= ref.coarse_selection_type.isin(selection)
    return ref.dms_id[m].tolist()


def load_assay(dms_id):
    """Return (dataframe, target_seq). The frame gains `subs` (list of (wt, pos, mut), 1-based)
    and `n_sub`. Variants whose stated wild-type residue disagrees with the target sequence are
    dropped, as are duplicated variants."""
    ref = reference()
    row = ref[ref.dms_id == dms_id]
    if not len(row):
        raise KeyError(dms_id)
    seq = row.iloc[0].target_seq
    df = pd.read_parquet(_dl(f"data/{dms_id}.parquet"))
    df = df[["mutant", "dms_score"]].dropna().drop_duplicates(subset="mutant")

    subs, keep = [], []
    for mut in df.mutant:
        parsed, ok = [], True
        for tok in str(mut).split(":"):
            m = MUT_RE.match(tok.strip())
            if not m:
                ok = False
                break
            w, p, a = m.group(1), int(m.group(2)), m.group(3)
            if not (1 <= p <= len(seq)) or seq[p - 1] != w:
                ok = False
                break
            parsed.append((w, p, a))
        keep.append(ok and bool(parsed))
        subs.append(parsed)
    df = df.assign(subs=subs)[keep].reset_index(drop=True)
    df["n_sub"] = df.subs.map(len)
    df["sites"] = df.subs.map(lambda s: {p for _, p, _ in s})
    return df, seq


def split_random(df, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    k = int(frac * len(df))
    return idx[:k], idx[k:]


def split_heldout_positions(df, seed, test_site_frac=0.2):
    """Partition positions; a variant joins a side only if every position it mutates is on that
    side. Variants spanning the partition are discarded, exactly as in the RBD study."""
    rng = np.random.default_rng(seed)
    sites = sorted({p for s in df.sites for p in s})
    if len(sites) < 10:
        return np.array([], int), np.array([], int)
    perm = rng.permutation(sites)
    n_test = max(2, int(test_site_frac * len(perm)))
    test_sites, train_sites = set(perm[:n_test]), set(perm[n_test:])
    tr = np.array([i for i, s in enumerate(df.sites) if s <= train_sites], int)
    te = np.array([i for i, s in enumerate(df.sites) if s <= test_sites], int)
    return tr, te


def summarise(ids=None, max_assays=None):
    """Quick inventory: size, position count, and how many variants carry multiple substitutions."""
    ids = ids or assay_ids()
    rows = []
    for i, d in enumerate(ids[:max_assays] if max_assays else ids):
        try:
            df, seq = load_assay(d)
        except Exception as e:
            rows.append(dict(dms_id=d, error=str(e)[:60]))
            continue
        rows.append(dict(dms_id=d, n_variants=len(df), seq_len=len(seq),
                         n_positions=len({p for s in df.sites for p in s}),
                         frac_multi=float((df.n_sub > 1).mean()),
                         max_nsub=int(df.n_sub.max())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# ProteinGym's own cross-validation schemes.
#
# The supervised ProteinGym benchmark (ProteinNPT, Notin et al. 2023) defines three fold
# variables: `fold_random_5`, `fold_contiguous_5` and `fold_modulo_5`. The latter two are
# position-based, so withholding positions is an established part of this benchmark and not
# something proposed here. The published fold assignments ship in a separate archive that is not
# reachable from this environment, but both position schemes are deterministic functions of
# residue position and are reconstructed exactly below, so results here are directly comparable
# to the leaderboard rather than to a private split.
#
#   modulo:     fold(position) = position mod 5
#   contiguous: the sequence is cut into 5 consecutive blocks of positions, one per fold
#
# A variant belongs to the test fold only if every position it mutates does, which matches the
# treatment of multi-substitution variants used throughout this study.
N_FOLDS = 5


def fold_modulo(df, seq_len, test_fold=0, n_folds=N_FOLDS):
    """ProteinGym `fold_modulo_5`: positions are assigned to folds by position mod n_folds."""
    test_sites = {p for p in range(1, seq_len + 1) if (p - 1) % n_folds == test_fold}
    return _by_sites(df, test_sites)


def fold_contiguous(df, seq_len, test_fold=0, n_folds=N_FOLDS):
    """ProteinGym `fold_contiguous_5`: the sequence is cut into n_folds consecutive blocks."""
    edges = np.linspace(0, seq_len, n_folds + 1).astype(int)
    lo, hi = edges[test_fold], edges[test_fold + 1]
    test_sites = set(range(lo + 1, hi + 1))
    return _by_sites(df, test_sites)


def _by_sites(df, test_sites):
    tr, te = [], []
    for i, s in enumerate(df.sites):
        if s <= test_sites:
            te.append(i)
        elif not (s & test_sites):
            tr.append(i)
    return np.array(tr, int), np.array(te, int)
