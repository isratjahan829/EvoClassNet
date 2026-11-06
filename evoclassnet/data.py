"""Loading and splitting the public SARS-CoV-2 RBD deep-mutational-scanning data
(Starr et al. 2020, Cell; github.com/jbloomlab/SARS-CoV-2-RBD_DMS).

Each barcoded variant carries 0-9 amino-acid substitutions on the 201-residue RBD
(Spike 331-531) and has a measured ACE2 binding (log10 Ka, delta vs wild type) and
expression (delta mean fluorescence) value. Barcodes sharing a genotype are collapsed
so that identical sequences never straddle the train/test boundary.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "bloom_rbd_dms"
_SUB = re.compile(r"^([A-Z])(\d+)([A-Z*])$")


def wildtype():
    s = pd.read_csv(RAW / "RBD_sites.csv").sort_values("site_RBD")
    return "".join(s.amino_acid_SARS2), s.site_SARS2.min() - 1  # offset: RBD idx -> Spike


def parse_subs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for t in s.split():
        m = _SUB.match(t)
        out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def load_genotypes(min_barcodes=1):
    """One row per unique amino-acid genotype with mean binding / expression."""
    wt, _ = wildtype()
    b = pd.read_csv(RAW / "binding_Kds.csv", usecols=["library", "barcode", "aa_substitutions", "delta_log10Ka", "variant_class"])
    e = pd.read_csv(RAW / "expression_meanFs.csv", usecols=["library", "barcode", "aa_substitutions", "delta_ML_meanF"])
    d = b.merge(e[["library", "barcode", "delta_ML_meanF"]], on=["library", "barcode"], how="outer")
    d["aa_substitutions"] = d["aa_substitutions"].fillna("")
    d = d[~d.variant_class.isin(["stop"]) & ~d.aa_substitutions.str.contains(r"\*", regex=True)]
    g = d.groupby("aa_substitutions").agg(bind=("delta_log10Ka", "mean"), expr=("delta_ML_meanF", "mean"),
                                          n_bc=("barcode", "size")).reset_index()
    g = g[g.n_bc >= min_barcodes]
    g["subs"] = g.aa_substitutions.map(parse_subs)
    for a, p, _ in (x for s in g.subs for x in s):
        pass
    # sanity: every wild-type residue must agree with RBD_sites
    for s in g.subs.head(2000):
        for w, p, m in s:
            assert wt[p - 1] == w, (w, p, wt[p - 1])
    g["n_sub"] = g.subs.map(len)
    g["sites"] = g.subs.map(lambda s: frozenset(p for _, p, _ in s))
    return g.reset_index(drop=True)


def encode(subs_list, wt=None):
    wt = wt or wildtype()[0]
    base = np.array([AA2I[a] for a in wt], dtype=np.int64)
    X = np.tile(base, (len(subs_list), 1))
    for i, s in enumerate(subs_list):
        for _, p, m in s:
            X[i, p - 1] = AA2I[m]
    return X


def split_random(g, seed, frac=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(g)); n = len(g)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def split_heldout_sites(g, seed, test_site_frac=0.15, val_site_frac=0.10):
    """Strict generalisation split: test variants only touch held-out sites; no training
    variant mutates a held-out site (neither test nor validation sites). Variants spanning
    both held-out and training sites are discarded."""
    rng = np.random.default_rng(seed)
    sites = np.arange(1, 202); rng.shuffle(sites)
    nt, nv = int(test_site_frac * 201), int(val_site_frac * 201)
    test_s, val_s, train_s = set(sites[:nt]), set(sites[nt:nt + nv]), set(sites[nt + nv:])
    tr, va, te = [], [], []
    for i, s in enumerate(g.sites):
        if not s:  # wild type -> train
            tr.append(i); continue
        if s <= train_s: tr.append(i)
        elif s <= val_s: va.append(i)
        elif s <= test_s: te.append(i)
    return np.array(tr), np.array(va), np.array(te)
