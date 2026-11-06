"""Coronavirus genome classification: data loading, features and splits.

This is the second task in the study and it exists to test whether the evaluation lesson from the
deep-mutational-scanning benchmark generalises beyond that one assay. The structure is deliberately
parallel:

  random split            genomes are shuffled. Near-identical sequences from the same species sit
                          on both sides, so the test measures retrieval of memorised sequences.
  held-out-species split  no species appears in both training and test, so a test genome must be
                          classified from a genus-level signal rather than by resemblance to a
                          specific relative already seen.

and the trivial controls play the role the mutation-count baseline plays there: genome length and
GC content require no training and no sequence model at all, yet carry real class information
because coronavirus genera differ systematically in both.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "genomes"
GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
NTS = "ACGT"


def read_fasta(path):
    recs, acc, buf = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                recs.append((acc, "".join(buf)))
            acc = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        recs.append((acc, "".join(buf)))
    return recs


def load(min_len=25000, max_len=35000):
    """Return a dataframe of accession, genus, species, sequence."""
    import pandas as pd
    rows = []
    for gi, genus in enumerate(GENERA):
        fa, mj = RAW / f"{genus}.fasta", RAW / f"{genus}.json"
        if not fa.exists():
            continue
        meta = json.load(open(mj))
        for acc, seq in read_fasta(fa):
            seq = re.sub(r"[^ACGTacgt]", "", seq).upper()
            if not (min_len <= len(seq) <= max_len):
                continue
            m = meta.get(acc, {})
            rows.append(dict(accession=acc, genus=genus, label=gi,
                             species=m.get("organism", "unknown"), seq=seq, length=len(seq)))
    df = pd.DataFrame(rows)
    # drop exact duplicate sequences, which would otherwise straddle any split
    return df.drop_duplicates(subset="seq").reset_index(drop=True)


def kmer_features(seqs, k=5, log=None):
    """Normalised k-mer frequency vectors. 4^k columns; k=5 gives 1024."""
    idx = {}
    for i in range(4 ** k):
        s, n = [], i
        for _ in range(k):
            s.append(NTS[n % 4]); n //= 4
        idx["".join(reversed(s))] = i
    X = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        c = Counter(s[i:i + k] for i in range(len(s) - k + 1))
        tot = sum(c.values()) or 1
        for km, v in c.items():
            j = idx.get(km)
            if j is not None:
                X[r, j] = v / tot
        if log and r % 200 == 0:
            log(f"  k-mers {r}/{len(seqs)}")
    return X


def trivial_features(df):
    """Genome length and base composition: no training, no sequence model."""
    out = np.zeros((len(df), 6), dtype=np.float32)
    for i, s in enumerate(df.seq):
        n = len(s) or 1
        counts = [s.count(b) / n for b in NTS]
        out[i] = [len(s), (counts[1] + counts[2])] + counts
    return out


def split_random(df, seed, frac=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    a, b = int(frac[0] * len(df)), int((frac[0] + frac[1]) * len(df))
    return idx[:a], idx[a:b], idx[b:]


def split_heldout_species(df, seed, frac=(0.7, 0.15, 0.15)):
    """Partition species, then assign every genome of a species to that species' partition.

    Species are allocated genus by genus so that each partition keeps all four classes; without
    that, a random species partition can leave a rare genus with no test genomes at all.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for g in sorted(df.genus.unique()):
        sub = df[df.genus == g]
        sp = rng.permutation(sorted(sub.species.unique()))
        a, b = max(1, int(frac[0] * len(sp))), max(2, int((frac[0] + frac[1]) * len(sp)))
        sets = (set(sp[:a]), set(sp[a:b]), set(sp[b:]))
        for dest, keep in zip((tr, va, te), sets):
            dest.extend(sub.index[sub.species.isin(keep)].tolist())
    return np.array(tr), np.array(va), np.array(te)
