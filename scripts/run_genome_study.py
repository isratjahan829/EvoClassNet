"""Coronavirus genus classification under a random split and a held-out-species split.

This is the second half of the study. The question is the same one the deep-mutational-scanning
benchmark asks: how much of a reported accuracy survives when the evaluation stops letting the
model recognise something it has already seen, and how much of what survives is explained by a
control that needs no model at all.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import genomes as G

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def scores(y, pred, proba=None, n_class=4):
    out = dict(accuracy=float(accuracy_score(y, pred)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
               weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
               mcc=float(matthews_corrcoef(y, pred)) if len(set(y)) > 1 else float("nan"),
               n=int(len(y)))
    per = f1_score(y, pred, average=None, labels=list(range(n_class)), zero_division=0)
    out["per_class_f1"] = {G.GENERA[i]: float(v) for i, v in enumerate(per)}
    out["confusion"] = confusion_matrix(y, pred, labels=list(range(n_class))).tolist()
    if proba is not None:
        try:
            out["log_loss"] = float(log_loss(y, proba, labels=list(range(n_class))))
        except Exception:
            out["log_loss"] = float("nan")
    return out


def cnn_fit_predict(seqs_tr, y_tr, seqs_te, seed, max_len=6000, epochs=12):
    """Small 1D CNN over the nucleotide sequence, the genome-level analogue of EvoClassNet."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    idx = {b: i + 1 for i, b in enumerate(G.NTS)}          # 0 = padding

    def enc(seqs):
        X = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            X[i, :len(s)] = [idx.get(c, 0) for c in s]
        return torch.tensor(X)

    Xtr, Xte = enc(seqs_tr), enc(seqs_te)
    ytr = torch.tensor(np.asarray(y_tr), dtype=torch.long)

    net = nn.Sequential(
        nn.Embedding(5, 16, padding_idx=0),
        _Transpose(),
        nn.Conv1d(16, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(64, 96, 7, padding=3), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(96, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, len(G.GENERA)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    w = torch.tensor([1.0 / max(1, (np.asarray(y_tr) == c).sum()) for c in range(len(G.GENERA))],
                     dtype=torch.float32)
    w = w / w.sum() * len(G.GENERA)
    lossf = nn.CrossEntropyLoss(weight=w)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            out = net(Xtr[b])
            loss = lossf(out, ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    with torch.no_grad():
        logits = torch.cat([net(Xte[i:i + 32]) for i in range(0, len(Xte), 32)])
        proba = torch.softmax(logits, -1).numpy()
    return proba.argmax(1), proba


class _Transpose(__import__("torch").nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


def one_run(df, Xk, Xt, split, seed):
    if split == "random":
        tr, va, te = G.split_random(df, seed)
    else:
        tr, va, te = G.split_heldout_species(df, seed)
    tr = np.concatenate([tr, va])                 # no early stopping is used; fold validation in
    y = df.label.to_numpy()
    res = {"_sizes": [int(len(tr)), int(len(te))]}

    maj = np.bincount(y[tr], minlength=len(G.GENERA)).argmax()
    res["majority_class"] = scores(y[te], np.full(len(te), maj))

    for name, F in (("length_gc", Xt), ("kmer5", Xk)):
        sc = StandardScaler().fit(F[tr])
        A, B = sc.transform(F[tr]), sc.transform(F[te])
        for mdl_name, mdl in (("logreg", LogisticRegression(max_iter=3000, C=1.0,
                                                            class_weight="balanced")),
                              ("gbm", HistGradientBoostingClassifier(max_iter=200, random_state=seed)),
                              ("rf", RandomForestClassifier(n_estimators=300, n_jobs=1,
                                                            random_state=seed,
                                                            class_weight="balanced"))):
            if name == "length_gc" and mdl_name != "logreg":
                continue
            mdl.fit(A, y[tr])
            p = mdl.predict(B)
            pr = mdl.predict_proba(B) if hasattr(mdl, "predict_proba") else None
            res[f"{name}_{mdl_name}"] = scores(y[te], p, pr)

    pred, proba = cnn_fit_predict(df.seq.iloc[tr].tolist(), y[tr], df.seq.iloc[te].tolist(), seed)
    res["cnn"] = scores(y[te], pred, proba)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    df = G.load()
    log(f"{len(df)} genomes, {df.species.nunique()} species, classes "
        + ", ".join(f"{g}={int((df.genus == g).sum())}" for g in G.GENERA))
    log("building k-mer features")
    Xk = G.kmer_features(df.seq.tolist(), k=a.k, log=log)
    Xt = G.trivial_features(df)
    log(f"k-mer matrix {Xk.shape}")

    allres = {}
    for split in ("random", "heldout_species"):
        for s in range(a.seeds):
            t0 = time.time()
            r = one_run(df, Xk, Xt, split, s)
            r["_seconds"] = round(time.time() - t0, 1)
            allres[f"{split}_s{s}"] = r
            log(f"{split} s{s}: " + "  ".join(
                f"{k}={v['balanced_accuracy']:.3f}" for k, v in r.items()
                if isinstance(v, dict) and "balanced_accuracy" in v))
    meta = dict(n_genomes=int(len(df)), n_species=int(df.species.nunique()), k=a.k,
                class_counts={g: int((df.genus == g).sum()) for g in G.GENERA},
                median_length={g: float(df.length[df.genus == g].median()) for g in G.GENERA})
    json.dump(dict(meta=meta, runs=allres), open(OUT / "genome_study.json", "w"), indent=1)
    log("saved results/genome_study.json")
