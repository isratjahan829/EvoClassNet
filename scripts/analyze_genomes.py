"""Aggregate the genome classification study and draw its figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import genomes as G, stats as ST

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
MODELS = [("majority_class", "Majority class", "untrained"),
          ("length_gc_logreg", "Genome length + base composition", "trivial"),
          ("kmer5_logreg", "5-mer + logistic regression", "sequence"),
          ("kmer5_rf", "5-mer + random forest", "sequence"),
          ("kmer5_gbm", "5-mer + gradient boosting", "sequence"),
          ("cnn", "1D CNN on nucleotides", "sequence")]
COL = {"untrained": "#8a8a8a", "trivial": "#2e7d32", "sequence": "#d9541e"}
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc", "log_loss"]
MARGIN = 0.03
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def series(runs, mdl, metric):
    return [r[mdl][metric] for r in runs.values()
            if mdl in r and metric in r[mdl] and not np.isnan(r[mdl][metric])]


def fmt(v, d=3):
    if not v:
        return "—"
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def main():
    p = OUT / "genome_study.json"
    if not p.exists():
        print("results/genome_study.json not found"); return
    blob = json.load(open(p))
    meta, allr = blob["meta"], blob["runs"]
    splits = {s: {k: v for k, v in allr.items() if k.startswith(s + "_s")}
              for s in ("random", "heldout_species")}

    rows = []
    for sp, runs in splits.items():
        if not runs:
            continue
        for mdl, nm, fam in MODELS:
            row = dict(split=sp, family=fam, model=nm, seeds=len(series(runs, mdl, "accuracy")))
            for m in METRICS:
                row[m] = fmt(series(runs, mdl, m))
            for g in G.GENERA:
                v = [r[mdl]["per_class_f1"][g] for r in runs.values() if mdl in r]
                row[f"f1_{g}"] = fmt(v)
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_genomes.csv", index=False)

    comp = []
    for sp, runs in splits.items():
        if len(runs) < 5:
            continue
        ks = sorted(runs)
        for ref in ("length_gc_logreg", "majority_class"):
            for mdl, nm, fam in MODELS:
                if mdl == ref or fam in ("untrained", "trivial"):
                    continue
                A = np.array([runs[k][mdl]["balanced_accuracy"] for k in ks])
                B = np.array([runs[k][ref]["balanced_accuracy"] for k in ks])
                w = ST.wilcoxon_paired(A, B)
                te = ST.tost(A, B, MARGIN)
                d, mag = ST.cliffs_delta(A, B)
                lo, hi = ST.bootstrap_ci(A - B)
                comp.append(dict(split=sp, model=nm, reference=dict(MODELS and
                                 {m: n for m, n, _ in MODELS})[ref], n_seeds=len(ks),
                                 median_diff=round(w["median"], 4), p_raw=w["p"],
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4), tost_p=te["p"],
                                 equivalent=te["equivalent"], cliffs_delta=round(d, 3), effect=mag))
    cdf = pd.DataFrame(comp)
    if len(cdf):
        q, rej = ST.benjamini_hochberg(cdf.p_raw.to_numpy())
        cdf["p_fdr"] = np.round(q, 5); cdf["significant_fdr"] = rej
        cdf.to_csv(OUT / "table_genome_comparisons.csv", index=False)

    # ---- figure: balanced accuracy by split
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    for ax, (sp, title) in zip(axes, (("random", "Random split"),
                                      ("heldout_species", "Held-out-species split"))):
        runs = splits[sp]
        labs, mu, sd, cols = [], [], [], []
        for mdl, nm, fam in MODELS:
            v = series(runs, mdl, "balanced_accuracy")
            if not v:
                continue
            labs.append(nm); mu.append(np.mean(v))
            sd.append(np.std(v, ddof=1) if len(v) > 1 else 0); cols.append(COL[fam])
        y = np.arange(len(labs))
        ax.barh(y, mu, xerr=sd, color=cols, height=0.66, error_kw=dict(lw=0.8, capsize=2))
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7.6)
        ax.axvline(0.25, color="k", lw=0.8, ls=":")
        ax.text(0.252, len(labs) - 0.4, "chance", fontsize=7, va="top")
        ax.set_xlim(0, 1.03); ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("Balanced accuracy")
    fig.tight_layout(); fig.savefig(FIG / "fig6_genomes.png", dpi=240); plt.close(fig)

    # ---- figure: confusion matrices, held-out-species, pooled over seeds
    runs = splits["heldout_species"]
    if runs:
        show = [m for m in ("length_gc_logreg", "kmer5_logreg", "cnn")
                if any(m in r for r in runs.values())]
        fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.2))
        axes = np.atleast_1d(axes)
        short = [g.replace("coronavirus", "") for g in G.GENERA]
        for ax, mdl in zip(axes, show):
            C = np.sum([np.array(r[mdl]["confusion"]) for r in runs.values()], axis=0).astype(float)
            Cn = C / np.clip(C.sum(1, keepdims=True), 1, None)
            ax.imshow(Cn, cmap="Oranges", vmin=0, vmax=1)
            for i in range(len(G.GENERA)):
                for j in range(len(G.GENERA)):
                    ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if Cn[i, j] > 0.55 else "#2f3640")
            ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=7)
            ax.set_title(dict((m, n) for m, n, _ in MODELS)[mdl], fontsize=8.5)
            ax.set_xlabel("predicted", fontsize=8)
        axes[0].set_ylabel("true", fontsize=8)
        fig.suptitle("Held-out-species split, pooled over seeds (row-normalised)", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / "fig7_genome_confusion.png", dpi=240); plt.close(fig)

    print(f"{meta['n_genomes']} genomes, {meta['n_species']} species, k={meta['k']}")
    print(tab[["split", "family", "model", "accuracy", "balanced_accuracy", "macro_f1"]]
          .to_string(index=False))
    if len(cdf):
        print("\n=== vs the trivial length + composition model ===")
        print(cdf[cdf.reference.str.startswith("Genome length")]
              [["split", "model", "median_diff", "ci_lo", "ci_hi", "p_fdr", "tost_p",
                "equivalent"]].to_string(index=False))


if __name__ == "__main__":
    main()
