"""Place this study's baselines against ProteinGym's published supervised leaderboard.

ProteinGym reports its supervised benchmark separately for the three cross-validation schemes,
and the comparison matters for what can honestly be claimed here. Their pure one-hot encoding
baseline falls from 0.582 under fold_random_5 to 0.022 under fold_modulo_5 and 0.059 under
fold_contiguous_5, while every model that adds a transferable representation - sequence
embeddings, or structure in the case of Kermut - retains most of its performance. The collapse of
position-indexed models under position-based cross-validation is therefore already visible in the
published leaderboard, and is not a discovery of this work.

What this script does is put the two on the same axes, so the manuscript can state plainly where
its own baselines sit: well below the published models, which is the honest reading.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
BENCH = ROOT / "data" / "raw" / "proteingym_bench" / "sup_summary.csv"
SCHEMES = [("Average_Spearman_fold_random_5", "random", "Random split"),
           ("Average_Spearman_fold_modulo_5", "pg_modulo_5", "fold_modulo_5"),
           ("Average_Spearman_fold_contiguous_5", "pg_contiguous_5", "fold_contiguous_5")]
OURS = [("position_mean", "Position mean (this work)"),
        ("ridge_additive", "Additive ridge (this work)"),
        ("physchem", "Chemistry ridge (this work)"),
        ("residue_only", "Residue identity (this work)"),
        ("blosum_zeroshot", "BLOSUM62 (this work)")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    if not BENCH.exists():
        print("published leaderboard not downloaded"); return
    lb = pd.read_csv(BENCH)
    per = OUT / "proteingym_per_assay.csv"
    mine = pd.read_csv(per) if per.exists() else pd.DataFrame()

    rows = []
    for _, r in lb.iterrows():
        for col, _, label in SCHEMES:
            rows.append(dict(source="ProteinGym leaderboard", model=r["Model_name"],
                             model_type=r["Model type"], scheme=label,
                             spearman=round(float(r[col]), 3)))
    if len(mine):
        for key, label in OURS:
            for _, split, lab in SCHEMES:
                d = mine[mine.split == split]
                if not len(d) or key not in d.columns:
                    continue
                v = d[key].dropna()
                if len(v):
                    rows.append(dict(source="This work", model=label, model_type="baseline",
                                     scheme=lab, spearman=round(float(v.mean()), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_leaderboard_comparison.csv", index=False)

    piv = T.pivot_table(index=["source", "model_type", "model"], columns="scheme",
                        values="spearman")
    cols = [lab for _, _, lab in SCHEMES if lab in piv.columns]
    piv = piv[cols]
    if "Random split" in piv.columns and "fold_modulo_5" in piv.columns:
        piv["drop (random → modulo)"] = (piv["Random split"] - piv["fold_modulo_5"]).round(3)
    piv = piv.sort_values(cols[-1], ascending=False)
    piv.to_csv(OUT / "table_leaderboard_pivot.csv")

    oh = lb[lb.Model_name == "One-Hot Encodings"]
    best = lb.sort_values("Average_Spearman_fold_modulo_5", ascending=False).iloc[0]
    summary = dict(
        n_published_models=int(len(lb)),
        one_hot=dict(random=float(oh.Average_Spearman_fold_random_5.iloc[0]),
                     modulo=float(oh.Average_Spearman_fold_modulo_5.iloc[0]),
                     contiguous=float(oh.Average_Spearman_fold_contiguous_5.iloc[0])) if len(oh) else None,
        best_under_modulo=dict(model=best.Model_name, type=best["Model type"],
                               random=float(best.Average_Spearman_fold_random_5),
                               modulo=float(best.Average_Spearman_fold_modulo_5)),
    )
    if len(mine):
        d = mine[mine.split == "pg_modulo_5"]
        summary["ours_under_modulo"] = {k: round(float(d[k].dropna().mean()), 3)
                                        for k, _ in OURS if k in d.columns}
        r = mine[mine.split == "random"]
        if "position_mean" in r.columns:
            summary["ours_under_modulo"]["position_mean_random"] = round(
                float(r.position_mean.dropna().mean()), 3)
    json.dump(summary, open(OUT / "leaderboard_comparison.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    labels, randv, modv, cols_ = [], [], [], []
    for _, r in piv.reset_index().iterrows():
        if "Random split" not in piv.columns or "fold_modulo_5" not in piv.columns:
            break
        labels.append(r["model"])
        randv.append(r["Random split"]); modv.append(r["fold_modulo_5"])
        cols_.append("#8c62aa" if r["source"] == "This work" else "#4c78a8")
    y = np.arange(len(labels))
    ax.barh(y - 0.2, randv, 0.38, color=cols_, alpha=0.45, label="random split")
    ax.barh(y + 0.2, modv, 0.38, color=cols_, label="fold_modulo_5")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Mean Spearman ρ across assays")
    ax.set_title("Published ProteinGym supervised leaderboard (blue) and this study's baselines "
                 "(purple)", fontsize=9.5)
    h = [plt.Rectangle((0, 0), 1, 1, color="#4c78a8", alpha=a) for a in (0.45, 1.0)]
    ax.legend(h, ["random split", "position-based fold"], frameon=False, fontsize=8.5,
              loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig11_leaderboard.png", dpi=240); plt.close(fig)

    print(piv.round(3).to_string())
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
