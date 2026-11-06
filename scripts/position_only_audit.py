"""Audit the published ProteinGym supervised leaderboard against a position-only baseline.

The leaderboard reports, per assay and per cross-validation scheme, the Spearman correlation of
eleven supervised models. This script asks one question of those published numbers: on how many
assays does a given model fail to beat a ridge regression that is told only *which position* a
variant mutates, and nothing about which substitution is made?

That baseline cannot in principle know whether a substitution is conservative or catastrophic. If
a published model does not beat it on an assay under the random split, the random-split score for
that assay is not evidence that the model understands substitutions there. The point is not that
the models are bad - under the position-based schemes they are far ahead, which is the whole
argument - but that the conventional protocol does not show it.

Published numbers come from the ProteinGym repository; the baseline is computed here on the same
assays and the same reconstructed schemes.
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
BENCH = ROOT / "data" / "raw" / "proteingym_bench"
SCHEMES = [("random_5", "random", "Random split"),
           ("modulo_5", "pg_modulo_5", "fold_modulo_5"),
           ("contiguous_5", "pg_contiguous_5", "fold_contiguous_5")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main():
    mine_path = OUT / "proteingym_per_assay.csv"
    if not mine_path.exists():
        print("results/proteingym_per_assay.csv missing"); return
    mine = pd.read_csv(mine_path)

    rows, per_assay = [], []
    for sch_file, sch_mine, label in SCHEMES:
        f = BENCH / f"dms_level_fold_{sch_file}.csv"
        if not f.exists():
            continue
        pub = pd.read_csv(f)
        base = (mine[mine.split == sch_mine][["dms_id", "position_mean", "physchem"]]
                .rename(columns={"dms_id": "DMS_id"}))
        m = pub.merge(base, on="DMS_id", how="inner")
        models = [c for c in pub.columns if c != "DMS_id"]
        for mod in models:
            d = m[[mod, "position_mean"]].dropna()
            if not len(d):
                continue
            beats = (d[mod] > d.position_mean).mean()
            rows.append(dict(scheme=label, model=mod, n_assays=len(d),
                             model_median=round(float(d[mod].median()), 3),
                             position_only_median=round(float(d.position_mean.median()), 3),
                             median_margin=round(float((d[mod] - d.position_mean).median()), 3),
                             frac_assays_model_beats_position_only=round(float(beats), 3),
                             n_assays_not_beaten=int((d[mod] <= d.position_mean).sum())))
        keep = ["DMS_id", "position_mean", "physchem"] + models
        per_assay.append(m[keep].assign(scheme=label))

    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_position_only_audit.csv", index=False)
    if per_assay:
        pd.concat(per_assay).to_csv(OUT / "position_only_audit_per_assay.csv", index=False)

    summary = {}
    for label in T.scheme.unique():
        d = T[T.scheme == label].sort_values("frac_assays_model_beats_position_only")
        summary[label] = {
            "n_models": int(len(d)),
            "worst": dict(model=d.iloc[0].model,
                          frac_beats=float(d.iloc[0].frac_assays_model_beats_position_only),
                          n_not_beaten=int(d.iloc[0].n_assays_not_beaten)),
            "best": dict(model=d.iloc[-1].model,
                         frac_beats=float(d.iloc[-1].frac_assays_model_beats_position_only),
                         n_not_beaten=int(d.iloc[-1].n_assays_not_beaten)),
            "median_frac_beats": round(float(d.frac_assays_model_beats_position_only.median()), 3),
        }
    json.dump(summary, open(OUT / "position_only_audit.json", "w"), indent=1)

    piv = T.pivot(index="model", columns="scheme",
                  values="frac_assays_model_beats_position_only")
    cols = [l for _, _, l in SCHEMES if l in piv.columns]
    piv = piv[cols].sort_values(cols[0])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(piv))
    w = 0.26
    for i, c in enumerate(cols):
        ax.barh(y + (i - 1) * w, piv[c], w, label=c,
                color=["#8a8a8a", "#4c78a8", "#2e7d32"][i])
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("Fraction of assays where the published model beats a position-only baseline")
    ax.set_xlim(0, 1.02); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Published supervised models against a model told only which position is mutated",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig12_position_only_audit.png", dpi=240)
    plt.close(fig)

    print(T.to_string(index=False))
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
