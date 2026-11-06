"""Aggregate the 217-assay ProteinGym study into tables and figures.

The question is whether the result obtained on one protein is a property of the benchmark. Three
things are reported per split and per model family: the distribution of rank correlations across
assays, a paired comparison against the untrained controls using the assay as the unit of
analysis, and the fraction of assays in which the mutation-count confound is present at all.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import stats as ST

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
JOBS = OUT / "pg_jobs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [("mutation_count", "Mutation count", "untrained"),
          ("blosum_zeroshot", "BLOSUM62", "untrained"),
          ("position_mean", "Position mean", "position-indexed"),
          ("ridge_additive", "Additive ridge", "position-indexed"),
          ("residue_only", "Residue identity", "transferable"),
          ("physchem", "Chemistry ridge", "transferable")]
NAME = {k: n for k, n, _ in MODELS}
FAM = {k: f for k, _, f in MODELS}
COL = {"untrained": "#2e7d32", "position-indexed": "#d9541e", "transferable": "#8c62aa"}
SPLITS = [("random", "Random split"), ("heldout_positions", "Held-out-position split")]
MARGIN = 0.05
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    rows = []
    for f in sorted(JOBS.glob("*.json")):
        r = json.load(open(f))
        if "error" in r or not r.get("splits"):
            continue
        for sp, _ in SPLITS:
            blk = r["splits"].get(sp, {})
            if not blk:
                continue
            row = dict(dms_id=r["dms_id"], split=sp, n_variants=r["n_variants"],
                       seq_len=r["seq_len"], n_positions=r["n_positions"],
                       frac_multi=r["frac_multi"], max_nsub=r["max_nsub"])
            for m, _, _ in MODELS:
                row[m] = blk.get(m, {}).get("spearman", np.nan)
                row[m + "_partial"] = blk.get(m, {}).get("partial", np.nan)
                row[m + "_nseeds"] = blk.get(m, {}).get("n_seeds", 0)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    D = load()
    if not len(D):
        print("no ProteinGym jobs yet"); return
    D.to_csv(OUT / "proteingym_per_assay.csv", index=False)
    n_assay = D.dms_id.nunique()

    # ---- Table: distribution across assays
    rows = []
    for sp, spn in SPLITS:
        d = D[D.split == sp]
        if not len(d):
            continue
        for m, nm, fam in MODELS:
            v = d[m].dropna()
            if not len(v):
                continue
            rows.append(dict(split=spn, family=fam, model=nm, n_assays=len(v),
                             median=round(float(v.median()), 3),
                             mean=round(float(v.mean()), 3),
                             q25=round(float(v.quantile(0.25)), 3),
                             q75=round(float(v.quantile(0.75)), 3),
                             frac_above_0_2=round(float((v > 0.2).mean()), 3),
                             frac_at_zero=round(float((v.abs() < 1e-6).mean()), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_proteingym.csv", index=False)

    # ---- Table: paired comparisons, assay as unit
    comp = []
    for sp, spn in SPLITS:
        d = D[D.split == sp]
        for ref in ("mutation_count", "blosum_zeroshot", "position_mean"):
            for m, nm, fam in MODELS:
                if m == ref:
                    continue
                ok = d[[m, ref]].dropna()
                if len(ok) < 10:
                    continue
                A, B = ok[m].to_numpy(), ok[ref].to_numpy()
                w = ST.wilcoxon_paired(A, B)
                te = ST.tost(A, B, MARGIN)
                cd, mag = ST.cliffs_delta(A, B)
                lo, hi = ST.bootstrap_ci(A - B)
                comp.append(dict(split=spn, model=nm, reference=NAME[ref], n_assays=len(ok),
                                 median_diff=round(w["median"], 4), p_raw=w["p"],
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4), tost_p=te["p"],
                                 equivalent=te["equivalent"], cliffs_delta=round(cd, 3),
                                 effect=mag,
                                 frac_model_better=round(float((A > B).mean()), 3)))
    C = pd.DataFrame(comp)
    if len(C):
        q, rej = ST.benjamini_hochberg(C.p_raw.to_numpy())
        C["p_fdr"] = np.round(q, 6)
        C["significant_fdr"] = rej
        C.to_csv(OUT / "table_proteingym_comparisons.csv", index=False)

    # ---- how often is the mutation-count confound present at all?
    d = D[D.split == "heldout_positions"]
    multi = d[d.frac_multi > 0]
    conf = dict(n_assays=int(n_assay),
                n_with_multi_mutants=int((d.frac_multi > 0).sum()),
                frac_with_multi_mutants=round(float((d.frac_multi > 0).mean()), 3),
                n_count_above_0_2=int((d.mutation_count > 0.2).sum()),
                median_count_rho_where_multi=round(float(multi.mutation_count.median()), 3)
                if len(multi) else None)
    json.dump(conf, open(OUT / "proteingym_confound.json", "w"), indent=1)

    # ---- Figure: distribution of rho per model, both splits
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, (sp, spn) in zip(axes, SPLITS):
        d = D[D.split == sp]
        data, labs, cols = [], [], []
        for m, nm, fam in MODELS:
            v = d[m].dropna().to_numpy()
            if not len(v):
                continue
            data.append(v); labs.append(nm); cols.append(COL[fam])
        pos = np.arange(len(data))
        bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k", lw=1.2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("none")
        for i, v in enumerate(data):
            ax.scatter(v, np.full(len(v), i) + np.random.default_rng(0).normal(0, 0.07, len(v)),
                       s=5, color="k", alpha=0.25, zorder=3)
        ax.set_yticks(pos); ax.set_yticklabels(labs, fontsize=8.5)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("Spearman ρ per assay"); ax.set_title(spn, fontsize=10)
        ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[f], alpha=0.75) for f in COL]
    fig.legend(handles, list(COL), loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"ProteinGym: {n_assay} deep mutational scanning assays", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "fig9_proteingym.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure: per-assay scatter, position_mean vs chemistry, both splits
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    for ax, (sp, spn) in zip(axes, SPLITS):
        d = D[D.split == sp].dropna(subset=["position_mean", "physchem"])
        ax.scatter(d.position_mean, d.physchem, s=16, alpha=0.6, color="#4c78a8",
                   edgecolor="none")
        lim = [-0.1, 1.0]
        ax.plot(lim, lim, color="k", lw=0.8, ls=":")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Position-mean model ρ"); ax.set_ylabel("Chemistry ridge ρ")
        ax.set_title(f"{spn}  (n = {len(d)})", fontsize=9.5)
    fig.suptitle("Which model wins depends entirely on the split", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig10_proteingym_scatter.png", dpi=240)
    plt.close(fig)

    print(f"{n_assay} assays\n")
    print(T.to_string(index=False))
    if len(C):
        print("\n=== held-out positions, against the position-mean model ===")
        print(C[(C.split == "Held-out-position split") & (C.reference == "Position mean")]
              [["model", "n_assays", "median_diff", "ci_lo", "ci_hi", "p_fdr",
                "frac_model_better", "effect"]].to_string(index=False))
    print("\nconfound:", json.dumps(conf))


if __name__ == "__main__":
    main()
