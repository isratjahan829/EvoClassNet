"""Aggregate the expanded study into the manuscript's tables, statistics and figures.

Differences from the first-generation scripts/analyze.py:
  * eleven model families instead of three;
  * the full metric panel, including the partial Spearman that removes the mutation-count
    confound directly rather than inferring it from a baseline comparison;
  * equivalence tests, so "indistinguishable from the untrained baseline" is a claim the
    statistics support rather than a failure to reject;
  * false-discovery-rate correction across the whole family of model-by-task comparisons.
"""
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
from evoclassnet.metrics import PANEL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
JOBS = OUT / "jobs_v2"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TASKS = [("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")]
MODELS = [
    ("mutation_count",   "Mutation count",        "untrained"),
    ("blosum_zeroshot",  "BLOSUM62",              "untrained"),
    ("esm_zeroshot",     "ESM-2 zero-shot",       "untrained"),
    ("physchem",         "Physicochemical ridge", "transferable"),
    ("knn",              "k-NN (features)",       "transferable"),
    ("gbm",              "Gradient boosting",     "transferable"),
    ("struct_gnn",       "Structure GNN",         "transferable"),
    ("ridge_additive",   "Additive ridge",        "position-indexed"),
    ("global_epistasis", "Global epistasis",      "position-indexed"),
    ("evoclassnet",      "EvoClassNet",           "position-indexed"),
    ("evoclassnet_plus", "EvoClassNet + ESM/str", "position-indexed"),
]
NAME = {k: n for k, n, _ in MODELS}
FAMILY = {k: f for k, _, f in MODELS}
COL = {"untrained": "#2e7d32", "transferable": "#8c62aa", "position-indexed": "#d9541e"}
# Equivalence margin: a difference in Spearman rho smaller than this is treated as practically
# nil. 0.05 is well below the between-seed spread of every trained model here, so declaring
# equivalence at this margin is a meaningful statement rather than a formality.
MARGIN = 0.05
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load(prefix):
    return {f.stem: json.load(open(f)) for f in sorted(JOBS.glob(f"{prefix}*.json"))}


def vals(runs, mdl, task, metric="spearman"):
    out = []
    for r in runs.values():
        v = r.get(mdl, {}).get(task, {}).get(metric, np.nan)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out.append(float(v))
    return out


def paired(runs, a, b, task, metric="spearman"):
    ka = sorted(runs)
    A = [runs[k].get(a, {}).get(task, {}).get(metric, np.nan) for k in ka]
    B = [runs[k].get(b, {}).get(task, {}).get(metric, np.nan) for k in ka]
    A = [np.nan if v is None else float(v) for v in A]
    B = [np.nan if v is None else float(v) for v in B]
    return np.array(A), np.array(B)


def fmt(v, d=3):
    if not v:
        return "—"
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def main():
    main_runs = {sp: load(f"main_{sp}_s") for sp in ("random", "heldout_sites")}
    abl = {sp: load(f"abl_perm_{sp}_s") for sp in ("random", "heldout_sites")}
    sweep = {f: load(f"sweep_f{f}_s") for f in (5, 15, 30, 50)}
    if not any(main_runs.values()):
        print("no jobs in results/jobs_v2 yet"); return

    # ---------------- Table 1: full metric panel -------------------------------
    rows = []
    for sp, runs in main_runs.items():
        for mdl, nm, fam in MODELS:
            for t, tn in TASKS:
                if not vals(runs, mdl, t):
                    continue
                row = dict(split=sp, family=fam, model=nm, task=tn, seeds=len(vals(runs, mdl, t)))
                for m in PANEL:
                    row[m] = fmt(vals(runs, mdl, t, m)) if m != "n" else int(
                        np.median(vals(runs, mdl, t, "n")))
                for s in (1, 2, 3):
                    v = vals(runs, mdl, t, f"spearman_nsub{s}")
                    if v:
                        row[f"spearman_nsub{s}"] = fmt(v)
                rows.append(row)
    panel = pd.DataFrame(rows)
    panel.to_csv(OUT / "table_panel.csv", index=False)

    # ---------------- Table 2: comparisons with equivalence testing -------------
    comp = []
    for sp, runs in main_runs.items():
        for t, tn in TASKS:
            for ref in ("mutation_count", "blosum_zeroshot", "esm_zeroshot"):
                for mdl, nm, fam in MODELS:
                    if mdl == ref or fam == "untrained":
                        continue
                    for metric in ("spearman", "spearman_partial"):
                        A, B = paired(runs, mdl, ref, t, metric)
                        ok = ~np.isnan(A) & ~np.isnan(B)
                        if ok.sum() < 5:
                            continue
                        w = ST.wilcoxon_paired(A[ok], B[ok])
                        te = ST.tost(A[ok], B[ok], MARGIN)
                        d, mag = ST.cliffs_delta(A[ok], B[ok])
                        lo, hi = ST.bootstrap_ci(A[ok] - B[ok])
                        comp.append(dict(split=sp, task=tn, metric=metric, model=nm,
                                         reference=NAME[ref], n_seeds=int(ok.sum()),
                                         median_diff=round(w["median"], 4), p_raw=w["p"],
                                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                         tost_p=te["p"], equivalent=te["equivalent"],
                                         cliffs_delta=round(d, 3), effect=mag))
    cdf = pd.DataFrame(comp)
    if len(cdf):
        q, rej = ST.benjamini_hochberg(cdf.p_raw.to_numpy())
        cdf["p_fdr"] = np.round(q, 5)
        cdf["significant_fdr"] = rej
        cdf.to_csv(OUT / "table_comparisons.csv", index=False)

    # -------- Table 2b: is the partial correlation different from zero? ----------
    # The partial correlation already removes the substitution-count confound, so the
    # informative question is not whether a model beats the untrained baseline on it - the
    # baseline's own partial correlation is undefined, being a function of the variable that was
    # partialled out - but whether the model's exceeds zero at all.
    zrows = []
    for sp, runs in main_runs.items():
        for mdl, nm, fam in MODELS:
            for t, tn in TASKS:
                v = np.array(vals(runs, mdl, t, "spearman_partial"), float)
                v = v[~np.isnan(v)]
                if len(v) < 5:
                    continue
                from scipy.stats import wilcoxon as _w
                pval = float(_w(v).pvalue) if not np.allclose(v, 0) else float("nan")
                lo, hi = ST.bootstrap_ci(v)
                zrows.append(dict(split=sp, family=fam, model=nm, task=tn, n_seeds=len(v),
                                  mean_partial=round(float(v.mean()), 4),
                                  ci_lo=round(lo, 4), ci_hi=round(hi, 4), p_raw=pval))
    zdf = pd.DataFrame(zrows)
    if len(zdf):
        q, rej = ST.benjamini_hochberg(zdf.p_raw.to_numpy())
        zdf["p_fdr"] = np.round(q, 5)
        zdf["nonzero_fdr"] = rej
        zdf.to_csv(OUT / "table_partial_vs_zero.csv", index=False)

    # ---------------- Table 3: permutation ablation -----------------------------
    # The permutation acts on the integer-encoded sequence, which is the input to the neural
    # models only. The untrained scorers and the feature-based models are computed from the
    # substitution list and are therefore untouched by construction; including them would show
    # a delta of exactly zero and invite the reading that permuting had no effect on them.
    PERMUTED_INPUT = {"Global epistasis", "EvoClassNet", "EvoClassNet + ESM/str",
                      "Structure GNN"}
    arows = []
    for sp in ("random", "heldout_sites"):
        if not abl[sp]:
            continue
        seeds = {int(k.split("_s")[-1]) for k in abl[sp]}
        base = {k: v for k, v in main_runs[sp].items() if int(k.split("_s")[-1]) in seeds}
        for mdl, nm, fam in MODELS:
            if nm not in PERMUTED_INPUT:
                continue
            for t, tn in TASKS:
                b, a = vals(base, mdl, t), vals(abl[sp], mdl, t)
                if not (a and b):
                    continue
                arows.append(dict(split=sp, model=nm, task=tn, intact=fmt(b), permuted=fmt(a),
                                  delta=round(np.mean(a) - np.mean(b), 3)))
    if arows:
        pd.DataFrame(arows).to_csv(OUT / "table_ablation_v2.csv", index=False)

    # ---------------- Figures ---------------------------------------------------
    for metric, fname, ylab in (
            ("spearman", "fig1_panel_spearman.png", "Test Spearman ρ"),
            ("spearman_partial", "fig2_panel_partial.png",
             "Partial Spearman ρ\n(substitution count removed)")):
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharey="row")
        for r, (sp, sname) in enumerate((("random", "Random genotype split"),
                                         ("heldout_sites", "Held-out-position split"))):
            runs = main_runs[sp]
            for c, (t, tn) in enumerate(TASKS):
                ax = axes[r, c]
                labs, means, sds, cols = [], [], [], []
                for mdl, nm, fam in MODELS:
                    v = vals(runs, mdl, t, metric)
                    if not v:
                        continue
                    labs.append(nm); means.append(np.mean(v))
                    sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                    cols.append(COL[fam])
                yy = np.arange(len(labs))
                ax.barh(yy, means, xerr=sds, color=cols, height=0.68,
                        error_kw=dict(lw=0.8, capsize=2))
                ax.set_yticks(yy); ax.set_yticklabels(labs if c == 0 else [], fontsize=7.5)
                ax.axvline(0, color="k", lw=0.7)
                ax.invert_yaxis()
                ax.set_title(f"{tn} — {sname}" if c == 1 else tn, fontsize=8.5)
                if r == 1:
                    ax.set_xlabel(ylab, fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COL[f]) for f in COL]
        fig.legend(handles, list(COL), loc="lower center", ncol=3, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(FIG / fname, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fname)

    # sweep over withheld fraction
    if any(sweep.values()):
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1), sharex=True)
        for ax, (t, tn) in zip(axes, TASKS):
            for mdl, nm, fam in MODELS:
                xs, ys, es = [], [], []
                for f, runs in sweep.items():
                    v = vals(runs, mdl, t)
                    if v:
                        xs.append(f); ys.append(np.mean(v))
                        es.append(np.std(v, ddof=1) if len(v) > 1 else 0)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, lw=1.1, capsize=2,
                                color=COL[fam], alpha=0.85, label=nm)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_title(tn, fontsize=9); ax.set_xlabel("% of positions withheld")
        axes[0].set_ylabel("Test Spearman ρ")
        axes[2].legend(frameon=False, fontsize=6.4, ncol=2)
        fig.tight_layout(); fig.savefig(FIG / "fig3_sweep_v2.png", dpi=240); plt.close(fig)
        print("wrote fig3_sweep_v2.png")

    summary = dict(n_runs=len(list(JOBS.glob("*.json"))),
                   seeds={k: len(v) for k, v in main_runs.items()},
                   models=[m for m, _, _ in MODELS], margin=MARGIN)
    json.dump(summary, open(OUT / "run_summary_v2.json", "w"), indent=1)

    print("\n=== headline: Spearman and partial Spearman, held-out positions ===")
    h = panel[(panel.split == "heldout_sites")]
    print(h[["family", "model", "task", "spearman", "spearman_partial", "n"]].to_string(index=False))
    if len(zdf):
        print("\n=== partial Spearman against zero, held-out positions ===")
        print(zdf[(zdf.split == "heldout_sites")]
              [["family", "model", "task", "n_seeds", "mean_partial", "ci_lo", "ci_hi",
                "p_fdr", "nonzero_fdr"]].to_string(index=False))
    if len(cdf):
        print("\n=== equivalence vs mutation count (held-out positions, raw Spearman) ===")
        sel = cdf[(cdf.split == "heldout_sites") & (cdf.reference == "Mutation count")
                  & (cdf.metric == "spearman")]
        print(sel[["task", "model", "median_diff", "ci_lo", "ci_hi", "p_fdr", "tost_p",
                   "equivalent", "effect"]].to_string(index=False))


if __name__ == "__main__":
    main()
