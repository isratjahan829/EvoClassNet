"""Aggregate the per-run job files into the tables, statistics and figures used in the manuscript."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"; JOBS = OUT / "jobs"; FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
TASKS = [("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")]
MODELS = [("ridge_additive", "Additive ridge"), ("global_epistasis", "Global epistasis"), ("evoclassnet", "EvoClassNet")]
COL = {"ridge_additive": "#8a8a8a", "global_epistasis": "#4c78a8", "evoclassnet": "#d9541e"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def load(prefix):
    out = {}
    for f in sorted(JOBS.glob(f"{prefix}*.json")):
        out[f.stem] = json.load(open(f))
    return out


def series(runs, mdl, task, metric="spearman"):
    return [r[mdl][task][metric] for r in runs.values() if not np.isnan(r[mdl][task][metric])]


def fmt(v, d=3):
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else (f"{v[0]:.{d}f}" if v else "n/a")


def paired(runs, a, b, task):
    ka = sorted(runs); A = [runs[k][a][task]["spearman"] for k in ka]; B = [runs[k][b][task]["spearman"] for k in ka]
    ok = [i for i in range(len(A)) if not (np.isnan(A[i]) or np.isnan(B[i]))]
    A, B = [A[i] for i in ok], [B[i] for i in ok]
    d = np.array(A) - np.array(B)
    p = float(wilcoxon(d).pvalue) if len(d) >= 5 and not np.allclose(d, 0) else float("nan")
    return float(np.median(d)), p, len(d)


res = {}
main = {sp: load(f"main_{sp}_s") for sp in ("random", "heldout_sites")}
abl = {sp: load(f"abl_perm_{sp}_s") for sp in ("random", "heldout_sites")}
sweep = {f: load(f"sweep_f{f}_s") for f in (5, 15, 30, 50)}

# ---------------- Tables 1-2: main results ----------------
rows = []
for sp, runs in main.items():
    for mdl, mname in MODELS:
        for t, tname in TASKS:
            v = series(runs, mdl, t); n = [r[mdl][t]["n"] for r in runs.values()]
            rows.append(dict(split=sp, model=mname, task=tname, seeds=len(v), n_test=int(np.median(n)),
                             spearman=fmt(v), pearson=fmt(series(runs, mdl, t, "pearson")),
                             rmse=fmt(series(runs, mdl, t, "rmse"))))
main_df = pd.DataFrame(rows); main_df.to_csv(OUT / "table_main.csv", index=False)

# ---------------- Table 3: paired tests vs global epistasis ----------------
rows = []
for sp, runs in main.items():
    for t, tname in TASKS:
        d, p, n = paired(runs, "evoclassnet", "global_epistasis", t)
        d2, p2, _ = paired(runs, "evoclassnet", "ridge_additive", t)
        rows.append(dict(split=sp, task=tname, n_seeds=n,
                         vs_global_epistasis=f"{d:+.3f} (P = {p:.3g})" if not np.isnan(p) else f"{d:+.3f} (n/a)",
                         vs_additive=f"{d2:+.3f} (P = {p2:.3g})" if not np.isnan(p2) else f"{d2:+.3f} (n/a)"))
stat_df = pd.DataFrame(rows); stat_df.to_csv(OUT / "table_stats.csv", index=False)

# ---------------- Table 4: amino-acid permutation ablation ----------------
rows = []
for sp in ("random", "heldout_sites"):
    for t, tname in TASKS:
        base = series({k: v for k, v in main[sp].items() if int(k.split("_s")[-1]) < len(abl[sp])}, "evoclassnet", t)
        ab = series(abl[sp], "evoclassnet", t)
        rows.append(dict(split=sp, task=tname, intact=fmt(base), permuted=fmt(ab),
                         delta=f"{np.mean(ab) - np.mean(base):+.3f}" if ab and base else "n/a"))
abl_df = pd.DataFrame(rows); abl_df.to_csv(OUT / "table_ablation.csv", index=False)

# ---------------- Figures ----------------
for sp, title, fn in (("random", "Random genotype split", "fig1_random_split.png"),
                      ("heldout_sites", "Held-out-site split (unseen positions)", "fig2_heldout_sites.png")):
    runs = main[sp]; fig, ax = plt.subplots(figsize=(5.8, 3.3)); w = 0.26
    for j, (mdl, mname) in enumerate(MODELS):
        for ti, (t, tname) in enumerate(TASKS):
            v = series(runs, mdl, t)
            if not v: continue
            ax.bar(ti + (j - 1) * w, np.mean(v), w, color=COL[mdl], label=mname if ti == 0 else None,
                   yerr=np.std(v, ddof=1), capsize=2, error_kw=dict(lw=.8))
            ax.scatter(np.full(len(v), ti + (j - 1) * w) + np.random.default_rng(0).normal(0, .02, len(v)), v, color="k", s=5, zorder=3)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([n for _, n in TASKS])
    ax.set_ylabel("Test Spearman ρ"); ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fn, dpi=220); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharex=True)
for ax, (t, tname) in zip(axes, TASKS):
    for mdl, mname in MODELS:
        xs, ys, es = [], [], []
        for f, runs in sweep.items():
            v = series(runs, mdl, t)
            if v: xs.append(f); ys.append(np.mean(v)); es.append(np.std(v, ddof=1))
        if xs: ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.3, capsize=2, color=COL[mdl], label=mname)
    ax.axhline(0, color="k", lw=.6, ls=":"); ax.set_title(tname, fontsize=9.5); ax.set_xlabel("% of positions held out")
axes[0].set_ylabel("Test Spearman ρ"); axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig4_sweep.png", dpi=220); plt.close(fig)

summary = dict(n_runs=len(list(JOBS.glob("*.json"))),
               seeds_main={k: len(v) for k, v in main.items()},
               median_seconds=float(np.median([r["_seconds"] for r in main["random"].values()])) if main["random"] else None)
json.dump(summary, open(OUT / "run_summary.json", "w"), indent=1)
for name, df in (("Main results", main_df), ("Paired tests", stat_df), ("AA-permutation ablation", abl_df)):
    print(f"\n=== {name} ===\n{df.to_string(index=False)}")
