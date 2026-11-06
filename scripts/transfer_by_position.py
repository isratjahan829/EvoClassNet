"""Which positions does the transferable model get right, and why?

The feature-based model predicts substitution effects at positions it never saw mutated. This
asks where that works. For every held-out-position seed we score every single substitution at the
withheld positions and compute, per position, the rank correlation between prediction and
measurement. Those per-position correlations are then related to the structural descriptors, and
compared against the sequence network and an untrained substitution matrix on identical
instances.

Per-position values recur across seeds and are not independent, so the test that the median
exceeds zero uses a random-intercept model over positions rather than a one-sample test over
rows.
"""
import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, escape as ESC, metrics, model as M, plm, stats as ST, structure as S, zoo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TASKS = ["bind", "expr", "escape"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    wt, off = data.wildtype()
    g = data.load_genotypes()
    Y = g[TASKS].to_numpy(dtype=np.float32)
    nsub = g.n_sub.to_numpy()
    sfeat, smask, adj, snames = S.features()
    mm, hid = plm.load()
    F = zoo.genotype_features(g.subs.tolist(), esm_mm=mm, struct=sfeat, struct_names=snames)

    singles = [(wt[p - 1], p, a) for p in range(1, 202) for a in zoo.AA if a != wt[p - 1]]
    Fs = zoo.genotype_features([[s] for s in singles], esm_mm=mm, struct=sfeat, struct_names=snames)
    sm = pd.read_csv(data.RAW / "single_mut_effects.csv")
    meas = {(int(r.site_RBD), r.mutant): (r.bind_avg, r.expr_avg) for r in sm.itertuples()}
    esc = ESC.escape_lookup(20)
    BL = zoo.blosum_matrix()

    recs = []
    for seed in range(nseeds):
        tr, va, te = data.split_heldout_sites(g, seed)
        tr = np.concatenate([tr, va])
        test_sites = set().union(*[g.sites.iloc[i] for i in te]) if len(te) else set()
        P = zoo.fit_predict_sklearn("physchem", np.vstack([F, Fs]), Y, tr,
                                    np.arange(len(F), len(F) + len(Fs)), seed=seed)
        for i, (w, p, m) in enumerate(singles):
            if p not in test_sites:
                continue
            mb, me = meas.get((p, m), (np.nan, np.nan))
            recs.append(dict(seed=seed, site=p, spike=p + off, wt=w, mut=m,
                             pred_bind=P[i, 0], pred_expr=P[i, 1], pred_escape=P[i, 2],
                             meas_bind=mb, meas_expr=me, meas_escape=esc.get((w, p, m), np.nan),
                             blosum=float(BL[zoo.AA2I[w], zoo.AA2I[m]])))
        log(f"seed {seed}: {len(test_sites)} withheld positions scored")

    D = pd.DataFrame(recs)
    D.to_csv(OUT / "transfer_predictions.csv", index=False)

    per = []
    for (seed, site), d in D.groupby(["seed", "site"]):
        for task, pc, mc in (("bind", "pred_bind", "meas_bind"),
                             ("expr", "pred_expr", "meas_expr"),
                             ("escape", "pred_escape", "meas_escape")):
            ok = d[[pc, mc, "blosum"]].dropna()
            if len(ok) >= 8 and ok[pc].std() > 1e-9 and ok[mc].std() > 1e-9:
                per.append(dict(seed=seed, site=site, task=task,
                                rho=spearmanr(ok[pc], ok[mc])[0],
                                rho_blosum=spearmanr(ok.blosum, ok[mc])[0], n=len(ok)))
    PS = pd.DataFrame(per)
    for j, name in enumerate(snames):
        PS[name] = PS.site.map({i + 1: float(sfeat[i, j]) for i in range(201)})
    PS.to_csv(OUT / "transfer_per_site.csv", index=False)

    res = {}
    for task, d in PS.groupby("task"):
        dd = d.dropna(subset=["rho"])
        me = ST.mixed_effects_per_site(dd, value="rho", group="site")
        vb = dd.dropna(subset=["rho_blosum"])
        diff = vb.rho - vb.rho_blosum
        wil = ST.wilcoxon_paired(vb.rho.to_numpy(), vb.rho_blosum.to_numpy())
        corrs = {}
        for name in snames:
            c = dd.dropna(subset=[name])
            if len(c) > 10 and c[name].std() > 0:
                r, pv = spearmanr(c[name], c.rho)
                corrs[name] = [float(r), float(pv)]
        res[task] = dict(n_site_seed=int(len(dd)), median_rho=float(dd.rho.median()),
                         frac_positive=float((dd.rho > 0).mean()),
                         mixed_effects=me,
                         vs_blosum=dict(model_median=float(vb.rho.median()),
                                        blosum_median=float(vb.rho_blosum.median()),
                                        median_diff=float(diff.median()), p=wil["p"],
                                        n=int(len(vb))),
                         rho_vs_structure=corrs)
        log(f"{task}: median per-position rho {dd.rho.median():+.3f}, "
            f"{100*(dd.rho>0).mean():.0f}% positive, mixed-effects P={me.get('p', float('nan')):.3g}")

    json.dump(res, open(OUT / "transfer_by_position.json", "w"), indent=1, default=float)

    show = [t for t in ("bind", "escape") if t in set(PS.task)]
    if show:
        fig, axes = plt.subplots(1, len(show) * 2, figsize=(4.0 * len(show), 3.1))
        axes = np.atleast_1d(axes)
        k = 0
        for task in show:
            d = PS[PS.task == task]
            for feat_name, lab in (("rsa_unbound", "Relative solvent accessibility (unbound)"),
                                   ("min_dist_ace2", "Distance to ACE2 (Å)")):
                ax = axes[k]; k += 1
                dd = d.dropna(subset=[feat_name, "rho"])
                ax.scatter(dd[feat_name], dd.rho, s=11, alpha=0.5, color="#8c62aa",
                           edgecolor="none")
                if len(dd) > 8:
                    z = np.polyfit(dd[feat_name], dd.rho, 1)
                    xs = np.linspace(dd[feat_name].min(), dd[feat_name].max(), 50)
                    ax.plot(xs, np.polyval(z, xs), color="k", lw=1)
                ax.axhline(0, color="k", lw=0.6, ls=":")
                ax.set_xlabel(lab, fontsize=8)
                ax.set_title(f"{task}", fontsize=9)
        axes[0].set_ylabel("Per-position Spearman ρ\n(withheld position)", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "fig8_transfer_by_position.png", dpi=240)
        plt.close(fig)
        log("wrote fig8_transfer_by_position.png")
    log("saved results/transfer_by_position.json")
