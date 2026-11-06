"""Leakage-free analyses based on held-out-site models.

For each seed we train EvoClassNet on a held-out-site split and predict every single substitution
at the *test* positions, i.e. positions whose mutational effects the model has never seen. This
supports two analyses that a model trained on all positions could not support honestly:

(a) Variant-of-Concern ranking without leakage. VOC-defining substitutions are present in the
    escape training data, so ranking them with a model that saw them measures memorisation. Here
    a VOC substitution contributes only in the seeds where its position was held out.
(a2) A hand-crafted chemistry prior. The model's transfer is carried by residue identity shared
    across positions (Section on the permutation ablation), so the natural question is whether it
    beats a substitution matrix, which encodes exactly that prior without any training. We score
    each substitution by its BLOSUM62 exchangeability and compare per-position rank accuracy.

(b) Which positions transfer? Per-position Spearman between predicted and measured phenotype,
    related to structural annotations (relative solvent accessibility, evolutionary entropy,
    receptor-binding-motif membership) from the source data set.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata, spearmanr, mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Bio.Align import substitution_matrices
from evoclassnet import data, model as M, escape as ESC
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
torch.set_num_threads(4)
TASKS = ["bind", "expr", "escape"]
VOC = ["G339D", "S371L", "S373P", "S375F", "K417N", "K417T", "N440K", "G446S", "L452R", "S477N",
       "T478K", "E484A", "E484K", "Q493R", "G496S", "Q498R", "N501Y", "Y505H"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def auroc_ci(score, lab, rng, n=2000):
    ok = ~np.isnan(score); s, l = score[ok], lab[ok].astype(bool)
    if l.sum() < 3 or (~l).sum() < 3: return float("nan"), [float("nan")] * 2, int(l.sum())
    f = lambda s, l: (rankdata(s)[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum())
    a = f(s, l); bs = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        if 0 < l[i].sum() < len(i): bs.append(f(s[i], l[i]))
    return float(a), [float(x) for x in np.percentile(bs, [2.5, 97.5])], int(l.sum())


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(OUT / "ga_hyperparams.json"))["best"].items()}
    wt, off = data.wildtype()
    voc = [(v[0], int(v[1:-1]) - off, v[-1]) for v in VOC]
    for w, p, m in voc: assert wt[p - 1] == w
    g = data.load_genotypes(); X = data.encode(g.subs.tolist()); Y = g[TASKS].to_numpy(dtype=np.float32)
    # measured single-mutant reference values
    sm = pd.read_csv(data.RAW / "single_mut_effects.csv")
    meas = {(int(r.site_RBD), r.mutant): (r.bind_avg, r.expr_avg) for r in sm.itertuples()}
    esc = ESC.escape_lookup(20)
    BL = substitution_matrices.load("BLOSUM62")
    singles = [(wt[p - 1], p, a) for p in range(1, 202) for a in data.AA if a != wt[p - 1]]
    Xs = data.encode([[s] for s in singles])

    recs = []
    for seed in range(nseeds):
        tr, va, te = data.split_heldout_sites(g, seed)
        test_sites = sorted(set().union(*[g.sites.iloc[i] for i in te])) if len(te) else []
        net = M.EvoClassNetLite(n_tasks=3, **{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
        net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=8, bs=256, lr=hp["lr"], wd=hp["wd"], seed=seed, log=lambda *a: None)
        P = M.predict(net, Xs)
        ts = set(test_sites)
        for i, (w, p, m) in enumerate(singles):
            if p not in ts: continue
            mb, me = meas.get((p, m), (np.nan, np.nan))
            recs.append(dict(seed=seed, site=p, spike_site=p + off, wt=w, mut=m,
                             pred_bind=P[i, 0], pred_expr=P[i, 1], pred_escape=P[i, 2],
                             meas_bind=mb, meas_expr=me, meas_escape=esc.get((w, p, m), np.nan),
                             blosum=float(BL[w, m]) if (w in BL.alphabet and m in BL.alphabet) else np.nan,
                             is_voc=(w, p, m) in set(voc)))
        log(f"seed {seed}: {len(ts)} held-out positions, {sum(r['seed']==seed for r in recs)} substitutions scored")
    D = pd.DataFrame(recs); D.to_csv(OUT / "holdout_predictions.csv", index=False)
    rng = np.random.default_rng(0); res = {}

    # ---- (a) leakage-free VOC ranking
    res["voc"] = {}
    for key, col in (("EvoClassNet predicted escape", "pred_escape"),
                     ("EvoClassNet predicted binding", "pred_bind"),
                     ("BLOSUM62 exchangeability", "blosum"),
                     ("Measured escape (held-out positions)", "meas_escape"),
                     ("Measured binding DMS (held-out positions)", "meas_bind")):
        a, ci, n = auroc_ci(D[col].to_numpy(float), D.is_voc.to_numpy(), rng)
        res["voc"][key] = dict(auroc=a, ci95=ci, n_voc=n, n_total=int(D[col].notna().sum()))
        log(f"VOC AUROC {key:42s} {a:.3f} [{ci[0]:.3f},{ci[1]:.3f}] (n_VOC={n})")
    res["voc_n_unique_voc_subs"] = int(D[D.is_voc].groupby(["site", "mut"]).ngroups)

    # ---- (b) which positions transfer
    sites = pd.read_csv(data.RAW / "RBD_sites.csv")
    per = []
    for (seed, site), d in D.groupby(["seed", "site"]):
        for task, pc, mc in (("bind", "pred_bind", "meas_bind"), ("escape", "pred_escape", "meas_escape")):
            ok = d[[pc, mc, "blosum"]].dropna()
            if len(ok) >= 8 and ok[pc].std() > 1e-9 and ok[mc].std() > 1e-9:
                per.append(dict(seed=seed, site=site, task=task, rho=spearmanr(ok[pc], ok[mc])[0],
                                rho_blosum=spearmanr(ok.blosum, ok[mc])[0], n=len(ok)))
    PS = pd.DataFrame(per)
    if len(PS):
        ann = sites.set_index("site_RBD")
        PS["RSA"] = PS.site.map(ann.RSA_unbound); PS["entropy"] = PS.site.map(ann.entropy); PS["RBM"] = PS.site.map(ann.RBM)
        PS.to_csv(OUT / "per_site_transfer.csv", index=False)
        res["per_site"] = {}
        res["vs_blosum"] = {}
        for task, d in PS.groupby("task"):
            dd = d.dropna(subset=["rho", "rho_blosum"])
            if len(dd) >= 8:
                from scipy.stats import wilcoxon as _w
                diff = dd.rho - dd.rho_blosum
                res["vs_blosum"][task] = dict(model_median=float(dd.rho.median()), blosum_median=float(dd.rho_blosum.median()),
                                              median_diff=float(diff.median()), n=len(dd),
                                              p=float(_w(diff).pvalue) if not np.allclose(diff, 0) else float("nan"))
                log(f"vs BLOSUM62 [{task}]: model {dd.rho.median():.3f} vs BLOSUM {dd.rho_blosum.median():.3f} "
                    f"(median diff {diff.median():+.3f}, P={res['vs_blosum'][task]['p']:.3g}, n={len(dd)})")
        for task, d in PS.groupby("task"):
            dd = d.dropna(subset=["RSA"])
            r_rsa = spearmanr(dd.RSA, dd.rho) if len(dd) > 8 else (np.nan, np.nan)
            de = d.dropna(subset=["entropy"])
            r_ent = spearmanr(de.entropy, de.rho) if len(de) > 8 else (np.nan, np.nan)
            rb = d.dropna(subset=["RBM"])
            u = mannwhitneyu(rb.rho[rb.RBM], rb.rho[~rb.RBM]) if rb.RBM.sum() > 3 and (~rb.RBM).sum() > 3 else None
            res["per_site"][task] = dict(n_site_seed=len(d), median_rho=float(d.rho.median()),
                                         frac_positive=float((d.rho > 0).mean()),
                                         rho_vs_RSA=[float(r_rsa[0]), float(r_rsa[1])],
                                         rho_vs_entropy=[float(r_ent[0]), float(r_ent[1])],
                                         RBM_vs_core=[float(rb.rho[rb.RBM].median()), float(rb.rho[~rb.RBM].median()),
                                                      float(u.pvalue)] if u else None)
            log(f"per-position {task}: median rho {d.rho.median():.3f}, {100*(d.rho>0).mean():.0f}% positive, "
                f"rho~RSA {r_rsa[0]:.3f} (P={r_rsa[1]:.3g})")
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for ax, (task, tname) in zip(axes, (("bind", "ACE2 binding"), ("escape", "Antibody escape"))):
            d = PS[PS.task == task].dropna(subset=["RSA"])
            ax.scatter(d.RSA, d.rho, s=11, alpha=.55, color="#d9541e", edgecolor="none")
            ax.axhline(0, color="k", lw=.6, ls=":"); ax.set_xlabel("Relative solvent accessibility (unbound)")
            ax.set_title(tname, fontsize=9.5)
            if len(d) > 8:
                z = np.polyfit(d.RSA, d.rho, 1); xs = np.linspace(d.RSA.min(), d.RSA.max(), 50)
                ax.plot(xs, np.polyval(z, xs), color="k", lw=1)
        axes[0].set_ylabel("Per-position Spearman ρ\n(held-out position)")
        fig.tight_layout(); fig.savefig(FIG / "fig5_per_site.png", dpi=220); plt.close(fig)
    json.dump(res, open(OUT / "holdout_analysis.json", "w"), indent=1, default=float)
    log("saved results/holdout_analysis.json")
