"""Real, CPU-scale study on public SARS-CoV-2 RBD deep-mutational-scanning data.

Tasks   : predict ACE2 binding (delta log10 Ka) and RBD expression (delta mean F) of
          multi-mutant RBD variants from amino-acid sequence.
Splits  : (A) random genotype split, (B) held-out *sites* split (strict generalisation).
Models  : additive ridge, global-epistasis, EvoClassNet-lite (CNN-BiLSTM-attention, multitask,
          with GA-tuned hyper-parameters).
Everything printed / saved here is measured, nothing is hard-coded.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, spearmanr
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, model as M, ga

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parents[1] / "results"; OUT.mkdir(exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(y, p):
    m = ~np.isnan(y)
    if np.ptp(p[m]) < 1e-9:  # constant prediction carries no rank information
        return dict(pearson=0.0, spearman=0.0, rmse=float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), n=int(m.sum()), constant=True)
    return dict(pearson=float(pearsonr(y[m], p[m])[0]), spearman=float(spearmanr(y[m], p[m])[0]),
                rmse=float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), n=int(m.sum()))


def onehot_mut(subs_list):
    r, c = [], []
    for i, s in enumerate(subs_list):
        for _, p, m in s: r.append(i); c.append((p - 1) * 20 + data.AA2I[m])
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(subs_list), 201 * 20))


def run_split(g, X, Y, split, seed, hp, epochs, tag):
    tr, va, te = split
    res = {}
    # --- additive ridge
    S = onehot_mut(g.subs.tolist())
    for t, name in enumerate(["bind", "expr"]):
        m = ~np.isnan(Y[tr, t])
        r = Ridge(alpha=1.0).fit(S[tr][m], Y[tr, t][m])
        res.setdefault("ridge_additive", {})[name] = metrics(Y[te, t], r.predict(S[te]))
    # --- global epistasis
    ge = M.fit(M.GlobalEpistasis(), X[tr], Y[tr], X[va], Y[va], epochs=15, bs=512, lr=1e-2, seed=seed, log=lambda *a: None)
    P = M.predict(ge, X[te])
    res["global_epistasis"] = {n: metrics(Y[te, t], P[:, t]) for t, n in enumerate(["bind", "expr"])}
    # --- EvoClassNet-lite
    net = M.EvoClassNetLite(**{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
    net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, bs=256, lr=hp["lr"], wd=hp["wd"], seed=seed, log=log)
    P = M.predict(net, X[te])
    res["evoclassnet_lite"] = {n: metrics(Y[te, t], P[:, t]) for t, n in enumerate(["bind", "expr"])}
    res["_sizes"] = [int(len(tr)), int(len(va)), int(len(te))]
    log(tag, json.dumps({k: {n: round(v["spearman"], 3) for n, v in d.items()} for k, d in res.items() if k != "_sizes"}))
    return res, net, (te, P)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ga_pop", type=int, default=6); ap.add_argument("--ga_gens", type=int, default=3)
    ap.add_argument("--ga_sub", type=int, default=12000); ap.add_argument("--ga_epochs", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=0, help="debug: use only N genotypes")
    a = ap.parse_args()
    g = data.load_genotypes()
    if a.subsample: g = g.sample(a.subsample, random_state=0).reset_index(drop=True)
    X = data.encode(g.subs.tolist()); Y = g[["bind", "expr"]].to_numpy(dtype=np.float32)
    log(f"{len(g)} genotypes; bind NaN {np.isnan(Y[:,0]).mean():.2%} expr NaN {np.isnan(Y[:,1]).mean():.2%}")

    # ---- GA hyper-parameter search (validation only; test never touched)
    tr, va, _ = data.split_random(g, 123)
    sub = np.random.default_rng(0).choice(tr, min(a.ga_sub, len(tr)), replace=False)
    def fitness(hp):
        net = M.EvoClassNetLite(**{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
        net = M.fit(net, X[sub], Y[sub], X[va[:3000]], Y[va[:3000]], epochs=a.ga_epochs, lr=hp["lr"], wd=hp["wd"], log=lambda *x: None, patience=9)
        P = M.predict(net, X[va[:3000]])
        return float(np.mean([spearmanr(Y[va[:3000], t][~np.isnan(Y[va[:3000], t])], P[~np.isnan(Y[va[:3000], t]), t])[0] for t in range(2)]))
    best_hp, best_f, hist = ga.GAHyperSearch(fitness, pop=a.ga_pop, gens=a.ga_gens, log=log).run()
    log("GA best hyper-parameters", best_hp, "val spearman", round(best_f, 4))
    json.dump(dict(best=best_hp, fitness=best_f, history=hist), open(OUT / "ga_hyperparams.json", "w"), default=str, indent=1)

    allres = {"random": [], "heldout_sites": []}
    for name, fn in (("random", data.split_random), ("heldout_sites", data.split_heldout_sites)):
        for s in range(a.seeds):
            r, net, _ = run_split(g, X, Y, fn(g, s), s, best_hp, a.epochs, f"[{name} seed{s}]")
            allres[name].append(r)
            if name == "random" and s == 0: torch.save(net.state_dict(), OUT / "evoclassnet_lite_seed0.pt")
    json.dump(allres, open(OUT / "study_results.json", "w"), indent=1)
    # summary table
    rows = []
    for sp, lst in allres.items():
        for mdl in ("ridge_additive", "global_epistasis", "evoclassnet_lite"):
            for task in ("bind", "expr"):
                v = [r[mdl][task] for r in lst]
                rows.append(dict(split=sp, model=mdl, task=task, **{k: f"{np.mean([x[k] for x in v]):.3f} ± {np.std([x[k] for x in v]):.3f}" for k in ("spearman", "pearson", "rmse")}))
    df = pd.DataFrame(rows); df.to_csv(OUT / "summary.csv", index=False); print(df.to_string(index=False))
