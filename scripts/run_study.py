"""Main study: multi-task prediction of ACE2 binding, expression and antibody escape for
SARS-CoV-2 RBD variants, with a leakage-controlled evaluation of generalisation to positions
never mutated during training.

Jobs
  main      3 models x {random, held-out-site} splits x N seeds       -> Tables 1-2
  sweep     held-out-site fraction 5/15/30/50 %                        -> Figure 4
  ablation  per-position amino-acid permutation (breaks cross-position -> Table 3
            residue-identity sharing while preserving all other structure)

Every number printed or saved here is measured; nothing is hard-coded.
"""
import argparse, json, os, sys, time
from functools import partial
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from scipy import sparse
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"; JOBS = OUT / "jobs"
TASKS = ["bind", "expr", "escape"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(y, p):
    m = ~np.isnan(y)
    if m.sum() < 3:
        return dict(spearman=float("nan"), pearson=float("nan"), rmse=float("nan"), n=int(m.sum()))
    if np.ptp(p[m]) < 1e-9:            # constant prediction carries no rank information
        return dict(spearman=0.0, pearson=0.0, rmse=float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), n=int(m.sum()), constant=True)
    return dict(spearman=float(spearmanr(y[m], p[m])[0]), pearson=float(pearsonr(y[m], p[m])[0]),
                rmse=float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), n=int(m.sum()))


def onehot_mut(subs_list, L=201):
    r, c = [], []
    from evoclassnet.data import AA2I
    for i, s in enumerate(subs_list):
        for _, p, m in s: r.append(i); c.append((p - 1) * 20 + AA2I[m])
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(subs_list), L * 20))


def permute_positions(X, seed, L=201):
    """Apply an independent random permutation of the amino-acid alphabet at every position.
    Information content per position is unchanged, but a residue's identity at one position no
    longer says anything about the same residue elsewhere, so cross-position transfer is removed."""
    rng = np.random.default_rng(10_000 + seed)
    Xp = X.copy()
    for p in range(L):
        Xp[:, p] = rng.permutation(20)[X[:, p]]
    return Xp


def one_run(split_name, seed, hp, epochs, kind="main", test_frac=0.15, permute=False, nseed=None):
    """Fit all three models on one split and return their test metrics."""
    import torch; torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    from evoclassnet import data, model as M
    g = CACHE["g"]; X = CACHE["X"]; Y = CACHE["Y"]
    if split_name == "random":
        tr, va, te = data.split_random(g, seed)
    else:
        tr, va, te = data.split_heldout_sites(g, seed, test_site_frac=test_frac, val_site_frac=0.10)
    if permute:
        X = permute_positions(X, seed)
    res = {"_sizes": [int(len(tr)), int(len(va)), int(len(te))], "_test_frac": test_frac}
    # additive ridge (not meaningful under position permutation of a one-hot basis -> same model)
    S = CACHE["S"]
    for t, task in enumerate(TASKS):
        m = ~np.isnan(Y[tr, t])
        if m.sum() < 10: res.setdefault("ridge_additive", {})[task] = metrics(Y[te, t], np.zeros(len(te))); continue
        r = Ridge(alpha=1.0).fit(S[tr][m], Y[tr, t][m])
        res.setdefault("ridge_additive", {})[task] = metrics(Y[te, t], r.predict(S[te]))
    ge = M.fit(M.GlobalEpistasis(n_tasks=3), X[tr], Y[tr], X[va], Y[va], epochs=15, bs=512, lr=1e-2, seed=seed, log=lambda *a: None)
    P = M.predict(ge, X[te]); res["global_epistasis"] = {t: metrics(Y[te, i], P[:, i]) for i, t in enumerate(TASKS)}
    net = M.EvoClassNetLite(n_tasks=3, **{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
    net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, bs=256, lr=hp["lr"], wd=hp["wd"], seed=seed, log=lambda *a: None)
    P = M.predict(net, X[te]); res["evoclassnet"] = {t: metrics(Y[te, i], P[:, i]) for i, t in enumerate(TASKS)}
    if kind == "main" and split_name == "random" and seed == 0:
        import torch as _t; _t.save(net.state_dict(), OUT / "evoclassnet_seed0.pt")
    return res


CACHE = {}


def init_cache(hp=None):
    from evoclassnet import data
    g = data.load_genotypes()
    CACHE["g"] = g
    CACHE["X"] = data.encode(g.subs.tolist())
    CACHE["Y"] = g[TASKS].to_numpy(dtype=np.float32)
    CACHE["S"] = onehot_mut(g.subs.tolist())
    return g


def _worker(spec):
    name, kw = spec
    f = JOBS / f"{name}.json"
    if f.exists(): return name, json.load(open(f))
    t0 = time.time(); r = one_run(**kw); r["_seconds"] = round(time.time() - t0, 1)
    json.dump(r, open(f, "w"), indent=1); return name, r


def paired_test(A, B):
    """Wilcoxon signed-rank on per-seed differences (A - B); returns (median diff, p)."""
    d = np.array(A) - np.array(B)
    if len(d) < 5 or np.allclose(d, 0): return float(np.median(d)), float("nan")
    return float(np.median(d)), float(wilcoxon(d).pvalue)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sweep_seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--torch_threads", type=int, default=1)
    ap.add_argument("--jobs", nargs="+", default=["main", "sweep", "ablation"])
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_THREADS"] = str(a.torch_threads)
    hp = json.load(open(OUT / "ga_hyperparams.json"))["best"]
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}
    g = init_cache()
    log(f"{len(g):,} genotypes; labelled " + ", ".join(f"{t}={int(g[t].notna().sum()):,}" for t in TASKS))

    specs = []
    if "main" in a.jobs:
        for sp in ("random", "heldout_sites"):
            for s in range(a.seeds):
                specs.append((f"main_{sp}_s{s}", dict(split_name=sp, seed=s, hp=hp, epochs=a.epochs)))
    if "sweep" in a.jobs:
        for fr in (0.05, 0.15, 0.30, 0.50):
            for s in range(a.sweep_seeds):
                specs.append((f"sweep_f{int(fr*100)}_s{s}", dict(split_name="heldout_sites", seed=s, hp=hp,
                                                                epochs=a.epochs, kind="sweep", test_frac=fr)))
    if "ablation" in a.jobs:
        for sp in ("random", "heldout_sites"):
            for s in range(a.sweep_seeds):
                specs.append((f"abl_perm_{sp}_s{s}", dict(split_name=sp, seed=s, hp=hp, epochs=a.epochs,
                                                          kind="ablation", permute=True)))
    log(f"{len(specs)} runs, {sum(1 for n,_ in specs if (JOBS/f'{n}.json').exists())} already done")

    import multiprocessing as mp
    results = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, (name, r) in enumerate(pool.imap_unordered(_worker, specs), 1):
            results[name] = r; log(f"[{i}/{len(specs)}] {name} " +
                                   " ".join(f"{t[:4]}={r['evoclassnet'][t]['spearman']:.3f}" for t in TASKS))
    json.dump(results, open(OUT / "study_results.json", "w"), indent=1)
    log("all runs finished")
