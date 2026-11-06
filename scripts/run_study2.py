"""Expanded study: nine model families on three phenotypes under two evaluation designs.

The first-generation study (scripts/run_study.py, results/jobs/) compared an additive one-hot
model, a global-epistasis fit and a sequence network, and found that at held-out positions none
of them beat an untrained mutation-count baseline. That result had a weakness: all three
represent a substitution by which position it occupies, so none has a parameter that applies to a
position absent from training. Their failure was partly a property of the representation rather
than a statement about proteins.

This run closes that gap. It adds families whose representation is defined at unseen positions:

  mutation_count     untrained; ranks by -n_substitutions
  blosum_zeroshot    untrained; sums BLOSUM62 exchangeability
  esm_zeroshot       untrained on the assay; sums ESM-2 masked-marginal log-likelihood ratios
  physchem           ridge on how the residue changes (hydrophobicity, volume, charge, ...)
  gbm / knn          gradient boosting and nearest neighbours on the transferable feature block
                     including per-position structural descriptors from PDB 6M0J
  ridge_additive     the original one-hot additive model
  global_epistasis   additive latent phenotype with a learned monotone link
  evoclassnet        the sequence network
  evoclassnet_plus   the same network given ESM embeddings and structural features per position
  struct_gnn         message passing over the residue contact graph

Results go to results/jobs_v2/ with the full metric panel from evoclassnet.metrics.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
JOBS = OUT / "jobs_v2"
TASKS = ["bind", "expr", "escape"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
CACHE = {}


def onehot_mut(subs_list, L=201):
    from evoclassnet.data import AA2I
    r, c = [], []
    for i, s in enumerate(subs_list):
        for _, p, m in s:
            r.append(i); c.append((p - 1) * 20 + AA2I[m])
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(subs_list), L * 20))


def permute_positions(X, seed, L=201):
    rng = np.random.default_rng(10_000 + seed)
    Xp = X.copy()
    for p in range(L):
        Xp[:, p] = rng.permutation(20)[X[:, p]]
    return Xp


def init_cache():
    from evoclassnet import data, plm, zoo, structure as S
    g = data.load_genotypes()
    subs = g.subs.tolist()
    CACHE["g"] = g
    CACHE["X"] = data.encode(subs)
    CACHE["Y"] = g[TASKS].to_numpy(dtype=np.float32)
    CACHE["nsub"] = g.n_sub.to_numpy()
    CACHE["S"] = onehot_mut(subs)
    sfeat, smask, adj, snames = S.features()
    CACHE["sfeat"], CACHE["adj"], CACHE["snames"] = sfeat, adj, snames
    mm, hid = plm.load()
    CACHE["mm"] = mm
    CACHE["side"] = (np.concatenate([sfeat, plm.embedding_features(hid, 16)], 1)
                     if mm is not None else sfeat)
    CACHE["esm"] = zoo.esm_scores(subs, mm) if mm is not None else None
    CACHE["blosum"] = zoo.blosum_scores(subs)
    CACHE["F"] = zoo.genotype_features(subs, esm_mm=mm, struct=sfeat, struct_names=snames)
    return g


def _ev(te, P_col, t):
    from evoclassnet.metrics import evaluate
    return evaluate(CACHE["Y"][te, t], P_col, CACHE["nsub"][te])


def one_run(split_name, seed, hp, epochs=8, test_frac=0.15, permute=False, models=None):
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    from evoclassnet import data, model as M, zoo
    g, X, Y = CACHE["g"], CACHE["X"], CACHE["Y"]
    if split_name == "random":
        tr, va, te = data.split_random(g, seed)
    else:
        tr, va, te = data.split_heldout_sites(g, seed, test_site_frac=test_frac, val_site_frac=0.10)
    if permute:
        X = permute_positions(X, seed)
    want = set(models or ["mutation_count", "blosum_zeroshot", "esm_zeroshot", "physchem", "gbm",
                          "knn", "ridge_additive", "global_epistasis", "evoclassnet",
                          "evoclassnet_plus", "struct_gnn"])
    res = {"_sizes": [int(len(tr)), int(len(va)), int(len(te))], "_test_frac": test_frac}

    def record(name, P):
        res[name] = {t: _ev(te, P[:, i], i) for i, t in enumerate(TASKS)}

    # ---- untrained scorers: one score per variant, shared across tasks
    if "mutation_count" in want:
        record("mutation_count", np.repeat(-CACHE["nsub"][te][:, None].astype(float), 3, axis=1))
    if "blosum_zeroshot" in want:
        record("blosum_zeroshot", np.repeat(CACHE["blosum"][te][:, None], 3, axis=1))
    if "esm_zeroshot" in want and CACHE["esm"] is not None:
        record("esm_zeroshot", np.repeat(CACHE["esm"][te][:, None], 3, axis=1))

    # ---- feature models on the transferable block
    F = CACHE["F"]
    for name in ("physchem", "gbm", "knn"):
        if name in want:
            record(name, zoo.fit_predict_sklearn(name, F, Y, tr, te, seed=seed))

    # ---- original additive one-hot
    if "ridge_additive" in want:
        S = CACHE["S"]
        P = np.zeros((len(te), 3))
        for t in range(3):
            m = ~np.isnan(Y[tr, t])
            P[:, t] = (Ridge(alpha=1.0).fit(S[tr][m], Y[tr, t][m]).predict(S[te])
                       if m.sum() >= 10 else 0.0)
        record("ridge_additive", P)

    if "global_epistasis" in want:
        ge = M.fit(M.build(M.GlobalEpistasis, seed, n_tasks=3), X[tr], Y[tr], X[va], Y[va],
                   epochs=15, bs=512, lr=1e-2, seed=seed, log=lambda *a: None)
        record("global_epistasis", M.predict(ge, X[te]))

    nk = {k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")}
    if "evoclassnet" in want:
        net = M.build(M.EvoClassNetLite, seed, n_tasks=3, **nk)
        net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, bs=256, lr=hp["lr"],
                    wd=hp["wd"], seed=seed, log=lambda *a: None)
        record("evoclassnet", M.predict(net, X[te]))
        if split_name == "random" and seed == 0 and not permute:
            torch.save(net.state_dict(), OUT / "evoclassnet_seed0.pt")

    if "evoclassnet_plus" in want:
        net = zoo.build_evoclassnet_plus(seed, CACHE["side"], n_tasks=3, **nk)
        net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, bs=256, lr=hp["lr"],
                    wd=hp["wd"], seed=seed, log=lambda *a: None)
        record("evoclassnet_plus", M.predict(net, X[te]))

    if "struct_gnn" in want:
        net = zoo.build_gnn(seed, CACHE["adj"], CACHE["sfeat"], n_tasks=3)
        net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=epochs, bs=256, lr=2e-3,
                    wd=1e-4, seed=seed, log=lambda *a: None)
        record("struct_gnn", M.predict(net, X[te]))
    return res


def _worker(spec):
    name, kw = spec
    f = JOBS / f"{name}.json"
    if f.exists():
        return name, json.load(open(f))
    t0 = time.time()
    r = one_run(**kw)
    r["_seconds"] = round(time.time() - t0, 1)
    json.dump(r, open(f, "w"), indent=1, default=float)
    return name, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--sweep_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--torch_threads", type=int, default=1)
    ap.add_argument("--jobs", nargs="+", default=["main", "sweep", "ablation"])
    ap.add_argument("--splits", nargs="+", default=["heldout_sites", "random"],
                    help="order matters: the first split's jobs are queued first")
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_THREADS"] = str(a.torch_threads)
    hp = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in json.load(open(OUT / "ga_hyperparams.json"))["best"].items()}
    g = init_cache()
    log(f"{len(g):,} genotypes; " + ", ".join(f"{t}={int(g[t].notna().sum()):,}" for t in TASKS))
    log(f"feature block {CACHE['F'].shape}, side info {CACHE['side'].shape}, "
        f"ESM {'loaded' if CACHE['esm'] is not None else 'MISSING'}")

    specs = []
    if "main" in a.jobs:
        for sp in a.splits:
            for s in range(a.seeds):
                specs.append((f"main_{sp}_s{s}", dict(split_name=sp, seed=s, hp=hp, epochs=a.epochs)))
    if "sweep" in a.jobs:
        for fr in (0.05, 0.15, 0.30, 0.50):
            for s in range(a.sweep_seeds):
                specs.append((f"sweep_f{int(fr * 100)}_s{s}",
                              dict(split_name="heldout_sites", seed=s, hp=hp, epochs=a.epochs,
                                   test_frac=fr)))
    if "ablation" in a.jobs:
        for sp in a.splits:
            for s in range(a.sweep_seeds):
                specs.append((f"abl_perm_{sp}_s{s}",
                              dict(split_name=sp, seed=s, hp=hp, epochs=a.epochs, permute=True)))
    done = sum(1 for n, _ in specs if (JOBS / f"{n}.json").exists())
    log(f"{len(specs)} runs, {done} already cached")

    import multiprocessing as mp
    results = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, (name, r) in enumerate(pool.imap_unordered(_worker, specs), 1):
            results[name] = r
            ev = r.get("evoclassnet", {})
            log(f"[{i}/{len(specs)}] {name} " +
                " ".join(f"{t[:4]}={ev.get(t, {}).get('spearman', float('nan')):.3f}" for t in TASKS))
    json.dump(results, open(OUT / "study_results_v2.json", "w"), indent=1, default=float)
    log("all runs finished")
