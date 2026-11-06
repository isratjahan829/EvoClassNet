"""Which feature block carries the transfer to unseen positions?

A ridge regression on transferable features reaches a far higher rank correlation at held-out
positions than any model in the first-generation study. Because that reverses the earlier
conclusion, the result needs to be decomposed rather than announced. Three questions:

1. Is it chemistry, structure, or the language model? The blocks are switched on separately.
2. Is the structural contribution trivially circular? The descriptors come from the RBD *bound to
   ACE2*, so `d_rsa` and `is_interface` mark exactly the residues that touch the receptor, which
   is close to knowing the answer for the binding task. The `struct_nointerface` arm drops those
   two columns and keeps only descriptors of the unbound domain.
3. Does any of it survive the mutation-count confound? Every arm is scored with the partial
   Spearman as well as the raw one.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, metrics, plm, structure as S, zoo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TASKS = ["bind", "expr", "escape"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# column layout produced by zoo.genotype_features
N_PROP = len(zoo.PROP_NAMES)
IDX = {}
k = 0
IDX["delta_props"] = list(range(k, k + N_PROP)); k += N_PROP
IDX["wt_props"] = list(range(k, k + N_PROP)); k += N_PROP
IDX["blosum"] = [k]; k += 1
IDX["wt_onehot"] = list(range(k, k + 20)); k += 20
IDX["mut_onehot"] = list(range(k, k + 20)); k += 20
IDX["esm"] = [k]; k += 1
STRUCT0 = k


def arms(struct_names):
    s_all = list(range(STRUCT0, STRUCT0 + len(struct_names)))
    drop = {struct_names.index("d_rsa"), struct_names.index("is_interface")}
    s_safe = [STRUCT0 + i for i in range(len(struct_names)) if i not in drop]
    chem = IDX["delta_props"] + IDX["wt_props"] + IDX["blosum"] + IDX["wt_onehot"] + IDX["mut_onehot"]
    nsub = [-1]
    return {
        "nsub_only":            nsub,
        "chem_only":            chem + nsub,
        "esm_only":             IDX["esm"] + nsub,
        "struct_only":          s_all + nsub,
        "struct_nointerface":   s_safe + nsub,
        "chem+struct_nointerface": chem + s_safe + nsub,
        "chem+struct":          chem + s_all + nsub,
        "chem+struct+esm":      chem + s_all + IDX["esm"] + nsub,
    }


if __name__ == "__main__":
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    g = data.load_genotypes()
    Y = g[TASKS].to_numpy(dtype=np.float32)
    nsub = g.n_sub.to_numpy()
    sfeat, smask, adj, snames = S.features()
    mm, hid = plm.load()
    F = zoo.genotype_features(g.subs.tolist(), esm_mm=mm, struct=sfeat, struct_names=snames)
    log(f"feature block {F.shape}; structural columns start at {STRUCT0}, names {snames}")
    A = arms(snames)

    res = {}
    for seed in range(nseeds):
        tr, va, te = data.split_heldout_sites(g, seed)
        tr = np.concatenate([tr, va])
        for arm, cols in A.items():
            cols = [c if c >= 0 else F.shape[1] - 1 for c in cols]
            P = zoo.fit_predict_sklearn("physchem", F[:, cols], Y, tr, te, seed=seed)
            for i, t in enumerate(TASKS):
                m = metrics.evaluate(Y[te, i], P[:, i], nsub[te])
                res.setdefault(arm, {}).setdefault(t, []).append(
                    dict(seed=seed, spearman=m["spearman"], partial=m["spearman_partial"],
                         n=m["n"]))
        log(f"seed {seed}: " + "  ".join(
            f"{a}={res[a]['bind'][-1]['spearman']:+.3f}" for a in list(A)[:4]))

    summary = {}
    for arm in A:
        summary[arm] = {}
        for t in TASKS:
            sp = [r["spearman"] for r in res[arm][t]]
            pa = [r["partial"] for r in res[arm][t] if not np.isnan(r["partial"])]
            summary[arm][t] = dict(spearman_mean=float(np.mean(sp)),
                                   spearman_sd=float(np.std(sp, ddof=1)) if len(sp) > 1 else 0.0,
                                   partial_mean=float(np.mean(pa)) if pa else float("nan"),
                                   partial_sd=float(np.std(pa, ddof=1)) if len(pa) > 1 else 0.0,
                                   n_seeds=len(sp))
    json.dump(dict(summary=summary, per_seed=res, columns={k: len(v) for k, v in A.items()}),
              open(OUT / "feature_ablation.json", "w"), indent=1, default=float)

    print(f"\n{'arm':26s} " + "  ".join(f"{t:>22s}" for t in TASKS))
    print(f"{'':26s} " + "  ".join(f"{'rho':>10s}{'partial':>12s}" for _ in TASKS))
    for arm in A:
        line = f"{arm:26s} "
        for t in TASKS:
            s = summary[arm][t]
            line += f"  {s['spearman_mean']:+.3f}±{s['spearman_sd']:.3f} {s['partial_mean']:+.3f}±{s['partial_sd']:.3f}"
        print(line)
    log("saved results/feature_ablation.json")
