"""Is the occasional failed run caused by a hard partition or by unstable optimisation?

In the main study the split seed and the weight-initialisation seed are the same number, so the
two explanations are confounded. Here we hold the partition fixed and vary only the
initialisation, for a partition that failed and for one that succeeded.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, model as M
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"
torch.set_num_threads(4)
TASKS = ["bind", "expr", "escape"]
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

if __name__ == "__main__":
    split_seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2", "0"])]
    init_seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2"])]
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(OUT / "ga_hyperparams.json"))["best"].items()}
    g = data.load_genotypes(); X = data.encode(g.subs.tolist()); Y = g[TASKS].to_numpy(dtype=np.float32)
    res = []
    for ss in split_seeds:
        tr, va, te = data.split_heldout_sites(g, ss)
        for init in init_seeds:
            net = M.build(M.EvoClassNetLite, init, n_tasks=3, **{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
            net = M.fit(net, X[tr], Y[tr], X[va], Y[va], epochs=8, bs=256, lr=hp["lr"], wd=hp["wd"], seed=init, log=lambda *a: None)
            P = M.predict(net, X[te])
            r = {}
            for i, t in enumerate(TASKS):
                m = ~np.isnan(Y[te, i])
                r[t] = float(spearmanr(Y[te, i][m], P[m, i])[0]) if m.sum() > 3 and np.ptp(P[m, i]) > 1e-9 else 0.0
            vl = M.evaluate_loss(net, X[va], Y[va])
            res.append(dict(split_seed=ss, init_seed=init, val_loss=vl, **r))
            log(f"split {ss} init {init}: " + " ".join(f"{t}={r[t]:+.3f}" for t in TASKS) + f"  val_loss={vl:.4f}")
    json.dump(res, open(OUT / "stability_check.json", "w"), indent=1)
    for ss in split_seeds:
        v = [x["bind"] for x in res if x["split_seed"] == ss]
        log(f"split {ss}: binding rho over {len(v)} inits = " + ", ".join(f"{x:+.3f}" for x in v))
