"""In-silico variant evolution with a genetic algorithm, used here as a control.

A GA maximises the trained model's own predicted phenotype, starting from the wild-type RBD. We ask
whether the variants it favours resemble substitutions that later defined Variants of Concern. Any
positive result would be confounded, because VOC substitutions are present in the escape training
data; the leakage-free test is in holdout_analysis.py. The purpose here is the opposite one: to
show what naive optimisation of a learned fitness surface actually produces.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, model as M, ga
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"; torch.set_num_threads(4)
VOC = ["G339D", "S371L", "S373P", "S375F", "K417N", "K417T", "N440K", "G446S", "L452R", "S477N",
       "T478K", "E484A", "E484K", "Q493R", "G496S", "Q498R", "N501Y", "Y505H"]

if __name__ == "__main__":
    wt, off = data.wildtype()
    voc = {(v[0], int(v[1:-1]) - off, v[-1]) for v in VOC}
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(OUT / "ga_hyperparams.json"))["best"].items()}
    net = M.EvoClassNetLite(n_tasks=3, **{k: hp[k] for k in ("emb", "ch", "ks", "lstm", "dropout")})
    net.load_state_dict(torch.load(OUT / "evoclassnet_seed0.pt")); net.eval()
    wt_i = np.array([data.AA2I[a] for a in wt])
    res = {}

    for obj, w in (("escape", (0.0, 0.0, 1.0)), ("escape_and_binding", (1.0, 0.0, 1.0))):
        score = lambda X, w=w: (lambda P: w[0] * P[:, 0] + w[1] * P[:, 1] + w[2] * P[:, 2])(M.predict(net, X))
        arch = ga.VariantGA(wt, score, pop=400, gens=20, seed=0, log=lambda *a: None).run()
        items = sorted(arch.values(), key=lambda t: -t[0]); top = items[:max(1, len(items) // 100)]
        nv = lambda x: sum((wt[p], p + 1, data.AA[x[p]]) in voc for p in np.flatnonzero(x != wt_i))
        obs = float(np.mean([nv(x) for _, x in top]))
        rng = np.random.default_rng(0)
        rnd = [np.mean([nv(items[i][1]) for i in rng.choice(len(items), len(top), replace=False)]) for _ in range(500)]
        p = float((np.sum(np.array(rnd) >= obs) + 1) / (len(rnd) + 1))
        muts = pd.Series([f"{wt[q]}{q+1+off}{data.AA[x[q]]}" for _, x in top for q in np.flatnonzero(x != wt_i)]).value_counts()
        res[obj] = dict(archive=len(items), top_n=len(top), top_mean_voc=obs, random_mean_voc=float(np.mean(rnd)),
                        perm_p=p, top_mutations=muts.head(12).to_dict(),
                        frac_top_muts_in_terminal_20=float(np.mean([int(m[1:-1]) >= 512 for m in muts.index[:20]])))
        print(f"[{obj}] archive {len(items)}, top-1% VOC subs/variant {obs:.3f} vs random {np.mean(rnd):.3f} (P={p:.3f})")
        print("  top mutations:", ", ".join(list(muts.index[:8])))
    json.dump(res, open(OUT / "variant_analysis.json", "w"), indent=1, default=float)
