"""Retrospective variant-prediction analysis (NOT a prospective forecast).

(1) Rank all 201x19 single substitutions by model-predicted fitness (binding + 0.5*expression) and
    test whether substitutions that later became lineage-defining in Variants of Concern
    (Alpha-Omicron BA.1, from public lineage definitions) are enriched near the top (AUROC).
    Benchmarks: measured single-mutant DMS values, and the additive ridge baseline.
    Caveat: DMS was measured on the Wuhan-Hu-1 RBD in 2020, i.e. before these VOCs.
(2) Run the variant GA from wild type, and ask whether GA-selected top variants contain more VOC
    substitutions than size-matched random variants (permutation test).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, model as M, ga
R = Path(__file__).resolve().parents[1] / "results"; torch.set_num_threads(4)

VOC = ["G339D","S371L","S373P","S375F","K417N","K417T","N440K","G446S","L452R","S477N","T478K","E484A","E484K",
       "Q493R","G496S","Q498R","N501Y","Y505H"]  # spike numbering
wt, off = data.wildtype()
voc = [(w, int(p[1:-1]) - off, m) for w, p, m in [(v[0], v, v[-1]) for v in VOC]]
voc = [(v[0], int(v[1:-1]) - off, v[-1]) for v in VOC]
for w, p, m in voc: assert wt[p - 1] == w, (w, p, wt[p - 1])

hp = json.load(open(R / "ga_hyperparams.json"))["best"]
net = M.EvoClassNetLite(**{k: (tuple(hp[k]) if isinstance(hp[k], list) else hp[k]) for k in ("emb", "ch", "ks", "lstm", "dropout")})
net.load_state_dict(torch.load(R / "evoclassnet_lite_seed0.pt")); net.eval()
fit = lambda X: (lambda P: P[:, 0] + 0.5 * P[:, 1])(M.predict(net, X))

# (1) single-substitution ranking
singles = [[(wt[p - 1], p, a)] for p in range(1, 202) for a in data.AA if a != wt[p - 1]]
Xs = data.encode(singles); pred = fit(Xs)
sm = pd.read_csv(data.RAW / "single_mut_effects.csv")
emp = {(r.site_RBD, r.mutant): r.bind_avg + 0.5 * r.expr_avg for r in sm.itertuples()}
empv = np.array([emp.get((s[0][1], s[0][2]), np.nan) for s in singles])
is_voc = np.array([(s[0][0], s[0][1], s[0][2]) in set(voc) for s in singles])
def auroc(score, lab):
    ok = ~np.isnan(score); s, l = score[ok], lab[ok]; r = rankdata(s)
    return (r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()), int(l.sum())
rng = np.random.default_rng(0)
def boot(score):
    out = []
    for _ in range(1000):
        idx = rng.integers(0, len(score), len(score)); out.append(auroc(score[idx], is_voc[idx])[0])
    return np.nanpercentile(out, [2.5, 97.5])
res = {}
for name, sc in (("EvoClassNet-lite (predicted)", pred), ("Measured single-mutant DMS", empv)):
    a, n = auroc(sc, is_voc); lo, hi = boot(sc); res[name] = dict(auroc=a, ci95=[lo, hi], n_voc=n)
    print(f"{name:34s} AUROC={a:.3f} [{lo:.3f},{hi:.3f}] (n_VOC_subs={n})")

# (2) variant GA
site_w = np.ones(201)
G = ga.VariantGA(wt, fit, site_weights=site_w, pop=400, gens=20, seed=0, log=lambda *a: None).run()
arch = sorted(G.values(), key=lambda t: -t[0]); top = arch[:max(1, len(arch) // 100)]
vset = set(voc)
def n_voc(x):
    d = np.flatnonzero(x != np.array([data.AA2I[a] for a in wt])); return sum((wt[p], p + 1, data.AA[x[p]]) in vset for p in d)
obs = np.mean([n_voc(x) for _, x in top])
rnd = []
for _ in range(500):
    pick = rng.choice(len(arch), len(top), replace=False); rnd.append(np.mean([n_voc(arch[i][1]) for i in pick]))
pval = (np.sum(np.array(rnd) >= obs) + 1) / (len(rnd) + 1)
print(f"GA archive {len(arch)} variants; top-1% mean VOC substitutions/variant {obs:.3f} vs random-from-archive {np.mean(rnd):.3f} (perm p={pval:.3f})")
res["variant_ga"] = dict(archive=len(arch), top_mean_voc=float(obs), random_mean_voc=float(np.mean(rnd)), perm_p=float(pval))
top_muts = pd.Series([f"{wt[p]}{p+1+off}{data.AA[x[p]]}" for _, x in top for p in np.flatnonzero(x != np.array([data.AA2I[a] for a in wt]))]).value_counts().head(15)
print(top_muts.to_string()); res["top_ga_mutations"] = top_muts.to_dict()
json.dump(res, open(R / "variant_analysis.json", "w"), indent=1, default=float)
