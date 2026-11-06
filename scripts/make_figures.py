import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1] / "results"; F = R / "figures"; F.mkdir(exist_ok=True)
S = json.load(open(R / "study_results.json")); V = json.load(open(R / "variant_analysis.json"))
MOD = [("ridge_additive", "Additive ridge", "#8a8a8a"), ("global_epistasis", "Global epistasis", "#4c78a8"), ("evoclassnet_lite", "EvoClassNet-lite", "#d9541e")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
for split, title, fn in (("random", "Random genotype split", "fig1_random_split.png"), ("heldout_sites", "Held-out-site split (unseen positions)", "fig2_heldout_sites.png")):
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); w = 0.26
    for j, (k, lab, c) in enumerate(MOD):
        for t, task in enumerate(("bind", "expr")):
            v = [r[k][task]["spearman"] for r in S[split]]
            ax.bar(t + (j - 1) * w, np.mean(v), w, color=c, label=lab if t == 0 else None)
            ax.scatter([t + (j - 1) * w] * len(v), v, color="k", s=9, zorder=3)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["ACE2 binding", "Expression"])
    ax.set_ylabel("Test Spearman ρ"); ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(F / fn, dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
names = ["EvoClassNet-lite (predicted)", "Measured single-mutant DMS"]
for i, n in enumerate(names):
    r = V[n]; ax.errorbar(r["auroc"], i, xerr=[[r["auroc"] - r["ci95"][0]], [r["ci95"][1] - r["auroc"]]], fmt="o", color="#d9541e" if i == 0 else "#4c78a8", capsize=3)
ax.axvline(0.5, color="k", ls=":", lw=.8); ax.set_yticks([0, 1]); ax.set_yticklabels(["Model", "Measured DMS"]); ax.set_xlim(0.4, 1); ax.set_xlabel("AUROC for VOC substitutions (95% bootstrap CI)")
fig.tight_layout(); fig.savefig(F / "fig3_voc_auroc.png", dpi=200)
print("figures ok")
