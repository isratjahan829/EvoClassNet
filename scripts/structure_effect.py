"""How much does a predicted structure add, and where?

The receptor-binding-domain analysis finds that structural descriptors carry most of the
transfer to unseen positions, but it uses a crystal structure of a bound complex. Across
ProteinGym almost nothing has that, so the structural arm there uses AlphaFold models of the
monomer. This script asks what those are worth, per assay rather than on average, and whether
the gain tracks how confident the prediction is.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import figstyle as F, stats as ST

F.use()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIGF = R / "figures" / "final"
AF = ROOT / "data" / "raw" / "alphafold"
SPLITS = [("random", "Random split"), ("heldout_positions", "Held-out positions"),
          ("pg_modulo_5", "fold_modulo_5"), ("pg_contiguous_5", "fold_contiguous_5")]


def main():
    per = R / "proteingym_per_assay.csv"
    if not per.exists():
        print("proteingym_per_assay.csv missing"); return
    D = pd.read_csv(per)
    if "chem_struct" not in D.columns:
        print("no structural arm in the per-assay table"); return

    cov = {c["dms_id"]: c for c in json.loads((AF / "coverage.json").read_text())} \
        if (AF / "coverage.json").exists() else {}
    D["plddt"] = D.dms_id.map(lambda d: (cov.get(d) or {}).get("plddt"))

    rows, summary = [], {}
    for sp, lab in SPLITS:
        d = D[(D.split == sp)].dropna(subset=["chem_struct", "physchem"])
        if len(d) < 10:
            continue
        gain = (d.chem_struct - d.physchem).to_numpy()
        lo, hi = ST.bootstrap_ci(gain)
        w = wilcoxon(gain) if not np.allclose(gain, 0) else None
        r_pl, p_pl = (np.nan, np.nan)
        dd = d.dropna(subset=["plddt"])
        if len(dd) > 10:
            r_pl, p_pl = spearmanr(dd.plddt, dd.chem_struct - dd.physchem)
        rows.append(dict(split=lab, n_assays=len(d),
                         median_physchem=round(float(d.physchem.median()), 4),
                         median_chem_struct=round(float(d.chem_struct.median()), 4),
                         median_gain=round(float(np.median(gain)), 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         p=float(w.pvalue) if w is not None else float("nan"),
                         frac_assays_improved=round(float((gain > 0).mean()), 3),
                         frac_gain_above_0_05=round(float((gain > 0.05).mean()), 3),
                         gain_vs_plddt_rho=round(float(r_pl), 3),
                         gain_vs_plddt_p=float(p_pl)))
    T = pd.DataFrame(rows)
    T.to_csv(R / "table_structure_effect.csv", index=False)
    summary = T.to_dict(orient="records")
    json.dump(summary, open(R / "structure_effect.json", "w"), indent=1, default=float)

    # figure: per-assay gain, and whether it tracks model confidence
    show = [s for s in ("heldout_positions", "pg_modulo_5") if (D.split == s).any()]
    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE * 0.78, 2.9))
    d = D[D.split == show[0]].dropna(subset=["chem_struct", "physchem"])
    gain = (d.chem_struct - d.physchem).to_numpy()
    ax = axes[0]
    ax.hist(gain, bins=28, color=F.OI["blue"], alpha=0.85, lw=0)
    F.zeroline(ax)
    ax.axvline(np.median(gain), color=F.OI["vermillion"], lw=1.1)
    ax.set_xlabel("Δ Spearman ρ from adding structure")
    ax.set_ylabel("assays")
    ax.set_title(f"Per-assay effect of an AlphaFold model\n{len(d)} assays, "
                 f"{100*(gain>0).mean():.0f}% improved", loc="left")
    F.panel(ax, "a", dx=-0.17, dy=1.18)
    ax = axes[1]
    dd = d.dropna(subset=["plddt"])
    ax.scatter(dd.plddt, dd.chem_struct - dd.physchem, s=11, alpha=0.6, linewidths=0,
               color=F.OI["blue"])
    F.zeroline(ax, x=False)
    if len(dd) > 10:
        z = np.polyfit(dd.plddt, dd.chem_struct - dd.physchem, 1)
        xs = np.linspace(dd.plddt.min(), dd.plddt.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=F.INK, lw=0.9)
        r, pv = spearmanr(dd.plddt, dd.chem_struct - dd.physchem)
        ax.text(0.03, 0.95, f"ρ = {r:+.2f}, P = {pv:.2g}", transform=ax.transAxes,
                va="top", fontsize=7)
    ax.set_xlabel("AlphaFold pLDDT of the model")
    ax.set_ylabel("Δ Spearman ρ")
    ax.set_title("Does the gain track prediction confidence?", loc="left")
    F.panel(ax, "b", dx=-0.20, dy=1.18)
    fig.tight_layout()
    F.save(fig, FIGF / "FigureS1_structure_effect.png")

    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
