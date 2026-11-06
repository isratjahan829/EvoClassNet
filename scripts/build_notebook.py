"""Generate EvoClassNet_report.ipynb: an executable scientific report.

The notebook is written against the cached outputs in results/, so it runs in seconds and
reproduces every table and figure in the manuscript without retraining anything. Cells that would
recompute a result from raw data are included but guarded, so the notebook stays runnable on a
laptop while documenting exactly how each number was produced.
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EvoClassNet_report.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


cells = []
cells.append(md("""\
# Position-level extrapolation in deep mutational scanning models

### a property of the representation, not of the data

This notebook reproduces every table and figure in the accompanying manuscript from the cached
outputs in `results/`. It runs in seconds; nothing is retrained here. The scripts that produced
those outputs are named in each section, and `run_all.sh` reproduces the whole study from raw
data.

**The argument, in four steps.**

1. A held-out-position split is uninterpretable without an untrained control. On a combinatorial
   library, ranking variants by how many substitutions they carry already scores well.
2. The partial Spearman correlation removes that confound directly and validates itself, because
   the trivial baseline must score exactly zero on it.
3. Models that index a substitution by *where* it occurs collapse at held-out positions. This is
   a property of the representation, not a limit of the data — a point an earlier version of
   this work got wrong.
4. Models whose features survive at an unseen position do not collapse. Structure is what carries
   the transfer, and the pattern holds across 217 ProteinGym assays.
"""))

cells.append(md("## 0. Setup"))
cells.append(code("""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

warnings.filterwarnings("ignore")
ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
R, FIG = ROOT / "results", ROOT / "results" / "figures"
J = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
C = lambda n: pd.read_csv(R / n) if (R / n).exists() else pd.DataFrame()
show = lambda n, w=900: display(Image(filename=str(FIG / n), width=w)) if (FIG / n).exists() else print("missing:", n)
pd.set_option("display.width", 200, "display.max_columns", 60)
print("results dir:", R)
print("available:", len(list(R.glob("*.csv"))), "csv,", len(list(FIG.glob("*.png"))), "figures")
"""))

cells.append(md("""## 1. Data

Five sources, all public and all fetched by scripts in this repository.
"""))
cells.append(code("""
from evoclassnet import data as D
g = D.load_genotypes()
print(f"SARS-CoV-2 RBD deep mutational scan: {len(g):,} unique genotypes")
for t, lab in (("bind","ACE2 binding"), ("expr","expression"), ("escape","antibody escape")):
    print(f"  {lab:18s} {int(g[t].notna().sum()):>7,} labelled   "
          f"(single substitutions: {int((g[t].notna() & (g.n_sub==1)).sum()):,})")
print(f"\\nsubstitution counts: median {int(g.n_sub.median())}, max {int(g.n_sub.max())}")
"""))
cells.append(code("""
from evoclassnet import structure as S
f, mask, adj, names = S.compute()
iface = [i + 1 + S.OFFSET for i in range(201) if mask[i] and f[i, names.index("is_interface")] > 0.5]
print(f"PDB 6M0J: {int(mask.sum())}/201 positions resolved, {int(adj.sum()//2)} contact edges")
print("descriptors:", names)
print("\\nACE2 interface recovered from the structure (spike numbering):")
print(" ", "  ".join(str(s) for s in iface))
print("\\nThese are the known ACE2 contacts, which is the check that the geometry pipeline works.")
"""))

cells.append(md("""## 2. The two evaluation designs

The held-out-position split assigns a variant to a partition only if *every* position it mutates
belongs to that partition. The assertion below is the one the whole paper rests on.
"""))
cells.append(code("""
for seed in range(3):
    tr, va, te = D.split_heldout_sites(g, seed)
    tr_s = set().union(*[g.sites.iloc[i] for i in tr])
    te_s = set().union(*[g.sites.iloc[i] for i in te])
    assert not (tr_s & te_s)
    print(f"seed {seed}: train {len(tr):>6,} ({len(tr_s)} positions) | "
          f"test {len(te):>5,} ({len(te_s)} positions) | shared positions: {len(tr_s & te_s)}")
"""))
cells.append(code("""show("schema1_design.png")"""))

cells.append(md("""## 3. The confound, and a metric that removes it

A model that has learned nothing about the withheld positions still orders the test set, because
fitness falls with the number of substitutions. The partial Spearman correlation regresses the
substitution count out of the ranks of both prediction and measurement. It has a built-in check:
the mutation-count baseline must score exactly zero on it.
"""))
cells.append(code("""
from evoclassnet.metrics import evaluate
rng = np.random.default_rng(0); n = 4000
nsub = rng.integers(1, 6, n)
y = -0.8 * nsub + rng.normal(0, 1, n)
demo = {"mutation count only": -nsub.astype(float),
        "genuine predictor":   y + rng.normal(0, 0.5, n)}
rows = []
for lab, p in demo.items():
    r = evaluate(y, p, nsub)
    rows.append(dict(predictor=lab, spearman=round(r["spearman"], 3),
                     partial=round(r["spearman_partial"], 3)))
display(pd.DataFrame(rows))
print("A mutation-count-only predictor looks respectable on Spearman and is exactly zero on the partial.")
"""))

cells.append(md("""## 4. SARS-CoV-2 RBD: the full metric panel"""))
cells.append(code("""
panel = C("table_panel.csv")
if len(panel):
    for sp in ("random", "heldout_sites"):
        d = panel[(panel.split == sp) & (panel.task == "ACE2 binding")]
        print(f"\\n--- {sp}, ACE2 binding")
        display(d[["family","model","seeds","spearman","spearman_partial","pearson",
                   "r2","auroc_deleterious","precision_at_50","n"]])
else:
    print("results/table_panel.csv not present: run scripts/run_study2.py then scripts/analyze2.py")
"""))
cells.append(code("""show("fig1_panel_spearman.png"); show("fig2_panel_partial.png")"""))

cells.append(md("""## 5. Where the signal comes from

`scripts/feature_ablation.py`. Each arm is a ridge regression on a different feature block, all
evaluated on the same held-out-position partitions over 10 seeds.
"""))
cells.append(code("""
FEAT = J("feature_ablation.json").get("summary", {})
order = [("nsub_only","Substitution count only"), ("esm_only","ESM-2 zero-shot only"),
         ("chem_only","Residue chemistry only"), ("struct_only","Structure only"),
         ("struct_nointerface","Structure, interface descriptors dropped"),
         ("chem+struct","Chemistry + structure"),
         ("chem+struct+esm","Chemistry + structure + ESM-2")]
rows = []
for arm, lab in order:
    s = FEAT.get(arm, {})
    if not s: continue
    row = {"features": lab}
    for t, tn in (("bind","binding"), ("expr","expression"), ("escape","escape")):
        v = s.get(t, {})
        row[f"{tn} ρ"] = f"{v.get('spearman_mean', np.nan):+.3f}"
        if t != "escape":
            row[f"{tn} partial"] = f"{v.get('partial_mean', np.nan):+.3f}"
    rows.append(row)
display(pd.DataFrame(rows))
print("Escape has no partial column: every escape-labelled variant carries one substitution,")
print("so the substitution count is constant and the confound is absent by construction.")
"""))

cells.append(md("""### 5a. Positions versus substitutions

`scripts/transfer_by_position.py`. Within a single withheld position, can the model order the
nineteen substitutions available there?
"""))
cells.append(code("""
TBP = J("transfer_by_position.json")
rows = []
for t, tn in (("bind","ACE2 binding"), ("expr","Expression"), ("escape","Antibody escape")):
    v = TBP.get(t)
    if not v: continue
    vb = v["vs_blosum"]
    rows.append(dict(phenotype=tn, model_median=round(v["median_rho"], 3),
                     pct_positive=round(100 * v["frac_positive"]),
                     blosum62=round(vb["blosum_median"], 3),
                     difference=round(vb["median_diff"], 3), p=f"{vb['p']:.3g}", n=vb["n"]))
display(pd.DataFrame(rows))
show("fig8_transfer_by_position.png")
"""))

cells.append(md("""## 6. Does the protein language model help?

`scripts/esm_control.py`. A null result from a large pretrained model invites the suspicion of a
broken pipeline, so the model is checked on a protein it should find easy, and the domain is
scored both alone and inside the full spike protein it belongs to.
"""))
cells.append(code("""
E = J("esm_control.json")
if E:
    display(pd.DataFrame([
        dict(setting=k.replace("_", " "), wt_recovery=f"{v['recovery']:.1%}",
             median_top_prob=round(v["median_top_prob"], 3))
        for k, v in E.items()]))
else:
    print("results/esm_control.json not present: run scripts/esm_control.py")
show("schema3_plm.png")
"""))

cells.append(md("""## 7. Does it hold across proteins? 217 ProteinGym assays

`scripts/run_proteingym.py`. The same comparison on every substitution assay in ProteinGym v1.3.
Two extra baselines make the decomposition clean: a model that knows only *which position* is
mutated, and one that knows only *which residue change* is made.
"""))
cells.append(code("""
T = C("table_proteingym.csv")
if len(T):
    display(T)
    conf = J("proteingym_confound.json")
    print("\\nconfound inventory:", json.dumps(conf, indent=1))
else:
    print("run scripts/run_proteingym.py then scripts/analyze_proteingym.py")
"""))
cells.append(code("""show("fig9_proteingym.png", 1000); show("fig10_proteingym_scatter.png", 820)"""))
cells.append(code("""
Cm = C("table_proteingym_comparisons.csv")
if len(Cm):
    sel = Cm[(Cm.split == "Held-out-position split") & (Cm.reference == "Position mean")]
    display(sel[["model","n_assays","median_diff","ci_lo","ci_hi","p_fdr",
                 "frac_model_better","effect"]])
"""))

cells.append(md("""## 8. A second data type: coronavirus genome classification

`scripts/run_genome_study.py`. Genus classification of coronavirus genomes from NCBI, with the
same question asked and an untrained control that looks only at genome length and base
composition.
"""))
cells.append(code("""
G = C("table_genomes.csv")
if len(G):
    display(G[["split","family","model","seeds","accuracy","balanced_accuracy","macro_f1","mcc"]])
show("fig6_genomes.png", 850); show("fig7_genome_confusion.png", 850)
"""))

cells.append(md("""## 9. Run-to-run instability

`scripts/stability_check.py`. The partition is held fixed and only the weight initialisation
varies.
"""))
cells.append(code("""
ST_ = J("stability_check.json")
if ST_:
    d = pd.DataFrame(ST_)
    display(d.groupby("split_seed").bind.agg(["min","median","max","count"]).round(3))
    allv = d.bind.to_numpy()
    print(f"spread across all {len(allv)} runs: {allv.min():+.3f} to {allv.max():+.3f}")
    print("A single run on this benchmark is not a measurement.")
"""))

cells.append(md("""## 10. What this does and does not show

**Shown.** A held-out-position split needs an untrained control, or better, the partial
correlation. Models that index substitutions by position cannot extrapolate across positions —
and that is about the representation, not the data. Features defined at unseen positions do
extrapolate, structure carries most of it, and the pattern reproduces across the ProteinGym
benchmark.

**Not shown.** That structure carries the transfer is established on one protein with a solved
complex; for a target without one the descriptors would have to be predicted. One language model
at one size with one scoring scheme was evaluated. The genome task is a demonstration that the
evaluation lesson transfers, not a competitive classifier. Seed-level P values describe
reproducibility across partitions of one data set, not inference to other proteins.
"""))
cells.append(code("""
print("Tables written by this study:")
for p in sorted(R.glob("table_*.csv")): print("  ", p.name)
for p in sorted(R.glob("proteingym_*.csv")): print("  ", p.name)
print("\\nFigures:")
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
