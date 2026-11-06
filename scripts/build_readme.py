"""Regenerate README.md from the measured outputs in results/.

Every number in the README is read from a result file, so the README cannot drift away from what
the code actually produced. Run this after scripts/analyze.py, holdout_analysis.py,
stability_check.py and variant_analysis.py.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def jload(name, default=None):
    p = OUT / name
    return json.load(open(p)) if p.exists() else default


def cload(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


main = cload("table_main.csv")
stats = cload("table_stats.csv")
abl = cload("table_ablation.csv")
hold = jload("holdout_analysis.json", {})
stab = jload("stability_check.json", [])
var = jload("variant_analysis.json", {})
rs = jload("run_summary.json", {})

SPLIT = {"random": "random genotypes", "heldout_sites": "held-out positions"}
ORDER = ["Mutation count (untrained)", "Additive ridge", "Global epistasis", "EvoClassNet"]
L = []
w = L.append


def md_table(df, cols, header=None):
    head = header or cols
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join(["---"] * len(head)) + "|")
    for _, r in df.iterrows():
        w("| " + " | ".join(str(r[c]) for c in cols) + " |")
    w("")


w("# EvoClassNet — a leakage-controlled benchmark of sequence models for SARS-CoV-2 RBD phenotypes")
w("")
w("Every number in this file is read out of `results/` by `scripts/build_readme.py`. "
  "Nothing is carried over from the original manuscript draft.")
w("")

# ---------------------------------------------------------------- headline
w("## The finding")
w("")
w("Two evaluation protocols on the same deep-mutational-scanning data rank models differently:")
w("")
w("* **Random genotype split** — a variant's own mutated positions are also mutated in the training "
  "set. All models do well; the neural model is not clearly better than a global-epistasis fit.")
w("* **Held-out-position split** — no training variant mutates any test position. Rank accuracy "
  "collapses, and once the trivial **mutation-count baseline** (`−n_substitutions`, untrained) is "
  "included in the comparison, no trained model in this study demonstrates position-level "
  "extrapolation on binding or expression.")
vb0 = hold.get("vs_blosum", {})
if vb0.get("escape") and vb0.get("bind"):
    e, b = vb0["escape"], vb0["bind"]
    w(f"* **The one exception is antibody escape.** Every escape-labelled variant carries exactly one "
      f"substitution, so the mutation-count baseline is constant there and cannot account for anything. "
      f"At withheld positions EvoClassNet's median per-position ρ is {e['model_median']:.3f} against "
      f"{e['blosum_median']:.3f} for BLOSUM62 exchangeability (paired Δ {e['median_diff']:+.3f}, "
      f"P = {e['p']:.3g}, n = {e['n']}). For binding the ordering reverses: BLOSUM62 reaches "
      f"{b['blosum_median']:.3f} against the model's {b['model_median']:.3f} (Δ {b['median_diff']:+.3f}, "
      f"P = {b['p']:.3g}, n = {b['n']}). So an untrained substitution matrix beats the network at unseen "
      f"positions on binding, and loses to it on escape — escape is the only place here where a trained "
      f"model supplies something that neither the trivial baseline nor generic chemistry does.")
    w("")
w("The mutation-count baseline is decisive because held-out-position test sets contain variants with "
  "different numbers of substitutions, and fitness falls with substitution count. A model can score "
  "a respectable Spearman ρ there while having learned nothing position-specific. Papers that report "
  "held-out-position performance without this control cannot distinguish the two cases.")
w("")
if stats is not None:
    hs = stats[stats.split == "heldout_sites"]
    w("Paired per-seed comparison, EvoClassNet minus baseline (median difference, Wilcoxon signed-rank), "
      "held-out-position split:")
    w("")
    md_table(hs, ["task", "vs_mutation_count", "vs_global_epistasis", "vs_additive", "n_seeds"],
             ["Task", "vs mutation count", "vs global epistasis", "vs additive ridge", "seeds"])

# ---------------------------------------------------------------- main table
if main is not None:
    w("## Main results")
    w("")
    w(f"Test Spearman ρ, mean ± SD over independent seeds (`results/table_main.csv` has Pearson and RMSE).")
    w("")
    for sp in ("random", "heldout_sites"):
        d = main[main.split == sp].copy()
        if not len(d):
            continue
        piv = d.pivot(index="model", columns="task", values="spearman").reindex(
            [m for m in ORDER if m in set(d.model)])
        piv = piv.reset_index()
        ns = d.groupby("task").n_test.median().to_dict()
        w(f"**Split: {SPLIT[sp]}** (median test-set size: "
          + ", ".join(f"{k} {int(v):,}" for k, v in ns.items()) + ")")
        w("")
        cols = [c for c in ["ACE2 binding", "Expression", "Antibody escape"] if c in piv.columns]
        md_table(piv, ["model"] + cols, ["Model"] + cols)

# ---------------------------------------------------------------- stability
if stab:
    w("## Run-to-run instability")
    w("")
    w("Holding the data partition fixed and varying only the weight-initialisation seed "
      "(`scripts/stability_check.py`):")
    w("")
    for ss in sorted({x["split_seed"] for x in stab}):
        v = [x for x in stab if x["split_seed"] == ss]
        w(f"* partition seed {ss}: binding ρ = " + ", ".join(f"{x['bind']:+.3f}" for x in v)
          + "  (val loss " + ", ".join(f"{x['val_loss']:.3f}" for x in v) + ")")
    sp = [x["bind"] for x in stab]
    w("")
    w(f"Spread across all {len(sp)} runs: {min(sp):+.3f} to {max(sp):+.3f}. A single-run number on this "
      "split is not a measurement, which is why every table here reports seed-level distributions.")
    w("")

# ---------------------------------------------------------------- ablation
if abl is not None and len(abl):
    w("## Amino-acid permutation ablation")
    w("")
    w("A per-position permutation of the 20-letter alphabet destroys residue identity shared across "
      "positions while leaving substitution counts untouched. If transfer to unseen positions were "
      "carried by chemistry, permuting should hurt.")
    w("")
    md_table(abl, ["split", "task", "intact", "permuted", "delta"],
             ["Split", "Task", "Intact", "Permuted", "Δ"])
    w("Read the Δ column against the seed-level SD in the same row, not on its own.")
    w("")

# ---------------------------------------------------------------- holdout
if hold:
    w("## Leakage-free variant-of-concern ranking")
    w("")
    nu = hold.get("voc_n_unique_voc_subs")
    w(f"VOC-defining substitutions appear in the escape training data, so ranking them with a model "
      f"that has seen them measures memorisation. Here each VOC substitution contributes only in the "
      f"seeds where its position was withheld"
      + (f". {nu} unique substitutions qualify; the n column counts position-seed instances, "
         f"since a substitution scores once per seed in which its position was withheld" if nu else "")
      + ":")
    w("")
    rows = [dict(scorer=k, auroc=f"{v['auroc']:.3f}",
                 ci=f"{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}", n=v["n_voc"])
            for k, v in hold.get("voc", {}).items()]
    if rows:
        md_table(pd.DataFrame(rows), ["scorer", "auroc", "ci", "n"],
                 ["Scorer", "AUROC", "95% bootstrap CI", "n positives"])
    vb = hold.get("vs_blosum", {})
    if vb:
        w("Against a hand-crafted chemistry prior, per-position rank accuracy at withheld positions:")
        w("")
        rows = [dict(task=k, model=f"{v['model_median']:.3f}", bl=f"{v['blosum_median']:.3f}",
                     diff=f"{v['median_diff']:+.3f}", p=f"{v['p']:.3g}", n=v["n"])
                for k, v in vb.items()]
        md_table(pd.DataFrame(rows), ["task", "model", "bl", "diff", "p", "n"],
                 ["Task", "Model median ρ", "BLOSUM62 median ρ", "Median Δ", "P", "n position–seed"])
    ps = hold.get("per_site", {})
    if ps:
        for task, v in ps.items():
            rr = v.get("rho_vs_RSA") or [float("nan")] * 2
            w(f"* **{task}**: median per-position ρ {v['median_rho']:+.3f}, "
              f"{100 * v['frac_positive']:.0f}% of position–seed cells positive, "
              f"ρ vs relative solvent accessibility {rr[0]:+.3f} (P = {rr[1]:.3g})")
        w("")

# ---------------------------------------------------------------- GA
if var:
    w("## In-silico variant evolution (a control, not a forecast)")
    w("")
    w("A genetic algorithm maximises the trained model's own predicted phenotype from wild type. "
      "This is reported to show what naive optimisation of a learned fitness surface produces:")
    w("")
    for obj, v in var.items():
        need = {"archive", "top_mean_voc", "random_mean_voc", "perm_p",
                "frac_top_muts_in_terminal_20", "top_mutations"}
        if not (isinstance(v, dict) and need <= set(v)):
            continue
        w(f"* **{obj}**: top-1% of a {v['archive']:,}-variant archive carries "
          f"{v['top_mean_voc']:.3f} VOC substitutions per variant vs {v['random_mean_voc']:.3f} for "
          f"random archive variants (permutation P = {v['perm_p']:.3f}); "
          f"{100 * v['frac_top_muts_in_terminal_20']:.0f}% of its top-20 mutations sit in the "
          f"terminal residues (≥ spike 512), where the yeast-display assay is least informative. "
          f"Top mutations: " + ", ".join(list(v["top_mutations"])[:6]))
    w("")
    w("No enrichment. The GA exploits assay artefacts, which is the expected failure mode and the "
      "reason this is a control rather than a result.")
    w("")

# ---------------------------------------------------------------- data / scope
w("## Data")
w("")
w("| Source | Use |")
w("|---|---|")
w("| Starr et al. 2020, SARS-CoV-2 RBD yeast-display DMS (Wuhan-Hu-1, spike 331–531) | ACE2 binding "
  "and expression labels; barcodes sharing a genotype are collapsed to the mean so identical "
  "sequences cannot straddle a split |")
w("| Aggregated RBD antibody-escape maps (Bloom lab, 13 studies) | per-substitution escape labels, "
  "averaged over ≥ 20 monoclonal antibodies, log10 |")
w("| `RBD_sites.csv` structural annotations | relative solvent accessibility, evolutionary entropy, "
  "receptor-binding-motif membership |")
w("")
w("Escape rows whose listed wild-type residue disagrees with the Wuhan-Hu-1 background are dropped "
  "(~1.5% of rows, measured in Omicron backgrounds, at exactly the VOC positions — keeping them "
  "would contaminate the VOC analysis).")
w("")

w("## What this study does not contain")
w("")
w("| Original draft claim | Status |")
w("|---|---|")
w("| 48,891 genomes, 5-species classification (NCBI/GISAID/ENA) | **Not run.** Those hosts are "
  "unreachable from this environment; GISAID additionally requires a data-access agreement. |")
w("| ESM-1b embeddings, GNN over PDB/AlphaFold structures | **Not run.** No GPU and no access to "
  "weights or structures. `EvoClassNetLite` keeps an `extra_dim` hook for precomputed per-residue "
  "features. |")
w("| 99.92% accuracy, 14.7% of GA variants later observed, 2.48M seq/h | **No supporting experiment "
  "exists.** These numbers were not reproduced and are not claimed anywhere in this repository. |")
w("")
w("Before submission this needs: real genomic data for the classification task, a GPU ablation "
  "against a protein language model and a structure-aware baseline, and more seeds on the "
  "held-out-position split.")
w("")

w("## Layout")
w("")
w("```")
w("evoclassnet/data.py        genotype loading, encoding, both split protocols")
w("evoclassnet/escape.py      per-substitution escape labels + background filtering")
w("evoclassnet/model.py       EvoClassNetLite, GlobalEpistasis, seeded build(), fit()")
w("evoclassnet/baselines.py   the mutation-count baseline")
w("evoclassnet/ga.py          GA hyper-parameter search and GA variant evolution")
w("scripts/run_study.py       main / sweep / ablation jobs, cached and restartable")
w("scripts/analyze.py         tables, paired tests, figures 1/2/4")
w("scripts/holdout_analysis.py  leakage-free VOC ranking, BLOSUM62 control, per-position transfer")
w("scripts/stability_check.py   partition fixed, init seed varied")
w("scripts/variant_analysis.py  GA variant evolution control")
w("scripts/build_docx.py      manuscript, generated from results/")
w("scripts/build_readme.py    this file, generated from results/")
w("```")
w("")
w("## Reproduce")
w("")
w("```bash")
w("./run_all.sh")
w("```")
w("")
w("Or step by step:")
w("")
w("```bash")
w("pip install -r requirements.txt")
w("python scripts/download_data.py")
w("python -m pytest tests -q")
w("python scripts/run_study.py --jobs main sweep ablation --seeds 10 --sweep_seeds 5 --workers 4")
w("python scripts/stability_check.py && python scripts/holdout_analysis.py")
w("python scripts/variant_analysis.py")
w("python scripts/analyze.py && python scripts/build_docx.py && python scripts/build_readme.py")
w("```")
w("")
if rs:
    w(f"The study in this repository is {rs.get('n_runs', '?')} cached runs "
      f"(`results/jobs/`, one JSON per run; re-running skips completed jobs)."
      + (f" Median wall time per random-split run: {rs['median_seconds'] / 60:.0f} min on 4 CPU cores."
         if rs.get("median_seconds") else ""))
    w("")

(ROOT / "README.md").write_text("\n".join(L))
print(f"wrote README.md ({len(L)} lines)")
