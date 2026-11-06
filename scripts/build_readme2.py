"""Regenerate README.md from the measured outputs. Every number is read from results/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
J = lambda n: json.load(open(OUT / n)) if (OUT / n).exists() else {}
C = lambda n: pd.read_csv(OUT / n) if (OUT / n).exists() else pd.DataFrame()

PANEL, COMPAR = C("table_panel.csv"), C("table_comparisons.csv")
GEN, GENC = C("table_genomes.csv"), C("table_genome_comparisons.csv")
FEAT, HO, STAB = J("feature_ablation.json"), J("holdout_analysis.json"), J("stability_check.json")
ESMC, SUM2 = J("esm_control.json"), J("run_summary_v2.json")
TBP = J("transfer_by_position.json")
PGT, PGA = C("table_proteingym.csv"), C("proteingym_per_assay.csv")
PGC, PGCONF = C("table_proteingym_comparisons.csv"), J("proteingym_confound.json")
GMETA = J("genome_study.json").get("meta", {})

L = []
w = L.append


def md(df, cols, head=None):
    if df is None or not len(df):
        return
    d = df[[c for c in cols if c in df.columns]]
    h = head or list(d.columns)
    w("| " + " | ".join(h) + " |")
    w("|" + "|".join(["---"] * len(h)) + "|")
    for _, r in d.iterrows():
        w("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    w("")


def fpm(arm, task, kind="partial"):
    s = FEAT.get("summary", {}).get(arm, {}).get(task, {})
    m, d = s.get(f"{kind}_mean", float("nan")), s.get(f"{kind}_sd", float("nan"))
    return f"{m:+.3f} ± {d:.3f}" if not np.isnan(m) else "—"


def pv(split, task, model, metric="spearman"):
    if not len(PANEL):
        return None
    s = PANEL[(PANEL.split == split) & (PANEL.task == task) & (PANEL.model == model)]
    return None if not len(s) else s.iloc[0].get(metric)


w("# What a held-out-position split actually measures")
w("")
w("Every number here is read out of `results/` by `scripts/build_readme2.py`.")
w("")
w("## The finding, in three steps")
w("")
w("**1. A hard split needs an untrained control.** On a combinatorial mutagenesis library, "
  "ranking variants by `−n_substitutions` — no training, no parameters, no idea which positions "
  "are involved — already scores well, because most substitutions are deleterious and their "
  "effects accumulate. A model that learned nothing about the withheld positions inherits that "
  "score. Better still, report the **partial Spearman** correlation, which removes substitution "
  "count from both sides: it measures position-level signal directly, and it validates itself, "
  "because the trivial baseline must score exactly zero on it. It does: "
  f"{FEAT.get('summary', {}).get('nsub_only', {}).get('bind', {}).get('partial_mean', float('nan')):+.3f}.")
w("")
w("**2. Sequence models indexed by position collapse there.** The additive model has no "
  "parameter for an unseen position and emits a constant; global epistasis goes "
  "anti-correlated; the neural network keeps a positive raw ρ that is almost entirely the "
  "substitution-count effect.")
w("")
w("**3. But that is a fact about the representation, not about the data — and this corrects an "
  "earlier version of this repository.** Every model in step 2 encodes a substitution by *where* "
  "it occurs, so none holds a parameter that applies to a held-out position. Give a plain ridge "
  "regression features that survive at an unseen position — how the residue changes chemically, "
  "and descriptors of the position taken from the crystal structure — and extrapolation is "
  "substantial:")
w("")
if FEAT:
    rows = []
    for arm, lab in [("nsub_only", "Substitution count only (untrained)"),
                     ("esm_only", "ESM-2 zero-shot only"),
                     ("chem_only", "Residue chemistry only"),
                     ("struct_only", "Structure only"),
                     ("struct_nointerface", "Structure, interface descriptors dropped"),
                     ("chem+struct", "Chemistry + structure"),
                     ("chem+struct+esm", "Chemistry + structure + ESM-2")]:
        if arm not in FEAT.get("summary", {}):
            continue
        rows.append({"Features": lab, "binding ρ": fpm(arm, "bind", "spearman"),
                     "binding partial ρ": fpm(arm, "bind"),
                     "expression partial ρ": fpm(arm, "expr"),
                     "escape ρ": fpm(arm, "escape", "spearman")})
    md(pd.DataFrame(rows), ["Features", "binding ρ", "binding partial ρ",
                            "expression partial ρ", "escape ρ"])
    w("Held-out-position split, 10 seeds. Escape has no partial column: every escape-labelled "
      "variant carries one substitution, so the confound is absent there by construction.")
    w("")
    w(f"Two things to note. **Structure carries most of it** "
      f"({fpm('struct_only','bind')} partial for binding against {fpm('chem_only','bind')} for "
      f"chemistry), and it is not an artefact of knowing the receptor interface: the two "
      f"descriptors derived from the ACE2-bound complex mark exactly the contact residues, which "
      f"is nearly circular for binding, and dropping them leaves "
      f"{fpm('struct_nointerface','bind')}. **ESM-2 contributes nothing on this protein** — its "
      f"raw {fpm('esm_only','bind','spearman')} is the substitution-count effect in full, with "
      f"{fpm('esm_only','bind')} surviving the control.")
    w("")
if ESMC:
    u, rf, rs = (ESMC.get(k, {}) for k in ("ubiquitin", "rbd_fragment", "rbd_in_spike"))
    w(f"That null result was checked rather than assumed: ESM-2 wild-type recovery was "
      f"{u.get('recovery', float('nan')):.1%} on human ubiquitin, "
      f"{rf.get('recovery', float('nan')):.1%} on the isolated RBD and "
      f"{rs.get('recovery', float('nan')):.1%} on the same positions inside the full spike "
      f"protein, with no weights missing at load time.")
    w("")

if len(PGT):
    n_assay = int(PGA.dms_id.nunique()) if len(PGA) else 0
    w(f"## It is not one protein: {n_assay} ProteinGym assays")
    w("")
    w("Two baselines make the point unambiguous. A **position-mean** model is a ridge regression "
      "on position indicators alone — it knows which position a variant mutates and nothing at "
      "all about which substitution is made. A **residue-identity** model is its mirror image.")
    w("")
    md(PGT, ["split", "family", "model", "n_assays", "median", "q25", "q75", "frac_above_0_2",
             "frac_at_zero"],
       ["Split", "Family", "Model", "Assays", "Median ρ", "Q1", "Q3", "Frac ρ>0.2", "Frac ρ=0"])
    def pgv(split, model, col="median"):
        r = PGT[(PGT.split == split) & (PGT.model == model)]
        return None if not len(r) else r.iloc[0][col]
    pm_r, pm_h = pgv("Random split", "Position mean"), pgv("Held-out-position split", "Position mean")
    ch_r, ch_h = pgv("Random split", "Chemistry ridge"), pgv("Held-out-position split", "Chemistry ridge")
    if None not in (pm_r, pm_h, ch_r, ch_h):
        w(f"Under a **random split** the position-mean model has a median per-assay ρ of "
          f"**{pm_r}**, against {ch_r} for chemistry: knowing only *where* beats knowing only "
          f"*what*. Under the **held-out-position split** it falls to **{pm_h}** — not a "
          f"degradation but an identity, since it holds no parameter for a position it never "
          f"saw — while chemistry retains {ch_h}.")
        w("")
        w("The standard evaluation rewards memorising per-position means. That capability is "
          "worth exactly nothing at a position no experiment has probed.")
        w("")
    if PGCONF:
        w(f"Worth noting: the mutation-count confound that dominates the RBD analysis is mostly "
          f"*absent* here — only {PGCONF.get('n_with_multi_mutants',0)} of "
          f"{PGCONF.get('n_assays',0)} assays contain any multi-substitution variant. So the "
          f"separation between representations does not depend on the confound existing.")
        w("")
    cs_h = pgv("Held-out-position split", "Chemistry + structure")
    if cs_h is not None and ch_h is not None:
        n_s = int(PGA[PGA.split == "heldout_positions"].chem_struct.notna().sum()) if "chem_struct" in PGA.columns else 0
        w(f"AlphaFold models were obtained for {n_s} of the assays. Adding their per-residue "
          f"descriptors moves the median held-out-position ρ from {ch_h} to **{cs_h}**. This is "
          f"a weaker instrument than the RBD analysis, where the structure is a crystal of the "
          f"bound complex and supplies descriptors a predicted monomer cannot.")
        w("")

if TBP:
    w("## Where the transfer comes from: positions, not substitutions")
    w("")
    w("A narrower question separates two components. Within a *single* withheld position, can the "
      "model order the nineteen substitutions available there?")
    w("")
    rows = []
    for t, tn in (("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")):
        v = TBP.get(t)
        if not v:
            continue
        vb, me = v["vs_blosum"], v["mixed_effects"]
        rows.append({"Phenotype": tn, "Model median ρ": f"{v['median_rho']:+.3f}",
                     "% positive": f"{100*v['frac_positive']:.0f}",
                     "BLOSUM62 median ρ": f"{vb['blosum_median']:+.3f}",
                     "Difference": f"{vb['median_diff']:+.3f}",
                     "P": f"{vb['p']:.3g}",
                     "n position–seed": vb["n"]})
    md(pd.DataFrame(rows), ["Phenotype", "Model median ρ", "% positive", "BLOSUM62 median ρ",
                            "Difference", "P", "n position–seed"])
    b, e = TBP.get("bind", {}), TBP.get("escape", {})
    if b and e:
        w(f"For binding and expression the model matches an **untrained substitution matrix** "
          f"({b['median_rho']:+.3f} against {b['vs_blosum']['blosum_median']:+.3f}, "
          f"P = {b['vs_blosum']['p']:.3g}): within a position, the ordering is chemistry and "
          f"needs no model. The model's real advantage is in telling **which positions** tolerate "
          f"substitution at all, and that is what the structural descriptors supply — a "
          f"position-independent matrix cannot.")
        w("")
        w(f"Escape runs the other way. BLOSUM62 is not merely uninformative there but "
          f"anti-correlated ({e['vs_blosum']['blosum_median']:+.3f}), because the substitutions "
          f"that abolish antibody binding are not the chemically radical ones, while the model "
          f"reaches {e['median_rho']:+.3f} (difference {e['vs_blosum']['median_diff']:+.3f}, "
          f"P = {e['vs_blosum']['p']:.3g}). That within-position ordering is genuinely learned.")
        w("")

if len(PANEL):
    w("## Full metric panel")
    w("")
    for sp, title in (("random", "Random genotype split"),
                      ("heldout_sites", "Held-out-position split")):
        d = PANEL[(PANEL.split == sp) & (PANEL.task == "ACE2 binding")]
        if not len(d):
            continue
        w(f"**{title}** — ACE2 binding (`results/table_panel.csv` has all phenotypes and metrics)")
        w("")
        md(d, ["family", "model", "seeds", "spearman", "spearman_partial", "pearson", "kendall",
               "r2", "rmse", "auroc_deleterious", "precision_at_50", "ndcg_at_100",
               "calibration_slope", "n"],
           ["Family", "Model", "Seeds", "Spearman ρ", "Partial ρ", "Pearson r", "Kendall τ",
            "R²", "RMSE", "AUROC", "Prec@k", "NDCG", "Calib.", "n"])

if len(COMPAR):
    w("## Equivalence testing")
    w("")
    w(f"A non-significant difference is not evidence of no difference. Each trained model is "
      f"also tested for *equivalence* to the untrained baseline with two one-sided tests at a "
      f"±{SUM2.get('margin', 0.05)} margin in Spearman ρ, and P values are FDR-corrected across "
      f"the whole model-by-task family.")
    w("")
    sel = COMPAR[(COMPAR.split == "heldout_sites") & (COMPAR.reference == "Mutation count")
                 & (COMPAR.metric == "spearman")]
    md(sel, ["task", "model", "median_diff", "ci_lo", "ci_hi", "p_fdr", "tost_p", "equivalent",
             "cliffs_delta", "effect"],
       ["Phenotype", "Model", "Median Δρ", "CI low", "CI high", "P (FDR)", "TOST P", "Equivalent",
        "Cliff's δ", "Effect"])

if len(GEN):
    w("## Second task: coronavirus genus classification")
    w("")
    w(f"{GMETA.get('n_genomes','?')} complete genomes from {GMETA.get('n_species','?')} species "
      f"across four genera, pulled from NCBI. SARS-CoV-2 is excluded from Betacoronavirus: with "
      f"millions of deposited genomes against hundreds for whole genera, it would make the class "
      f"a measure of one outbreak's sequencing effort. The same question is asked — a random "
      f"split against a held-out-species split, with an untrained control.")
    w("")
    md(GEN, ["split", "family", "model", "seeds", "accuracy", "balanced_accuracy", "macro_f1",
             "mcc"],
       ["Split", "Family", "Model", "Seeds", "Accuracy", "Balanced acc.", "Macro F1", "MCC"])
    w("Genome length and base composition need no training and no sequence model. They are the "
      "score any reported accuracy on this task has to be read against.")
    w("")

if STAB:
    w("## Run-to-run instability")
    w("")
    for ss in sorted({r["split_seed"] for r in STAB}):
        v = [r["bind"] for r in STAB if r["split_seed"] == ss]
        w(f"* partition seed {ss}, varying only the weight initialisation: "
          + ", ".join(f"{x:+.3f}" for x in v) + f"  (range {max(v)-min(v):.3f})")
    allv = [r["bind"] for r in STAB]
    w("")
    w(f"Spread across all {len(allv)} runs: {min(allv):+.3f} to {max(allv):+.3f}. A single run "
      f"on this benchmark is not a measurement.")
    w("")

w("## Data")
w("")
w("| Source | Use |")
w("|---|---|")
w("| Starr et al. 2020 RBD yeast-display DMS | binding and expression; barcodes sharing a "
  "genotype collapsed to the mean so one sequence cannot straddle a split |")
w("| Bloom lab aggregated RBD escape maps | per-substitution escape over ≥ 20 antibodies, log10; "
  "rows measured in a non-Wuhan background dropped, since those positions are exactly the VOC "
  "ones |")
w("| PDB 6M0J (RBD–ACE2 complex) | solvent accessibility bound and unbound, interface burial, "
  "distance to ACE2, contact number, burial, B-factor, contact graph |")
w("| UniProt P0DTC2 | full spike sequence, for scoring the RBD in its native context |")
w("| NCBI nucleotide | coronavirus genomes for the classification task |")
w("| facebook/esm2_t33_650M_UR50D | masked-marginal substitution scores and per-residue "
  "embeddings |")
w("")

w("## What is still missing")
w("")
for t in ["One protein, one genetic background, two assay platforms. The evaluation lesson is "
          "general; that structure carries the transfer is shown only here.",
          "The structural descriptors come from a solved complex. For a target without one they "
          "would have to be predicted, and that substitution is untested.",
          "One language model at one size with one scoring scheme. Fine-tuning on viral "
          "sequences was not tried.",
          "Hyper-parameters were tuned on random-split validation, which if anything understates "
          "the trained models at held-out positions.",
          "The genome task is a demonstration that the lesson transfers, not a competitive "
          "classifier."]:
    w(f"* {t}")
w("")

w("## Layout")
w("")
w("```")
for line in [
    "evoclassnet/data.py        genotype loading, encoding, both split protocols",
    "evoclassnet/escape.py      per-substitution escape labels and background filtering",
    "evoclassnet/structure.py   per-residue features and contact graph from PDB 6M0J",
    "evoclassnet/plm.py         ESM-2 masked marginals and per-residue embeddings",
    "evoclassnet/zoo.py         transferable feature block, untrained scorers, GNN",
    "evoclassnet/genomes.py     genome loading, k-mer features, species-aware splits",
    "evoclassnet/model.py       EvoClassNetLite, GlobalEpistasis, seeded build(), fit()",
    "evoclassnet/baselines.py   the mutation-count baseline",
    "evoclassnet/metrics.py     twelve-metric panel including the partial correlation",
    "evoclassnet/stats.py       TOST equivalence, FDR, bootstrap, Cliff's delta, mixed effects",
    "scripts/run_study2.py      eleven models x two splits, cached per run",
    "scripts/feature_ablation.py  which feature block carries the transfer",
    "scripts/analyze2.py        tables, statistics, figures",
    "scripts/make_schematics.py architecture figures drawn from the real configuration",
    "scripts/build_docx2.py     manuscript, generated from results/",
    "scripts/build_readme2.py   this file, generated from results/",
]:
    w(line)
w("```")
w("")
w("## Reproduce")
w("")
w("```bash")
w("./run_all.sh")
w("```")
w("")
if SUM2:
    w(f"The study is {SUM2.get('n_runs', '?')} cached runs in `results/jobs_v2/`, one JSON per "
      f"run; re-running skips completed jobs.")
    w("")

(ROOT / "README.md").write_text("\n".join(L))
print(f"wrote README.md ({len(L)} lines)")
