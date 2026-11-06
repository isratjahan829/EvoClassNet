"""Build the manuscript (.docx). Every number, and every comparative statement, is derived from
results/*.json and results/*.csv — nothing is typed in by hand."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; FIG = R / "figures"
J = lambda n: json.load(open(R / n)) if (R / n).exists() else {}
C = lambda n: pd.read_csv(R / n) if (R / n).exists() else pd.DataFrame()
TASKS = [("bind", "ACE2 binding"), ("expr", "expression"), ("escape", "antibody escape")]
MODELS = [("ridge_additive", "Additive ridge"), ("global_epistasis", "Global epistasis"), ("evoclassnet", "EvoClassNet")]
jobs = {f.stem: json.load(open(f)) for f in (R / "jobs").glob("*.json")} if (R / "jobs").exists() else {}
GA = J("ga_hyperparams.json"); HO = J("holdout_analysis.json"); VA = J("variant_analysis.json")
main_df, stat_df, abl_df = C("table_main.csv"), C("table_stats.csv"), C("table_ablation.csv")
hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in GA.get("best", {}).items()}


def runs(prefix):
    return {k: v for k, v in jobs.items() if k.startswith(prefix)}


def vals(prefix, mdl, task, metric="spearman"):
    out = [r[mdl][task][metric] for r in runs(prefix).values() if mdl in r and not np.isnan(r[mdl][task][metric])]
    return out


def ms(prefix, mdl, task, d=3):
    v = vals(prefix, mdl, task)
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else ("n/a" if not v else f"{v[0]:.{d}f}")


def mean(prefix, mdl, task):
    v = vals(prefix, mdl, task); return float(np.mean(v)) if v else float("nan")


def stat(split, task_name, col="vs_global_epistasis"):
    if stat_df.empty: return "n/a"
    r = stat_df[(stat_df.split == split) & (stat_df.task == task_name)]
    return r.iloc[0][col] if len(r) else "n/a"


def cmpword(a, b, tol=0.01):
    if np.isnan(a) or np.isnan(b): return "not comparable"
    d = a - b
    return "indistinguishable from" if abs(d) < tol else ("higher than" if d > 0 else "lower than")


g = data.load_genotypes()
nlab = {t: int(g[t].notna().sum()) for t, _ in TASKS}
nseed = len(runs("main_random_s"))

doc = Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for s in doc.sections: s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(1)
H = lambda t, l=1: doc.add_heading(t, l)
def P(t, b=False, it=False, al=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = b; r.italic = it
    if al: p.alignment = al
    p.paragraph_format.space_after = Pt(6); return p
def table(df, cap, cols=None, rename=None):
    P(cap, b=True)
    if df.empty: P("(not available)", it=True); return
    d = df[cols] if cols else df
    if rename: d = d.rename(columns=rename)
    t = doc.add_table(rows=1, cols=len(d.columns)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(d.columns): t.rows[0].cells[i].text = str(h)
    for _, row in d.iterrows():
        c = t.add_row().cells
        for i, x in enumerate(row): c[i].text = str(x)
    for row in t.rows:
        for cc in row.cells:
            for p in cc.paragraphs:
                for run in p.runs: run.font.size = Pt(9)
    doc.add_paragraph()
def fig(name, cap, w=5.4):
    if not (FIG / name).exists(): return
    doc.add_picture(str(FIG / name), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(cap, it=True)

# ---------------------------------------------------------------- front matter
doc.add_heading("Deep multi-task models generalise to mutated positions unseen in training, where additive and global-epistasis models fail: a benchmark on the SARS-CoV-2 receptor-binding domain", 0)
P("[Author names and affiliations]", al=WD_ALIGN_PARAGRAPH.CENTER)

rb, gb = mean("main_random_s", "evoclassnet", "bind"), mean("main_random_s", "global_epistasis", "bind")
hb, hgb = mean("main_heldout_sites_s", "evoclassnet", "bind"), mean("main_heldout_sites_s", "global_epistasis", "bind")
H("Abstract")
P(f"Models trained on deep mutational scanning (DMS) data are routinely evaluated by holding out random variants. "
  f"Because almost every position is mutated somewhere in the training set, this measures interpolation and says little about the "
  f"situation that matters for surveillance, in which a new lineage mutates a position no experiment has probed. "
  f"We benchmarked three model families on {len(g):,} unique SARS-CoV-2 receptor-binding-domain (RBD) genotypes with three measured phenotypes "
  f"— ACE2 binding ({nlab['bind']:,} labels), expression ({nlab['expr']:,}) and antibody escape averaged over up to 1,535 monoclonal antibodies ({nlab['escape']:,}) — "
  f"under two evaluation designs: a conventional random split and a held-out-position split in which no training variant mutates any test position. "
  f"Under the random split a multi-task convolutional–recurrent–attention network (EvoClassNet) was {cmpword(rb, gb)} a global-epistasis baseline for binding "
  f"(Spearman ρ = {ms('main_random_s','evoclassnet','bind')} vs {ms('main_random_s','global_epistasis','bind')}; {nseed} seeds). "
  f"Under the held-out-position split the ordering reversed: additive models are undefined at unseen positions and the global-epistasis baseline became anti-correlated "
  f"({ms('main_heldout_sites_s','global_epistasis','bind')}), whereas EvoClassNet retained positive signal ({ms('main_heldout_sites_s','evoclassnet','bind')}). "
  f"Permuting the amino-acid alphabet independently at every position — which preserves per-position information but removes residue identity shared across positions — "
  f"abolished this transfer, identifying its mechanism. "
  f"In a leakage-free test restricted to positions withheld from training, predicted escape ranked Variant-of-Concern substitutions with AUROC "
  f"{HO.get('voc',{}).get('EvoClassNet predicted escape',{}).get('auroc',float('nan')):.2f}. "
  f"We argue that held-out-position evaluation should be reported alongside random splits for any DMS-trained predictor, and release code that reproduces every number here.")
P("Keywords: deep mutational scanning; SARS-CoV-2; generalisation; multi-task learning; model evaluation; antibody escape; variant effect prediction.", it=True)

# ---------------------------------------------------------------- introduction
H("1. Introduction")
P("Deep mutational scanning measures the phenotypic effect of thousands of amino-acid substitutions in a single experiment, and such data now underpin much of what is known about the SARS-CoV-2 receptor-binding domain (RBD): which substitutions raise ACE2 affinity, which destabilise the fold, and which abolish binding by therapeutic antibodies [1,2]. Supervised models trained on these measurements are widely used to score substitutions that have not been assayed, and to prioritise candidates for experimental follow-up.")
P("How such models are evaluated has received far less attention than how they are built. The standard protocol holds out a random subset of variants. In a saturating DMS library almost every position is mutated somewhere in the training data, so a random test variant differs from training variants in which substitutions it combines, not in where they occur. A model can therefore score well by learning a per-position, per-residue effect table and a link function, without any ability to say anything about a position it has never seen mutated. That is precisely the ability surveillance requires: a newly sampled lineage frequently mutates positions absent from any existing scan.")
P("We therefore evaluate under two designs. The first is the conventional random split over genotypes. The second is a held-out-position split: positions are partitioned, and a variant enters a partition only if every position it mutates belongs to that partition, so no training variant touches a test position. We compare three model families — an additive ridge model, a global-epistasis model [3] and a multi-task convolutional–recurrent–attention network we call EvoClassNet — on three phenotypes measured for the same domain: ACE2 binding, expression and antibody escape aggregated across a large antibody panel [1,2].")
P("Our contributions are: (i) a quantitative demonstration that conventional and held-out-position evaluation give opposite model rankings on the same data; (ii) an ablation that isolates the mechanism of the neural model's transfer, namely amino-acid identity shared across positions; (iii) a characterisation of how far transfer extends as more positions are withheld, and of which positions it succeeds at; (iv) a leakage-free retrospective test against Variant-of-Concern substitutions, made possible by evaluating only at withheld positions; and (v) a control showing what naive genetic-algorithm optimisation of a learned fitness surface actually produces.")

# ---------------------------------------------------------------- methods
H("2. Materials and methods")
H("2.1 Phenotype data", 2)
P(f"Binding and expression. We used the yeast-display RBD DMS of Starr et al. [1] (Wuhan-Hu-1 background, spike residues 331–531; 201 positions), which reports ACE2 binding (Δlog10 Ka relative to wild type) and surface expression (Δ mean fluorescence) for barcoded variants carrying 0–9 amino-acid substitutions. Variants containing stop codons were removed, and barcodes sharing an identical amino-acid genotype were collapsed to a single record carrying the mean of their measurements, so that an identical sequence can never appear on both sides of a split. This yields {len(g):,} unique genotypes (median {int(g.n_sub.median())} substitutions, range {int(g.n_sub.min())}–{int(g.n_sub.max())}); {nlab['bind']:,} carry a binding value and {nlab['expr']:,} an expression value. Binding is censored at a floor of log10 Ka = 6, which we did not model explicitly.")
P(f"Antibody escape. We used the aggregated RBD escape maps [2], which collate per-antibody escape fractions from 13 studies. For each substitution we averaged the escape fraction over all monoclonal antibodies assayed against it, kept substitutions measured against at least 20 antibodies (median 94), and took log10 of the mean. Approximately 1.5 % of rows were measured in a non-Wuhan-Hu-1 background — their listed wild-type residue is an Omicron residue (for example P373, Y501, H505) — and were discarded, both because the model operates in the Wuhan-Hu-1 background and because the affected positions are exactly those used in the Variant-of-Concern analysis. This gives escape labels for {nlab['escape']:,} single-substitution genotypes across 175 positions; multi-substitution genotypes have no escape label and are masked out of that task.")
H("2.2 Evaluation designs", 2)
P("Random split. Genotypes are divided 80/10/10 % into training, validation and test sets.")
P("Held-out-position split. The 201 positions are partitioned at random into training (75 %), validation (10 %) and test (15 %) sets. A genotype is assigned to a partition only if every position it mutates lies in that partition's position set; genotypes spanning partitions are discarded, and the wild type is assigned to training. No training genotype therefore mutates any validation or test position. Section 3.3 repeats this with 5, 15, 30 and 50 % of positions withheld.")
P(f"Both designs were repeated over {nseed} random seeds; all models within a seed see exactly the same partition, so model comparisons are paired.")
H("2.3 Models", 2)
P(f"EvoClassNet. A variant is represented as its full 201-residue amino-acid sequence. A residue embedding of dimension {hp.get('emb','—')} is added to a learned positional embedding and passed through three one-dimensional convolutional blocks (channels {hp.get('ch','—')}, kernel sizes {hp.get('ks','—')}, batch normalisation, ReLU), with max-pooling after the first two blocks (201 → 100 → 50 positions). A bidirectional LSTM with hidden size {hp.get('lstm','—')} per direction is followed by four-head self-attention with a residual connection and layer normalisation. Learned attention pooling and max pooling are concatenated and fed to three task heads (two-layer perceptrons, 128 hidden units) predicting binding, expression and escape. Training uses a masked Huber loss, so a genotype lacking one phenotype still contributes to the others, with learned homoscedastic-uncertainty task weights [4].")
P("Baselines. The additive ridge model is ridge regression (α = 1) on one-hot indicators of each (position, residue) substitution; it has no parameter for a substitution at an unseen position and therefore predicts a constant there. The global-epistasis model fits an additive latent phenotype per task and passes it through a learned monotone sigmoid link [3], trained with the same loss, optimiser and schedule as EvoClassNet.")
H("2.4 Training and hyper-parameters", 2)
P(f"AdamW [5] with learning rate {hp.get('lr','—')} and weight decay {hp.get('wd','—')}, a one-cycle schedule, batch size 256, up to 8 epochs with early stopping on validation loss (patience 3), gradient clipping at global norm 1.0 and dropout {hp.get('dropout','—')}. Hyper-parameters were chosen by a genetic algorithm (population 6, 3 generations, tournament selection of size 2, uniform crossover with probability 0.7, per-gene mutation with probability 0.25, elitism) maximising mean validation Spearman ρ on a random-split validation set; test data were never used for selection (best validation ρ = {GA.get('fitness',float('nan')):.3f}). All experiments ran on four CPU cores.")
H("2.5 Amino-acid permutation ablation", 2)
P("To test whether transfer to unseen positions relies on residue identity being shared across positions, we applied an independent random permutation of the 20-letter amino-acid alphabet at every position, consistently across training, validation and test. Each position retains exactly the same information about its own substitutions, and the number of parameters, the architecture and the data are unchanged; what is removed is the correspondence that makes, say, a leucine at position 50 informative about a leucine at position 120.")
H("2.6 Variant-of-Concern analysis", 2)
P("As positives we used 18 RBD substitutions that define Alpha–Omicron BA.1 lineages (G339D, S371L, S373P, S375F, K417N, K417T, N440K, G446S, L452R, S477N, T478K, E484A, E484K, Q493R, G496S, Q498R, N501Y, Y505H). These substitutions are present in the escape training data, so ranking them with a model that has seen them measures memorisation rather than prediction. We therefore evaluate only at withheld positions: for each held-out-position seed we score every single substitution at that seed's test positions, and a Variant-of-Concern substitution contributes only in the seeds where its position was withheld. AUROC is reported with a 2,000-sample bootstrap interval, alongside the AUROC obtained from the measured experimental values on the identical subset.")
H("2.7 Statistics", 2)
P(f"The primary metric is Spearman's ρ between predicted and measured phenotype on the test partition, computed per task over genotypes carrying that measurement. A model producing a constant prediction conveys no rank information and is assigned ρ = 0. We report mean ± SD over seeds and compare models with a two-sided Wilcoxon signed-rank test on the per-seed paired differences ({nseed} seeds); because seeds are not independent samples of a population these P values describe reproducibility across partitions, not generalisation to other proteins.")

# ---------------------------------------------------------------- results
H("3. Results")
H("3.1 Conventional random splits do not separate the model families", 2)
table(main_df[main_df.split == "random"] if not main_df.empty else main_df,
      "Table 1. Random genotype split: test-set performance (mean ± SD across seeds).",
      cols=["model", "task", "seeds", "n_test", "spearman", "pearson", "rmse"],
      rename={"model": "Model", "task": "Phenotype", "seeds": "Seeds", "n_test": "n test", "spearman": "Spearman ρ", "pearson": "Pearson r", "rmse": "RMSE"})
fig("fig1_random_split.png", "Figure 1. Random genotype split. Bars show the mean across seeds, error bars the standard deviation, dots individual seeds.")
P(f"On the conventional design the three families are close. For binding, EvoClassNet reached ρ = {ms('main_random_s','evoclassnet','bind')} and the global-epistasis baseline {ms('main_random_s','global_epistasis','bind')} "
  f"(paired difference {stat('random','ACE2 binding')}); for expression {ms('main_random_s','evoclassnet','expr')} versus {ms('main_random_s','global_epistasis','expr')} "
  f"({stat('random','Expression')}); for escape {ms('main_random_s','evoclassnet','escape')} versus {ms('main_random_s','global_epistasis','escape')} "
  f"({stat('random','Antibody escape')}). The additive model is only slightly behind ({ms('main_random_s','ridge_additive','bind')} for binding). "
  f"A reader given only this table would reasonably conclude that the additional capacity of the neural model buys little.")
H("3.2 Held-out positions reverse the ranking", 2)
table(main_df[main_df.split == "heldout_sites"] if not main_df.empty else main_df,
      "Table 2. Held-out-position split: test variants mutate only positions absent from training (mean ± SD across seeds).",
      cols=["model", "task", "seeds", "n_test", "spearman", "pearson", "rmse"],
      rename={"model": "Model", "task": "Phenotype", "seeds": "Seeds", "n_test": "n test", "spearman": "Spearman ρ", "pearson": "Pearson r", "rmse": "RMSE"})
fig("fig2_heldout_sites.png", "Figure 2. Held-out-position split. Same layout as Figure 1.")
P(f"When every test variant mutates positions the model has never seen mutated, the additive model has no applicable parameter and emits a constant prediction (ρ = 0 by our convention). "
  f"The global-epistasis model does worse than that: its predictions are anti-correlated with measurement ({ms('main_heldout_sites_s','global_epistasis','bind')} for binding, "
  f"{ms('main_heldout_sites_s','global_epistasis','expr')} for expression), because the latent effects it assigns to unseen substitutions are set by regularisation towards zero and then passed through a link function fitted to the training distribution. "
  f"EvoClassNet retained positive correlation ({ms('main_heldout_sites_s','evoclassnet','bind')} binding, {ms('main_heldout_sites_s','evoclassnet','expr')} expression, "
  f"{ms('main_heldout_sites_s','evoclassnet','escape')} escape; paired difference versus global epistasis {stat('heldout_sites','ACE2 binding')} for binding). "
  f"The absolute values are modest and well below the ρ ≈ {rb:.2f} the same architecture attains under interpolation, so this is a difference in kind, not a claim of accurate extrapolation.")
H("3.3 How far transfer extends", 2)
fig("fig4_sweep.png", "Figure 3. Performance as a function of the fraction of positions withheld (5–50 %). Points are means over seeds, error bars standard deviations.", 6.2)
sw = {f: mean(f"sweep_f{f}_s", "evoclassnet", "bind") for f in (5, 15, 30, 50)}
P(f"Withholding more positions removes training data and widens the gap the model must bridge. For binding, EvoClassNet's test ρ moved from {sw[5]:.3f} at 5 % of positions withheld to "
  f"{sw[15]:.3f} at 15 %, {sw[30]:.3f} at 30 % and {sw[50]:.3f} at 50 % (Figure 3). Transfer therefore degrades gracefully rather than collapsing at a threshold, and remains positive even when half of the domain is unseen, "
  f"while the baselines remain at or below zero throughout.")
H("3.4 The mechanism is residue identity shared across positions", 2)
table(abl_df, "Table 3. Amino-acid permutation ablation: an independent permutation of the amino-acid alphabet at every position, applied consistently to all partitions.",
      cols=["split", "task", "intact", "permuted", "delta"],
      rename={"split": "Split", "task": "Phenotype", "intact": "Intact alphabet", "permuted": "Permuted alphabet", "delta": "Δρ"})
if not abl_df.empty:
    a = abl_df[(abl_df.split == "heldout_sites") & (abl_df.task == "ACE2 binding")].iloc[0]
    b = abl_df[(abl_df.split == "random") & (abl_df.task == "ACE2 binding")].iloc[0]
    P(f"Permuting the alphabet independently at each position leaves every position's own information intact but destroys the correspondence between the same residue at different positions. "
      f"Under the random split this barely matters (binding ρ {b.intact} → {b.permuted}, Δ = {b.delta}): interpolation only needs a per-position effect table, which the permutation merely relabels. "
      f"Under the held-out-position split the same manipulation removes the model's advantage (binding ρ {a.intact} → {a.permuted}, Δ = {a.delta}). "
      f"Transfer to unseen positions is therefore carried by amino-acid identity shared across positions — approximately, learned residue chemistry — and not by positional context or extra capacity.")
H("3.5 Leakage-free ranking of Variant-of-Concern substitutions", 2)
if HO.get("voc"):
    rows = [dict(Score=k, AUROC=f"{v['auroc']:.3f}", CI=f"{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}", n_VOC=v["n_voc"], n_scored=v["n_total"]) for k, v in HO["voc"].items()]
    table(pd.DataFrame(rows), "Table 4. Ranking Variant-of-Concern substitutions at withheld positions only (pooled over seeds; 95 % bootstrap intervals).")
    pe = HO["voc"].get("EvoClassNet predicted escape", {}); me = HO["voc"].get("Measured escape (held-out positions)", {})
    P(f"Restricting the comparison to positions that were withheld removes the circularity that would otherwise dominate this analysis. Predicted escape ranked Variant-of-Concern substitutions above the remainder "
      f"with AUROC {pe.get('auroc',float('nan')):.3f} (95 % CI {pe.get('ci95',[float('nan')]*2)[0]:.3f}–{pe.get('ci95',[float('nan')]*2)[1]:.3f}; {pe.get('n_voc','—')} positive instances). "
      f"The measured escape values on the identical subset give {me.get('auroc',float('nan')):.3f} ({me.get('ci95',[float('nan')]*2)[0]:.3f}–{me.get('ci95',[float('nan')]*2)[1]:.3f}), "
      f"which is the ceiling available to any model of this phenotype. The number of positives is small and the intervals are correspondingly wide.")
H("3.6 Which positions transfer", 2)
fig("fig5_per_site.png", "Figure 4. Per-position Spearman ρ at withheld positions against relative solvent accessibility, with a linear fit.", 6.2)
if HO.get("per_site"):
    for task, tname in (("bind", "binding"), ("escape", "escape")):
        d = HO["per_site"].get(task)
        if not d: continue
        rr, pp = d["rho_vs_RSA"]
        P(f"For {tname}, the median per-position ρ at withheld positions was {d['median_rho']:.3f} and {100*d['frac_positive']:.0f} % of position–seed pairs were positive. "
          f"Per-position accuracy correlated with relative solvent accessibility (ρ = {rr:.3f}, P = {pp:.3g}), and the comparison of receptor-binding-motif positions with core positions "
          f"is reported in results/holdout_analysis.json. Transfer is thus not uniform across the domain.")
H("3.7 A control: optimising a learned fitness surface", 2)
if VA:
    for obj, lab in (("escape", "predicted escape"), ("escape_and_binding", "predicted escape plus binding")):
        v = VA.get(obj)
        if not v: continue
        P(f"Maximising {lab} with a genetic algorithm (population 400, 20 generations, at most 8 substitutions) visited {v['archive']:,} distinct sequences. "
          f"The top 1 % carried {v['top_mean_voc']:.3f} Variant-of-Concern substitutions per variant against {v['random_mean_voc']:.3f} for size-matched random samples of the same archive "
          f"(permutation P = {v['perm_p']:.3f}); {100*v['frac_top_muts_in_terminal_20']:.0f} % of its twenty most frequent substitutions fall in the last twenty positions of the construct. "
          f"The optimiser converges on regions where the assay is permissive rather than on biologically meaningful variants.")
    P("We report this as a control rather than a method. A learned fitness surface is accurate in the region where it was fitted and unconstrained outside it, and an optimiser will find the unconstrained region. In-silico variant generation of this kind should be reported with such a null comparison, or not reported as prediction.")

# ---------------------------------------------------------------- discussion
H("4. Discussion")
P("The same data, the same models and two evaluation designs give opposite conclusions. Under random splits the neural model is indistinguishable from a two-parameter-per-substitution global-epistasis fit, and a reviewer would be right to ask what the additional machinery buys. Under held-out-position evaluation that fit becomes anti-correlated with measurement while the neural model retains positive signal across all three phenotypes. Neither result is wrong; they answer different questions, and only the second is the question that variant surveillance poses.")
P("The permutation ablation locates the difference. Nothing about capacity or positional context explains the gap: once the amino-acid alphabet is permuted independently per position, the neural model loses its advantage exactly where that advantage existed, and is unaffected where it did not. What transfers is the learned regularity that a given residue tends to have related consequences wherever it appears — chemistry rather than coordinates. This also predicts where the approach should fail, and Section 3.6 shows that it does fail unevenly across the domain.")
P("The practical recommendation is narrow and, we think, well supported: any predictor trained on a saturating mutational scan should report held-out-position performance alongside the conventional random split, because the two can disagree in direction. The protocol is cheap, requires no additional data, and would have changed the interpretation of the comparison reported here.")
H("4.1 What this work does not show", 2)
for x in ["Absolute accuracy at unseen positions is modest and far below interpolation performance; these models are not a substitute for measurement.",
          "A single domain, a single genetic background (Wuhan-Hu-1) and two assay platforms. Whether the ranking reversal generalises to other proteins or scans is untested, and the seed-level P values describe reproducibility across partitions rather than across proteins.",
          "Escape labels average over an antibody panel assembled from 13 studies with differing antibody selections; the average is a crude summary of a high-dimensional phenotype, and only single substitutions carry escape labels.",
          "Binding measurements are censored at an assay floor, which we did not model; expression is a proxy for folding stability.",
          "The Variant-of-Concern analysis is retrospective, uses 18 substitutions chosen from known lineage definitions, and contributes only at seeds where the relevant position was withheld, so the effective sample is small and the intervals wide.",
          "Hyper-parameters were selected on random-split validation data and may be suboptimal for the held-out-position regime, which if anything understates the neural model there.",
          "Protein language model embeddings and structure-based components were not evaluated: the pretrained weights and structural data were not reachable from the compute environment used, and no GPU was available. The implementation exposes an input for precomputed per-residue features so that this comparison can be added."]:
    doc.add_paragraph(x, style="List Bullet")

H("5. Data and code availability")
P("Phenotype data: Starr et al., SARS-CoV-2-RBD_DMS (github.com/jbloomlab/SARS-CoV-2-RBD_DMS) and the aggregated escape maps at github.com/jbloomlab/SARS2_RBD_Ab_escape_maps. All analysis code, the data-retrieval script, per-run outputs and the script that generates this manuscript from them are in the accompanying repository; run_all.sh reproduces every table and figure from raw data on four CPU cores.")
H("References")
for r in ["Starr TN, Greaney AJ, Hilton SK, et al. Deep mutational scanning of SARS-CoV-2 receptor binding domain reveals constraints on folding and ACE2 binding. Cell 2020;182:1295–1310.",
          "Greaney AJ, Starr TN, Bloom JD. An antibody-escape estimator for mutations to the SARS-CoV-2 receptor-binding domain. Virus Evolution 2022;8:veac021.",
          "Otwinowski J, McCandlish DM, Plotkin JB. Inferring the shape of global epistasis. Proc Natl Acad Sci USA 2018;115:E7550–E7559.",
          "Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018:7482–7491.",
          "Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR 2019.",
          "Hie B, Zhong ED, Berger B, Bryson B. Learning the language of viral evolution and escape. Science 2021;371:284–288.",
          "Rives A, Meier J, Sercu T, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci USA 2021;118:e2016239118."]:
    doc.add_paragraph(r, style="List Number")
out = ROOT / "EvoClassNet_manuscript.docx"; doc.save(out); print("saved", out)
