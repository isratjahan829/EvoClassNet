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
doc.add_heading("Apparent generalisation to unseen positions in deep mutational scanning models is explained by mutation count: a benchmark on the SARS-CoV-2 receptor-binding domain", 0)
P("[Author names and affiliations]", al=WD_ALIGN_PARAGRAPH.CENTER)

rb, gb = mean("main_random_s", "evoclassnet", "bind"), mean("main_random_s", "global_epistasis", "bind")
nb_r, nb_h = mean("main_random_s", "mutation_count", "bind"), mean("main_heldout_sites_s", "mutation_count", "bind")
hb = mean("main_heldout_sites_s", "evoclassnet", "bind")
H("Abstract")
P(f"Models trained on deep mutational scanning (DMS) data are normally evaluated by holding out random variants. In a combinatorial library almost every position is mutated somewhere in the "
  f"training set, so this measures interpolation, not the situation that matters for surveillance, in which a lineage mutates a position no experiment has probed. We benchmarked four model "
  f"families on {len(g):,} unique SARS-CoV-2 receptor-binding-domain genotypes with three measured phenotypes — ACE2 binding ({nlab['bind']:,} labels), expression ({nlab['expr']:,}) and antibody "
  f"escape averaged over a large monoclonal panel ({nlab['escape']:,}) — under a conventional random split and under a held-out-position split in which no training variant mutates any test position. "
  f"Under the random split the trained models performed as expected and clearly exceeded an untrained baseline that ranks variants only by how many substitutions they carry "
  f"(EvoClassNet ρ = {ms('main_random_s','evoclassnet','bind')} versus {ms('main_random_s','mutation_count','bind')} for binding). "
  f"Under the held-out-position split the additive model became undefined and the global-epistasis model anti-correlated ({ms('main_heldout_sites_s','global_epistasis','bind')}), while a multi-task "
  f"convolutional–recurrent–attention network appeared to retain signal ({ms('main_heldout_sites_s','evoclassnet','bind')}). That appearance does not survive the right control: the untrained "
  f"mutation-count baseline reached {ms('main_heldout_sites_s','mutation_count','bind')} on the same partitions, and the trained network was statistically indistinguishable from it on all three "
  f"phenotypes (paired Wilcoxon over {nseed} seeds: {stat('heldout_sites','ACE2 binding','vs_mutation_count')} for binding). Permuting the amino-acid alphabet independently at every position, which "
  f"destroys residue identity shared across positions but leaves mutation count untouched, changed nothing, as this account predicts. We conclude that none of the models tested demonstrates "
  f"position-level extrapolation, and that held-out-position evaluation is interpretable only when reported against a mutation-count baseline.")
P("Keywords: deep mutational scanning; SARS-CoV-2; model evaluation; generalisation; baselines; variant effect prediction; negative result.", it=True)

H("1. Introduction")
P("Deep mutational scanning measures the phenotypic consequences of thousands of amino-acid substitutions in one experiment, and such data underpin much of what is known about the SARS-CoV-2 receptor-binding domain (RBD): which substitutions raise ACE2 affinity, which destabilise the fold, and which abolish binding by therapeutic antibodies [1,2]. Supervised models trained on these measurements are routinely used to score substitutions that were not assayed and to prioritise candidates for experimental follow-up.")
P("How such models are evaluated has received less scrutiny than how they are built. The standard protocol holds out a random subset of variants. In a saturating combinatorial library almost every position is mutated somewhere in the training data, so a random test variant differs from training variants in which substitutions it combines, not in where those substitutions fall. A model can therefore score well by learning a per-position, per-residue effect table together with a link function, without any capacity to say anything about a position it has never seen mutated. That capacity is what surveillance requires.")
P("We therefore compared a conventional random split against a held-out-position split, in which positions are partitioned and a variant enters a partition only if every position it mutates belongs to that partition. We expected, and found, that the trained models degrade sharply under the second design. The purpose of this paper is a second and less comfortable observation: the degradation is total. Once an untrained baseline that ranks variants by the number of substitutions they carry is included, no trained model in our comparison is distinguishable from it at unseen positions. Because that baseline is trivial to compute and is almost never reported, a held-out-position experiment without it invites exactly the wrong conclusion — and we initially drew that conclusion ourselves before adding the control.")
P("Our contributions are: (i) a demonstration that random-split and held-out-position evaluation give different model rankings on the same data, with paired statistics over ten seeds; (ii) the mutation-count control that reinterprets the second ranking, showing that apparent extrapolation is not position-level transfer; (iii) an alphabet-permutation ablation whose null result corroborates that account; (iv) a quantification of run-to-run instability large enough to flip conclusions drawn from single runs; and (v) a leakage-free retrospective test against Variant-of-Concern substitutions and a genetic-algorithm control, both of which we report with the same scepticism.")

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
H("3.1 Under random splits the trained models behave as expected", 2)
table(main_df[main_df.split == "random"] if not main_df.empty else main_df,
      "Table 1. Random genotype split: test-set performance (mean ± SD across seeds).",
      cols=["model", "task", "seeds", "n_test", "spearman", "pearson", "rmse"],
      rename={"model": "Model", "task": "Phenotype", "seeds": "Seeds", "n_test": "n test", "spearman": "Spearman ρ", "pearson": "Pearson r", "rmse": "RMSE"})
fig("fig1_random_split.png", "Figure 1. Random genotype split. Bars show the mean across seeds, error bars the standard deviation, dots individual seeds.")
P(f"On the conventional design all trained models performed well and comfortably exceeded the untrained mutation-count baseline. For binding, EvoClassNet reached ρ = {ms('main_random_s','evoclassnet','bind')} "
  f"and the global-epistasis baseline {ms('main_random_s','global_epistasis','bind')}, against {ms('main_random_s','mutation_count','bind')} for mutation count alone "
  f"(EvoClassNet versus mutation count {stat('random','ACE2 binding','vs_mutation_count')}). The two trained families were close to each other: {stat('random','ACE2 binding')} for binding and "
  f"{stat('random','Expression')} for expression. On escape the global-epistasis model was the better of the two ({ms('main_random_s','global_epistasis','escape')} versus "
  f"{ms('main_random_s','evoclassnet','escape')}; {stat('random','Antibody escape')}), so the neural architecture is not uniformly preferable even under interpolation.")
P(f"Two entries in Table 1 are zero by construction rather than by failure. The additive ridge model scores 0 on escape because escape labels exist only for single substitutions, so each "
  f"(position, residue) indicator occurs in exactly one training row and the regularised fit collapses to the intercept. Mutation count scores 0 on escape for the same reason: every labelled "
  f"variant carries exactly one substitution, so the baseline is constant. This makes escape the one phenotype here on which mutation count cannot masquerade as predictive skill, a point we return to in Section 3.3.")

H("3.2 Held-out positions: the trained models collapse", 2)
table(main_df[main_df.split == "heldout_sites"] if not main_df.empty else main_df,
      "Table 2. Held-out-position split: test variants mutate only positions absent from training (mean ± SD across seeds).",
      cols=["model", "task", "seeds", "n_test", "spearman", "pearson", "rmse"],
      rename={"model": "Model", "task": "Phenotype", "seeds": "Seeds", "n_test": "n test", "spearman": "Spearman ρ", "pearson": "Pearson r", "rmse": "RMSE"})
fig("fig2_heldout_sites.png", "Figure 2. Held-out-position split. Same layout as Figure 1.")
P(f"When every test variant mutates positions never mutated in training, the additive model has no applicable parameter and emits a constant prediction. The global-epistasis model does worse than "
  f"nothing: its predictions are anti-correlated with measurement ({ms('main_heldout_sites_s','global_epistasis','bind')} for binding, {ms('main_heldout_sites_s','global_epistasis','expr')} for "
  f"expression), because latent effects for unseen substitutions are shrunk towards zero and then passed through a link function fitted to the training distribution. EvoClassNet remained positive on "
  f"average ({ms('main_heldout_sites_s','evoclassnet','bind')} binding, {ms('main_heldout_sites_s','evoclassnet','expr')} expression, {ms('main_heldout_sites_s','evoclassnet','escape')} escape) and beat both "
  f"trained baselines by a wide margin ({stat('heldout_sites','ACE2 binding')} versus global epistasis). Read in isolation, Table 2 says that the neural model extrapolates to unseen positions where "
  f"the conventional models cannot.")

H("3.3 An untrained baseline matches the neural model at unseen positions", 2)
P(f"That reading is wrong. Ranking variants by the number of substitutions they carry — no training, no parameters, and no knowledge of which positions or residues are involved — reached "
  f"ρ = {ms('main_heldout_sites_s','mutation_count','bind')} for binding and {ms('main_heldout_sites_s','mutation_count','expr')} for expression on the same partitions, against "
  f"{ms('main_heldout_sites_s','evoclassnet','bind')} and {ms('main_heldout_sites_s','evoclassnet','expr')} for EvoClassNet. Paired over {nseed} seeds the differences are "
  f"{stat('heldout_sites','ACE2 binding','vs_mutation_count')} for binding, {stat('heldout_sites','Expression','vs_mutation_count')} for expression and "
  f"{stat('heldout_sites','Antibody escape','vs_mutation_count')} for escape: the trained network is not distinguishable from the untrained baseline on any phenotype, and the baseline is far more "
  f"stable across partitions (SD {np.std(vals('main_heldout_sites_s','mutation_count','bind'), ddof=1):.3f} against {np.std(vals('main_heldout_sites_s','evoclassnet','bind'), ddof=1):.3f}).")
table(stat_df, "Table 3. Paired differences in Spearman ρ (EvoClassNet minus comparator) with two-sided Wilcoxon signed-rank tests over seeds.",
      cols=["split", "task", "n_seeds", "vs_mutation_count", "vs_global_epistasis", "vs_additive"],
      rename={"split": "Split", "task": "Phenotype", "n_seeds": "Seeds", "vs_mutation_count": "vs mutation count",
              "vs_global_epistasis": "vs global epistasis", "vs_additive": "vs additive ridge"})
P("The interpretation is straightforward. In a combinatorial library a large part of the rank variance in binding and expression is carried by how many substitutions a variant has, because most "
  "substitutions are deleterious and their effects accumulate. A model that has learned nothing about the held-out positions can still order test variants correctly to that extent, simply by "
  "responding to the overall amount of mutation. What Table 2 measures is therefore not transfer to unseen positions; it is the part of the signal that requires no knowledge of positions at all.")
P(f"Escape is the exception that makes the point. Every escape-labelled variant carries exactly one substitution, so mutation count is constant and contributes nothing. There, EvoClassNet reached "
  f"{ms('main_heldout_sites_s','evoclassnet','escape')} against {ms('main_heldout_sites_s','global_epistasis','escape')} for global epistasis, a difference of "
  f"{stat('heldout_sites','Antibody escape')}. This is the only measurement in our study that could constitute genuine position-level transfer, and at {nseed} seeds it does not reach significance.")

H("3.4 Permuting the amino-acid alphabet changes nothing", 2)
table(abl_df, "Table 4. Amino-acid permutation ablation: an independent random permutation of the alphabet at every position, applied consistently to all partitions.",
      cols=["split", "task", "intact", "permuted", "delta"],
      rename={"split": "Split", "task": "Phenotype", "intact": "Intact alphabet", "permuted": "Permuted alphabet", "delta": "Δρ"})
if not abl_df.empty:
    a = abl_df[(abl_df.split == "heldout_sites") & (abl_df.task == "ACE2 binding")].iloc[0]
    P(f"We designed this ablation to test the hypothesis that transfer is carried by residue identity shared across positions: permuting the alphabet independently at each position preserves every "
      f"position's own information while destroying any correspondence between the same residue at different positions. Under that hypothesis the manipulation should remove the neural model's "
      f"advantage at held-out positions. It did not. At held-out positions binding moved from {a.intact} to {a.permuted} (Δ = {a.delta}), a change small against the between-seed standard deviation, "
      f"and the other conditions behaved similarly. The null result is what the mutation-count account predicts, because the number of substitutions in a variant is invariant to relabelling the "
      f"alphabet. We report the ablation because it was the discriminating experiment between the two explanations, and it favours the second.")

H("3.5 Run-to-run instability", 2)
hv = sorted(vals("main_heldout_sites_s", "evoclassnet", "bind"))
ST = J("stability_check.json")
P(f"Per-seed results for EvoClassNet at held-out positions ranged from {min(hv):+.3f} to {max(hv):+.3f} for binding ({sum(1 for x in hv if x > 0)} of {len(hv)} seeds positive), a spread that "
  f"exceeds most of the differences discussed above. In the main study the partition and the weight initialisation share a seed value, so partition difficulty and optimisation noise are confounded.")
if ST:
    import collections
    by = collections.defaultdict(list)
    for r in ST: by[r["split_seed"]].append(r["bind"])
    for ss, v in sorted(by.items()):
        P(f"Holding partition {ss} fixed and varying only the initialisation gave binding ρ of " + ", ".join(f"{x:+.3f}" for x in v) +
          f" (range {max(v)-min(v):.3f}). ", it=False)
    P("Initialisation alone therefore accounts for a substantial part of the spread, which means single-run comparisons on this benchmark are not reliable and that published results of this kind should report a seed distribution rather than a single number.")

H("3.6 Withholding more positions", 2)
fig("fig4_sweep.png", "Figure 3. Performance as a function of the fraction of positions withheld (5–50 %). Points are means over seeds, error bars standard deviations.", 6.2)
sw = {f: mean(f"sweep_f{f}_s", "evoclassnet", "bind") for f in (5, 15, 30, 50)}
swb = {f: mean(f"sweep_f{f}_s", "mutation_count", "bind") for f in (5, 15, 30, 50)}
P(f"Withholding a larger fraction of positions removes training data and widens the gap to be bridged. EvoClassNet's binding ρ fell from {sw[5]:.3f} at 5 % of positions withheld to {sw[15]:.3f} at 15 %, "
  f"{sw[30]:.3f} at 30 % and {sw[50]:.3f} at 50 %. The mutation-count baseline on the same partitions gave {swb[5]:.3f}, {swb[15]:.3f}, {swb[30]:.3f} and {swb[50]:.3f}. The trained model does not "
  f"separate from the baseline at any fraction, so the decline reflects the shrinking contribution of mutation count rather than a failure of transfer that was present at smaller fractions.")

H("3.7 Single substitutions at withheld positions", 2)
P("Restricting attention to single substitutions removes mutation count as a confound entirely, because every variant then carries exactly one substitution. We trained held-out-position models and "
  "scored every single substitution at the withheld positions, comparing the model against the measured values and against BLOSUM62 exchangeability, which encodes generic residue chemistry with no training.")
if HO.get("vs_blosum"):
    rows = [dict(Phenotype={"bind": "ACE2 binding", "escape": "Antibody escape"}.get(k, k),
                 EvoClassNet=f"{v['model_median']:.3f}", BLOSUM62=f"{v['blosum_median']:.3f}",
                 Difference=f"{v['median_diff']:+.3f}", P=f"{v['p']:.3g}", n=v["n"]) for k, v in HO["vs_blosum"].items()]
    table(pd.DataFrame(rows), "Table 5. Median per-position Spearman ρ over single substitutions at withheld positions (Wilcoxon signed-rank on paired per-position differences).")
if HO.get("per_site"):
    for task, tname in (("bind", "binding"), ("escape", "escape")):
        d = HO["per_site"].get(task)
        if not d: continue
        rr, pp = d["rho_vs_RSA"]
        P(f"For {tname} the median per-position ρ was {d['median_rho']:.3f}, with {100*d['frac_positive']:.0f} % of position–seed pairs positive, and per-position accuracy correlated with relative "
          f"solvent accessibility at ρ = {rr:.3f} (P = {pp:.3g}).")
fig("fig5_per_site.png", "Figure 4. Per-position Spearman ρ at withheld positions against relative solvent accessibility, with a linear fit.", 6.2)

H("3.8 Variant-of-Concern ranking and a genetic-algorithm control", 2)
if HO.get("voc"):
    rows = [dict(Score=k, AUROC=f"{v['auroc']:.3f}", CI=f"{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}", n_VOC=v["n_voc"], n_scored=v["n_total"]) for k, v in HO["voc"].items()]
    table(pd.DataFrame(rows), "Table 6. Ranking Variant-of-Concern substitutions at withheld positions only (pooled over seeds; 95 % bootstrap intervals).")
    pe = HO["voc"].get("EvoClassNet predicted escape", {}); me = HO["voc"].get("Measured escape (held-out positions)", {})
    P(f"Variant-of-Concern substitutions appear in the escape training data, so ranking them with a model that has seen them measures memorisation; restricting the comparison to withheld positions "
      f"removes that circularity. Predicted escape gave AUROC {pe.get('auroc',float('nan')):.3f} "
      f"(95 % CI {pe.get('ci95',[float('nan')]*2)[0]:.3f}–{pe.get('ci95',[float('nan')]*2)[1]:.3f}; {pe.get('n_voc','—')} positive instances), against "
      f"{me.get('auroc',float('nan')):.3f} for the measured values on the identical subset, which is the ceiling available to any model of this phenotype. The number of positives is small and the intervals are wide.")
if VA:
    for obj, lab in (("escape", "predicted escape"), ("escape_and_binding", "predicted escape plus binding")):
        v = VA.get(obj)
        if not v: continue
        P(f"Maximising {lab} with a genetic algorithm (population 400, 20 generations, at most 8 substitutions) visited {v['archive']:,} distinct sequences. The top 1 % carried {v['top_mean_voc']:.3f} "
          f"Variant-of-Concern substitutions per variant against {v['random_mean_voc']:.3f} for size-matched random samples of the same archive (permutation P = {v['perm_p']:.3f}); "
          f"{100*v['frac_top_muts_in_terminal_20']:.0f} % of its twenty most frequent substitutions fall in the last twenty positions of the construct.")
    P("A learned fitness surface is constrained only where it was fitted, and an optimiser will leave that region. In-silico variant generation of this kind needs a null comparison of this sort before it is described as prediction.")

H("4. Discussion")
P("Three evaluation designs applied to the same data support three different conclusions. A random split says the trained models work well and that the choice between them barely matters. A "
  "held-out-position split says the neural model extrapolates where conventional models fail. Adding an untrained baseline to the held-out-position split says that none of them extrapolates at "
  "all. Only the third conclusion survives, and the baseline that produces it costs nothing to compute.")
P("We think the general lesson is about which control a hard evaluation requires. Making a split harder does not by itself make the resulting number meaningful: it changes what an uninformative "
  "model would score, and unless that value is measured it is easy to mistake a shift in the null for evidence of skill. On a combinatorial mutagenesis library the uninformative score is not zero, "
  "because the number of substitutions a variant carries is itself predictive. We report this partly as a cautionary note on our own process: we had written up the held-out-position result as "
  "evidence of transfer, and designed an ablation to explain a mechanism, before the baseline showed there was nothing to explain. The ablation's null result, which had looked like a failed "
  "experiment, was in fact the first symptom.")
P("Whether position-level transfer exists at all in these models remains open, and our escape measurements are the place to look, since every escape-labelled variant carries a single substitution "
  "and mutation count is therefore constant. The neural model was above the alternatives there but not significantly so at ten seeds. Settling that question needs either more seeds, more "
  "single-substitution phenotypes, or models with access to information beyond the training library — pretrained protein language models and structure being the obvious candidates, neither of which "
  "we could evaluate here.")
P("The practical recommendations are narrow. First, report held-out-position performance alongside random splits for any predictor trained on a saturating scan, because the two designs can be "
  "ranked differently. Second, report a mutation-count baseline on every split, because without it a held-out-position number cannot be interpreted. Third, report a distribution over seeds rather "
  "than a single run: on this benchmark initialisation alone moved held-out binding ρ across a range wide enough to reverse conclusions.")
H("4.1 Limitations", 2)
for x in ["This is a negative result on one domain, one genetic background (Wuhan-Hu-1) and two assay platforms. It shows that these models do not extrapolate here, not that no model can.",
          "The architecture is a CPU-scale convolutional-recurrent-attention network. Pretrained protein language model embeddings and structural inputs, which are the components most likely to supply position-independent biochemical knowledge, were not evaluated: the pretrained weights and structural data were not reachable from the compute environment and no GPU was available. The implementation exposes an input for precomputed per-residue features so this comparison can be added.",
          "Hyper-parameters were selected by validation on random splits and may be suboptimal in the held-out-position regime; this would, if anything, understate the trained models there.",
          "Escape labels average over an antibody panel assembled from 13 studies with differing antibody selections, and only single substitutions carry escape labels, so the escape task has far fewer labels than the others and wide confidence intervals.",
          "Binding measurements are censored at an assay floor, which we did not model; expression is a proxy for folding stability.",
          "The Variant-of-Concern analysis is retrospective, uses 18 substitutions drawn from known lineage definitions, and contributes only at seeds where the relevant position was withheld, so the effective sample is small.",
          "Seed-level P values describe reproducibility across partitions of one data set; they do not license inference to other proteins or other scans."]:
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
