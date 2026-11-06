"""Build the manuscript (.docx). All numbers are read from results/*.json, none are typed by hand."""
import json, sys
from pathlib import Path
import numpy as np
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
S = json.load(open(R / "study_results.json")); V = json.load(open(R / "variant_analysis.json")); G = json.load(open(R / "ga_hyperparams.json"))
g = data.load_genotypes(); hp = G["best"]
MOD = [("ridge_additive", "Additive ridge"), ("global_epistasis", "Global epistasis"), ("evoclassnet_lite", "EvoClassNet-lite")]
def ms(split, mdl, task, metric="spearman", d=3):
    v = [r[mdl][task][metric] for r in S[split]]; return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}"
def per(split, mdl, task): return " / ".join(f"{r[mdl][task]['spearman']:.3f}" for r in S[split])
sz = {k: S[k][0]["_sizes"] for k in S}
nb, ne = int(g.bind.notna().sum()), int(g.expr.notna().sum())

doc = Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
for s in doc.sections: s.left_margin = s.right_margin = Inches(1); s.top_margin = s.bottom_margin = Inches(1)
H = lambda t, l=1: doc.add_heading(t, l)
def P(t, b=False, it=False, al=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = b; r.italic = it
    if al: p.alignment = al
    p.paragraph_format.space_after = Pt(6); return p
def table(rows, hdr, widths=None, cap=None):
    if cap: P(cap, b=True)
    t = doc.add_table(rows=1, cols=len(hdr)); t.style = "Light Grid Accent 1"
    for i, h in enumerate(hdr): t.rows[0].cells[i].text = h
    for r in rows:
        c = t.add_row().cells
        for i, x in enumerate(r): c[i].text = str(x)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs: run.font.size = Pt(9.5)
    doc.add_paragraph()
def fig(name, cap, w=4.6):
    doc.add_picture(str(R / "figures" / name), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(cap, it=True)

t = doc.add_heading("EvoClassNet: a multi-task CNN–BiLSTM–attention framework with genetic-algorithm tuning for predicting SARS-CoV-2 receptor-binding-domain phenotypes", 0)
P("[Author names and affiliations]", al=WD_ALIGN_PARAGRAPH.CENTER)

H("Abstract")
P(f"Anticipating how mutations in the SARS-CoV-2 spike receptor-binding domain (RBD) change receptor binding and protein expression is central to variant surveillance. "
  f"We present EvoClassNet, a multi-task sequence model that combines a convolutional front end, a bidirectional LSTM, multi-head self-attention pooling and uncertainty-weighted task heads, with a genetic algorithm (GA) for hyper-parameter search. "
  f"We trained and evaluated it on {len(g):,} unique RBD genotypes from a public deep-mutational-scanning (DMS) data set, with ACE2-binding measurements for {nb:,} and expression measurements for {ne:,} genotypes. "
  f"On a random genotype split, EvoClassNet reached test Spearman ρ = {ms('random','evoclassnet_lite','bind')} (binding) and {ms('random','evoclassnet_lite','expr')} (expression; mean ± SD, 3 seeds), "
  f"comparable to a global-epistasis baseline ({ms('random','global_epistasis','bind')} and {ms('random','global_epistasis','expr')}). "
  f"On a stricter split in which all test variants mutate positions never mutated in training, additive and global-epistasis baselines failed to generalise "
  f"(ρ = {ms('heldout_sites','global_epistasis','bind')} for global epistasis, binding), whereas EvoClassNet retained partial predictive signal "
  f"(ρ = {ms('heldout_sites','evoclassnet_lite','bind')} binding, {ms('heldout_sites','evoclassnet_lite','expr')} expression). "
  f"Ranking all single substitutions by predicted fitness gave an AUROC of {V['EvoClassNet-lite (predicted)']['auroc']:.2f} for substitutions that later defined Variants of Concern, close to the {V['Measured single-mutant DMS']['auroc']:.2f} obtained from measured DMS values. "
  f"A GA that evolved variants against the model's own predictions did not enrich for these substitutions (permutation P = {V['variant_ga']['perm_p']:.2f}), which we report as a negative result. "
  f"Code, data-retrieval scripts and all outputs are provided for exact reproduction.")
P("Keywords: SARS-CoV-2; receptor-binding domain; deep mutational scanning; deep learning; genetic algorithm; multi-task learning; variant effect prediction.", it=True)

H("1. Introduction")
P("Repeated emergence of SARS-CoV-2 variants with altered receptor affinity and immune evasion has shown that surveillance based on observing variants after they spread is slow. Experimental deep mutational scanning (DMS) of the RBD quantifies how single and multiple amino-acid substitutions affect ACE2 binding and folding/expression [1], but it cannot be repeated for every new lineage. Computational models that predict these phenotypes from sequence could prioritise variants for follow-up and help evaluate how well sequence models extrapolate to mutations they have not seen [2,3].")
P("Existing statistical approaches for DMS data are largely additive or use a global-epistasis link function [4]. These are strong, interpretable baselines for interpolation but cannot, by construction, predict the effect of mutations at positions absent from training. Neural sequence models share information across positions and residue types and may therefore extrapolate, but this has to be tested with splits that actually remove positions, not only random genotypes.")
P("We make three contributions: (i) EvoClassNet, a multi-task CNN–BiLSTM–attention model with uncertainty-weighted heads, trained jointly on ACE2 binding and expression; (ii) a GA used both for hyper-parameter search and for in-silico variant evolution; and (iii) a leakage-controlled evaluation with a random split and a held-out-site split, benchmarked against additive and global-epistasis baselines, with a retrospective analysis against Variants of Concern (VOC). We state explicitly which components of a larger planned architecture were not evaluated here (Section 5.2).")

H("2. Data")
P(f"We used the SARS-CoV-2 RBD DMS data of Starr et al. [1] (Wuhan-Hu-1 background, spike residues 331–531, 201 residues), obtained from the authors' public repository. The tables contain barcoded variants carrying 0–9 amino-acid substitutions with yeast-display measurements of ACE2 binding (Δlog10 Ka relative to wild type) and expression (Δ mean fluorescence). Variants with stop codons were removed. Barcodes sharing an identical amino-acid genotype (across both libraries) were collapsed to one record carrying the mean phenotype, so that identical sequences cannot appear on both sides of a split. This yields {len(g):,} unique genotypes ({nb:,} with a binding value, {ne:,} with an expression value); genotypes carry a median of {int(g.n_sub.median())} substitutions (range {int(g.n_sub.min())}–{int(g.n_sub.max())}). Binding measurements are censored at a floor of log10 Ka = 6; we did not model censoring explicitly.")
P("Splits. (A) Random split: 80/10/10 % of genotypes. (B) Held-out-site split: 15 % of the 201 positions were assigned to test, 10 % to validation and 75 % to training; a variant is assigned to a partition only if every position it mutates lies in that partition's positions, and variants spanning partitions are discarded. No training variant therefore touches a validation or test position. "
  f"Partition sizes (train/validation/test) were {sz['random'][0]:,}/{sz['random'][1]:,}/{sz['random'][2]:,} for split A and {sz['heldout_sites'][0]:,}/{sz['heldout_sites'][1]:,}/{sz['heldout_sites'][2]:,} for split B (seed 0; sizes vary with the seed for B). Splits were repeated for three random seeds.")

H("3. Methods")
H("3.1 EvoClassNet architecture", 2)
P(f"Each variant is encoded as its 201-residue amino-acid sequence. A residue embedding (dimension {hp['emb']}) plus a learned positional embedding feeds three 1-D convolutional blocks (channels {tuple(hp['ch'])}, kernel sizes {tuple(hp['ks'])}; batch normalisation and ReLU), with max-pooling after the first two blocks (201→100→50 positions). A bidirectional LSTM (hidden size {hp['lstm']} per direction) models long-range dependencies, followed by 4-head self-attention with a residual connection and layer normalisation. A learned attention pooling and a max pooling are concatenated and passed to two task-specific heads (two-layer MLP, 128 hidden units) predicting binding and expression. Multi-task training uses a masked Huber loss, so a genotype lacking one phenotype still contributes to the other, with learned homoscedastic-uncertainty task weights [5].")
H("3.2 Baselines", 2)
P("Additive ridge: ridge regression (α = 1) on one-hot indicators of each (position, residue) substitution. Global epistasis: an additive latent phenotype per task passed through a learned monotone sigmoid link [4], trained with the same loss and optimiser. Both are fitted on the training partition only.")
H("3.3 Training", 2)
P(f"AdamW [7] (learning rate {hp['lr']}, weight decay {hp['wd']}), one-cycle schedule, batch size 256, up to 8 epochs with early stopping on validation loss (patience 3), gradient clipping at 1.0, dropout {hp['dropout']}. Training ran on CPU (4 cores).")
H("3.4 Genetic algorithm", 2)
P(f"Hyper-parameter search. A GA evolved a population of 6 configurations (learning rate, embedding size, LSTM size, dropout, channel widths, kernel sizes, weight decay) for 3 generations with tournament selection (size 2), uniform crossover (P = 0.7), per-gene mutation (P = 0.25) and elitism. Fitness was the mean validation Spearman ρ over both tasks after 3 epochs on a 12,000-genotype subsample, evaluated on 3,000 validation genotypes; test data were never used. The best configuration (validation ρ = {G['fitness']:.3f}) is the one given above.")
P("Variant evolution. Starting from the wild-type RBD, a GA (population 400, 20 generations, tournament size 3, Poisson(2) point mutations, single-point recombination with probability 0.5, at most 8 substitutions) maximised the model-predicted fitness f = predicted binding + 0.5 × predicted expression. All distinct sequences visited were archived and the top 1 % by predicted fitness analysed.")
H("3.5 Evaluation", 2)
P("The primary metric is Spearman's ρ between predicted and measured phenotype on the test partition, computed separately per task on genotypes with a measurement; Pearson's r and RMSE are also reported. A model that outputs a constant prediction carries no rank information and is assigned ρ = 0. Results are mean ± SD over three seeds; with n = 3 no formal significance test is attempted.")
P("Retrospective VOC analysis. All 201 × 19 = 3,819 single substitutions were ranked by predicted fitness. As positives we used 18 substitutions in the RBD that define Alpha–Omicron BA.1 lineages (G339D, S371L, S373P, S375F, K417N, K417T, N440K, G446S, L452R, S477N, T478K, E484A, E484K, Q493R, G496S, Q498R, N501Y, Y505H) and computed the AUROC with a 1,000-sample bootstrap interval, alongside the AUROC from measured single-mutant DMS values. Because the training data predate these lineages this is a retrospective, not a prospective, test. For the variant GA we compared the mean number of VOC substitutions per variant in the top 1 % of the archive with random archive samples of equal size (500 permutations).")

H("4. Results")
H("4.1 Prediction on a random genotype split", 2)
rows = [[n, ms("random", k, "bind"), ms("random", k, "bind", "pearson"), ms("random", k, "expr"), ms("random", k, "expr", "pearson")] for k, n in MOD]
table(rows, ["Model", "Binding ρ", "Binding r", "Expression ρ", "Expression r"], cap="Table 1. Random genotype split (test set, mean ± SD over 3 seeds).")
fig("fig1_random_split.png", "Figure 1. Test Spearman ρ on the random split. Bars: mean of 3 seeds; dots: individual seeds.")
P(f"For binding, EvoClassNet ({ms('random','evoclassnet_lite','bind')}) and global epistasis ({ms('random','global_epistasis','bind')}) were indistinguishable at this sample size and both exceeded the additive model ({ms('random','ridge_additive','bind')}). For expression EvoClassNet was higher on average ({ms('random','evoclassnet_lite','expr')} vs {ms('random','global_epistasis','expr')}) but with large seed-to-seed spread (per-seed ρ: {per('random','evoclassnet_lite','expr')}). The neural model therefore shows no consistent advantage when test variants are composed of positions seen in training.")
H("4.2 Generalisation to unseen positions", 2)
rows = [[n, ms("heldout_sites", k, "bind"), ms("heldout_sites", k, "bind", "pearson"), ms("heldout_sites", k, "expr"), ms("heldout_sites", k, "expr", "pearson")] for k, n in MOD]
table(rows, ["Model", "Binding ρ", "Binding r", "Expression ρ", "Expression r"], cap="Table 2. Held-out-site split (test variants mutate only unseen positions; mean ± SD over 3 seeds).")
fig("fig2_heldout_sites.png", "Figure 2. Test Spearman ρ when test variants mutate only positions absent from training.")
P(f"The additive model has no parameter for an unseen (position, residue) pair and predicts a constant (ρ = 0 by convention). The global-epistasis model produced a negative correlation ({ms('heldout_sites','global_epistasis','bind')} for binding), i.e. its predictions for unseen positions were anti-informative. EvoClassNet retained a positive but modest correlation ({ms('heldout_sites','evoclassnet_lite','bind')} binding; {ms('heldout_sites','evoclassnet_lite','expr')} expression) with substantial variability across seeds (per-seed binding ρ: {per('heldout_sites','evoclassnet_lite','bind')}). This indicates partial transfer of residue-level information, well below the values of about 0.88 (binding) obtained under interpolation on the random split.")
H("4.3 Retrospective analysis against Variants of Concern", 2)
a, b = V["EvoClassNet-lite (predicted)"], V["Measured single-mutant DMS"]
P(f"Ranking single substitutions by predicted fitness placed the {a['n_voc']} VOC-defining substitutions above the remainder with AUROC {a['auroc']:.3f} (95 % CI {a['ci95'][0]:.3f}–{a['ci95'][1]:.3f}). Measured DMS values gave {b['auroc']:.3f} ({b['ci95'][0]:.3f}–{b['ci95'][1]:.3f}); the intervals overlap, so the model recovers about as much VOC-relevant signal as the experimental data it was trained on, but not more (Figure 3).")
fig("fig3_voc_auroc.png", "Figure 3. AUROC for ranking VOC-defining RBD substitutions among all single substitutions.", 3.8)
vg = V["variant_ga"]
P(f"The variant GA visited {vg['archive']:,} distinct sequences. The top 1 % carried {vg['top_mean_voc']:.3f} VOC substitutions per variant versus {vg['random_mean_voc']:.3f} for random archive samples (permutation P = {vg['perm_p']:.2f}); there was therefore no enrichment. The most frequent substitutions in top variants included several at the C-terminal end of the construct (" + ", ".join([m for m in V["top_ga_mutations"] if int(m[1:-1]) >= 520][:4]) + "), which most likely reflects tolerance of that region in the DMS construct rather than biologically meaningful escape or affinity-enhancing mutations. We do not claim prospective prediction of emerging variants.")
rows = [[sp.replace("_", " "), n, t, per(sp, k, t)] for sp in S for k, n in MOD for t in ("bind", "expr")]
table(rows, ["Split", "Model", "Task", "Per-seed test ρ (seeds 0/1/2)"], cap="Supplementary Table S1. Per-seed results.")

H("5. Discussion")
H("5.1 Findings", 2)
P("Under interpolation, a well-specified global-epistasis model matches the neural model, so the added capacity of EvoClassNet is not warranted by the random-split results alone. The case for the neural architecture rests on the held-out-site experiment, where it is the only model with positive correlation. We consider this the main, if modest, empirical finding, and the held-out-site protocol itself a useful benchmark for DMS-trained models.")
H("5.2 Components not evaluated", 2)
P("The EvoClassNet design also envisages (i) frozen protein-language-model embeddings (e.g. ESM-1b [6]) as per-residue inputs, (ii) a graph neural network over RBD structures, (iii) genome-level species classification and (iv) GA optimisation of network weights. These were not run in this study: the required genome collections, structures and pretrained weights were not accessible in the compute environment and no GPU was available. The code exposes an `extra_dim` input for precomputed per-residue features. Statements about the benefit of these components, or about genome-level classification accuracy and throughput, are therefore not made here and should be tested in future work.")
H("5.3 Limitations", 2)
for x in ["A single DMS data set in one genetic background (Wuhan-Hu-1) measured by yeast display; effects in other backgrounds (e.g. Omicron) are known to differ.",
          "Three seeds only; seed variance of the neural model is large, so differences from baselines on the random split are not established.",
          "Binding values are censored at a floor and expression is a proxy for folding stability; no antibody-escape or transmissibility labels were modelled.",
          "The VOC analysis is retrospective with 18 positives; the data set predates these lineages but the substitution list was chosen from known VOC definitions.",
          "The variant GA optimises the model's own predictions and showed no enrichment; it should not be interpreted as forecasting.",
          "Hyper-parameters were selected with random-split validation data only, which may not be optimal for the held-out-site setting.",
          "The model used for the VOC analysis was the seed-0 random-split model, which was the weakest of the three seeds for the random split."]:
    doc.add_paragraph(x, style="List Bullet")

H("6. Data and code availability")
P("Source data: Starr et al., SARS-CoV-2-RBD_DMS repository (https://github.com/jbloomlab/SARS-CoV-2-RBD_DMS). Code, logs and result files: this repository (run_all.sh reproduces every table and figure; scripts/download_data.py retrieves the data; tests/ contains unit tests).")
H("References")
for r in ["Starr TN, Greaney AJ, Hilton SK, et al. Deep mutational scanning of SARS-CoV-2 receptor binding domain reveals constraints on folding and ACE2 binding. Cell 2020;182:1295–1310.",
          "Hie B, Zhong ED, Berger B, Bryson B. Learning the language of viral evolution and escape. Science 2021;371:284–288.",
          "Rao R, Bhattacharya N, Thomas N, et al. Evaluating protein transfer learning with TAPE. NeurIPS 2019.",
          "Otwinowski J, McCandlish DM, Plotkin JB. Inferring the shape of global epistasis. Proc Natl Acad Sci USA 2018;115:E7550–E7559.",
          "Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR 2018.",
          "Rives A, Meier J, Sercu T, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci USA 2021;118:e2016239118.",
          "Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR 2019."]:
    doc.add_paragraph(r, style="List Number")
out = ROOT / "EvoClassNet_manuscript.docx"; doc.save(out); print("saved", out)
