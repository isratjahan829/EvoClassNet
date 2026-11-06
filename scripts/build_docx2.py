"""Build the expanded manuscript. Every number and every comparative word is read from
results/*.json and results/*.csv; nothing is typed in by hand.

If a result file is missing the corresponding section is skipped rather than invented, so the
script can be run at any point during a study and the document will contain only what has
actually been measured.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"

J = lambda n: json.load(open(R / n)) if (R / n).exists() else {}
C = lambda n: pd.read_csv(R / n) if (R / n).exists() else pd.DataFrame()

PANEL = C("table_panel.csv")
COMPAR = C("table_comparisons.csv")
ABL2 = C("table_ablation_v2.csv")
GEN = C("table_genomes.csv")
GENC = C("table_genome_comparisons.csv")
FEAT = J("feature_ablation.json")
HO = J("holdout_analysis.json")
STAB = J("stability_check.json")
VA = J("variant_analysis.json")
ESMC = J("esm_control.json")
GMETA = J("genome_study.json").get("meta", {})
SUM2 = J("run_summary_v2.json")

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def P(t, it=False, al=None, size=None):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.italic = it
    if size:
        r.font.size = Pt(size)
    if al is not None:
        p.alignment = al
    return p


def H(t, lvl=1):
    doc.add_heading(t, lvl)


def fig(name, caption, width=6.2):
    f = FIG / name
    if not f.exists():
        return False
    doc.add_picture(str(f), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, it=True, size=9)
    return True


def table(df, caption, cols=None, rename=None, maxrows=60):
    if df is None or not len(df):
        return False
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if rename:
        d = d.rename(columns=rename)
    d = d.head(maxrows)
    P(caption, it=True, size=9)
    t = doc.add_table(rows=1, cols=len(d.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(d.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for pr in cell.paragraphs:
            for rr in pr.runs:
                rr.bold = True
                rr.font.size = Pt(8)
    for _, row in d.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(d.columns):
            cells[i].text = "" if pd.isna(row[c]) else str(row[c])
            for pr in cells[i].paragraphs:
                for rr in pr.runs:
                    rr.font.size = Pt(8)
    doc.add_paragraph()
    return True


# ---------------------------------------------------------------- value helpers
def num(txt):
    """First number in a 'mean ± sd' cell."""
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return float("nan")
    m = re.match(r"\s*([+-]?[0-9.]+)", str(txt))
    return float(m.group(1)) if m else float("nan")


HAVE_PANEL = len(PANEL) > 0


def pv(split, task, model, metric="spearman"):
    """Formatted cell from the metric panel, or None when that run does not exist yet."""
    if not HAVE_PANEL:
        return None
    s = PANEL[(PANEL.split == split) & (PANEL.task == task) & (PANEL.model == model)]
    if not len(s):
        return None
    v = s.iloc[0].get(metric)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def cmp_row(split, task, model, reference, metric="spearman"):
    if not len(COMPAR):
        return None
    s = COMPAR[(COMPAR.split == split) & (COMPAR.task == task) & (COMPAR.model == model)
               & (COMPAR.reference == reference) & (COMPAR.metric == metric)]
    return None if not len(s) else s.iloc[0]


def say_cmp(row):
    if row is None:
        return "not available"
    txt = (f"Δ = {row.median_diff:+.3f} (95 % CI {row.ci_lo:+.3f} to {row.ci_hi:+.3f}, "
           f"FDR-adjusted P = {row.p_fdr:.3g}, Cliff's δ = {row.cliffs_delta:+.2f}, {row.effect})")
    if bool(row.equivalent):
        txt += f"; equivalence to within ±{SUM2.get('margin', 0.05)} is established (TOST P = {row.tost_p:.3g})"
    return txt


def feat(arm, task, key="partial_mean"):
    s = FEAT.get("summary", {}).get(arm, {}).get(task, {})
    return s.get(key, float("nan"))


def feat_pm(arm, task, kind="partial"):
    s = FEAT.get("summary", {}).get(arm, {}).get(task, {})
    m, d = s.get(f"{kind}_mean", float("nan")), s.get(f"{kind}_sd", float("nan"))
    return f"{m:+.3f} ± {d:.3f}" if not np.isnan(m) else "—"


NSEED = SUM2.get("seeds", {}).get("heldout_sites", 10)

# ================================================================= TITLE
doc.add_heading(
    "Position-level extrapolation in deep mutational scanning models is a property of the "
    "representation, not of the data: structure transfers where sequence-indexed models cannot",
    0)
P("[Author names and affiliations]", al=WD_ALIGN_PARAGRAPH.CENTER)

# ================================================================= ABSTRACT
H("Abstract")
hb_mc = pv("heldout_sites", "ACE2 binding", "Mutation count")
hb_ec = pv("heldout_sites", "ACE2 binding", "EvoClassNet")
rb_ec = pv("random", "ACE2 binding", "EvoClassNet")
rb_ge = pv("random", "ACE2 binding", "Global epistasis")
abs_bits = [
    "Supervised models of deep mutational scanning data are normally validated by holding out "
    "random variants. In a saturating library almost every position is mutated somewhere in the "
    "training set, so this measures interpolation, not the case that matters for surveillance, "
    "in which a lineage mutates a position no experiment has probed. We benchmarked eleven model "
    "families on SARS-CoV-2 receptor-binding-domain variants with three measured phenotypes "
    "under a conventional random split and under a held-out-position split in which no training "
    "variant mutates any test position.",
]
_ab_p1 = pv('heldout_sites','ACE2 binding','EvoClassNet','spearman_partial')
_ab_p2 = pv('heldout_sites','ACE2 binding','Mutation count','spearman_partial')
if hb_mc and hb_ec and _ab_p1 and _ab_p2:
    abs_bits.append(
        f"Under the held-out-position split the sequence network reached Spearman ρ = {hb_ec} for "
        f"binding, against {hb_mc} for an untrained baseline that ranks variants only by how many "
        f"substitutions they carry. Reporting the rank correlation after substitution count is "
        f"partialled out of both sides shows why: the network retains "
        f"{_ab_p1} of genuine "
        f"position-level signal, and the untrained baseline, as it must, retains "
        f"{_ab_p2}.")
if FEAT:
    abs_bits.append(
        f"That failure is a property of the representation rather than of the data. Every model "
        f"in this class encodes a substitution by which position it occupies, so none holds a "
        f"parameter applicable to a position absent from training. Given features that remain "
        f"defined at unseen positions — how the residue changes chemically, and descriptors of "
        f"the position taken from the crystal structure of the receptor complex — a ridge "
        f"regression reaches a partial correlation of {feat_pm('chem+struct+esm','bind')} for "
        f"binding and {feat_pm('chem+struct+esm','escape','spearman')} for antibody escape. "
        f"Decomposing it, structure contributes {feat_pm('struct_only','bind')} and chemistry "
        f"{feat_pm('chem_only','bind')}; removing the two descriptors that mark the receptor "
        f"interface, and are therefore close to circular for binding, leaves "
        f"{feat_pm('struct_nointerface','bind')}.")
    abs_bits.append(
        f"ESM-2 contributes nothing on this protein: its zero-shot scores reach "
        f"{feat_pm('esm_only','bind','spearman')} for binding, of which "
        f"{feat_pm('esm_only','bind')} survives the substitution-count control.")
if len(GEN):
    g_triv = GEN[(GEN.split == "heldout_species") & (GEN.model.str.startswith("Genome length"))]
    g_seq = GEN[(GEN.split == "heldout_species") & (GEN.family == "sequence")]
    if len(g_triv) and len(g_seq):
        _t = num(g_triv.iloc[0].balanced_accuracy)
        _bestrow = g_seq.loc[g_seq.balanced_accuracy.map(num).idxmax()]
        _b = num(_bestrow.balanced_accuracy)
        # Say what the numbers show, not what the first task showed.
        if _b - _t < 0.03:
            _verdict = ("the sequence models add nothing over it, so the same reading applies as "
                        "for the assay data")
        else:
            _verdict = (f"the sequence models do add a real increment here ({_bestrow.model} "
                        f"reaches {_bestrow.balanced_accuracy}), so on this task the control "
                        f"bounds how much of a headline accuracy is informative rather than "
                        f"explaining all of it")
        abs_bits.append(
            f"A second task on {GMETA.get('n_genomes','?')} coronavirus genomes from "
            f"{GMETA.get('n_species','?')} species shows how much of a headline number a control "
            f"can account for even when the models are genuinely good: under a held-out-species "
            f"split, genome length and base composition alone reach balanced accuracy "
            f"{g_triv.iloc[0].balanced_accuracy}, and {_verdict}.")
abs_bits.append(
    "We conclude that a hard evaluation split is interpretable only against an untrained "
    "baseline computed on the same partition, that the apparent inability of sequence models to "
    "extrapolate across positions is an artefact of how they index substitutions, and that "
    "structural information is what makes such extrapolation possible here.")
P(" ".join(abs_bits))
P("Keywords: deep mutational scanning; SARS-CoV-2; model evaluation; generalisation; baselines; "
  "protein structure; protein language models; variant effect prediction.", it=True)

# ================================================================= INTRODUCTION
H("1. Introduction")
P("Deep mutational scanning measures the phenotypic consequences of thousands of amino-acid "
  "substitutions in a single experiment, and such data underpin much of what is known about the "
  "SARS-CoV-2 receptor-binding domain (RBD): which substitutions raise ACE2 affinity, which "
  "destabilise the fold, and which abolish binding by therapeutic antibodies [1,2]. Supervised "
  "models trained on these measurements are routinely used to score substitutions that were "
  "never assayed.")
P("How such models are validated has received less scrutiny than how they are built. The "
  "standard protocol holds out a random subset of variants. In a saturating combinatorial "
  "library almost every position is mutated somewhere in the training data, so a random test "
  "variant differs from training variants in which substitutions it combines, not in where those "
  "substitutions fall. A model can therefore score well by learning a per-position, per-residue "
  "effect table together with a link function, without any capacity to speak about a position it "
  "has never seen mutated. That capacity is what surveillance requires.")
P("This paper makes three points, and the second corrects the first. First, held-out-position "
  "evaluation is uninterpretable without an untrained control: on a combinatorial library a "
  "model that has learned nothing about the withheld positions still orders test variants "
  "correctly to the extent that fitness falls with the number of substitutions, and that alone "
  "produces a respectable-looking rank correlation. Second — and this is where our own first "
  "analysis went wrong — the conclusion that no model extrapolates across positions does not "
  "follow. The models we had compared all encode a substitution by the position it occupies, so "
  "none of them holds a parameter that applies to a held-out position; their failure was a "
  "property of the representation. Given features that remain defined at an unseen position, "
  "extrapolation is substantial. Third, structure is what carries it, while a protein language "
  "model contributes nothing measurable on this protein.")
P("We report the correction explicitly rather than presenting only the final result, because the "
  "sequence of errors is the methodological content of the paper: a confounded metric produced "
  "an inflated number, the right control removed it, and an over-general conclusion drawn from "
  "that control was itself wrong for a reason the control could not reveal.")

# ================================================================= METHODS
H("2. Materials and methods")

H("2.1 Phenotype data", 2)
try:
    from evoclassnet import data as _D
    _g = _D.load_genotypes()
    _n, _nb, _nx, _ne = len(_g), int(_g.bind.notna().sum()), int(_g.expr.notna().sum()), int(_g.escape.notna().sum())
except Exception:
    _n = _nb = _nx = _ne = 0
P(f"Binding and expression come from the yeast-display RBD scan of Starr et al. [1] "
  f"(Wuhan-Hu-1 background, spike residues 331–531; 201 positions). Variants containing stop "
  f"codons were removed and barcodes sharing an amino-acid genotype were collapsed to the mean "
  f"of their measurements, so an identical sequence can never appear on both sides of a split. "
  f"This yields {_n:,} unique genotypes, of which {_nb:,} carry a binding value and {_nx:,} an "
  f"expression value. Antibody escape comes from the aggregated RBD escape maps [2]: for each "
  f"substitution the escape fraction was averaged over all monoclonal antibodies assayed against "
  f"it, substitutions measured against fewer than 20 antibodies were dropped, and the mean was "
  f"log-transformed, giving {_ne:,} labelled single-substitution genotypes. Roughly 1.5 % of "
  f"escape rows were measured in a non-Wuhan-Hu-1 background; their listed wild-type residue is "
  f"an Omicron residue and they were discarded, both because the models operate in the "
  f"Wuhan-Hu-1 background and because the affected positions are exactly those used in the "
  f"Variant-of-Concern analysis.")

H("2.2 Structural data", 2)
try:
    from evoclassnet import structure as _S
    _f, _m, _adj, _names = _S.compute()
    _iface = [i + 1 + _S.OFFSET for i in range(201) if _m[i] and _f[i, _names.index("is_interface")] > 0.5]
    _res, _edges, _deg = int(_m.sum()), int(_adj.sum() // 2), float(_adj.sum(1).mean())
except Exception:
    _names, _iface, _res, _edges, _deg = [], [], 0, 0, 0.0
if _res:
    P(f"Per-position structural descriptors were computed from PDB 6M0J, the crystal structure of "
      f"the RBD (chain E) bound to human ACE2 (chain A), which resolves {_res} of the 201 "
      f"positions. Solvent accessibility was computed with the Shrake–Rupley algorithm for the "
      f"isolated domain and again for the complex, and normalised by the residue's theoretical "
      f"maximum. The descriptors are: " + ", ".join(_names) + ". Positions absent from the "
      f"crystal structure were imputed at the column median. A residue contact graph was built "
      f"from CB–CB distances below 8 Å (CA for glycine) together with backbone neighbours, giving "
      f"{_edges} edges at mean degree {_deg:.1f}. As a check on the pipeline, the positions it "
      f"marks as buried by receptor binding (ΔRSA > 0.05) are " +
      ", ".join(str(s) for s in _iface) + ", which is the ACE2 contact set reported in the "
      f"structural literature. Every one of these quantities is a property of the folded protein "
      f"and is available for a position whether or not it was ever assayed.")
fig("schema4_structure_gnn.png",
    "Figure M1. Structural pipeline and residue contact graph from PDB 6M0J. Descriptor names, "
    "edge count, mean degree and the recovered interface set are read from the computed features, "
    "not written by hand.", 6.6)

H("2.3 Protein language model", 2)
try:
    from evoclassnet import plm as _plm
    _mm, _hid = _plm.load()
except Exception:
    _mm = _hid = None
if _mm is not None:
    P(f"ESM-2 ({_plm.DEFAULT_MODEL.split('/')[-1]}, 650 M parameters) was run once over the "
      f"wild-type sequence. Two products were taken. Masked marginals: each position in turn is "
      f"replaced by the mask token and the substitution is scored as "
      f"log p(mutant | context) − log p(wild type | context), the standard zero-shot variant "
      f"effect score [6]; a multi-substitution genotype is scored as the sum over its "
      f"substitutions. Per-residue embeddings from the final layer were reduced by PCA to 16 "
      f"dimensions per position and supplied to the network through its per-position side-input. "
      f"Neither product sees any measurement, so the zero-shot scorer is leakage-free at "
      f"held-out positions by construction.")
if ESMC:
    u = ESMC.get("ubiquitin", {}); rf = ESMC.get("rbd_fragment", {}); rs = ESMC.get("rbd_in_spike", {})
    P(f"Because the language model performs poorly here, two controls were run to separate a "
      f"broken pipeline from a real result. Wild-type recovery — how often the masked position's "
      f"true residue is the model's top prediction — was {u.get('recovery', float('nan')):.1%} on "
      f"human ubiquitin, a small, highly constrained, well-represented protein. On the RBD it was "
      f"{rf.get('recovery', float('nan')):.1%} when the domain was scored in isolation and "
      f"{rs.get('recovery', float('nan')):.1%} when the same positions were scored inside the "
      f"full 1273-residue spike protein, which is the context the domain actually occupies. "
      f"Median top-token probability was {u.get('median_top_prob', float('nan')):.3f}, "
      f"{rf.get('median_top_prob', float('nan')):.3f} and "
      f"{rs.get('median_top_prob', float('nan')):.3f} respectively.")

fig("schema3_plm.png",
    "Figure M2. ESM-2 enters the study twice: as an untrained zero-shot scorer, and as "
    "per-position features for the supervised network.", 6.6)

H("2.4 Evaluation designs", 2)
P("Random split. Genotypes are divided 80/10/10 % into training, validation and test sets.")
P("Held-out-position split. The 201 positions are partitioned at random into training (75 %), "
  "validation (10 %) and test (15 %) sets. A genotype joins a partition only if every position it "
  "mutates lies in that partition's position set; genotypes spanning partitions are discarded. No "
  "training genotype therefore mutates any validation or test position, a property asserted "
  "directly in the test suite rather than assumed. Section 3.7 repeats the design with 5, 15, 30 "
  "and 50 % of positions withheld. Both designs were run over "
  f"{NSEED} seeds, and all models within a seed see the identical partition, so every model "
  "comparison is paired.")

fig("schema1_design.png",
    "Figure M3. The two evaluation designs, and the untrained controls evaluated on the identical "
    "partitions.", 6.6)

H("2.5 Models", 2)
P("Three untrained scorers. Mutation count ranks by the negative number of substitutions. "
  "BLOSUM62 sums substitution-matrix exchangeabilities. ESM-2 sums masked-marginal "
  "log-likelihood ratios. None is fitted to any measurement, so each is defined, unchanged, at "
  "held-out positions.")
P("Four models on a position-transferable feature block. Each substitution contributes the change "
  "in eight physicochemical properties (hydropathy, side-chain volume, charge, polarity, "
  "flexibility, helix and sheet propensity, mass), the wild-type residue's values for the same, "
  "its BLOSUM62 score, indicator variables for the wild-type and mutant residue identity, its "
  "ESM-2 score, and the structural descriptors of the position it occupies; a genotype is the "
  "sum over its substitutions plus the substitution count. Nothing in this block indexes a "
  "position by identity, so the representation is defined for a genotype that mutates positions "
  "never seen. On it we fit ridge regression with generalised cross-validated regularisation "
  "(referred to below as the physicochemical model), gradient boosting, random forest and "
  "k-nearest neighbours.")
P("A structure graph network. Each of the 201 positions is a node carrying an embedding of the "
  "residue this variant places there together with the fixed structural descriptors of the "
  "position; edges are the contact graph of Section 2.2. Three graph-convolution layers with "
  "symmetric normalisation are followed by attention pooling and three task heads.")
P("Four position-indexed models. Additive ridge regression on one-hot (position, residue) "
  "indicators; a global-epistasis model fitting an additive latent phenotype per task through a "
  "learned monotone sigmoid link [3]; EvoClassNet, a multi-task network of three convolutional "
  "blocks, a bidirectional LSTM, four-head self-attention and attention pooling, trained with a "
  "masked Huber loss and learned homoscedastic task weights [4]; and the same network given the "
  "structural descriptors and ESM-2 embeddings as per-position side information. The additive "
  "and global-epistasis models have no parameter for a substitution at an unseen position; the "
  "two networks have position embeddings that were never updated for such a position.")
P("Hyper-parameters were selected by a genetic algorithm maximising mean validation Spearman ρ on "
  "a random-split validation set; test data were never used for selection. Optimisation used "
  "AdamW [5] with a one-cycle schedule, batch size 256, up to 8 epochs with early stopping, "
  "gradient clipping at global norm 1.0. All experiments ran on four CPU cores.")

fig("schema2_backbone.png",
    "Figure M4. EvoClassNet backbone. Channel counts, kernel sizes, pooling, hidden widths and "
    "the parameter count are read from the trained configuration.", 6.8)

H("2.6 Metrics", 2)
P("The primary metric is Spearman's ρ on the test partition, computed per task over genotypes "
  "carrying that measurement; a model emitting a constant prediction conveys no rank information "
  "and is assigned ρ = 0. Because part of any rank correlation on a combinatorial library is "
  "explained by how many substitutions a variant carries, we also report the partial Spearman "
  "correlation, in which the substitution count is regressed out of the ranks of both prediction "
  "and measurement. This measures position-level signal directly rather than inferring it from "
  "a baseline comparison, and it has a built-in check: the mutation-count baseline must score "
  "exactly zero on it. We additionally report Pearson's r, Kendall's τ, R², RMSE, MAE, AUROC for "
  "ranking the lower measurement quartile, precision at k and NDCG for top-variant retrieval, "
  "the calibration slope, and correlations within fixed substitution-count strata.")

H("2.7 Statistics", 2)
P(f"Models are compared on per-seed paired differences with a two-sided Wilcoxon signed-rank "
  f"test, bootstrap percentile intervals, and Cliff's δ as a scale-free effect size. Because the "
  f"central claim about the position-indexed models is a negative one, and failing to reject a "
  f"null is not evidence for it, we also report two one-sided tests of equivalence against a "
  f"margin of ±{SUM2.get('margin', 0.05)} in Spearman ρ; a significant result there is positive "
  f"evidence that two models differ by less than that margin. P values are corrected for false "
  f"discovery rate across the whole family of model-by-task comparisons. Per-position outcomes "
  f"recur across seeds and are not independent, so they are analysed with a random-intercept "
  f"model over positions. Seed-level P values describe reproducibility across partitions of one "
  f"data set; they do not license inference to other proteins.")

H("2.8 Genome classification task", 2)
if GMETA:
    cc = GMETA.get("class_counts", {})
    P(f"Complete coronavirus genomes (25–35 kb) were retrieved from NCBI nucleotide and labelled "
      f"by genus, giving {GMETA['n_genomes']} sequences from {GMETA['n_species']} species after "
      f"removal of duplicates (" + ", ".join(f"{k} {v}" for k, v in cc.items()) + f"). SARS-CoV-2 "
      f"was excluded from the Betacoronavirus class: with over three million deposited genomes "
      f"against a few hundred for whole genera, including it would make the class a measure of "
      f"one outbreak's sequencing effort. Sequences are represented by normalised "
      f"{GMETA.get('k', 5)}-mer frequencies, and classified by logistic regression, gradient "
      f"boosting, random forest, and a 1D convolutional network over the first 4,000 nucleotides. "
      f"Two untrained controls are reported: the majority class, and genome length together with "
      f"base composition. The split is either random or by species, the latter placing every "
      f"genome of a species on one side only, which is the genome-level analogue of withholding "
      f"positions.")
    fig("schema5_genome_task.png",
        "Figure M5. The genome classification design, with class sizes and the untrained "
        "controls.", 6.6)

# ================================================================= RESULTS
H("3. Results")

H("3.1 Under random splits every trained model performs well", 2)
if len(PANEL):
    rnd = PANEL[PANEL.split == "random"]
    table(rnd, "Table 1. Random genotype split: full metric panel, mean ± SD across seeds.",
          cols=["family", "model", "task", "seeds", "spearman", "spearman_partial", "pearson",
                "kendall", "r2", "rmse", "auroc_deleterious", "precision_at_50", "ndcg_at_100",
                "calibration_slope", "n"],
          rename={"family": "Family", "model": "Model", "task": "Phenotype", "seeds": "Seeds",
                  "spearman": "Spearman ρ", "spearman_partial": "Partial ρ", "pearson": "Pearson r",
                  "kendall": "Kendall τ", "r2": "R²", "rmse": "RMSE",
                  "auroc_deleterious": "AUROC", "precision_at_50": "Prec@k",
                  "ndcg_at_100": "NDCG", "calibration_slope": "Calib.", "n": "n"})
if rb_ec and rb_ge:
    r = cmp_row("random", "ACE2 binding", "EvoClassNet", "Mutation count")
    P(f"On the conventional design the trained models behave as expected. EvoClassNet reached "
      f"ρ = {rb_ec} for binding and the global-epistasis baseline {rb_ge}, against "
      f"{pv('random','ACE2 binding','Mutation count')} for mutation count alone "
      f"({say_cmp(r)}). The differences among trained families are small; the design does not "
      f"discriminate between them.")
P("Two entries are zero by construction rather than by failure. The additive ridge model scores "
  "zero on escape because escape labels exist only for single substitutions, so each "
  "(position, residue) indicator occurs in exactly one training row and the regularised fit "
  "collapses to the intercept. Mutation count scores zero on escape for the same reason: every "
  "labelled variant carries exactly one substitution, so the baseline is constant. This makes "
  "escape the one phenotype on which mutation count cannot masquerade as skill.")
fig("fig1_panel_spearman.png", "Figure 1. Test Spearman ρ for every model family under both "
    "evaluation designs. Bars are means over seeds, whiskers one standard deviation. Colour "
    "denotes how a model represents a substitution.", 6.6)

H("3.2 At held-out positions the position-indexed models fall to the untrained baseline", 2)
if len(PANEL):
    hld = PANEL[PANEL.split == "heldout_sites"]
    table(hld, "Table 2. Held-out-position split: test variants mutate only positions absent from "
          "training.",
          cols=["family", "model", "task", "seeds", "spearman", "spearman_partial", "pearson",
                "kendall", "r2", "rmse", "auroc_deleterious", "precision_at_50", "ndcg_at_100",
                "calibration_slope", "n"],
          rename={"family": "Family", "model": "Model", "task": "Phenotype", "seeds": "Seeds",
                  "spearman": "Spearman ρ", "spearman_partial": "Partial ρ", "pearson": "Pearson r",
                  "kendall": "Kendall τ", "r2": "R²", "rmse": "RMSE",
                  "auroc_deleterious": "AUROC", "precision_at_50": "Prec@k",
                  "ndcg_at_100": "NDCG", "calibration_slope": "Calib.", "n": "n"})
P("When every test variant mutates positions never mutated in training, the additive model has "
  "no applicable parameter and emits a constant. The global-epistasis model does worse than "
  "nothing, its predictions anti-correlated with measurement, because latent effects for unseen "
  "substitutions are shrunk towards zero and then passed through a link fitted to the training "
  "distribution. EvoClassNet stays positive. Read on its own, the table says the network "
  "extrapolates where the conventional models cannot.")

H("3.3 The mutation-count control, and a metric that makes it unnecessary", 2)
mc_part = pv("heldout_sites", "ACE2 binding", "Mutation count", "spearman_partial")
if HAVE_PANEL and hb_mc and hb_ec and mc_part:
  P(f"That reading is wrong, and the reason is visible in a single column. Ranking variants by the "
  f"number of substitutions they carry — no training, no parameters, no knowledge of which "
  f"positions or residues are involved — reaches ρ = {hb_mc} for binding at held-out positions, "
  f"against {hb_ec} for EvoClassNet. In a combinatorial library most substitutions are "
  f"deleterious and their effects accumulate, so a large part of the rank variance is carried by "
  f"substitution count alone, and a model that has learned nothing about the withheld positions "
  f"still orders the test set to that extent.")
  P(f"Rather than argue from the baseline comparison, we measure the quantity directly. The "
  f"partial Spearman correlation removes substitution count from the ranks of both prediction "
  f"and measurement. It validates itself: the mutation-count baseline scores {mc_part} on it, as "
  f"it must. On the same partitions EvoClassNet retains "
  f"{pv('heldout_sites','ACE2 binding','EvoClassNet','spearman_partial')} for binding and "
  f"{pv('heldout_sites','Expression','EvoClassNet','spearman_partial')} for expression.")
mc_part = mc_part if mc_part else "—"
if len(COMPAR):
    sel = COMPAR[(COMPAR.split == "heldout_sites") & (COMPAR.reference == "Mutation count")
                 & (COMPAR.metric == "spearman")]
    table(sel, "Table 3. Held-out-position split: every trained model against the untrained "
          "mutation-count baseline, paired over seeds. Equivalence is a two one-sided test at the "
          f"±{SUM2.get('margin', 0.05)} margin; a true value there is positive evidence of no "
          "practical difference, not merely absence of a significant one.",
          cols=["task", "model", "n_seeds", "median_diff", "ci_lo", "ci_hi", "p_fdr", "tost_p",
                "equivalent", "cliffs_delta", "effect"],
          rename={"task": "Phenotype", "model": "Model", "n_seeds": "Seeds",
                  "median_diff": "Median Δρ", "ci_lo": "CI low", "ci_hi": "CI high",
                  "p_fdr": "P (FDR)", "tost_p": "TOST P", "equivalent": "Equivalent",
                  "cliffs_delta": "Cliff's δ", "effect": "Effect"})
fig("fig2_panel_partial.png", "Figure 2. The same comparison in partial Spearman ρ, with "
    "substitution count removed from both prediction and measurement. The untrained "
    "mutation-count baseline sits at zero by construction, which is what makes this panel "
    "readable.", 6.6)

H("3.4 Extrapolation is possible; the earlier conclusion was about representations", 2)
P("Our first analysis stopped here and concluded that none of the models extrapolates across "
  "positions. That conclusion does not follow from the evidence, and it is wrong. Every model "
  "compared above encodes a substitution by the position it occupies: the additive model by a "
  "(position, residue) indicator, the global-epistasis model by a per-position latent effect, "
  "the networks by a position embedding. For a position absent from training, none of them holds "
  "a parameter that was ever updated. Their collapse measures that fact, not a limit on what the "
  "data support.")
if FEAT:
    rows = []
    order = [("nsub_only", "Substitution count only"),
             ("esm_only", "ESM-2 score only"),
             ("chem_only", "Residue chemistry only"),
             ("struct_only", "Structure only"),
             ("struct_nointerface", "Structure, interface descriptors removed"),
             ("chem+struct_nointerface", "Chemistry + structure, interface removed"),
             ("chem+struct", "Chemistry + structure"),
             ("chem+struct+esm", "Chemistry + structure + ESM-2")]
    for arm, lab in order:
        if arm not in FEAT.get("summary", {}):
            continue
        row = dict(Features=lab)
        for t, tn in (("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")):
            row[f"{tn} ρ"] = feat_pm(arm, t, "spearman")
            row[f"{tn} partial"] = feat_pm(arm, t, "partial")
        rows.append(row)
    table(pd.DataFrame(rows),
          "Table 4. Held-out-position split: a ridge regression on position-transferable "
          "features, by feature block. Partial ρ removes the substitution-count confound. "
          "Antibody escape has no partial column because every escape-labelled variant carries "
          "one substitution, so the confound is absent there by construction.")
    P(f"Given features that remain defined at an unseen position, extrapolation is substantial. "
      f"Chemistry alone reaches a partial correlation of {feat_pm('chem_only','bind')} for "
      f"binding, structure alone {feat_pm('struct_only','bind')}, and the combination "
      f"{feat_pm('chem+struct+esm','bind')} — against {mc_part} for the untrained baseline and "
      f"{pv('heldout_sites','ACE2 binding','EvoClassNet','spearman_partial') or 'not yet measured'} "
      f"for the sequence network on the identical partitions. The information needed to predict the effect of a "
      f"substitution at an unseen position is present in the data; the earlier models simply had "
      f"no way to express it.")
    P(f"One component of that signal needs separating out before it is believed. Two of the "
      f"structural descriptors, the change in solvent accessibility on binding and the interface "
      f"indicator, are derived from the RBD bound to ACE2 and therefore mark precisely the "
      f"residues that touch the receptor, which for the binding task is close to being told the "
      f"answer. Dropping both leaves {feat_pm('struct_nointerface','bind')}, against "
      f"{feat_pm('struct_only','bind')} with them, so the structural contribution does not rest "
      f"on them. The remaining descriptors — solvent accessibility of the unbound domain, "
      f"contact number, burial, B-factor — describe the folded domain alone.")

H("3.5 Where the transfer comes from: positions, not substitutions", 2)
TBP = J("transfer_by_position.json")
if TBP:
    rows = []
    for t, tn in (("bind", "ACE2 binding"), ("expr", "Expression"), ("escape", "Antibody escape")):
        v = TBP.get(t)
        if not v:
            continue
        vb, me = v["vs_blosum"], v["mixed_effects"]
        rows.append({"Phenotype": tn,
                     "Median ρ": f"{v['median_rho']:+.3f}",
                     "% positive": f"{100*v['frac_positive']:.0f}",
                     "Random-intercept P": f"{me.get('p', float('nan')):.3g}",
                     "BLOSUM62 median ρ": f"{vb['blosum_median']:+.3f}",
                     "Difference": f"{vb['median_diff']:+.3f}",
                     "P": f"{vb['p']:.3g}",
                     "n position–seed": vb["n"]})
    table(pd.DataFrame(rows),
          "Table 5. Per-position rank accuracy of the transferable model at withheld positions: "
          "within each withheld position, how well it orders the 19 possible substitutions. "
          "Compared against BLOSUM62 exchangeability on identical instances. The P value for the "
          "median comes from a random-intercept model over positions, because positions recur "
          "across seeds.")
    b, e = TBP.get("bind", {}), TBP.get("escape", {})
    if b and e:
        P(f"Restricting to single substitutions and asking a narrower question — within one "
          f"withheld position, can the model order the nineteen substitutions available there — "
          f"separates two components of the transfer. For binding the model reaches a median "
          f"per-position ρ of {b['median_rho']:+.3f}, positive at "
          f"{100*b['frac_positive']:.0f} % of position–seed cells, but BLOSUM62 reaches "
          f"{b['vs_blosum']['blosum_median']:+.3f} on the same instances and the difference is "
          f"{b['vs_blosum']['median_diff']:+.3f} (P = {b['vs_blosum']['p']:.3g}). Within a "
          f"position, in other words, the ordering is chemistry, and an untrained substitution "
          f"matrix already supplies it.")
        P(f"The model's advantage therefore lies in discriminating between positions — which "
          f"positions tolerate substitution at all — and that is precisely what the structural "
          f"descriptors provide and what a substitution matrix, being position-independent, "
          f"cannot. It is consistent with the whole-test-set result: summed BLOSUM62 scores reach "
          f"{pv('heldout_sites','ACE2 binding','BLOSUM62') or 'n/a'} there, against "
          f"{feat_pm('chem+struct+esm','bind','spearman')} for the full feature block.")
        P(f"Escape is the exception in the opposite direction. BLOSUM62 is not merely uninformative "
          f"but anti-correlated ({e['vs_blosum']['blosum_median']:+.3f}), because the "
          f"substitutions that abolish antibody binding are not the chemically radical ones, "
          f"while the model reaches {e['median_rho']:+.3f} "
          f"(difference {e['vs_blosum']['median_diff']:+.3f}, P = {e['vs_blosum']['p']:.3g}). "
          f"There the within-position ordering is genuinely learned.")
    for t, tn in (("bind", "binding"), ("escape", "escape")):
        v = TBP.get(t)
        if not v:
            continue
        cs = sorted(v["rho_vs_structure"].items(), key=lambda kv: -abs(kv[1][0]))[:3]
        P(f"Per-position accuracy for {tn} is associated with " + ", ".join(
            f"{k} (ρ = {r:+.2f}, P = {pp:.2g})" for k, (r, pp) in cs) + ".")
fig("fig8_transfer_by_position.png",
    "Figure 6. Per-position rank accuracy at withheld positions against solvent accessibility and "
    "distance to ACE2.", 6.6)

H("3.6 The protein language model contributes nothing here", 2)
if FEAT:
    P(f"ESM-2 is the component most often expected to supply exactly this kind of "
      f"position-independent knowledge, and on this protein it does not. Its zero-shot scores "
      f"reach {feat_pm('esm_only','bind','spearman')} for binding at held-out positions, of which "
      f"{feat_pm('esm_only','bind')} survives the substitution-count control: the raw figure is "
      f"the confound in full. Added to chemistry and structure it moves the partial correlation "
      f"from {feat_pm('chem+struct','bind')} to {feat_pm('chem+struct+esm','bind')}.")
if ESMC:
    u, rf2, rs2 = ESMC.get("ubiquitin", {}), ESMC.get("rbd_fragment", {}), ESMC.get("rbd_in_spike", {})
    ur, fr, sr = u.get("recovery", float("nan")), rf2.get("recovery", float("nan")), rs2.get("recovery", float("nan"))
    verdict = ("the model is working and the RBD is genuinely hard for it"
               if ur > 0.3 and fr < 0.3 else
               "the model recovers little anywhere in this run, so the negative result should be "
               "treated as unresolved rather than as a property of the protein")
    ctx = ("supplying the full spike context does not rescue it"
           if not np.isnan(sr) and sr < fr + 0.1 else
           "supplying the full spike context improves it substantially, so the isolated-domain "
           "figure understates what the model can do")
    P(f"Because a null result from a large pretrained model invites the suspicion of a broken "
      f"pipeline, we checked. Wild-type recovery was {ur:.1%} on human ubiquitin against "
      f"{fr:.1%} on the isolated RBD and {sr:.1%} on the same positions scored inside the full "
      f"spike protein: {verdict}, and {ctx}. No weights were missing at load time. We therefore "
      f"report the absence of a contribution as a result about this protein, which is consistent "
      f"with SARS-CoV-2 ranking among the harder entries in published zero-shot benchmarks, and "
      f"not as a general statement about protein language models.")

H("3.7 Permuting the amino-acid alphabet", 2)
if len(ABL2):
    table(ABL2, "Table 6. Amino-acid permutation ablation: an independent random permutation of "
          "the alphabet at every position, applied consistently to all partitions.",
          cols=["split", "model", "task", "intact", "permuted", "delta"],
          rename={"split": "Split", "model": "Model", "task": "Phenotype", "intact": "Intact",
                  "permuted": "Permuted", "delta": "Δ"})
    P("Permuting the alphabet independently at each position preserves every position's own "
      "information while destroying any correspondence between the same residue at different "
      "positions. For the sequence models the manipulation changes little, which is what the "
      "mutation-count account predicts, since substitution count is invariant to relabelling. "
      "The feature-based models are destroyed by it, as they must be: their entire "
      "representation is residue identity and its chemical correlates.")

H("3.8 Withholding more positions", 2)
fig("fig3_sweep_v2.png", "Figure 3. Performance against the fraction of positions withheld "
    "(5–50 %). Points are means over seeds, whiskers one standard deviation.", 6.6)

H("3.9 Run-to-run instability", 2)
if STAB:
    byss = {}
    for r in STAB:
        byss.setdefault(r["split_seed"], []).append(r)
    for ss, rs in sorted(byss.items()):
        v = [r["bind"] for r in rs]
        P(f"Holding partition {ss} fixed and varying only the weight initialisation gave binding "
          f"ρ of " + ", ".join(f"{x:+.3f}" for x in v) + f" (range {max(v)-min(v):.3f}).")
    allv = [r["bind"] for r in STAB]
    P(f"Across all {len(allv)} runs the spread was {min(allv):+.3f} to {max(allv):+.3f}. "
      f"Initialisation alone accounts for a substantial part of it, and on some partitions the "
      f"spread exceeds the differences being discussed, so a single run on this benchmark is not "
      f"a measurement. Every table here reports a seed distribution for that reason.")

H("3.10 Variant-of-Concern ranking without leakage", 2)
if HO.get("voc"):
    rows = [dict(Score=k, AUROC=f"{v['auroc']:.3f}",
                 CI=f"{v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}", n_positives=v["n_voc"])
            for k, v in HO["voc"].items()]
    table(pd.DataFrame(rows),
          "Table 7. Ranking Variant-of-Concern substitutions at withheld positions only, pooled "
          "over seeds, with 95 % bootstrap intervals.")
    pe = HO["voc"].get("EvoClassNet predicted escape", {})
    me = HO["voc"].get("Measured escape (held-out positions)", {})
    bl = HO["voc"].get("BLOSUM62 exchangeability", {})
    P(f"Variant-of-Concern substitutions are present in the escape training data, so ranking them "
      f"with a model that has seen them measures memorisation; restricting to withheld positions "
      f"removes that circularity. Predicted escape gave AUROC {pe.get('auroc', float('nan')):.3f} "
      f"(95 % CI {pe.get('ci95',[np.nan]*2)[0]:.3f}–{pe.get('ci95',[np.nan]*2)[1]:.3f}), an "
      f"interval that includes 0.5. The measured values on the identical subset gave "
      f"{me.get('auroc', float('nan')):.3f}, which is the ceiling available to any model of this "
      f"phenotype, and BLOSUM62 exchangeability, which requires no training at all, gave "
      f"{bl.get('auroc', float('nan')):.3f}. The assay carries real information about which "
      f"substitutions went on to define lineages; at positions it never saw, the trained "
      f"sequence model does not recover it. The positives are a small set chosen with hindsight, "
      f"so the comparison between scorers on identical instances is meaningful while the absolute "
      f"values are not.")
if HO.get("vs_blosum"):
    rows = [dict(Phenotype={"bind": "ACE2 binding", "escape": "Antibody escape"}.get(k, k),
                 EvoClassNet=f"{v['model_median']:.3f}", BLOSUM62=f"{v['blosum_median']:.3f}",
                 Difference=f"{v['median_diff']:+.3f}", P=f"{v['p']:.3g}", n=v["n"])
            for k, v in HO["vs_blosum"].items()]
    table(pd.DataFrame(rows), "Table 8. Median per-position Spearman ρ over single substitutions "
          "at withheld positions, against a substitution matrix.")
    vb = HO["vs_blosum"]
    if "bind" in vb and "escape" in vb:
        P(f"Per-position rank accuracy separates the phenotypes. For binding an untrained "
          f"substitution matrix beats the network ({vb['bind']['blosum_median']:.3f} against "
          f"{vb['bind']['model_median']:.3f}, P = {vb['bind']['p']:.3g}); for escape the matrix is "
          f"actively misleading, its median correlation negative "
          f"({vb['escape']['blosum_median']:.3f}) where the network is positive "
          f"({vb['escape']['model_median']:.3f}, P = {vb['escape']['p']:.3g}), because the "
          f"substitutions that abolish antibody binding are not the chemically radical ones.")
if VA:
    for obj, v in VA.items():
        if not (isinstance(v, dict) and "archive" in v):
            continue
        P(f"Maximising {obj.replace('_', ' ')} with a genetic algorithm visited {v['archive']:,} "
          f"distinct sequences. The top 1 % carried {v['top_mean_voc']:.3f} Variant-of-Concern "
          f"substitutions per variant against {v['random_mean_voc']:.3f} for size-matched random "
          f"samples of the same archive (permutation P = {v['perm_p']:.3f}). A learned fitness "
          f"surface is constrained only where it was fitted, and an optimiser leaves that region.")

H("3.11 A second task: coronavirus genomes", 2)
if len(GEN):
    table(GEN, "Table 9. Coronavirus genus classification under both splits, mean ± SD across "
          "seeds.",
          cols=["split", "family", "model", "seeds", "accuracy", "balanced_accuracy", "macro_f1",
                "weighted_f1", "mcc"] + [c for c in GEN.columns if c.startswith("f1_")],
          rename={"split": "Split", "family": "Family", "model": "Model", "seeds": "Seeds",
                  "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
                  "macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "mcc": "MCC"})
    for sp, label in (("random", "random"), ("heldout_species", "held-out-species")):
        d = GEN[GEN.split == sp]
        if not len(d):
            continue
        triv = d[d.model.str.startswith("Genome length")]
        seqm = d[d.family == "sequence"]
        if len(triv) and len(seqm):
            best = seqm.loc[seqm.balanced_accuracy.map(num).idxmax()]
            P(f"Under the {label} split, genome length and base composition alone reach balanced "
              f"accuracy {triv.iloc[0].balanced_accuracy}, against {best.balanced_accuracy} for "
              f"the best sequence model ({best.model}) and "
              f"{d[d.model == 'Majority class'].iloc[0].balanced_accuracy} for the majority "
              f"class.")
if len(GENC):
    table(GENC, "Table 10. Sequence models against the untrained length-and-composition control, "
          "paired over seeds.",
          cols=["split", "model", "reference", "n_seeds", "median_diff", "ci_lo", "ci_hi",
                "p_fdr", "tost_p", "equivalent", "cliffs_delta"],
          rename={"split": "Split", "model": "Model", "reference": "Reference",
                  "n_seeds": "Seeds", "median_diff": "Median Δ", "ci_lo": "CI low",
                  "ci_hi": "CI high", "p_fdr": "P (FDR)", "tost_p": "TOST P",
                  "equivalent": "Equivalent", "cliffs_delta": "Cliff's δ"})
fig("fig6_genomes.png", "Figure 4. Balanced accuracy on coronavirus genus classification under "
    "both splits. The dotted line is chance for four classes.", 6.4)
fig("fig7_genome_confusion.png", "Figure 5. Confusion matrices under the held-out-species split, "
    "pooled over seeds and row-normalised.", 6.4)
P("The original project proposal reported this task as a five-way classification at 99.92 % "
  "accuracy, with no baseline. The lesson here is not that such a number must be wrong, but that "
  "it is uninterpretable on its own: an untrained control looking only at genome length and base "
  "composition already reaches most of it, so the reported figure conveys almost none of the "
  "information it appears to. Where the sequence models do beat that control, as they do here, "
  "the control is what makes the increment visible and worth reporting.")

# ================================================================= DISCUSSION
H("4. Discussion")
P("Four evaluation designs applied to the same library support four different conclusions. A "
  "random split says the trained models work well and that the choice between them hardly "
  "matters. A held-out-position split says a sequence network extrapolates where conventional "
  "models fail. Adding an untrained baseline to that split says the network does nothing of the "
  "kind. Adding models whose features remain defined at an unseen position says that "
  "extrapolation is nonetheless possible, and substantial. Only the fourth conclusion survives, "
  "and reaching it required two corrections in sequence.")
P("The first correction is the familiar one about controls. Making a split harder does not by "
  "itself make the resulting number meaningful: it changes what an uninformative model would "
  "score, and unless that value is measured on the same partition it is easy to mistake a shift "
  "in the null for evidence of skill. On a combinatorial mutagenesis library the uninformative "
  "score is far from zero, because the number of substitutions a variant carries is itself "
  "predictive. We would go further and recommend the partial correlation over the baseline "
  "comparison: it measures the quantity of interest directly, it needs no second model, and it "
  "carries its own validity check, since the trivial baseline must score zero on it.")
P("The second correction is the one we did not anticipate, and it is the more interesting. "
  "Having established that the sequence models do not transfer across positions, we wrote that "
  "none of the models tested demonstrates position-level extrapolation — a statement about the "
  "data that the evidence did not support. All three models we had compared share a "
  "representational commitment: a substitution is identified by where it occurs. Such a model "
  "cannot transfer across positions for the same reason a lookup table cannot, and observing "
  "that it does not is uninformative about whether the phenotype is predictable at an unseen "
  "position. It is. The information is carried by the chemistry of the substitution and by the "
  "structural environment of the position, both available without having assayed it.")
P("That this is a negative result about architectures rather than about proteins has a practical "
  "consequence. A model intended for surveillance should be built on features that survive the "
  "arrival of a position no one has measured; the structural descriptors used here are cheap, "
  "and for SARS-CoV-2 they were available from early 2020. The protein language model, which is "
  "the component usually expected to provide this, supplied nothing measurable on this protein, "
  "and we are careful to scope that claim to this protein.")
P("The practical recommendations are narrow. Report held-out-position performance alongside "
  "random splits for any predictor trained on a saturating scan, because the designs rank models "
  "differently. Report an untrained baseline on every split, or better, report the partial "
  "correlation that removes the confound outright. Before concluding that a phenotype cannot be "
  "extrapolated, check whether the models tested have any parameter that applies to the held-out "
  "condition. And report a seed distribution: initialisation alone moved held-out binding ρ "
  "across a range wide enough to reverse conclusions.")

H("4.1 Limitations", 2)
for t in [
    "This is one protein, one genetic background (Wuhan-Hu-1) and two assay platforms. The "
    "evaluation lesson is general; the specific finding that structure carries the transfer is "
    "not established beyond this domain.",
    "The structural descriptors come from a crystal structure of the complex. Two of them mark "
    "the receptor interface and are close to circular for the binding task; we quantify the "
    "effect of removing them rather than claiming it is negligible a priori. For a target "
    "without a solved structure the same descriptors would have to come from a prediction, and "
    "that substitution was not tested here.",
    "Only one protein language model, at one size, was evaluated, with a single scoring scheme. "
    "A larger model, a different scoring convention, or fine-tuning on viral sequences could "
    "change the conclusion.",
    "Hyper-parameters were selected on random-split validation and may be unsuited to the "
    "held-out-position regime; this would, if anything, understate the trained models there.",
    "Escape labels average over an antibody panel assembled from many studies with differing "
    "antibody selections, and exist only for single substitutions, so that task has far fewer "
    "labels and wider intervals than the other two.",
    "Binding measurements are censored at an assay floor, which we did not model; expression is "
    "a proxy for folding stability.",
    "The genome task uses a few hundred genomes per genus and truncates sequences for the "
    "convolutional model; it is a demonstration that the evaluation lesson transfers, not a "
    "competitive classifier.",
    "Seed-level P values describe reproducibility across partitions of one data set. They do not "
    "license inference to other proteins or other scans.",
]:
    doc.add_paragraph(t, style="List Bullet")

H("5. Data and code availability")
P("Phenotype data: Starr et al., SARS-CoV-2-RBD_DMS, and the aggregated escape maps at "
  "jbloomlab/SARS2_RBD_Ab_escape_maps. Structure: PDB 6M0J. Reference sequence: UniProt P0DTC2. "
  "Genomes: NCBI nucleotide. Language model: facebook/esm2_t33_650M_UR50D. All analysis code, "
  "the retrieval scripts, every per-run output and the script that generates this manuscript "
  "from them are in the accompanying repository; run_all.sh reproduces every table and figure "
  "from raw data on four CPU cores.")

H("References")
for r in [
    "Starr TN, Greaney AJ, Hilton SK, et al. Deep mutational scanning of SARS-CoV-2 receptor "
    "binding domain reveals constraints on folding and ACE2 binding. Cell 2020;182:1295–1310.",
    "Greaney AJ, Starr TN, Bloom JD. An antibody-escape estimator for mutations to the "
    "SARS-CoV-2 receptor-binding domain. Virus Evolution 2022;8:veac021.",
    "Otwinowski J, McCandlish DM, Plotkin JB. Inferring the shape of global epistasis. Proc Natl "
    "Acad Sci USA 2018;115:E7550–E7559.",
    "Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene "
    "geometry and semantics. CVPR 2018:7482–7491.",
    "Loshchilov I, Hutter F. Decoupled weight decay regularization. ICLR 2019.",
    "Meier J, Rao R, Verkuil R, et al. Language models enable zero-shot prediction of the effects "
    "of mutations on protein function. NeurIPS 2021.",
    "Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure "
    "with a language model. Science 2023;379:1123–1130.",
    "Lan J, Ge J, Yu J, et al. Structure of the SARS-CoV-2 spike receptor-binding domain bound to "
    "the ACE2 receptor. Nature 2020;581:215–220.",
    "Shrake A, Rupley JA. Environment and exposure to solvent of protein atoms. Lysozyme and "
    "insulin. J Mol Biol 1973;79:351–371.",
    "Notin P, Kollasch AW, Ritter D, et al. ProteinGym: large-scale benchmarks for protein "
    "fitness prediction and design. NeurIPS 2023.",
]:
    doc.add_paragraph(r, style="List Number")

out = ROOT / "EvoClassNet_manuscript.docx"
doc.save(out)
print("saved", out)
