# Figure generation prompts

Prompts for an AI figure generator or an illustrator, one per manuscript figure, numbered as
they appear in the manuscript.

**Every number below was measured by this repository.** Nothing is estimated. Do not let a
generator add hardware names, runtimes, throughput, FLOPs, parameter counts or accuracy figures
that are not written here — the uploaded sample schematic that inspired these carried several
such numbers (an A100 GPU, 8.7 ms inference, 115 structures per second, "+1.58 % accuracy",
"73 % attention weight") and none of them were ever measured.

**Method schematics (Figures 1–5) are the right use for a generator.** Result figures (6–12) are plots
of measured data; redrawing them is legitimate only if the exact values below are reproduced,
and the vector PDFs in `deliverables/figures/publication/` are preferable for submission.

---

## STYLE PREAMBLE — paste this before every prompt

```
Clean vector scientific figure in the style of a Nature or Briefings in Bioinformatics panel.
White background, thin grey panel borders, pale blue-grey header bars, sans-serif labels
(Arial or Helvetica), 7–9 pt label sizes, hairline axes with no top or right spine.
Colour-blind-safe palette only: steel blue #0072B2, teal green #009E73, vermillion #D55E00,
purple #CC79A7, sky #56B4E9, neutral grey #6E6E6E. Colour must never be the only cue — repeat
it with position or a direct text label. No 3D effects, no drop shadows, no photographic
texture, no decorative icons. Bold lower-case panel letters (a, b, c) at the top left of each
panel. Reproduce every label and number exactly as written; invent nothing.
```

---

## Figure 1 — Structural pipeline and contact graph

```
Left-to-right flow in four boxes.

Box 1 "PDB 6M0J": crystal structure of the SARS-CoV-2 receptor-binding domain (chain E) bound
to human ACE2 (chain A). Label: "194 of 201 positions resolved".

Box 2 "Per-position descriptors", listing exactly these eight names:
rsa_unbound, rsa_bound, d_rsa, min_dist_ace2, contact_number, depth, bfactor, is_interface.
Caption under the box: "properties of the folded protein, defined for positions never assayed".

Box 3 "Residue contact graph": edge rule "CB–CB < 8 Å (CA for glycine) plus backbone
neighbours". Stats: "913 edges, mean degree 9.08".

Box 4 "Graph neural network": node input "residue embedding 24 ‖ structural descriptors 8 =
32-dim"; three layers each drawn as "normalised adjacency Â × H × W → LayerNorm → ReLU",
hidden width 64, dropout 0.1; then "attention pooling ‖ max pooling"; then "three task heads".
Parameter box: "61,287 parameters".

Footer strip, bold, in teal green, titled "ACE2 interface positions recovered from ΔRSA > 0.05
(spike numbering)":
417  446  449  453  455  456  475  476  484  486  487  489  493  496  498  500  501  502  505
```

## Figure 2 — ESM-2 in two roles

```
One input box on the left, two output branches on the right.

Input: "Wild-type RBD, 201 residues, scored inside the full 1273-residue spike protein
(UniProt P0DTC2)".

Centre: "ESM-2, facebook/esm2_t33_650M_UR50D, 650 million parameters, masked language model".

Upper branch "Masked marginals, 201 × 20": formula line
"score = log p(mutant | context) − log p(wild type | context)";
note "a genotype is scored as the sum over its substitutions";
emphasis label in green: "untrained on the assay — leakage-free at held-out positions".

Lower branch "Per-residue embeddings, 201 × 1280": then "PCA → 16 dimensions per position";
note "concatenated to the residue embedding as per-position side information".

Footer strip: "Control — wild-type recovery: 80.3 % on human ubiquitin, 11.8 % on the isolated
domain, 17.6 % on the same positions inside full spike."
```

## Figure 3 — The two evaluation designs

```
A three-panel landscape scientific figure, title "Two evaluation designs on one saturating
mutational library". Panels a, b, c left to right, b the widest.

PANEL a — genomic context, drawn as a real annotated gene track.
A horizontal SARS-CoV-2 spike track, residues 1 to 1273, drawn to scale, with domain blocks
labelled exactly: "NTD 14-305", "RBD 331-531", "S1/S2 cleavage 685", "HR1", "CH",
"TM 1214-1237". The RBD block filled steel blue #0072B2, every other block pale grey #D9D9D9.
Two guide lines expand the RBD downward into a wider bar labelled
"201 positions, Wuhan-Hu-1 background", carrying a residue ruler ticked every 20 residues from
331 to 531 and small ticks marking 417, 453, 477, 484, 498, 501 labelled "sites mutated in
Variants of Concern".
Caption strip under panel a, small type:
"124,197 unique amino-acid genotypes - 105,526 with ACE2 binding - 116,258 with expression -
2,029 single substitutions with antibody escape".

PANEL b — the two designs as variant x position matrices, stacked vertically.
Rows are variants, columns are positions; a filled cell means that variant mutates that
position. Draw 14 rows and 16 columns, two filled cells per row. Column headers are a short
residue ruler, row label on the left reads "variants", column label below reads "positions".

Top matrix, header bar "Random split - 80 / 10 / 15 % of variants":
three whole rows filled vermillion #D55E00 (test), one row purple #CC79A7 (validation), the
rest sky #56B4E9 (train). A bracket spanning all 16 columns, annotated
"every column appears in training".

Bottom matrix, header bar "Held-out-position split - 75 / 10 / 15 % of positions":
a dashed vermillion rectangle around the four rightmost columns, labelled
"withheld positions". Rows whose both filled cells lie inside the rectangle are vermillion.
Rows with neither cell inside are sky blue. Two rows with one cell on each side are drawn as
hollow outlined cells with a diagonal hatch, labelled "spans the split, discarded".
A bracket under the withheld columns annotated
"no training variant mutates any withheld position - asserted in the test suite".

Shared legend beneath both matrices: train (sky #56B4E9), validation (purple #CC79A7),
test (vermillion #D55E00), discarded (hollow, hatched).

PANEL c — the same two designs painted onto the folded domain.
Two semi-transparent three-dimensional cartoon ribbons of the SARS-CoV-2 receptor-binding
domain, both in the same orientation, soft depth shading and a faint ambient-occlusion edge, no
hard specular highlight. Behind the upper right of each ribbon, a pale grey molecular surface
labelled "ACE2".
Upper ribbon, titled "Random split": residues coloured in a fine, evenly scattered mix of sky
blue and vermillion so no contiguous region stands out. Annotation with a thin leader line:
"every position is represented in training; a test variant differs only in which substitutions
it combines".
Lower ribbon, titled "Held-out-position split": the withheld 15 % of residues coloured
vermillion, falling as three contiguous surface patches and a few isolated residues, the rest
sky blue. Annotation: "a withheld position has no parameter in any position-indexed model".
A small shared colour key: "sky, positions seen in training - vermillion, positions withheld".

FOOTER STRIP, full width, pale green #009E73 outline, three compartments:
Left, titled "ProteinGym's own position-based schemes, reconstructed here": two small
sixteen-tooth combs over a position ruler. First comb labelled "fold_modulo_5 - fold = position
mod 5" with every fifth tooth filled. Second labelled "fold_contiguous_5 - five consecutive
blocks" with one contiguous fifth filled.
Middle, titled "Untrained controls, evaluated on the identical partitions":
"-n_substitutions", "position mean (no residue identity)", "BLOSUM62 exchangeability",
"ESM-2 masked marginals".
Right, titled "Replication": "10 seeds - every model within a seed sees the identical
partition, so every comparison is paired".

HARD CONSTRAINTS.
Reproduce every label and number exactly as written and invent nothing.
This figure shows experimental design only. It contains NO measured results, so do NOT add
any of the following: a training or validation loss curve, an accuracy or correlation bar
chart, a scatter plot, a precision-recall or ROC curve, a confusion matrix, a phylogenetic tree
or bootstrap values, a PCA or UMAP embedding, a clustered or structure-based third split, a
heat map of effects, a performance table, or any number not listed above.
No decorative icons, no drop shadows, no photographic texture, no gradient backgrounds, no
beaker or DNA-helix clip art. Colour is never the only cue: repeat it with a direct text label.
Colour-blind-safe palette only: steel blue #0072B2, teal #009E73, vermillion #D55E00,
purple #CC79A7, sky #56B4E9, grey #6E6E6E on white.
Sans-serif labels (Arial or Helvetica), 7-9 pt, bold lower-case panel letters a, b, c at each
panel's top left. Hairline axes, no top or right spines.
```

## Figure 4 — EvoClassNet backbone

```
A left-to-right chain of blocks, then a branch down to three heads.

Chain, with these exact labels:
  "Variant sequence, 201 residues (spike 331–531), integer coded"
  "Residue embedding 48-dim + learned positional embedding"
  "Conv1d 64 channels, kernel 5 · BatchNorm · ReLU · max-pool /2 → 100 positions"
  "Conv1d 96 channels, kernel 7 · BatchNorm · ReLU · max-pool /2 → 50 positions"
  "Conv1d 64 channels, kernel 5 · BatchNorm · ReLU → 50 positions"
  "BiLSTM, 96 per direction → 192-dim"
  "4-head self-attention, residual + LayerNorm"

Then down to "Pooling: learned attention pooling ‖ max pooling → 384-dim",
then three parallel heads in vermillion: "ACE2 binding head · MLP 128",
"Expression head · MLP 128", "Antibody escape head · MLP 128".

Right-hand note: "Masked Huber loss — a genotype missing one phenotype still trains the others.
Task weights are learned homoscedastic uncertainties, so the 2,029 escape labels are not
drowned by the two larger sets."

Footer strip: "521,815 parameters, trained on four CPU cores. Optional per-position side
information: 8 structural descriptors and 16 ESM-2 PCA components, concatenated before the
first convolution."
```

## Figure 5 — Genome classification design

```
A four-panel landscape scientific figure, title "A second data type: coronavirus genus
classification from whole genomes".

PANEL a — four real genome maps, drawn to scale against a shared nucleotide axis.
Four horizontal genome tracks stacked, sharing one axis from 0 to 32 kb ticked every 5 kb. Each
track drawn at its measured median length and labelled with that length:
  "Alphacoronavirus - 285 genomes - median 28,141 nt"
  "Betacoronavirus - 371 genomes - median 30,085 nt"
  "Gammacoronavirus - 71 genomes - median 27,684 nt"
  "Deltacoronavirus - 79 genomes - median 25,834 nt"
On each track, gene blocks in the canonical coronavirus order, sized proportionally and
labelled where they fit: "ORF1a", "ORF1b", "S", "E", "M", "N", with accessory ORFs as small
unlabelled blocks. ORF1ab in grey #D9D9D9, S in steel blue #0072B2, the structural genes E, M,
N in teal #009E73, accessory ORFs in pale purple. A 5' cap symbol at the left end and a
"3' poly(A)" label at the right end of each track.
A vertical dashed vermillion line at 4,000 nt across all four tracks, labelled
"1D CNN input window: first 4,000 nt".
Note beneath panel a: "SARS-CoV-2 excluded from Betacoronavirus - its millions of deposited
genomes would make the class a measure of one outbreak's sequencing effort, not of a genus."

PANEL b — how a genome becomes features.
A left-to-right chain. First, a short stretch of real-looking nucleotide sequence in monospace,
about 40 bases, with a 5-base sliding window outlined in vermillion shown at three successive
offsets, labelled "k = 5, stride 1". Then an arrow to a 1,024-bin histogram sketch labelled
"5-mer frequency vector, 4^5 = 1,024 features, normalised to the genome length". Below it, a
second, much smaller branch: a two-element box labelled "untrained control: genome length +
base composition (A, C, G, T fractions) - 5 features, no sequence model".

PANEL c — the two splits, drawn over a collapsed species tree.
A compact ladder-style cladogram of coronavirus species with four labelled clades matching the
genera above, 287 tips collapsed into triangles whose widths are proportional to the class
sizes. No branch lengths, no support values, no scale bar - this is a schematic of the grouping
structure, not an inferred phylogeny, and it should be labelled as such:
"species grouping, schematic - 287 species".
Upper copy, header "Random split": tips shaded in a fine mix of sky #56B4E9 and vermillion, so
both colours occur inside every triangle. Annotation: "genomes of the same species fall on both
sides".
Lower copy, header "Held-out-species split": whole triangles are either sky blue or vermillion,
never mixed. Annotation: "every genome of a species lies on one side only - the genome-level
analogue of withholding positions".

PANEL d — what is compared, with no results shown.
A single column of model labels in evaluation order, each as a small box, grouped and bracketed:
Bracket "untrained controls": "Majority class", "Genome length + base composition".
Bracket "sequence models": "5-mer + logistic regression", "5-mer + gradient boosting",
"5-mer + random forest", "1D CNN over the first 4,000 nt".
To the right of the column, a vertical label: "balanced accuracy, macro F1, MCC, log loss -
10 seeds, paired within seed".

FOOTER STRIP, full width, teal #009E73 outline:
"Genome length and base composition need no training and no sequence model. They are the score
any reported accuracy must be read against."

HARD CONSTRAINTS.
Reproduce every label and number exactly as written and invent nothing.
This figure shows experimental design only. It contains NO measured results, so do NOT add any
of the following: an accuracy or balanced-accuracy bar chart, a confusion matrix, a
precision-recall or ROC curve, a training or validation loss curve, a PCA, t-SNE or UMAP
scatter of k-mer space, branch lengths or bootstrap support values on the cladogram, a
per-class performance table, or any number not listed above. In particular do not label the
cladogram with support values and do not draw it as a radial or circular tree.
No decorative icons, no drop shadows, no photographic texture, no gradient backgrounds, no
virus particle or DNA-helix clip art. Colour is never the only cue: repeat it with a direct
text label.
Colour-blind-safe palette only: steel blue #0072B2, teal #009E73, vermillion #D55E00,
purple #CC79A7, sky #56B4E9, grey #6E6E6E on white.
Sans-serif labels (Arial or Helvetica), 7-9 pt, bold lower-case panel letters a, b, c, d at
each panel's top left. Hairline axes, no top or right spines.
```

---

## Figure 6 — What each evaluation protocol measures

```
Four panels. Panels a–c across the top, panel d full width below.

a — the two-matrix schematic exactly as described in Figure M3 above, drawn small.

b — scatter, titled "Random split, 216 assays". x "position-only model ρ", y "chemistry model ρ",
both axes 0 to 1.0 with a dashed 1:1 diagonal, grey points. Annotation: "chemistry ahead in
25 % of assays".

c — same axes, titled "ProteinGym fold_modulo_5, 214 assays", steel blue points forming a
vertical line at x = 0 spanning y = 0 to 0.68. Annotation: "chemistry ahead in 100 % of assays".

d — horizontal interval plot. Six model rows, three coloured bars each (grey = random split,
steel blue = fold_modulo_5, green = fold_contiguous_5). Plot the median as a white tick inside
a thick interquartile bar. Medians, in the order random / modulo / contiguous:

  Mutation count         0.000  0.000  0.000
  BLOSUM62               0.220  0.203  0.207
  Additive ridge         0.000  0.000  0.000
  Position mean          0.665  0.000  0.000
  Residue identity       0.402  0.252  0.251
  Chemistry ridge        0.455  0.309  0.314
  Chemistry + structure  0.590  0.474  0.429

Colour the row labels by family: Mutation count and BLOSUM62 green; Additive ridge and Position
mean vermillion; the rest steel blue.
x axis "Spearman ρ per assay", range −0.12 to 1.0.
Title: "Every ProteinGym substitution assay under three evaluation protocols".
```

## Figure 7 — The published leaderboard audited

```
Two panels.

a — horizontal grouped bars, "fraction of assays the published model wins" against a
position-only baseline, 0 to 1.0, dashed line at 0.5. Three bars per model
(grey random split, steel blue fold_modulo_5, green fold_contiguous_5):

  One-Hot Encodings                    0.264  0.603  0.601
  DeepSequence + One-Hot Encodings     0.278  0.995  0.995
  MSA_Transformer + One-Hot Encodings  0.282  0.986  0.990
  TranceptEVE + One-Hot Encodings      0.306  0.995  0.995
  Tranception + One-Hot Encodings      0.306  0.986  0.995
  ESM-1v + One-Hot Encodings           0.352  0.977  0.974
  ESM-1v Embeddings                    0.602  0.995  0.984
  MSA Transformer Embeddings           0.694  0.995  1.000
  Tranception Embeddings               0.736  0.995  0.995
  ProteinNPT                           0.903  0.995  1.000
  Kermut                               0.940  1.000  1.000

b — slope chart, two x positions labelled "random split" and "fold_modulo_5", y "mean Spearman
ρ across assays". One line per model, purple for published models (solid), orange for the
baselines added here (dashed, label suffixed with *). Right-hand labels spread vertically with
hairline leaders so none overlap.

  Kermut                               0.745 → 0.633
  ProteinNPT                           0.741 → 0.588
  MSA Transformer Embeddings           0.670 → 0.562
  Tranception Embeddings               0.681 → 0.525
  ESM-1v Embeddings                    0.614 → 0.514
  TranceptEVE + One-Hot Encodings      0.552 → 0.435
  Chemistry + structure*               0.586 → 0.449
  Tranception + One-Hot Encodings      0.550 → 0.413
  MSA_Transformer + One-Hot Encodings  0.540 → 0.409
  ESM-1v + One-Hot Encodings           0.565 → 0.394
  DeepSequence + One-Hot Encodings     0.523 → 0.393
  Chemistry ridge*                     0.493 → 0.318
  Residue identity*                    0.455 → 0.269
  BLOSUM62*                            0.209 → 0.202
  One-Hot Encodings                    0.582 → 0.022
  Additive ridge*                      0.220 → 0.000
  Position mean*                       0.630 → 0.000
```

## Figure 8 — Effect of adding a predicted structure

```
Two panels.

a — histogram, steel blue, about 28 bins, of the per-assay change in Spearman ρ at held-out
positions when AlphaFold descriptors are added to the chemistry block. x axis
"Δ Spearman ρ from adding structure", y axis "assays". A hairline at zero and a vermillion
vertical line at the median, +0.114. Title: "Per-assay effect of an AlphaFold model —
132 assays, 88 % improved".

b — scatter, x "AlphaFold pLDDT of the model", y "Δ Spearman ρ", steel blue points, a hairline
at zero and a thin linear fit. Annotation: "ρ = +0.10, P = 0.24".
Title: "Does the gain track prediction confidence?"

Footer values for the caption, four splits as
n assays / chemistry / chemistry+structure / median gain / fraction improved:
  Random split        135   0.455   0.590   +0.079   98.5 %
  Held-out positions  132   0.316   0.472   +0.114   87.9 %
  fold_modulo_5       133   0.308   0.474   +0.124   90.2 %
  fold_contiguous_5   116   0.315   0.430   +0.106   85.3 %
```

## Figure 9 — The receptor-binding-domain panel

```
Two panels, shared model order top to bottom. Colour the labels by family: green for Mutation
count, BLOSUM62, ESM-2 zero-shot; vermillion for Additive ridge, Global epistasis, EvoClassNet,
EvoClassNet + ESM/str; steel blue for Structure GNN, k-NN, Gradient boosting, Physicochemical
ridge.

a — horizontal grouped bars with error bars, "Spearman ρ, ACE2 binding".
Grey = random split, steel blue = held-out positions. Values mean ± SD:

  Mutation count          0.478 ± 0.005   0.386 ± 0.047
  BLOSUM62                0.417 ± 0.007   0.410 ± 0.032
  ESM-2 zero-shot         0.075 ± 0.004  −0.022 ± 0.084
  Additive ridge          0.874 ± 0.004   0.000 ± 0.000
  Global epistasis        0.887 ± 0.005  −0.370 ± 0.042
  EvoClassNet             0.890 ± 0.011   0.341 ± 0.118
  Structure GNN           0.733 ± 0.042   0.332 ± 0.172
  EvoClassNet + ESM/str   0.895 ± 0.004   0.449 ± 0.102
  k-NN (features)         0.646 ± 0.007   0.602 ± 0.052
  Gradient boosting       0.746 ± 0.004   0.698 ± 0.052
  Physicochemical ridge   0.726 ± 0.004   0.711 ± 0.053

b — forest plot, "partial Spearman ρ (substitution count removed)", held-out positions only.
Point with a 95 % interval; filled point where the interval excludes zero, open where it does
not:

  Mutation count          0.000  (zero by construction, no interval)
  BLOSUM62               +0.318  [+0.298, +0.338]  filled
  ESM-2 zero-shot        −0.052  [−0.095, −0.005]  open
  Additive ridge          0.000  [ 0.000,  0.000]  open
  Global epistasis       −0.015  [−0.059, +0.030]  open
  EvoClassNet            +0.188  [+0.104, +0.262]  filled
  Structure GNN          +0.259  [+0.164, +0.353]  filled
  EvoClassNet + ESM/str  +0.339  [+0.268, +0.401]  filled
  k-NN (features)        +0.530  [+0.485, +0.572]  filled
  Gradient boosting      +0.635  [+0.592, +0.678]  filled
  Physicochemical ridge  +0.648  [+0.600, +0.690]  filled
```

## Figure 10 — Where the transferable signal comes from

```
Two panels.

a — horizontal bars with error bars, "partial Spearman ρ, ACE2 binding, held-out positions",
mean ± SD over ten seeds. Colour the first bar green (untrained), the rest steel blue:

  substitution count only           −0.000 ± 0.001
  ESM-2 score only                  −0.004 ± 0.091
  residue chemistry                 +0.371 ± 0.060
  structure, no interface           +0.518 ± 0.095
  structure                         +0.554 ± 0.112
  chemistry + structure             +0.642 ± 0.085
  chemistry + structure + ESM-2     +0.650 ± 0.074

b — grouped vertical bars, "median per-position ρ" within a single withheld position.
Steel blue = transferable model, green = BLOSUM62. Significance marks above each pair:

  ACE2 binding      model +0.533   BLOSUM62 +0.528   n.s.
  Expression        model +0.510   BLOSUM62 +0.561   n.s.
  Antibody escape   model +0.570   BLOSUM62 −0.350   ***

Note that the BLOSUM62 bar for escape is negative and drops below the zero line.
```

## Figure 11 — Controls

```
Two panels.

a — grouped vertical bars, y axis "fraction", 0 to 1.0. Purple = wild-type recovery,
light grey = median top-token probability. Print the recovery percentage above each purple bar:

  ubiquitin (positive control)   recovery 0.803   top-token probability 0.796
  RBD alone                      recovery 0.118   top-token probability 0.096
  RBD inside full spike          recovery 0.176   top-token probability 0.099

Title: "ESM-2 works, but not on this domain".

b — horizontal grouped bars, "balanced accuracy", 0 to 1.05, dashed line at 0.25 labelled
"chance". Grey = random split, steel blue = held-out species. Values mean ± SD:

  Majority class                     0.250 ± 0.000   0.250 ± 0.000
  Genome length + base composition   0.953 ± 0.015   0.914 ± 0.037
  5-mer + logistic regression        1.000 ± 0.000   0.998 ± 0.004
  5-mer + gradient boosting          0.995 ± 0.009   0.976 ± 0.026
  5-mer + random forest              0.993 ± 0.013   0.962 ± 0.034
  1D CNN on nucleotides              0.992 ± 0.009   0.988 ± 0.017

Title: "Coronavirus genus classification".
```

## Figure 12 — Confusion matrices, held-out-species split

```
Three square row-normalised confusion matrices side by side, sequential orange colour map from
0 to 1, each cell annotated with its value to two decimals, white text on dark cells.
Axes labelled "true" and "predicted", tick labels Alpha, Beta, Gamma, Delta.
Panel titles: "Genome length + base composition", "5-mer + logistic regression",
"1D CNN on nucleotides".
Overall title: "Held-out-species split, pooled over seeds (row-normalised)".
Take the cell values from results/table_genomes.csv and the per-run confusion matrices in
results/genome_study.json; do not estimate them.
```
