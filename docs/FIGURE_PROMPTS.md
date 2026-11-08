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
Two stacked matrix diagrams. Rows are variants, columns are positions. A filled cell means that
variant mutates that position; each variant has two filled cells.

Top, titled "Random split": three whole rows coloured vermillion (test), the rest sky blue
(train).

Bottom, titled "Held-out-position split": a dashed vermillion rectangle around the four
rightmost columns labelled "withheld positions". Rows whose every filled cell lies inside the
rectangle are vermillion. Rows with no cell inside are sky blue. One row with cells on both
sides is drawn as an empty outlined cell, labelled "spans the split, discarded".

Legend: train (sky blue), test (vermillion), discarded (outline).
Axis note: "rows, variants — columns, positions".

Footer strip in green: "Untrained controls evaluated on the identical partitions:
−n_substitutions · BLOSUM62 exchangeability · ESM-2 masked marginals".

Side note: "ProteinGym also defines two position-based schemes, fold_modulo_5 and
fold_contiguous_5, both reconstructed here."
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
Left box "NCBI nucleotide": "806 complete genomes, 287 species, four genera".
Class sizes listed: Alphacoronavirus 285, Betacoronavirus 371, Gammacoronavirus 71,
Deltacoronavirus 79. Note: "SARS-CoV-2 excluded from Betacoronavirus — its millions of
deposited genomes would make the class a measure of one outbreak's sequencing effort."

Two branches: "Random split" (grey) and "Held-out-species split — no species on both sides"
(steel blue).

Right box "Models": majority class; genome length + base composition (untrained);
5-mer frequencies + logistic regression / gradient boosting / random forest;
1D CNN over the first 4,000 nucleotides.

Footer strip in green: "Genome length and base composition need no training and no sequence
model. They are the score any reported accuracy must be read against."
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
