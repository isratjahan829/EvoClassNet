# What a held-out-position split actually measures

Every number here is read out of `results/` by `scripts/build_readme2.py`.

## The finding, in three steps

**1. A hard split needs an untrained control.** On a combinatorial mutagenesis library, ranking variants by `−n_substitutions` — no training, no parameters, no idea which positions are involved — already scores well, because most substitutions are deleterious and their effects accumulate. A model that learned nothing about the withheld positions inherits that score. Better still, report the **partial Spearman** correlation, which removes substitution count from both sides: it measures position-level signal directly, and it validates itself, because the trivial baseline must score exactly zero on it. It does: -0.000.

**2. Sequence models indexed by position collapse there.** The additive model has no parameter for an unseen position and emits a constant; global epistasis goes anti-correlated; the neural network keeps a positive raw ρ that is almost entirely the substitution-count effect.

**3. But that is a fact about the representation, not about the data — and this corrects an earlier version of this repository.** Every model in step 2 encodes a substitution by *where* it occurs, so none holds a parameter that applies to a held-out position. Give a plain ridge regression features that survive at an unseen position — how the residue changes chemically, and descriptors of the position taken from the crystal structure — and extrapolation is substantial:

| Features | binding ρ | binding partial ρ | expression partial ρ | escape ρ |
|---|---|---|---|---|
| Substitution count only (untrained) | +0.386 ± 0.047 | -0.000 ± 0.001 | +0.000 ± 0.001 | +0.000 ± 0.000 |
| ESM-2 zero-shot only | +0.340 ± 0.067 | -0.004 ± 0.091 | -0.058 ± 0.097 | +0.108 ± 0.050 |
| Residue chemistry only | +0.508 ± 0.046 | +0.371 ± 0.060 | +0.257 ± 0.091 | +0.340 ± 0.123 |
| Structure only | +0.644 ± 0.075 | +0.554 ± 0.112 | +0.485 ± 0.101 | +0.470 ± 0.079 |
| Structure, interface descriptors dropped | +0.615 ± 0.061 | +0.518 ± 0.095 | +0.477 ± 0.094 | +0.361 ± 0.148 |
| Chemistry + structure | +0.706 ± 0.059 | +0.642 ± 0.085 | +0.494 ± 0.088 | +0.597 ± 0.071 |
| Chemistry + structure + ESM-2 | +0.712 ± 0.051 | +0.650 ± 0.074 | +0.506 ± 0.082 | +0.600 ± 0.069 |

Held-out-position split, 10 seeds. Escape has no partial column: every escape-labelled variant carries one substitution, so the confound is absent there by construction.

Two things to note. **Structure carries most of it** (+0.554 ± 0.112 partial for binding against +0.371 ± 0.060 for chemistry), and it is not an artefact of knowing the receptor interface: the two descriptors derived from the ACE2-bound complex mark exactly the contact residues, which is nearly circular for binding, and dropping them leaves +0.518 ± 0.095. **ESM-2 contributes nothing on this protein** — its raw +0.340 ± 0.067 is the substitution-count effect in full, with -0.004 ± 0.091 surviving the control.

## Where the transfer comes from: positions, not substitutions

A narrower question separates two components. Within a *single* withheld position, can the model order the nineteen substitutions available there?

| Phenotype | Model median ρ | % positive | BLOSUM62 median ρ | Difference | P | n position–seed |
|---|---|---|---|---|---|---|
| ACE2 binding | +0.533 | 92 | +0.528 | +0.005 | 0.407 | 300 |
| Expression | +0.510 | 90 | +0.561 | +0.003 | 0.926 | 300 |
| Antibody escape | +0.570 | 96 | -0.350 | +0.869 | 8.65e-30 | 178 |

For binding and expression the model matches an **untrained substitution matrix** (+0.533 against +0.528, P = 0.407): within a position, the ordering is chemistry and needs no model. The model's real advantage is in telling **which positions** tolerate substitution at all, and that is what the structural descriptors supply — a position-independent matrix cannot.

Escape runs the other way. BLOSUM62 is not merely uninformative there but anti-correlated (-0.350), because the substitutions that abolish antibody binding are not the chemically radical ones, while the model reaches +0.570 (difference +0.869, P = 8.65e-30). That within-position ordering is genuinely learned.

## Second task: coronavirus genus classification

806 complete genomes from 287 species across four genera, pulled from NCBI. SARS-CoV-2 is excluded from Betacoronavirus: with millions of deposited genomes against hundreds for whole genera, it would make the class a measure of one outbreak's sequencing effort. The same question is asked — a random split against a held-out-species split, with an untrained control.

| Split | Family | Model | Seeds | Accuracy | Balanced acc. | Macro F1 | MCC |
|---|---|---|---|---|---|---|---|
| random | untrained | Majority class | 10 | 0.457 ± 0.044 | 0.250 ± 0.000 | 0.157 ± 0.010 | 0.000 ± 0.000 |
| random | trivial | Genome length + base composition | 10 | 0.950 ± 0.010 | 0.953 ± 0.015 | 0.941 ± 0.016 | 0.924 ± 0.016 |
| random | sequence | 5-mer + logistic regression | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| random | sequence | 5-mer + random forest | 10 | 0.997 ± 0.006 | 0.993 ± 0.013 | 0.995 ± 0.009 | 0.995 ± 0.009 |
| random | sequence | 5-mer + gradient boosting | 10 | 0.998 ± 0.004 | 0.995 ± 0.009 | 0.997 ± 0.006 | 0.996 ± 0.006 |
| random | sequence | 1D CNN on nucleotides | 10 | 0.990 ± 0.015 | 0.992 ± 0.009 | 0.992 ± 0.010 | 0.985 ± 0.022 |
| heldout_species | untrained | Majority class | 10 | 0.497 ± 0.077 | 0.250 ± 0.000 | 0.165 ± 0.017 | 0.000 ± 0.000 |
| heldout_species | trivial | Genome length + base composition | 10 | 0.894 ± 0.044 | 0.914 ± 0.037 | 0.891 ± 0.038 | 0.843 ± 0.062 |
| heldout_species | sequence | 5-mer + logistic regression | 10 | 0.999 ± 0.003 | 0.998 ± 0.004 | 0.997 ± 0.007 | 0.998 ± 0.005 |
| heldout_species | sequence | 5-mer + random forest | 10 | 0.969 ± 0.038 | 0.962 ± 0.034 | 0.968 ± 0.031 | 0.953 ± 0.053 |
| heldout_species | sequence | 5-mer + gradient boosting | 10 | 0.986 ± 0.017 | 0.976 ± 0.026 | 0.982 ± 0.021 | 0.978 ± 0.026 |
| heldout_species | sequence | 1D CNN on nucleotides | 10 | 0.986 ± 0.018 | 0.988 ± 0.017 | 0.985 ± 0.019 | 0.978 ± 0.026 |

Genome length and base composition need no training and no sequence model. They are the score any reported accuracy on this task has to be read against.

## Run-to-run instability

* partition seed 0, varying only the weight initialisation: +0.374, +0.423, +0.387  (range 0.050)
* partition seed 2, varying only the weight initialisation: +0.236, +0.558, +0.385  (range 0.322)

Spread across all 6 runs: +0.236 to +0.558. A single run on this benchmark is not a measurement.

## Data

| Source | Use |
|---|---|
| Starr et al. 2020 RBD yeast-display DMS | binding and expression; barcodes sharing a genotype collapsed to the mean so one sequence cannot straddle a split |
| Bloom lab aggregated RBD escape maps | per-substitution escape over ≥ 20 antibodies, log10; rows measured in a non-Wuhan background dropped, since those positions are exactly the VOC ones |
| PDB 6M0J (RBD–ACE2 complex) | solvent accessibility bound and unbound, interface burial, distance to ACE2, contact number, burial, B-factor, contact graph |
| UniProt P0DTC2 | full spike sequence, for scoring the RBD in its native context |
| NCBI nucleotide | coronavirus genomes for the classification task |
| facebook/esm2_t33_650M_UR50D | masked-marginal substitution scores and per-residue embeddings |

## What is still missing

* One protein, one genetic background, two assay platforms. The evaluation lesson is general; that structure carries the transfer is shown only here.
* The structural descriptors come from a solved complex. For a target without one they would have to be predicted, and that substitution is untested.
* One language model at one size with one scoring scheme. Fine-tuning on viral sequences was not tried.
* Hyper-parameters were tuned on random-split validation, which if anything understates the trained models at held-out positions.
* The genome task is a demonstration that the lesson transfers, not a competitive classifier.

## Layout

```
evoclassnet/data.py        genotype loading, encoding, both split protocols
evoclassnet/escape.py      per-substitution escape labels and background filtering
evoclassnet/structure.py   per-residue features and contact graph from PDB 6M0J
evoclassnet/plm.py         ESM-2 masked marginals and per-residue embeddings
evoclassnet/zoo.py         transferable feature block, untrained scorers, GNN
evoclassnet/genomes.py     genome loading, k-mer features, species-aware splits
evoclassnet/model.py       EvoClassNetLite, GlobalEpistasis, seeded build(), fit()
evoclassnet/baselines.py   the mutation-count baseline
evoclassnet/metrics.py     twelve-metric panel including the partial correlation
evoclassnet/stats.py       TOST equivalence, FDR, bootstrap, Cliff's delta, mixed effects
scripts/run_study2.py      eleven models x two splits, cached per run
scripts/feature_ablation.py  which feature block carries the transfer
scripts/analyze2.py        tables, statistics, figures
scripts/make_schematics.py architecture figures drawn from the real configuration
scripts/build_docx2.py     manuscript, generated from results/
scripts/build_readme2.py   this file, generated from results/
```

## Reproduce

```bash
./run_all.sh
```
