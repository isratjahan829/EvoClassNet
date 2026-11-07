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

That null result was checked rather than assumed: ESM-2 wild-type recovery was 80.3% on human ubiquitin, 11.8% on the isolated RBD and 17.6% on the same positions inside the full spike protein, with no weights missing at load time.

## It is not one protein: 217 ProteinGym assays

Two baselines make the point unambiguous. A **position-mean** model is a ridge regression on position indicators alone — it knows which position a variant mutates and nothing at all about which substitution is made. A **residue-identity** model is its mirror image.

| Split | Family | Model | Assays | Median ρ | Q1 | Q3 | Frac ρ>0.2 | Frac ρ=0 |
|---|---|---|---|---|---|---|---|---|
| Random split | untrained | Mutation count | 216 | 0.0 | 0.0 | 0.178 | 0.245 | 0.681 |
| Random split | untrained | BLOSUM62 | 216 | 0.22 | 0.134 | 0.284 | 0.546 | 0.0 |
| Random split | position-indexed | Position mean | 216 | 0.665 | 0.564 | 0.732 | 0.968 | 0.0 |
| Random split | position-indexed | Additive ridge | 216 | 0.0 | 0.0 | 0.546 | 0.319 | 0.681 |
| Random split | transferable | Residue identity | 216 | 0.402 | 0.293 | 0.632 | 0.944 | 0.0 |
| Random split | transferable | Chemistry ridge | 216 | 0.455 | 0.344 | 0.65 | 0.972 | 0.0 |
| Random split | transferable | Chemistry + structure | 135 | 0.59 | 0.459 | 0.718 | 0.978 | 0.0 |
| Held-out-position split | untrained | Mutation count | 212 | 0.0 | 0.0 | 0.0 | 0.042 | 0.83 |
| Held-out-position split | untrained | BLOSUM62 | 212 | 0.198 | 0.123 | 0.27 | 0.486 | 0.0 |
| Held-out-position split | position-indexed | Position mean | 212 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Held-out-position split | position-indexed | Additive ridge | 212 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Held-out-position split | transferable | Residue identity | 212 | 0.253 | 0.189 | 0.352 | 0.712 | 0.0 |
| Held-out-position split | transferable | Chemistry ridge | 212 | 0.313 | 0.24 | 0.405 | 0.84 | 0.0 |
| Held-out-position split | transferable | Chemistry + structure | 132 | 0.472 | 0.347 | 0.553 | 0.917 | 0.0 |
| ProteinGym fold_modulo_5 | untrained | Mutation count | 214 | 0.0 | 0.0 | 0.0 | 0.061 | 0.818 |
| ProteinGym fold_modulo_5 | untrained | BLOSUM62 | 214 | 0.203 | 0.126 | 0.273 | 0.514 | 0.0 |
| ProteinGym fold_modulo_5 | position-indexed | Position mean | 214 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| ProteinGym fold_modulo_5 | position-indexed | Additive ridge | 214 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| ProteinGym fold_modulo_5 | transferable | Residue identity | 214 | 0.252 | 0.184 | 0.352 | 0.701 | 0.0 |
| ProteinGym fold_modulo_5 | transferable | Chemistry ridge | 214 | 0.309 | 0.237 | 0.411 | 0.827 | 0.0 |
| ProteinGym fold_modulo_5 | transferable | Chemistry + structure | 133 | 0.474 | 0.345 | 0.544 | 0.947 | 0.0 |
| ProteinGym fold_contiguous_5 | untrained | Mutation count | 193 | 0.0 | 0.0 | 0.0 | 0.057 | 0.782 |
| ProteinGym fold_contiguous_5 | untrained | BLOSUM62 | 193 | 0.207 | 0.132 | 0.268 | 0.523 | 0.0 |
| ProteinGym fold_contiguous_5 | position-indexed | Position mean | 193 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| ProteinGym fold_contiguous_5 | position-indexed | Additive ridge | 193 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| ProteinGym fold_contiguous_5 | transferable | Residue identity | 193 | 0.251 | 0.165 | 0.339 | 0.668 | 0.0 |
| ProteinGym fold_contiguous_5 | transferable | Chemistry ridge | 193 | 0.314 | 0.228 | 0.386 | 0.788 | 0.0 |
| ProteinGym fold_contiguous_5 | transferable | Chemistry + structure | 116 | 0.429 | 0.307 | 0.533 | 0.922 | 0.0 |

Under a **random split** the position-mean model has a median per-assay ρ of **0.665**, against 0.455 for chemistry: knowing only *where* beats knowing only *what*. Under the **held-out-position split** it falls to **0.0** — not a degradation but an identity, since it holds no parameter for a position it never saw — while chemistry retains 0.313.

The standard evaluation rewards memorising per-position means. That capability is worth exactly nothing at a position no experiment has probed.

Worth noting: the mutation-count confound that dominates the RBD analysis is mostly *absent* here — only 69 of 217 assays contain any multi-substitution variant. So the separation between representations does not depend on the confound existing.

AlphaFold models were obtained for 132 of the assays. Adding their per-residue descriptors moves the median held-out-position ρ from 0.313 to **0.472**. This is a weaker instrument than the RBD analysis, where the structure is a crystal of the bound complex and supplies descriptors a predicted monomer cannot.

## Where the transfer comes from: positions, not substitutions

A narrower question separates two components. Within a *single* withheld position, can the model order the nineteen substitutions available there?

| Phenotype | Model median ρ | % positive | BLOSUM62 median ρ | Difference | P | n position–seed |
|---|---|---|---|---|---|---|
| ACE2 binding | +0.533 | 92 | +0.528 | +0.005 | 0.407 | 300 |
| Expression | +0.510 | 90 | +0.561 | +0.003 | 0.926 | 300 |
| Antibody escape | +0.570 | 96 | -0.350 | +0.869 | 8.65e-30 | 178 |

For binding and expression the model matches an **untrained substitution matrix** (+0.533 against +0.528, P = 0.407): within a position, the ordering is chemistry and needs no model. The model's real advantage is in telling **which positions** tolerate substitution at all, and that is what the structural descriptors supply — a position-independent matrix cannot.

Escape runs the other way. BLOSUM62 is not merely uninformative there but anti-correlated (-0.350), because the substitutions that abolish antibody binding are not the chemically radical ones, while the model reaches +0.570 (difference +0.869, P = 8.65e-30). That within-position ordering is genuinely learned.

## Full metric panel

**Random genotype split** — ACE2 binding (`results/table_panel.csv` has all phenotypes and metrics)

| Family | Model | Seeds | Spearman ρ | Partial ρ | Pearson r | Kendall τ | R² | RMSE | AUROC | Prec@k | NDCG | Calib. | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | Mutation count | 5 | 0.478 ± 0.005 | -0.001 ± 0.001 | 0.470 ± 0.005 | 0.374 ± 0.005 | 0.151 ± 0.010 | 1.760 ± 0.012 | 0.735 ± 0.005 | 0.012 ± 0.018 | 0.770 ± 0.019 | 0.689 ± 0.015 | 10540 |
| untrained | BLOSUM62 | 5 | 0.417 ± 0.007 | 0.234 ± 0.008 | 0.420 ± 0.006 | 0.306 ± 0.005 | -1.821 ± 0.047 | 3.206 ± 0.028 | 0.712 ± 0.002 | 0.040 ± 0.020 | 0.692 ± 0.028 | 0.253 ± 0.005 | 10540 |
| untrained | ESM-2 zero-shot | 5 | 0.075 ± 0.004 | -0.005 ± 0.010 | 0.081 ± 0.004 | 0.051 ± 0.003 | -1.750 ± 0.041 | 3.166 ± 0.016 | 0.539 ± 0.003 | 0.012 ± 0.018 | 0.231 ± 0.027 | 0.110 ± 0.006 | 10540 |
| transferable | Physicochemical ridge | 5 | 0.726 ± 0.004 | 0.622 ± 0.006 | 0.689 ± 0.005 | 0.539 ± 0.004 | 0.475 ± 0.007 | 1.383 ± 0.011 | 0.863 ± 0.004 | 0.048 ± 0.018 | 0.937 ± 0.004 | 1.001 ± 0.016 | 10540 |
| transferable | k-NN (features) | 5 | 0.646 ± 0.007 | 0.533 ± 0.008 | 0.655 ± 0.006 | 0.471 ± 0.005 | 0.311 ± 0.008 | 1.585 ± 0.010 | 0.819 ± 0.005 | 0.036 ± 0.030 | 0.945 ± 0.007 | 1.162 ± 0.010 | 10540 |
| transferable | Gradient boosting | 5 | 0.746 ± 0.004 | 0.653 ± 0.006 | 0.775 ± 0.003 | 0.559 ± 0.004 | 0.599 ± 0.004 | 1.208 ± 0.010 | 0.873 ± 0.004 | 0.044 ± 0.026 | 0.950 ± 0.004 | 1.031 ± 0.005 | 10540 |
| transferable | Structure GNN | 5 | 0.733 ± 0.042 | 0.646 ± 0.052 | 0.760 ± 0.045 | 0.549 ± 0.039 | 0.459 ± 0.048 | 1.403 ± 0.063 | 0.864 ± 0.019 | 0.044 ± 0.033 | 0.930 ± 0.033 | 0.884 ± 0.027 | 10540 |
| position-indexed | Additive ridge | 5 | 0.874 ± 0.004 | 0.834 ± 0.006 | 0.806 ± 0.005 | 0.714 ± 0.005 | 0.649 ± 0.008 | 1.131 ± 0.013 | 0.933 ± 0.004 | 0.204 ± 0.064 | 0.966 ± 0.003 | 0.984 ± 0.015 | 10540 |
| position-indexed | Global epistasis | 5 | 0.887 ± 0.005 | 0.850 ± 0.006 | 0.935 ± 0.001 | 0.743 ± 0.006 | 0.875 ± 0.003 | 0.676 ± 0.007 | 0.939 ± 0.004 | 0.284 ± 0.075 | 0.975 ± 0.005 | 0.996 ± 0.005 | 10540 |
| position-indexed | EvoClassNet | 5 | 0.890 ± 0.011 | 0.857 ± 0.013 | 0.958 ± 0.021 | 0.748 ± 0.022 | 0.909 ± 0.059 | 0.558 ± 0.161 | 0.944 ± 0.009 | 0.160 ± 0.042 | 0.971 ± 0.005 | 0.991 ± 0.005 | 10540 |
| position-indexed | EvoClassNet + ESM/str | 5 | 0.895 ± 0.004 | 0.862 ± 0.005 | 0.967 ± 0.001 | 0.756 ± 0.004 | 0.934 ± 0.003 | 0.489 ± 0.011 | 0.947 ± 0.002 | 0.160 ± 0.037 | 0.971 ± 0.003 | 0.990 ± 0.004 | 10540 |

**Held-out-position split** — ACE2 binding (`results/table_panel.csv` has all phenotypes and metrics)

| Family | Model | Seeds | Spearman ρ | Partial ρ | Pearson r | Kendall τ | R² | RMSE | AUROC | Prec@k | NDCG | Calib. | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | Mutation count | 10 | 0.386 ± 0.047 | -0.000 ± 0.001 | 0.350 ± 0.044 | 0.311 ± 0.038 | 0.091 ± 0.079 | 1.630 ± 0.075 | 0.660 ± 0.026 | 0.046 ± 0.051 | 0.768 ± 0.113 | 0.971 ± 0.158 | 1386 |
| untrained | BLOSUM62 | 10 | 0.410 ± 0.032 | 0.318 ± 0.034 | 0.390 ± 0.028 | 0.299 ± 0.024 | -0.684 ± 0.361 | 2.202 ± 0.125 | 0.709 ± 0.023 | 0.142 ± 0.063 | 0.858 ± 0.033 | 0.325 ± 0.042 | 1386 |
| untrained | ESM-2 zero-shot | 10 | -0.022 ± 0.084 | -0.052 ± 0.077 | -0.012 ± 0.103 | -0.015 ± 0.056 | -0.912 ± 0.250 | 2.366 ± 0.232 | 0.500 ± 0.063 | 0.022 ± 0.039 | 0.556 ± 0.113 | -0.023 ± 0.175 | 1386 |
| transferable | Physicochemical ridge | 10 | 0.711 ± 0.053 | 0.648 ± 0.077 | 0.658 ± 0.045 | 0.518 ± 0.046 | 0.391 ± 0.071 | 1.335 ± 0.100 | 0.859 ± 0.024 | 0.150 ± 0.079 | 0.964 ± 0.015 | 1.256 ± 0.152 | 1386 |
| transferable | k-NN (features) | 10 | 0.602 ± 0.052 | 0.530 ± 0.075 | 0.568 ± 0.057 | 0.427 ± 0.042 | 0.287 ± 0.060 | 1.446 ± 0.120 | 0.803 ± 0.028 | 0.134 ± 0.072 | 0.956 ± 0.010 | 1.094 ± 0.110 | 1386 |
| transferable | Gradient boosting | 10 | 0.698 ± 0.052 | 0.635 ± 0.073 | 0.666 ± 0.061 | 0.506 ± 0.045 | 0.420 ± 0.082 | 1.299 ± 0.081 | 0.858 ± 0.027 | 0.104 ± 0.059 | 0.957 ± 0.009 | 0.909 ± 0.094 | 1386 |
| transferable | Structure GNN | 10 | 0.332 ± 0.172 | 0.259 ± 0.162 | 0.293 ± 0.159 | 0.227 ± 0.121 | -0.047 ± 0.114 | 1.753 ± 0.178 | 0.669 ± 0.079 | 0.048 ± 0.042 | 0.793 ± 0.168 | 3.747 ± 10.138 | 1386 |
| position-indexed | Additive ridge | 10 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — | 2.017 ± 0.241 | 0.500 ± 0.000 | — | — | — | 1386 |
| position-indexed | Global epistasis | 10 | -0.370 ± 0.042 | -0.015 ± 0.075 | -0.350 ± 0.045 | -0.289 ± 0.038 | -0.166 ± 0.217 | 1.849 ± 0.259 | 0.342 ± 0.024 | 0.002 ± 0.006 | 0.361 ± 0.103 | -19.605 ± 16.762 | 1386 |
| position-indexed | EvoClassNet | 10 | 0.341 ± 0.118 | 0.187 ± 0.134 | 0.293 ± 0.129 | 0.234 ± 0.083 | -0.184 ± 0.203 | 1.860 ± 0.219 | 0.664 ± 0.081 | 0.072 ± 0.053 | 0.795 ± 0.082 | 7.257 ± 18.051 | 1386 |
| position-indexed | EvoClassNet + ESM/str | 10 | 0.449 ± 0.102 | 0.339 ± 0.114 | 0.387 ± 0.123 | 0.314 ± 0.076 | -0.143 ± 0.290 | 1.821 ± 0.274 | 0.716 ± 0.066 | 0.084 ± 0.053 | 0.829 ± 0.086 | 2.383 ± 4.285 | 1386 |

## Equivalence testing

A non-significant difference is not evidence of no difference. Each trained model is also tested for *equivalence* to the untrained baseline with two one-sided tests at a ±0.05 margin in Spearman ρ, and P values are FDR-corrected across the whole model-by-task family.

| Phenotype | Model | Median Δρ | CI low | CI high | P (FDR) | TOST P | Equivalent | Cliff's δ | Effect |
|---|---|---|---|---|---|---|---|---|---|
| ACE2 binding | Physicochemical ridge | 0.3088 | 0.2797 | 0.3724 | 0.00601 | 0.999999164248084 | False | 1.0 | large |
| ACE2 binding | k-NN (features) | 0.2137 | 0.1723 | 0.2598 | 0.00601 | 0.9999695051538248 | False | 1.0 | large |
| ACE2 binding | Gradient boosting | 0.3081 | 0.2725 | 0.3544 | 0.00601 | 0.9999995483930094 | False | 1.0 | large |
| ACE2 binding | Structure GNN | -0.0264 | -0.1497 | 0.0347 | 0.33871 | 0.5313516250383241 | False | -0.16 | small |
| ACE2 binding | Additive ridge | -0.3939 | -0.4118 | -0.3573 | 0.00601 | 0.9999999985020248 | False | -1.0 | large |
| ACE2 binding | Global epistasis | -0.7771 | -0.8013 | -0.7051 | 0.00601 | 0.9999999996603752 | False | -1.0 | large |
| ACE2 binding | EvoClassNet | -0.0217 | -0.1215 | 0.0207 | 0.29702 | 0.4462284048570662 | False | -0.22 | small |
| ACE2 binding | EvoClassNet + ESM/str | 0.0703 | 0.0006 | 0.1199 | 0.09454 | 0.6569136834149025 | False | 0.48 | large |
| Expression | Physicochemical ridge | 0.203 | 0.1706 | 0.2693 | 0.00601 | 0.999928016037545 | False | 1.0 | large |
| Expression | k-NN (features) | 0.1488 | 0.0945 | 0.2013 | 0.00601 | 0.9956419163076456 | False | 0.96 | large |
| Expression | Gradient boosting | 0.2363 | 0.2088 | 0.3023 | 0.00601 | 0.9999897622022216 | False | 1.0 | large |
| Expression | Structure GNN | 0.0732 | -0.0953 | 0.0965 | 0.39015 | 0.2555906965152575 | False | 0.36 | medium |
| Expression | Additive ridge | -0.382 | -0.3938 | -0.3373 | 0.00601 | 0.9999999963204804 | False | -1.0 | large |
| Expression | Global epistasis | -0.7401 | -0.7779 | -0.6587 | 0.00601 | 0.9999999966239258 | False | -1.0 | large |
| Expression | EvoClassNet | -0.0177 | -0.0976 | 0.0052 | 0.33871 | 0.4250940411662657 | False | -0.4 | medium |
| Expression | EvoClassNet + ESM/str | 0.0242 | -0.0594 | 0.0823 | 0.56766 | 0.1852968019775392 | False | 0.34 | medium |
| Antibody escape | Physicochemical ridge | 0.5812 | 0.557 | 0.6414 | 0.00601 | 0.9999999990092736 | False | 1.0 | large |
| Antibody escape | k-NN (features) | 0.4058 | 0.3571 | 0.4569 | 0.00601 | 0.9999998550819889 | False | 1.0 | large |
| Antibody escape | Gradient boosting | 0.579 | 0.5204 | 0.6398 | 0.00601 | 0.9999999744333672 | False | 1.0 | large |
| Antibody escape | Structure GNN | 0.2542 | 0.1177 | 0.2848 | 0.01676 | 0.9972316420931856 | False | 0.8 | large |
| Antibody escape | Additive ridge | 0.0 | 0.0 | 0.0 |  |  | False | 0.0 | negligible |
| Antibody escape | Global epistasis | 0.0 | 0.0 | 0.0187 | 0.51244 | 7.399767135154628e-06 | True | 0.2 | small |
| Antibody escape | EvoClassNet | 0.2414 | 0.0602 | 0.2948 | 0.07193 | 0.9675099975291984 | False | 0.6 | large |
| Antibody escape | EvoClassNet + ESM/str | 0.3877 | 0.1304 | 0.4405 | 0.07193 | 0.9923486991895392 | False | 0.8 | large |

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

The study is 15 cached runs in `results/jobs_v2/`, one JSON per run; re-running skips completed jobs.
