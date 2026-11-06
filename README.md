# EvoClassNet — a leakage-controlled benchmark of sequence models for SARS-CoV-2 RBD phenotypes

Every number in this file is read out of `results/` by `scripts/build_readme.py`. Nothing is carried over from the original manuscript draft.

## The finding

Two evaluation protocols on the same deep-mutational-scanning data rank models differently:

* **Random genotype split** — a variant's own mutated positions are also mutated in the training set. All models do well; the neural model is not clearly better than a global-epistasis fit.
* **Held-out-position split** — no training variant mutates any test position. Rank accuracy collapses, and once the trivial **mutation-count baseline** (`−n_substitutions`, untrained) is included in the comparison, no trained model in this study demonstrates position-level extrapolation on binding or expression.
* **The one exception is antibody escape.** Every escape-labelled variant carries exactly one substitution, so the mutation-count baseline is constant there and cannot account for anything. At withheld positions EvoClassNet's median per-position ρ is 0.251 against -0.337 for BLOSUM62 exchangeability (paired Δ +0.598, P = 1.02e-15, n = 98). For binding the ordering reverses: BLOSUM62 reaches 0.538 against the model's 0.181 (Δ -0.269, P = 1.54e-14, n = 150). So an untrained substitution matrix beats the network at unseen positions on binding, and loses to it on escape — escape is the only place here where a trained model supplies something that neither the trivial baseline nor generic chemistry does.

The mutation-count baseline is decisive because held-out-position test sets contain variants with different numbers of substitutions, and fitness falls with substitution count. A model can score a respectable Spearman ρ there while having learned nothing position-specific. Papers that report held-out-position performance without this control cannot distinguish the two cases.

Paired per-seed comparison, EvoClassNet minus baseline (median difference, Wilcoxon signed-rank), held-out-position split:

| Task | vs mutation count | vs global epistasis | vs additive ridge | seeds |
|---|---|---|---|---|
| ACE2 binding | -0.022 (P = 0.275) | +0.704 (P = 0.00195) | +0.352 (P = 0.00195) | 10 |
| Expression | -0.018 (P = 0.322) | +0.702 (P = 0.00195) | +0.329 (P = 0.00195) | 10 |
| Antibody escape | +0.241 (P = 0.0273) | +0.241 (P = 0.0371) | +0.241 (P = 0.0273) | 10 |

## Main results

Test Spearman ρ, mean ± SD over independent seeds (`results/table_main.csv` has Pearson and RMSE).

**Split: random genotypes** (median test-set size: ACE2 binding 10,540, Antibody escape 202, Expression 11,641)

| Model | ACE2 binding | Expression | Antibody escape |
|---|---|---|---|
| Mutation count (untrained) | 0.478 ± 0.006 | 0.390 ± 0.007 | 0.000 ± 0.000 |
| Additive ridge | 0.874 ± 0.004 | 0.790 ± 0.003 | 0.000 ± 0.000 |
| Global epistasis | 0.886 ± 0.004 | 0.799 ± 0.004 | 0.754 ± 0.025 |
| EvoClassNet | 0.892 ± 0.008 | 0.855 ± 0.019 | 0.690 ± 0.104 |

**Split: held-out positions** (median test-set size: ACE2 binding 1,386, Antibody escape 300, Expression 1,432)

| Model | ACE2 binding | Expression | Antibody escape |
|---|---|---|---|
| Mutation count (untrained) | 0.386 ± 0.047 | 0.368 ± 0.049 | 0.000 ± 0.000 |
| Additive ridge | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Global epistasis | -0.370 ± 0.042 | -0.354 ± 0.057 | 0.008 ± 0.016 |
| EvoClassNet | 0.341 ± 0.118 | 0.323 ± 0.079 | 0.183 ± 0.200 |

## Run-to-run instability

Holding the data partition fixed and varying only the weight-initialisation seed (`scripts/stability_check.py`):

* partition seed 0: binding ρ = +0.374, +0.423, +0.387  (val loss 0.470, 0.422, 0.488)
* partition seed 2: binding ρ = +0.236, +0.558, +0.385  (val loss 0.418, 0.365, 0.426)

Spread across all 6 runs: +0.236 to +0.558. A single-run number on this split is not a measurement, which is why every table here reports seed-level distributions.

## Amino-acid permutation ablation

A per-position permutation of the 20-letter alphabet destroys residue identity shared across positions while leaving substitution counts untouched. If transfer to unseen positions were carried by chemistry, permuting should hurt.

| Split | Task | Intact | Permuted | Δ |
|---|---|---|---|---|
| random | ACE2 binding | 0.890 ± 0.011 | 0.870 ± 0.054 | -0.02 |
| random | Expression | 0.849 ± 0.026 | 0.808 ± 0.093 | -0.041 |
| random | Antibody escape | 0.650 ± 0.135 | 0.553 ± 0.230 | -0.097 |
| heldout_sites | ACE2 binding | 0.280 ± 0.118 | 0.260 ± 0.030 | -0.02 |
| heldout_sites | Expression | 0.287 ± 0.070 | 0.223 ± 0.036 | -0.064 |
| heldout_sites | Antibody escape | 0.161 ± 0.199 | 0.194 ± 0.306 | 0.033 |

Read the Δ column against the seed-level SD in the same row, not on its own.

## Leakage-free variant-of-concern ranking

VOC-defining substitutions appear in the escape training data, so ranking them with a model that has seen them measures memorisation. Here each VOC substitution contributes only in the seeds where its position was withheld. 12 unique substitutions qualify; the n column counts position-seed instances, since a substitution scores once per seed in which its position was withheld:

| Scorer | AUROC | 95% bootstrap CI | n positives |
|---|---|---|---|
| EvoClassNet predicted escape | 0.588 | 0.441–0.730 | 15 |
| EvoClassNet predicted binding | 0.469 | 0.291–0.638 | 15 |
| BLOSUM62 exchangeability | 0.718 | 0.602–0.820 | 15 |
| Measured escape (held-out positions) | 0.749 | 0.636–0.856 | 14 |
| Measured binding DMS (held-out positions) | 0.664 | 0.537–0.797 | 15 |

Against a hand-crafted chemistry prior, per-position rank accuracy at withheld positions:

| Task | Model median ρ | BLOSUM62 median ρ | Median Δ | P | n position–seed |
|---|---|---|---|---|---|
| bind | 0.181 | 0.538 | -0.269 | 1.54e-14 | 150 |
| escape | 0.251 | -0.337 | +0.598 | 1.02e-15 | 98 |

* **bind**: median per-position ρ +0.181, 76% of position–seed cells positive, ρ vs relative solvent accessibility -0.259 (P = 0.00168)
* **escape**: median per-position ρ +0.251, 80% of position–seed cells positive, ρ vs relative solvent accessibility -0.098 (P = 0.351)

## In-silico variant evolution (a control, not a forecast)

A genetic algorithm maximises the trained model's own predicted phenotype from wild type. This is reported to show what naive optimisation of a learned fitness surface produces:

* **escape**: top-1% of a 7,714-variant archive carries 0.000 VOC substitutions per variant vs 0.062 for random archive variants (permutation P = 1.000); 35% of its top-20 mutations sit in the terminal residues (≥ spike 512), where the yeast-display assay is least informative. Top mutations: T478F, K528H, K417H, T531R, L452Y, S477H
* **escape_and_binding**: top-1% of a 7,744-variant archive carries 0.000 VOC substitutions per variant vs 0.034 for random archive variants (permutation P = 1.000); 20% of its top-20 mutations sit in the terminal residues (≥ spike 512), where the yeast-display assay is least informative. Top mutations: T376H, K386N, S373R, K386D, A363F, D428N

No enrichment. The GA exploits assay artefacts, which is the expected failure mode and the reason this is a control rather than a result.

## Data

| Source | Use |
|---|---|
| Starr et al. 2020, SARS-CoV-2 RBD yeast-display DMS (Wuhan-Hu-1, spike 331–531) | ACE2 binding and expression labels; barcodes sharing a genotype are collapsed to the mean so identical sequences cannot straddle a split |
| Aggregated RBD antibody-escape maps (Bloom lab, 13 studies) | per-substitution escape labels, averaged over ≥ 20 monoclonal antibodies, log10 |
| `RBD_sites.csv` structural annotations | relative solvent accessibility, evolutionary entropy, receptor-binding-motif membership |

Escape rows whose listed wild-type residue disagrees with the Wuhan-Hu-1 background are dropped (~1.5% of rows, measured in Omicron backgrounds, at exactly the VOC positions — keeping them would contaminate the VOC analysis).

## What this study does not contain

| Original draft claim | Status |
|---|---|
| 48,891 genomes, 5-species classification (NCBI/GISAID/ENA) | **Not run.** Those hosts are unreachable from this environment; GISAID additionally requires a data-access agreement. |
| ESM-1b embeddings, GNN over PDB/AlphaFold structures | **Not run.** No GPU and no access to weights or structures. `EvoClassNetLite` keeps an `extra_dim` hook for precomputed per-residue features. |
| 99.92% accuracy, 14.7% of GA variants later observed, 2.48M seq/h | **No supporting experiment exists.** These numbers were not reproduced and are not claimed anywhere in this repository. |

Before submission this needs: real genomic data for the classification task, a GPU ablation against a protein language model and a structure-aware baseline, and more seeds on the held-out-position split.

## Layout

```
evoclassnet/data.py        genotype loading, encoding, both split protocols
evoclassnet/escape.py      per-substitution escape labels + background filtering
evoclassnet/model.py       EvoClassNetLite, GlobalEpistasis, seeded build(), fit()
evoclassnet/baselines.py   the mutation-count baseline
evoclassnet/ga.py          GA hyper-parameter search and GA variant evolution
scripts/run_study.py       main / sweep / ablation jobs, cached and restartable
scripts/analyze.py         tables, paired tests, figures 1/2/4
scripts/holdout_analysis.py  leakage-free VOC ranking, BLOSUM62 control, per-position transfer
scripts/stability_check.py   partition fixed, init seed varied
scripts/variant_analysis.py  GA variant evolution control
scripts/build_docx.py      manuscript, generated from results/
scripts/build_readme.py    this file, generated from results/
```

## Reproduce

```bash
./run_all.sh
```

Or step by step:

```bash
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest tests -q
python scripts/run_study.py --jobs main sweep ablation --seeds 10 --sweep_seeds 5 --workers 4
python scripts/stability_check.py && python scripts/holdout_analysis.py
python scripts/variant_analysis.py
python scripts/analyze.py && python scripts/build_docx.py && python scripts/build_readme.py
```

The study in this repository is 50 cached runs (`results/jobs/`, one JSON per run; re-running skips completed jobs). Median wall time per random-split run: 38 min on 4 CPU cores.
