# EvoClassNet — a leakage-controlled benchmark of sequence models for SARS-CoV-2 RBD phenotypes

Every number in this file is read out of `results/` by `scripts/build_readme.py`. Nothing is carried over from the original manuscript draft.

## The finding

Two evaluation protocols on the same deep-mutational-scanning data rank models differently:

* **Random genotype split** — a variant's own mutated positions are also mutated in the training set. All models do well; the neural model is not clearly better than a global-epistasis fit.
* **Held-out-position split** — no training variant mutates any test position. Rank accuracy collapses, and once the trivial **mutation-count baseline** (`−n_substitutions`, untrained) is included in the comparison, no trained model in this study demonstrates position-level extrapolation on binding or expression.

The mutation-count baseline is decisive because held-out-position test sets contain variants with different numbers of substitutions, and fitness falls with substitution count. A model can score a respectable Spearman ρ there while having learned nothing position-specific. Papers that report held-out-position performance without this control cannot distinguish the two cases.

Paired per-seed comparison, EvoClassNet minus baseline (median difference, Wilcoxon signed-rank), held-out-position split:

| Task | vs mutation count | vs global epistasis | vs additive ridge | seeds |
|---|---|---|---|---|
| ACE2 binding | +0.012 (P = 0.557) | +0.707 (P = 0.00391) | +0.363 (P = 0.0371) | 10 |
| Expression | -0.105 (P = 0.322) | +0.701 (P = 0.00195) | +0.307 (P = 0.0137) | 10 |
| Antibody escape | +0.274 (P = 0.084) | +0.257 (P = 0.105) | +0.274 (P = 0.084) | 10 |

## Main results

Test Spearman ρ, mean ± SD over independent seeds (`results/table_main.csv` has Pearson and RMSE).

**Split: random genotypes** (median test-set size: ACE2 binding 10,540, Antibody escape 202, Expression 11,641)

| Model | ACE2 binding | Expression | Antibody escape |
|---|---|---|---|
| Mutation count (untrained) | 0.478 ± 0.006 | 0.390 ± 0.007 | 0.000 ± 0.000 |
| Additive ridge | 0.874 ± 0.004 | 0.790 ± 0.003 | 0.000 ± 0.000 |
| Global epistasis | 0.886 ± 0.004 | 0.799 ± 0.004 | 0.754 ± 0.025 |
| EvoClassNet | 0.892 ± 0.009 | 0.852 ± 0.025 | 0.693 ± 0.140 |

**Split: held-out positions** (median test-set size: ACE2 binding 1,386, Antibody escape 300, Expression 1,432)

| Model | ACE2 binding | Expression | Antibody escape |
|---|---|---|---|
| Mutation count (untrained) | 0.386 ± 0.047 | 0.368 ± 0.049 | 0.000 ± 0.000 |
| Additive ridge | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Global epistasis | -0.370 ± 0.042 | -0.354 ± 0.057 | 0.008 ± 0.016 |
| EvoClassNet | 0.294 ± 0.265 | 0.271 ± 0.228 | 0.171 ± 0.281 |

## Amino-acid permutation ablation

A per-position permutation of the 20-letter alphabet destroys residue identity shared across positions while leaving substitution counts untouched. If transfer to unseen positions were carried by chemistry, permuting should hurt.

| Split | Task | Intact | Permuted | Δ |
|---|---|---|---|---|
| random | ACE2 binding | 0.890 ± 0.012 | 0.886 ± 0.018 | -0.003 |
| random | Expression | 0.845 ± 0.035 | 0.831 ± 0.037 | -0.013 |
| random | Antibody escape | 0.638 ± 0.189 | 0.585 ± 0.127 | -0.054 |
| heldout_sites | ACE2 binding | 0.171 ± 0.333 | 0.132 ± 0.227 | -0.039 |
| heldout_sites | Expression | 0.158 ± 0.275 | 0.086 ± 0.170 | -0.073 |
| heldout_sites | Antibody escape | 0.026 ± 0.266 | -0.034 ± 0.228 | -0.06 |

Read the Δ column against the seed-level SD in the same row, not on its own.

## In-silico variant evolution (a control, not a forecast)

A genetic algorithm maximises the trained model's own predicted phenotype from wild type. This is reported to show what naive optimisation of a learned fitness surface produces:


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

The study in this repository is 50 cached runs (`results/jobs/`, one JSON per run; re-running skips completed jobs). Median wall time per random-split run: 39 min on 4 CPU cores.
