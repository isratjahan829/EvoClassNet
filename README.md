# EvoClassNet — reproducible CPU-scale study

Everything below was **measured by the code in this repo** (`scripts/run_study.py`, `scripts/variant_analysis.py`);
raw outputs are in `results/`. Nothing is copied from the original manuscript draft.

## What was run, and what was not
| Draft claim | Status here |
|---|---|
| 48,891 genomes, 5-species classification (NCBI/GISAID/ENA) | **Not run**: those hosts are unreachable from this environment (GISAID also needs a data-access agreement). |
| ESM-1b embeddings, GNN on PDB/AlphaFold structures | **Not run**: weights/structures unreachable, no GPU. The model has an `extra_dim` hook for precomputed per-residue features. |
| Multi-task binding / expression prediction | **Run** on public SARS-CoV-2 RBD deep-mutational-scanning data (Starr et al. 2020), 124,197 unique genotypes. |
| CNN–BiLSTM–attention, uncertainty-weighted multitask loss | **Run** (`EvoClassNet-lite`, CPU). |
| GA optimisation | **Run**: GA hyper-parameter search (validation only) and GA variant generation. The draft's GA *weight* optimisation was not implemented. |
| 99.92 % accuracy, 14.7 % of GA variants later observed, 2.48 M seq/h | **Not reproduced / no supporting experiment exists in this repo.** |

## Results (test Spearman, mean ± SD over 3 seeds)
| Split | Model | ACE2 binding | Expression |
|---|---|---|---|
| random genotypes | additive ridge | 0.872 ± 0.004 | 0.790 ± 0.002 |
| random genotypes | global epistasis | 0.885 ± 0.003 | 0.797 ± 0.003 |
| random genotypes | EvoClassNet-lite | 0.883 ± 0.019 | 0.836 ± 0.049 |
| held-out sites | additive ridge | 0 (constant prediction) | 0 |
| held-out sites | global epistasis | −0.353 ± 0.053 | −0.342 ± 0.011 |
| held-out sites | EvoClassNet-lite | 0.394 ± 0.071 | 0.379 ± 0.129 |

Per-seed values are in `results/study_log.txt`, full metrics (Pearson, RMSE) in `results/summary.csv`.
Seed variance of EvoClassNet-lite is large (random split, binding: 0.856 / 0.898 / 0.894).

**Honest reading**
* On the random split EvoClassNet-lite is **not distinguishable from a global-epistasis baseline for binding**; it is better for expression in 2 of 3 seeds. With 3 seeds no significance claim is possible.
* The informative result is the **held-out-site split** (no training variant touches a test position): additive and global-epistasis models cannot generalise to unseen positions, EvoClassNet-lite reaches Spearman ≈ 0.4. That is modest, not near-perfect.

## Retrospective variant analysis (not a prospective forecast)
* Ranking all 3,819 single substitutions by predicted fitness (binding + 0.5·expression), 18 lineage-defining VOC substitutions (Alpha–Omicron BA.1) get **AUROC 0.753 (95 % bootstrap CI 0.662–0.832)**; measured single-mutant DMS values give 0.781 (0.688–0.860). The model uses the 2020 DMS, so this is a retrospective check only.
* **GA variant generation shows no enrichment**: top-1 % GA variants carry 0.013 VOC substitutions/variant vs 0.042 for random archive variants (permutation p = 0.95). The GA mostly exploits terminal residues (P527C, S530N, K529I) — likely a DMS artefact. The draft's "14.7 % of GA variants later observed" has no counterpart here.
* Caveat: this analysis used the seed-0 random-split model, the weakest seed.

## Deliverables
* `EvoClassNet_manuscript.docx` — revised manuscript; every number is generated from `results/*.json` by `scripts/build_docx.py`.
* `results/figures/` — figures used in the manuscript.
* `run_all.sh` — one-command reproduction (install, download data, unit tests, study, variant analysis, figures, docx).

## Reproduce
```bash
./run_all.sh          # ~1 h on 4 CPU cores; or step by step:
python scripts/download_data.py
python scripts/run_study.py --seeds 3 --epochs 8
python scripts/variant_analysis.py && python scripts/make_figures.py && python scripts/build_docx.py
```
