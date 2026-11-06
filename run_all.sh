#!/usr/bin/env bash
# Full reproduction from raw data on 4 CPU cores. The study step is the long one (several hours);
# individual runs are cached in results/jobs_v2/, so the script can be interrupted and restarted.
set -euo pipefail
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q tests
python scripts/compute_plm.py                 # ESM-2 features (downloads ~2.6 GB once)
python scripts/download_genomes.py            # coronavirus genomes from NCBI
python scripts/run_study2.py --jobs main sweep ablation --seeds 10 --sweep_seeds 5 --epochs 8 --workers 4 --torch_threads 1
python scripts/feature_ablation.py 10
python scripts/esm_control.py
python scripts/stability_check.py 2,0 0,1,2
python scripts/holdout_analysis.py 5
python scripts/variant_analysis.py
python scripts/run_genome_study.py --seeds 10
python scripts/analyze2.py
python scripts/analyze_genomes.py
python scripts/make_schematics.py
python scripts/build_docx2.py
python scripts/build_readme2.py
