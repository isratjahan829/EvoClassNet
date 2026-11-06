#!/usr/bin/env bash
# Full reproduction from raw data on 4 CPU cores. The study step is the long one (several hours);
# individual runs are cached in results/jobs/, so the script can be interrupted and restarted.
set -euo pipefail
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q tests
python scripts/run_study.py --jobs main sweep ablation --seeds 10 --sweep_seeds 5 --epochs 8 --workers 4 --torch_threads 1
python scripts/stability_check.py 2,0 0,1,2
python scripts/holdout_analysis.py 5
python scripts/variant_analysis.py
python scripts/analyze.py
python scripts/build_docx.py
python scripts/build_readme.py
