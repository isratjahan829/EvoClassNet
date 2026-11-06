#!/usr/bin/env bash
# Full reproduction (CPU, ~1 h on 4 cores). Every number in the manuscript is regenerated from these outputs.
set -euo pipefail
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q tests
python scripts/run_study.py --seeds 3 --epochs 8
python scripts/variant_analysis.py
python scripts/make_figures.py
python scripts/build_docx.py
