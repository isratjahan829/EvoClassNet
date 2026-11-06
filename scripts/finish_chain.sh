#!/usr/bin/env bash
# Runs the post-study analyses in order once the main study process has exited.
set -x
cd /home/user/EvoClassNet
while pgrep -f "run_study.py" >/dev/null; do sleep 30; done
python3 -u scripts/stability_check.py 2,0 0,1,2 > results/stability_log.txt 2>&1
python3 -u scripts/holdout_analysis.py 5          > results/holdout_log.txt   2>&1
python3 -u scripts/variant_analysis.py            > results/variant_log.txt   2>&1
python3 -u scripts/analyze.py                     > results/analyze_log.txt   2>&1
python3 -u scripts/build_docx.py                  > results/docx_log.txt      2>&1
echo CHAIN_COMPLETE
