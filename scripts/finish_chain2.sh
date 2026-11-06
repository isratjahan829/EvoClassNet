#!/usr/bin/env bash
# Runs every downstream analysis once the expanded study has finished, then rebuilds the
# manuscript and README. Safe to start at any time: it waits for run_study2.py to exit first.
set -x
cd /home/user/EvoClassNet
while pgrep -f "run_study2.py" >/dev/null; do sleep 60; done
while pgrep -f "run_genome_study.py" >/dev/null; do sleep 60; done
while pgrep -f "esm_control.py" >/dev/null; do sleep 60; done
python3 -u scripts/analyze2.py          > results/analyze2_log.txt        2>&1
python3 -u scripts/analyze_genomes.py   > results/analyze_genomes_log.txt 2>&1
python3 -u scripts/make_schematics.py   > results/schematics_log.txt      2>&1
python3 -u scripts/build_docx2.py       > results/docx2_log.txt           2>&1
python3 -u scripts/build_readme2.py     > results/readme2_log.txt         2>&1
echo CHAIN2_COMPLETE
