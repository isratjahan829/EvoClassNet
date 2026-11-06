#!/usr/bin/env bash
# Runs every downstream analysis once the long jobs have finished, then rebuilds the manuscript,
# the README and the executable notebook. Safe to start at any time.
set -x
cd /home/user/EvoClassNet
export HF_HUB_DISABLE_XET=1
for proc in run_study2.py run_genome_study.py esm_control.py run_proteingym.py download_alphafold.py; do
  while pgrep -f "$proc" >/dev/null; do sleep 60; done
done
python3 -u scripts/analyze2.py            > results/analyze2_log.txt         2>&1
python3 -u scripts/analyze_genomes.py     > results/analyze_genomes_log.txt  2>&1
python3 -u scripts/analyze_proteingym.py  > results/analyze_pg_log.txt       2>&1
python3 -u scripts/make_schematics.py     > results/schematics_log.txt       2>&1
python3 -u scripts/build_docx2.py         > results/docx2_log.txt            2>&1
python3 -u scripts/build_readme2.py       > results/readme2_log.txt          2>&1
python3 -u scripts/build_notebook.py      > results/notebook_log.txt         2>&1
python3 - <<'PY'                         >> results/notebook_log.txt         2>&1
import nbformat, nbclient
nb = nbformat.read("EvoClassNet_report.ipynb", as_version=4)
nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True,
                        resources={"metadata": {"path": "/home/user/EvoClassNet"}}).execute()
nbformat.write(nb, "EvoClassNet_report.ipynb")
errs = sum(1 for c in nb.cells if c.cell_type == "code"
           for o in c.get("outputs", []) if o.get("output_type") == "error")
print("notebook executed, error cells:", errs)
PY
python3 -u scripts/package_deliverables.py   > results/package_log.txt          2>&1
echo CHAIN2_COMPLETE
