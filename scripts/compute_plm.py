"""Download ESM-2 and cache its masked-marginal scores and per-residue embeddings for the RBD.

By default the RBD is scored inside the full spike protein, which is the context it actually
occupies. Scoring the 201-residue domain in isolation measurably weakens the model: wild-type
recovery falls from roughly a third to around a tenth (scripts/esm_control.py), so the fragment
scores understate what ESM-2 knows and would make any negative result about it unfair.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, plm

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
SPIKE = Path(__file__).resolve().parents[1] / "data" / "raw" / "structure" / "spike_P0DTC2.fasta"

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else plm.DEFAULT_MODEL
    context = sys.argv[2] if len(sys.argv) > 2 else "spike"
    wt, off = data.wildtype()
    kw = {}
    if context == "spike":
        if not SPIKE.exists():
            raise SystemExit(f"{SPIKE} missing; run scripts/download_data.py")
        spike = "".join(l.strip() for l in open(SPIKE) if not l.startswith(">"))
        i = spike.find(wt)
        if i < 0:
            raise SystemExit("RBD not found inside the spike sequence")
        kw = dict(context_seq=spike, context_offset=i, context_name="spike")
        log(f"scoring RBD inside full spike: offset {i}, spike length {len(spike)}")
    else:
        log("scoring the RBD as an isolated fragment")
    log(f"model={model}  RBD length={len(wt)}  offset={off}")
    plm.compute(wt, model_name=model, batch=2, log=log, **kw)
    log("done")
