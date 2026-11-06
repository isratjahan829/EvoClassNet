"""Download ESM-2 and cache its masked-marginal scores and per-residue embeddings for the RBD."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, plm

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else plm.DEFAULT_MODEL
    wt, off = data.wildtype()
    log(f"model={model}  RBD length={len(wt)}  offset={off}")
    plm.compute(wt, model_name=model, batch=4, log=log)
    log("done")
