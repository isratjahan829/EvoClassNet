"""Download the public DMS tables (Starr et al. 2020) used by the study. Idempotent."""
import sys, urllib.request
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data" / "raw" / "bloom_rbd_dms"; D.mkdir(parents=True, exist_ok=True)
M = "https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/master/"
R = "https://raw.githubusercontent.com/jbloomlab/SARS-CoV-2-RBD_DMS/master/"
FILES = {"results/binding_Kds/binding_Kds.csv": (M, D), "results/expression_meanFs/expression_meanFs.csv": (M, D),
         "results/single_mut_effects/single_mut_effects.csv": (M, D), "data/RBD_sites.csv": (R, D)}
E2 = Path(__file__).resolve().parents[1] / "data" / "raw" / "bloom_escape"; E2.mkdir(parents=True, exist_ok=True)
ME = "https://media.githubusercontent.com/media/jbloomlab/SARS2_RBD_Ab_escape_maps/main/"
FILES["processed_data/escape_data.csv"] = (ME, E2)
for f, (base, dest) in FILES.items():
    out = dest / Path(f).name
    if out.exists() and out.stat().st_size > 10_000: print("have", out.name); continue
    print("downloading", f); urllib.request.urlretrieve(base + f, out)
    assert out.read_bytes()[:20] != b"version https://git-l", "got an LFS pointer, not data"
