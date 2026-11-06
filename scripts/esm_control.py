"""Sanity controls for the ESM-2 scorer.

The masked-marginal scores computed on the isolated RBD correlate almost not at all with the
measured phenotypes. Before that is reported as a finding about protein language models on this
protein, two things have to be ruled out: that the model is broken, and that the poor result is an
artefact of scoring a 201-residue domain with no surrounding context.

  (a) wild-type recovery on human ubiquitin, a small, highly constrained, well-represented protein
      on which any working ESM-2 should recover most residues;
  (b) wild-type recovery for RBD positions scored inside the full 1273-residue spike protein,
      which is the context the domain actually occupies.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import data, plm

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def recovery(mdl, tok, seq, positions, label):
    hit, top_p = 0, []
    enc = tok(seq, return_tensors="pt")
    for n, i in enumerate(positions, 1):
        rep = enc["input_ids"].clone()
        rep[0, i + 1] = tok.mask_token_id
        lp = torch.log_softmax(mdl(input_ids=rep).logits[0, i + 1], -1)
        if tok.convert_ids_to_tokens(int(lp.argmax())) == seq[i]:
            hit += 1
        top_p.append(float(lp.max().exp()))
        if n % 20 == 0:
            log(f"  {label}: {n}/{len(positions)}  running recovery {hit/n:.1%}")
    log(f"{label}: recovery {hit/len(positions):.1%} over {len(positions)} positions, "
        f"median top probability {np.median(top_p):.3f}")
    return hit / len(positions), float(np.median(top_p))


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    name = plm.DEFAULT_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForMaskedLM.from_pretrained(name).eval()
    out = {}

    out["ubiquitin"] = recovery(mdl, tok, UBQ, range(len(UBQ)), "ubiquitin (positive control)")

    wt, off = data.wildtype()
    idx = list(range(0, 201, 4))
    out["rbd_fragment"] = recovery(mdl, tok, wt, idx, "RBD alone (fragment)")

    spike = "".join(l.strip() for l in
                    open("data/raw/structure/spike_P0DTC2.fasta") if not l.startswith(">"))
    assert spike[330:531] == wt
    out["rbd_in_spike"] = recovery(mdl, tok, spike, [330 + i for i in idx], "RBD in full spike")

    import json
    json.dump({k: dict(recovery=v[0], median_top_prob=v[1]) for k, v in out.items()},
              open("results/esm_control.json", "w"), indent=1)
    log("saved results/esm_control.json")
