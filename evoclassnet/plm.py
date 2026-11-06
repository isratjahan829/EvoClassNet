"""ESM-2 protein language model features for the RBD.

Two products, both computed once from the wild-type sequence and cached:

`masked_marginals`  a 201 x 20 matrix of log-likelihood ratios. For position i the residue is
                    replaced by <mask>, the model is run, and the score of substitution i:wt->mut
                    is log p(mut | context) - log p(wt | context). This is the standard zero-shot
                    variant-effect score (Meier et al. 2021). A multi-substitution genotype scores
                    as the sum over its substitutions.

`embeddings`        per-residue representations of the wild-type sequence from the final layer,
                    reduced by PCA. These feed EvoClassNet's `extra_dim` input as a per-position
                    feature block.

Why this matters for the study: the masked-marginal scorer is *untrained on the assay*. It has
never seen a binding, expression or escape measurement, so it is leakage-free at held-out
positions by construction, and it is the natural test of whether information from outside the
training library supplies the position-level transfer that the supervised models do not.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "plm"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _paths(model_name):
    tag = model_name.split("/")[-1]
    return CACHE / f"{tag}_masked_marginals.npz", CACHE / f"{tag}_embeddings.npz"


def compute(wt_seq, model_name=DEFAULT_MODEL, batch=4, log=print):
    """Run ESM-2 over the wild-type sequence and cache both products. CPU-friendly."""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    CACHE.mkdir(parents=True, exist_ok=True)
    mm_path, emb_path = _paths(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    mdl.eval()
    torch.set_grad_enabled(False)

    enc = tok(wt_seq, return_tensors="pt")
    ids = enc["input_ids"]                      # [1, L+2] with BOS/EOS
    L = len(wt_seq)
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA])

    # ---- per-residue embeddings (one forward pass over the intact sequence)
    out = mdl(**enc, output_hidden_states=True)
    hid = out.hidden_states[-1][0, 1:L + 1].numpy().astype(np.float32)   # [L, d]
    np.savez_compressed(emb_path, hidden=hid, seq=wt_seq, model=model_name)
    log(f"embeddings {hid.shape} -> {emb_path.name}")

    # ---- masked marginals (one forward pass per position, batched)
    mm = np.zeros((L, 20), dtype=np.float32)
    mask_id = tok.mask_token_id
    for start in range(0, L, batch):
        idx = list(range(start, min(start + batch, L)))
        rep = ids.repeat(len(idx), 1).clone()
        for r, i in enumerate(idx):
            rep[r, i + 1] = mask_id                      # +1 for BOS
        logits = mdl(input_ids=rep).logits
        for r, i in enumerate(idx):
            lp = torch.log_softmax(logits[r, i + 1], dim=-1)
            wt_lp = lp[tok.convert_tokens_to_ids(wt_seq[i])]
            mm[i] = (lp[aa_ids] - wt_lp).numpy()
        if start % 40 == 0:
            log(f"masked marginals {min(start + batch, L)}/{L}")
    np.savez_compressed(mm_path, mm=mm, seq=wt_seq, alphabet=AA, model=model_name)
    log(f"masked marginals {mm.shape} -> {mm_path.name}")
    return mm, hid


def load(model_name=DEFAULT_MODEL):
    """Return (masked_marginals [L,20], embeddings [L,d]) from cache, or (None, None)."""
    mm_path, emb_path = _paths(model_name)
    if not (mm_path.exists() and emb_path.exists()):
        return None, None
    return np.load(mm_path)["mm"], np.load(emb_path)["hidden"]


def score_genotypes(subs_list, mm, offset_is_rbd=True):
    """Additive zero-shot score for each genotype.

    `subs_list` is a list of per-genotype substitution lists, each entry (wt, rbd_pos, mut) with
    `rbd_pos` 1-based within the RBD. Higher = the language model finds the variant more plausible.
    The wild type scores 0.
    """
    idx = {a: i for i, a in enumerate(AA)}
    out = np.zeros(len(subs_list), dtype=float)
    for k, subs in enumerate(subs_list):
        s = 0.0
        for w, p, m in subs:
            if m in idx and 1 <= p <= mm.shape[0]:
                s += float(mm[p - 1, idx[m]])
        out[k] = s
    return out


def embedding_features(hid, n_components=16, seed=0):
    """PCA-reduce per-residue embeddings to a compact per-position feature block."""
    from sklearn.decomposition import PCA
    n = min(n_components, hid.shape[0], hid.shape[1])
    X = PCA(n_components=n, random_state=seed).fit_transform(hid.astype(np.float64))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float32)
