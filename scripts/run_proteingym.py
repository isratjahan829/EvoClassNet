"""The evaluation comparison, run across every ProteinGym substitution assay.

For each assay and seed we build a random split and a held-out-position split and score four
model families on both:

  mutation_count    untrained; ranks by -n_substitutions. Constant, and therefore uninformative,
                    on any assay containing only single mutants, which is itself worth counting.
  blosum_zeroshot   untrained; sums BLOSUM62 exchangeabilities.
  physchem          ridge on position-transferable features: how the residue changes chemically,
                    its BLOSUM score, wild-type and mutant identity, and the substitution count.
                    Defined at a position the training set never touched.
  ridge_additive    one-hot (position, residue) indicators. Has no parameter for an unseen
                    position, and is the stand-in here for the whole family of position-indexed
                    models that the RBD study examines in depth.

Both the raw and the partial Spearman correlation are recorded, the latter removing the
substitution count from prediction and measurement so that position-level signal is measured
directly rather than inferred from a baseline comparison.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import proteingym as PG, zoo
from evoclassnet.metrics import evaluate

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
JOBS = OUT / "pg_jobs"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
MODELS = ["mutation_count", "blosum_zeroshot", "physchem", "ridge_additive"]


def _nanmean(xs):
    """Mean over the non-missing entries, or NaN if there are none.

    The partial correlation is undefined on an assay containing only single mutants, because the
    substitution count is then constant. That is the expected case for most of ProteinGym, not an
    error, so it is handled rather than warned about.
    """
    v = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def onehot(subs_list, L):
    r, c = [], []
    for i, s in enumerate(subs_list):
        for _, p, m in s:
            if m in zoo.AA2I:
                r.append(i); c.append((p - 1) * 20 + zoo.AA2I[m])
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(subs_list), L * 20))


def one_assay(dms_id, seeds, test_site_frac=0.2, max_variants=120_000):
    df, seq = PG.load_assay(dms_id)
    if len(df) > max_variants:                       # keep the largest assays affordable
        df = df.sample(max_variants, random_state=0).reset_index(drop=True)
    y = df.dms_score.to_numpy(float)
    nsub = df.n_sub.to_numpy()
    subs = df.subs.tolist()
    F = zoo.genotype_features(subs)                  # chemistry only: no structure, no PLM
    S = onehot(subs, len(seq))
    out = dict(dms_id=dms_id, n_variants=int(len(df)), seq_len=int(len(seq)),
               n_positions=int(len({p for s in df.sites for p in s})),
               frac_multi=float((nsub > 1).mean()), max_nsub=int(nsub.max()),
               splits={})

    for split in ("random", "heldout_positions"):
        per = {m: {"spearman": [], "spearman_partial": [], "n": []} for m in MODELS}
        for seed in range(seeds):
            if split == "random":
                tr, te = PG.split_random(df, seed)
            else:
                tr, te = PG.split_heldout_positions(df, seed, test_site_frac)
            if len(tr) < 50 or len(te) < 30:
                continue
            preds = {
                "mutation_count": -nsub[te].astype(float),
                "blosum_zeroshot": zoo.blosum_scores([subs[i] for i in te]),
            }
            sc = StandardScaler().fit(F[tr])
            preds["physchem"] = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(
                sc.transform(F[tr]), y[tr]).predict(sc.transform(F[te]))
            preds["ridge_additive"] = Ridge(alpha=1.0).fit(S[tr], y[tr]).predict(S[te])
            for m, p in preds.items():
                r = evaluate(y[te], np.asarray(p, float), nsub[te])
                per[m]["spearman"].append(r["spearman"])
                per[m]["spearman_partial"].append(r["spearman_partial"])
                per[m]["n"].append(r["n"])
        out["splits"][split] = {
            m: dict(spearman=float(np.mean(v["spearman"])) if v["spearman"] else float("nan"),
                    spearman_sd=float(np.std(v["spearman"], ddof=1)) if len(v["spearman"]) > 1 else 0.0,
                    partial=_nanmean(v["spearman_partial"]),
                    n_seeds=len(v["spearman"]),
                    n_test=int(np.median(v["n"])) if v["n"] else 0)
            for m, v in per.items()}
    return out


def _worker(spec):
    dms_id, seeds, frac = spec
    f = JOBS / f"{dms_id}.json"
    if f.exists():
        return dms_id, json.load(open(f))
    try:
        t0 = time.time()
        r = one_assay(dms_id, seeds, frac)
        r["_seconds"] = round(time.time() - t0, 1)
    except Exception as e:
        r = dict(dms_id=dms_id, error=f"{type(e).__name__}: {e}"[:200])
    json.dump(r, open(f, "w"), indent=1, default=float)
    return dms_id, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--test_site_frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    ids = PG.assay_ids()
    if a.limit:
        ids = ids[:a.limit]
    done = sum(1 for d in ids if (JOBS / f"{d}.json").exists())
    log(f"{len(ids)} assays, {done} cached, {a.seeds} seeds, {a.workers} workers")

    import multiprocessing as mp
    specs = [(d, a.seeds, a.test_site_frac) for d in ids]
    results = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, (d, r) in enumerate(pool.imap_unordered(_worker, specs), 1):
            results[d] = r
            if "error" in r:
                log(f"[{i}/{len(ids)}] {d} FAILED {r['error'][:70]}")
            else:
                h = r["splits"].get("heldout_positions", {})
                mc = h.get("mutation_count", {}).get("spearman", float("nan"))
                pc = h.get("physchem", {}).get("spearman", float("nan"))
                ra = h.get("ridge_additive", {}).get("spearman", float("nan"))
                log(f"[{i}/{len(ids)}] {d[:42]:42s} heldout: count={mc:+.3f} "
                    f"additive={ra:+.3f} physchem={pc:+.3f}")
    json.dump(results, open(OUT / "proteingym_study.json", "w"), indent=1, default=float)
    log("saved results/proteingym_study.json")
