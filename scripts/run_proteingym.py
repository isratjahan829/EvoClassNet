"""The evaluation comparison, run across every ProteinGym substitution assay.

For each assay and seed we build four splits and score six model families on each:

  random             an 80/20 shuffle of variants, the conventional protocol
  heldout_positions  a random 20 % of positions withheld
  pg_modulo_5        ProteinGym's `fold_modulo_5`: positions assigned by position mod 5
  pg_contiguous_5    ProteinGym's `fold_contiguous_5`: five consecutive blocks of positions

The last two are reconstructions of the cross-validation schemes the supervised ProteinGym
benchmark already defines, so nothing about withholding positions is proposed here; the point is
what the comparison shows once an untrained control and a position-only model are included.

  mutation_count    untrained; ranks by -n_substitutions. Constant, and therefore uninformative,
                    on any assay containing only single mutants, which is itself worth counting.
  blosum_zeroshot   untrained; sums BLOSUM62 exchangeabilities.
  physchem          ridge on position-transferable features: how the residue changes chemically,
                    its BLOSUM score, wild-type and mutant identity, and the substitution count.
                    Defined at a position the training set never touched.
  residue_only      ridge on wild-type and mutant residue identity alone, with no position
                    information at all. The minimal transferable model.
  chem_struct       the chemistry block plus per-residue descriptors of the position taken
                    from an AlphaFold model of the protein (relative solvent accessibility,
                    contact number, burial, and the model's own pLDDT confidence). Available
                    only for the assays whose target sequence could be located in a predicted
                    model; NaN elsewhere. This is the benchmark-wide version of the structural
                    arm that the RBD study runs against a crystal structure.
  position_mean     ridge on one-hot position indicators: the mean effect of mutating each
                    position, learning nothing about which substitution is made. This is the
                    cleanest representative of the position-indexed family, because it is
                    exactly what a random split rewards and exactly what cannot survive a
                    held-out-position split.
  ridge_additive    one-hot (position, residue) indicators. On the many ProteinGym assays that
                    contain only single mutants each indicator occurs in a single row, so the
                    regularised fit collapses to the intercept even under a random split; that
                    degenerate behaviour is itself worth recording.

Both the raw and the partial Spearman correlation are recorded, the latter removing the
substitution count from prediction and measurement so that position-level signal is measured
directly rather than inferred from a baseline comparison.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

# Must precede the numpy and sklearn imports: each worker is single-threaded so that the pool
# size, not the BLAS library, decides how many cores are used. Without this the machine ends up
# heavily oversubscribed and every job on it slows down.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evoclassnet import proteingym as PG, structure as ST, zoo
from evoclassnet.metrics import evaluate


def structure_block(dms_id, seq):
    """Per-position AlphaFold descriptors for this assay, or None if no model matched."""
    cov = AF / "coverage.json"
    if not cov.exists():
        return None
    if not hasattr(structure_block, "_map"):
        structure_block._map = {c["dms_id"]: c for c in json.load(open(cov))}
    c = structure_block._map.get(dms_id)
    if not c or c.get("status") != "matched" or c.get("offset") is None:
        return None
    pdb = AF / f"AF-{c['acc']}.pdb"
    if not pdb.exists():
        return None
    try:
        f, mask, _ = ST.features_from_pdb(pdb, seq, offset=int(c["offset"]))
    except Exception:
        return None
    return f if mask.sum() >= 0.5 * len(seq) else None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
JOBS = OUT / "pg_jobs"
AF = ROOT / "data" / "raw" / "alphafold"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
MODELS = ["mutation_count", "blosum_zeroshot", "residue_only", "physchem",
          "chem_struct", "position_mean", "ridge_additive"]


def _nanmean(xs):
    """Mean over the non-missing entries, or NaN if there are none.

    The partial correlation is undefined on an assay containing only single mutants, because the
    substitution count is then constant. That is the expected case for most of ProteinGym, not an
    error, so it is handled rather than warned about.
    """
    v = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def onehot_position(subs_list, L):
    r, c = [], []
    for i, s in enumerate(subs_list):
        for _, p, _ in s:
            r.append(i); c.append(p - 1)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(subs_list), L))


def residue_identity(subs_list):
    """Wild-type and mutant residue indicators, summed over a genotype. No position information."""
    X = np.zeros((len(subs_list), 41), dtype=np.float32)
    for i, s in enumerate(subs_list):
        for w, _, m in s:
            if w in zoo.AA2I:
                X[i, zoo.AA2I[w]] += 1
            if m in zoo.AA2I:
                X[i, 20 + zoo.AA2I[m]] += 1
        X[i, 40] = len(s)
    return X


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
    sf = structure_block(dms_id, seq)
    FS = (zoo.genotype_features(subs, struct=sf, struct_names=ST.GENERIC_FEATURES)
          if sf is not None else None)
    R = residue_identity(subs)
    S = onehot(subs, len(seq))
    Pm = onehot_position(subs, len(seq))
    out = dict(dms_id=dms_id, n_variants=int(len(df)), seq_len=int(len(seq)),
               n_positions=int(len({p for s in df.sites for p in s})),
               frac_multi=float((nsub > 1).mean()), max_nsub=int(nsub.max()),
               has_structure=bool(FS is not None), splits={})

    schemes = {
        "random": lambda seed: PG.split_random(df, seed),
        "heldout_positions": lambda seed: PG.split_heldout_positions(df, seed, test_site_frac),
        # ProteinGym's own position-based cross-validation schemes, reconstructed exactly, so
        # these numbers sit on the same footing as the published supervised leaderboard.
        "pg_modulo_5": lambda seed: PG.fold_modulo(df, len(seq), test_fold=seed % PG.N_FOLDS),
        "pg_contiguous_5": lambda seed: PG.fold_contiguous(df, len(seq), test_fold=seed % PG.N_FOLDS),
    }
    for split in schemes:
        per = {m: {"spearman": [], "spearman_partial": [], "n": []} for m in MODELS}
        for seed in range(seeds):
            tr, te = schemes[split](seed)
            if len(tr) < 50 or len(te) < 30:
                continue
            preds = {
                "mutation_count": -nsub[te].astype(float),
                "blosum_zeroshot": zoo.blosum_scores([subs[i] for i in te]),
            }
            sc = StandardScaler().fit(F[tr])
            preds["physchem"] = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(
                sc.transform(F[tr]), y[tr]).predict(sc.transform(F[te]))
            sr = StandardScaler().fit(R[tr])
            preds["residue_only"] = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(
                sr.transform(R[tr]), y[tr]).predict(sr.transform(R[te]))
            if FS is not None:
                ss = StandardScaler().fit(FS[tr])
                preds["chem_struct"] = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(
                    ss.transform(FS[tr]), y[tr]).predict(ss.transform(FS[te]))
            preds["position_mean"] = Ridge(alpha=1.0).fit(Pm[tr], y[tr]).predict(Pm[te])
            preds["ridge_additive"] = Ridge(alpha=1.0).fit(S[tr], y[tr]).predict(S[te])
            for m, p in preds.items():
                if m not in per:
                    continue
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
                rnd = r["splits"].get("random", {})
                gv = lambda blk, m: blk.get(m, {}).get("spearman", float("nan"))
                st_tag = "S" if r.get("has_structure") else "-"
                log(f"[{i}/{len(ids)}] {d[:36]:36s} [{st_tag}] "
                    f"rand pos={gv(rnd,'position_mean'):+.3f} | heldout pos={gv(h,'position_mean'):+.3f} "
                    f"res={gv(h,'residue_only'):+.3f} chem={gv(h,'physchem'):+.3f} "
                    f"chem+str={gv(h,'chem_struct'):+.3f}")
    json.dump(results, open(OUT / "proteingym_study.json", "w"), indent=1, default=float)
    log("saved results/proteingym_study.json")
