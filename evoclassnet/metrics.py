"""Evaluation metrics for variant-effect prediction.

The study's central question is whether a model carries information about a position it has never
seen mutated. A single rank correlation cannot answer that on a combinatorial library, because
part of any correlation is explained by how many substitutions a variant carries. The panel below
therefore reports, alongside the conventional metrics:

`spearman_partial`  Spearman correlation between prediction and measurement after the substitution
                    count has been regressed out of both. This is the direct measure of signal
                    beyond the mutation-count confound, and it is the number to read when
                    comparing models at held-out positions.
`spearman_by_nsub`  the same correlation computed within each substitution-count stratum, where
                    mutation count is constant by construction and so cannot contribute at all.
`precision_at_k`    whether the top-ranked variants really are the best ones, which is what a
                    prioritisation workflow actually uses, and which rank correlation can hide.
`auroc_deleterious` ranking performance treated as a binary decision at the lower measurement
                    quartile, so results are comparable to classification work.
`calibration_slope` the slope of measurement regressed on prediction; 1.0 means predictions are on
                    the right scale, not merely correctly ordered.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata

PANEL = ["spearman", "spearman_partial", "pearson", "kendall", "r2", "rmse", "mae",
         "auroc_deleterious", "precision_at_50", "ndcg_at_100", "calibration_slope", "n"]


def _safe(y, p):
    m = ~np.isnan(y) & ~np.isnan(p)
    return y[m], p[m]


def _auroc(score, lab):
    lab = lab.astype(bool)
    if lab.sum() < 3 or (~lab).sum() < 3:
        return float("nan")
    r = rankdata(score)
    return float((r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))


def _partial_spearman(y, p, z):
    """Spearman between y and p after removing the linear effect of z on the ranks of both."""
    if len(y) < 10 or np.ptp(z) < 1e-9:
        return float("nan")
    ry, rp, rz = rankdata(y), rankdata(p), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ep = rp - A @ np.linalg.lstsq(A, rp, rcond=None)[0]
    if np.std(ey) < 1e-12 or np.std(ep) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ey, ep)[0, 1])


def _ndcg(y, p, k):
    """Normalised discounted cumulative gain over the top k, with gains shifted to be positive."""
    if len(y) < k:
        k = len(y)
    if k < 2:
        return float("nan")
    g = y - y.min()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    actual = (g[np.argsort(-p)[:k]] * disc).sum()
    ideal = (np.sort(g)[::-1][:k] * disc).sum()
    return float(actual / ideal) if ideal > 0 else float("nan")


def evaluate(y, p, n_sub=None, k_prec=50, k_ndcg=100, strata=(1, 2, 3)):
    """Full metric panel for one task. `y` measured, `p` predicted, `n_sub` substitution counts."""
    m = ~np.isnan(y)
    out = {k: float("nan") for k in PANEL}
    out["n"] = int(m.sum())
    if m.sum() < 3:
        return out
    yy, pp = y[m], p[m]
    nn = n_sub[m] if n_sub is not None else None
    out["rmse"] = float(np.sqrt(np.mean((yy - pp) ** 2)))
    out["mae"] = float(np.mean(np.abs(yy - pp)))
    if np.ptp(pp) < 1e-9:                       # a constant prediction carries no rank information
        out.update(spearman=0.0, pearson=0.0, kendall=0.0, spearman_partial=0.0,
                   auroc_deleterious=0.5, constant=True)
        return out
    out["spearman"] = float(spearmanr(yy, pp)[0])
    out["pearson"] = float(pearsonr(yy, pp)[0])
    out["kendall"] = float(kendalltau(yy, pp)[0])
    ss = float(np.sum((yy - yy.mean()) ** 2))
    out["r2"] = float(1 - np.sum((yy - pp) ** 2) / ss) if ss > 0 else float("nan")
    out["auroc_deleterious"] = _auroc(-pp, yy <= np.quantile(yy, 0.25))
    kk = min(k_prec, max(2, len(yy) // 10))
    top_true = set(np.argsort(-yy)[:kk].tolist())
    out["precision_at_50"] = float(len(top_true & set(np.argsort(-pp)[:kk].tolist())) / kk)
    out["ndcg_at_100"] = _ndcg(yy, pp, min(k_ndcg, len(yy)))
    v = float(np.var(pp))
    out["calibration_slope"] = float(np.cov(yy, pp)[0, 1] / v) if v > 1e-12 else float("nan")
    if nn is not None:
        out["spearman_partial"] = _partial_spearman(yy, pp, nn.astype(float))
        for s in strata:
            sel = nn == s
            out[f"spearman_nsub{s}"] = (float(spearmanr(yy[sel], pp[sel])[0])
                                        if sel.sum() >= 10 and np.ptp(pp[sel]) > 1e-9 else float("nan"))
    return out
