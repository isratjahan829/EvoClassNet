"""Statistical procedures for comparing models across seeds.

The study's main claim is a *negative* one: that a trained model is no better than an untrained
baseline at held-out positions. A non-significant difference test cannot support that claim, since
failing to reject the null is not evidence for it, especially at ten seeds. The right instrument is
an equivalence test, which asks whether the difference is small enough to rule out an effect of
practical size. `tost` does that, and it is what the manuscript should cite whenever it says two
models are indistinguishable.

The rest of the module supplies the pieces a referee will expect around that: correction for
testing many model-by-task combinations, confidence intervals on the paired differences rather
than P values alone, a non-parametric effect size, and a mixed-effects model for the per-position
analyses, where observations are nested within seeds and positions and so are not independent.
"""
import numpy as np
from scipy import stats


def paired_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    return a[ok] - b[ok]


def wilcoxon_paired(a, b):
    d = paired_diff(a, b)
    if len(d) < 5 or np.allclose(d, 0):
        return dict(median=float(np.median(d)) if len(d) else float("nan"), p=float("nan"), n=len(d))
    return dict(median=float(np.median(d)), p=float(stats.wilcoxon(d).pvalue), n=len(d))


def tost(a, b, margin, alpha=0.05):
    """Two one-sided tests for equivalence of paired samples.

    The null is that the true mean difference lies *outside* +/- `margin`. Rejecting it (p < alpha)
    is positive evidence that the two models differ by less than `margin`. Reported alongside the
    difference test, this distinguishes "we found no difference" from "we showed there is no
    difference larger than `margin`".
    """
    d = paired_diff(a, b)
    n = len(d)
    out = dict(n=n, margin=float(margin), mean=float(np.mean(d)) if n else float("nan"))
    if n < 3 or np.std(d, ddof=1) < 1e-12:
        out.update(p=float("nan"), equivalent=False)
        return out
    se = np.std(d, ddof=1) / np.sqrt(n)
    df = n - 1
    t_lo = (np.mean(d) + margin) / se          # H0: diff <= -margin
    t_hi = (np.mean(d) - margin) / se          # H0: diff >= +margin
    p = max(stats.t.sf(t_lo, df), stats.t.cdf(t_hi, df))
    ci = stats.t.interval(1 - 2 * alpha, df, loc=np.mean(d), scale=se)
    out.update(p=float(p), equivalent=bool(p < alpha),
               ci90=[float(ci[0]), float(ci[1])])
    return out


def bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    """Percentile bootstrap interval for a statistic of the paired differences."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    bs = [stat(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    return [float(x) for x in np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])]


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]: P(a > b) - P(a < b). Scale-free, so it is comparable
    across tasks whose correlations live on different ranges."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan"), "undefined"
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    d = (gt - lt) / (len(a) * len(b))
    mag = ("negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33 else
           "medium" if abs(d) < 0.474 else "large")
    return float(d), mag


def benjamini_hochberg(pvals, alpha=0.05):
    """Return (adjusted p values, rejected flags) controlling the false discovery rate."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q, np.zeros(p.shape, bool)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        val = pv[order[rank]] * m / (rank + 1)
        prev = min(prev, val)
        adj[order[rank]] = prev
    q[ok] = np.minimum(adj, 1.0)
    return q, (q < alpha) & ok


def mixed_effects_per_site(df, value="rho", group="site", second="seed"):
    """Random-intercept model of a per-position outcome, to test whether its mean differs from zero
    while respecting that positions recur across seeds.

    Falls back to a one-sample Wilcoxon on position means if statsmodels is unavailable.
    """
    try:
        import statsmodels.formula.api as smf
        d = df.dropna(subset=[value]).copy()
        if d[group].nunique() < 5:
            raise RuntimeError("too few groups")
        m = smf.mixedlm(f"{value} ~ 1", d, groups=d[group]).fit(reml=True, method="lbfgs")
        return dict(method="mixedlm", intercept=float(m.params["Intercept"]),
                    se=float(m.bse["Intercept"]), p=float(m.pvalues["Intercept"]),
                    n_obs=int(len(d)), n_groups=int(d[group].nunique()),
                    group_var=float(m.cov_re.iloc[0, 0]))
    except Exception as e:
        d = df.dropna(subset=[value])
        per = d.groupby(group)[value].mean()
        if len(per) < 5:
            return dict(method="unavailable", reason=str(e)[:120])
        w = stats.wilcoxon(per.to_numpy())
        return dict(method="wilcoxon_on_site_means", intercept=float(per.mean()),
                    se=float(per.std(ddof=1) / np.sqrt(len(per))), p=float(w.pvalue),
                    n_obs=int(len(d)), n_groups=int(len(per)), note=f"statsmodels unavailable: {str(e)[:80]}")
