"""Layer-2 inference: spend superiority, default non-inferiority, CUPED.

Statistical reasoning, per pre-registered design (DESIGN_FREEZE.md §7 +
experiment_design_frozen.json):

- SPEND (superiority, two-sided α=0.05): spend is zero-inflated and
  heavy-tailed, so the pre-registered mean-difference estimate is accompanied
  by a bootstrap percentile CI (means of heavy-tailed data converge slowly; the
  normal-theory CI is a sanity check, not the headline), a winsorization
  sensitivity, and a two-part (hurdle) decomposition [P(spend>0) and
  E[spend | spend>0]] so a lift driven by activation vs intensification is
  distinguishable.
- CUPED with the pre-period spend covariate: Y_adj = Y - θ(X_pre - mean(X_pre)),
  θ = cov(Y, X_pre)/var(X_pre) pooled across arms. Unbiased for the ATE because
  X_pre is pre-treatment and θ is the same in both arms; variance shrinks by
  the squared correlation.
- DEFAULT (non-inferiority at margin δ): with rare events a Wald interval is
  unreliable, so the one-sided 95% upper bound uses the Newcombe score method
  (Wilson score interval per arm, combined). The guardrail passes ONLY if that
  upper bound < δ. "Not statistically significant" is never read as "safe":
  an underpowered test fails to reject H0 almost regardless of truth, which is
  why the verdict tracks the prospective power analysis.
- FDR: the primary metric and guardrail use their pre-registered thresholds —
  no multiplicity correction is applied to them (they are two co-primary,
  jointly required decisions, not a family of hypotheses to screen).
  Benjamini-Hochberg applies ONLY to the exploratory segment analyses, which
  are labeled exploratory and generate hypotheses, not decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ------------------------------------------------------------------ spend
def mean_diff(y_t: np.ndarray, y_c: np.ndarray) -> dict:
    d = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    return {"diff": float(d), "se": float(se),
            "ci_lo": float(d - 1.96 * se), "ci_hi": float(d + 1.96 * se),
            "rel_lift": float(d / y_c.mean()) if y_c.mean() > 0 else np.nan,
            "n_t": len(y_t), "n_c": len(y_c)}


def bootstrap_diff_ci(y_t: np.ndarray, y_c: np.ndarray, n_boot: int = 4000,
                      seed: int = config.SEED_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = (rng.choice(y_t, len(y_t)).mean()
                    - rng.choice(y_c, len(y_c)).mean())
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def winsorized_diff(y_t: np.ndarray, y_c: np.ndarray, q: float = 0.99) -> dict:
    cap = np.quantile(np.concatenate([y_t, y_c]), q)
    return mean_diff(np.minimum(y_t, cap), np.minimum(y_c, cap))


def hurdle_decomposition(y_t: np.ndarray, y_c: np.ndarray) -> dict:
    """Two-part view: activation P(spend>0) and intensity E[spend | spend>0]."""
    act_t, act_c = (y_t > 0).mean(), (y_c > 0).mean()
    int_t = y_t[y_t > 0].mean() if (y_t > 0).any() else 0.0
    int_c = y_c[y_c > 0].mean() if (y_c > 0).any() else 0.0
    se_act = np.sqrt(act_t * (1 - act_t) / len(y_t) + act_c * (1 - act_c) / len(y_c))
    return {"p_active_t": float(act_t), "p_active_c": float(act_c),
            "p_active_diff": float(act_t - act_c), "p_active_se": float(se_act),
            "intensity_t": float(int_t), "intensity_c": float(int_c),
            "intensity_diff": float(int_t - int_c)}


def cuped_adjust(y: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    """θ estimated pooled (treatment-blind), X_pre strictly pre-treatment —
    the two conditions that keep CUPED unbiased for the ATE."""
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / x_pre.var(ddof=1)
    return y - theta * (x_pre - x_pre.mean())


def cuped_bootstrap_ci(y: np.ndarray, x_pre: np.ndarray, treated: np.ndarray,
                       n_boot: int = 4000,
                       seed: int = config.SEED_BOOTSTRAP) -> tuple[float, float]:
    """Bootstrap of the FULL CUPED pipeline: resample (Y, X_pre) pairs within
    each arm, re-estimate the pooled θ inside every replicate, re-adjust,
    re-difference. Holding θ fixed across replicates would treat an estimated
    quantity as a known constant and understate the interval (in large samples
    the difference is small — but the resampling must match the estimator)."""
    rng = np.random.default_rng(seed)
    t_idx, c_idx = np.flatnonzero(treated == 1), np.flatnonzero(treated == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        bt = rng.choice(t_idx, len(t_idx))
        bc = rng.choice(c_idx, len(c_idx))
        idx = np.concatenate([bt, bc])
        adj = cuped_adjust(y[idx], x_pre[idx])
        diffs[b] = adj[: len(bt)].mean() - adj[len(bt):].mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def ancova_crosscheck(y: np.ndarray, x_pre: np.ndarray,
                      treated: np.ndarray) -> dict:
    """Regression cross-check: Y ~ treatment + centered pre-period covariate,
    heteroskedasticity-robust (HC3) SEs. Asymptotically equivalent to CUPED;
    a large discrepancy would indicate an implementation bug."""
    import statsmodels.api as sm

    X = np.column_stack([np.ones_like(x_pre), treated, x_pre - x_pre.mean()])
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    return {"diff": float(fit.params[1]), "se_hc3": float(fit.bse[1]),
            "ci_lo": float(fit.conf_int()[1][0]),
            "ci_hi": float(fit.conf_int()[1][1])}


# ----------------------------------------------------------------- default
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha,
                                                 method="wilson")
    return float(lo), float(hi)


def newcombe_upper_bound(k_t: int, n_t: int, k_c: int, n_c: int,
                         alpha_one_sided: float = 0.05) -> float:
    """One-sided (1 - α) upper bound for p_t - p_c, Newcombe score method:
    combine per-arm Wilson limits at the matching two-sided level.
    UB = (p̂_t - p̂_c) + sqrt((u_t - p̂_t)² + (p̂_c - l_c)²).
    """
    p_t, p_c = k_t / n_t, k_c / n_c
    two_sided = 2 * alpha_one_sided
    _, u_t = wilson_interval(k_t, n_t, alpha=two_sided)
    l_c, _ = wilson_interval(k_c, n_c, alpha=two_sided)
    return float((p_t - p_c) + np.sqrt((u_t - p_t) ** 2 + (p_c - l_c) ** 2))


def noninferiority_verdict(k_t: int, n_t: int, k_c: int, n_c: int,
                           deltas_pp: tuple, powered: dict[float, bool],
                           alpha_one_sided: float = 0.05) -> pd.DataFrame:
    """Guardrail table across margins.

    Non-inferiority is DEMONSTRATED only when the upper bound sits below the
    margin. Anything else is NOT DEMONSTRATED — never 'proven unsafe': failure
    to demonstrate is an absence of evidence, and its interpretation depends
    on the separate power-adequacy column (an underpowered design was unlikely
    to demonstrate non-inferiority regardless of the truth)."""
    ub = newcombe_upper_bound(k_t, n_t, k_c, n_c, alpha_one_sided)
    diff = k_t / n_t - k_c / n_c
    rows = []
    for d_pp in deltas_pp:
        demonstrated = ub < d_pp / 100
        rows.append({
            "delta_pp": d_pp,
            "observed_diff_pp": diff * 100,
            "one_sided_95_upper_pp": ub * 100,
            "noninferiority": ("DEMONSTRATED" if demonstrated
                               else "NOT DEMONSTRATED"),
            "power_adequate": bool(powered.get(d_pp, False)),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- exploratory
def bh_adjust(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (exploratory analyses ONLY)."""
    p = pvals.to_numpy()
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running_min = 1.0
    for rank_from_end, i in enumerate(order[::-1]):
        rank = m - rank_from_end
        running_min = min(running_min, p[i] * m / rank)
        adj[i] = running_min
    return pd.Series(adj, index=pvals.index, name="p_bh")
