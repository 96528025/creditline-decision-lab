"""Spec-required test: the X-learner must recover a KNOWN heterogeneous
effect within tolerance — for BOTH outcome types (continuous spend-like,
rare binary default-like). Also anchors the T-learner baseline."""

import numpy as np
import pandas as pd
import pytest

from src.cate import TLearner, XLearner, pehe


def _covariates(n, rng):
    return pd.DataFrame({
        "x1": rng.uniform(0, 1, n),      # effect modifier
        "x2": rng.normal(size=n),        # outcome-only signal
        "x3": rng.normal(size=n),        # noise
    })


def test_xlearner_recovers_continuous_heterogeneous_effect():
    rng = np.random.default_rng(31)
    n = 40_000
    X = _covariates(n, rng)
    treated = (rng.random(n) < 0.5).astype(int)
    tau_true = 30.0 * (1.0 - X["x1"]) - 5.0          # +25 .. -5, like the DGP
    y = (100 + 40 * X["x2"] + treated * tau_true
         + rng.normal(0, 30, n)).to_numpy()

    xl = XLearner().fit(X, y, treated)
    tl = TLearner().fit(X, y, treated)

    rng2 = np.random.default_rng(32)
    X_new = _covariates(20_000, rng2)
    tau_new = (30.0 * (1.0 - X_new["x1"]) - 5.0).to_numpy()

    err_x = pehe(xl.predict_tau(X_new), tau_new)
    err_t = pehe(tl.predict_tau(X_new), tau_new)
    # effect range is 30; recovery within a fraction of that range
    assert err_x < 5.0, f"X-learner PEHE {err_x}"
    assert err_t < 8.0, f"T-learner PEHE {err_t}"
    # the smooth-effect regime is where the X-learner should not lose
    assert err_x <= err_t * 1.2


def test_xlearner_recovers_rare_binary_effect_in_aggregate():
    """Individual τ_default estimates are unavoidably noisy for a ~2% outcome;
    the requirement is calibrated AGGREGATE recovery and correct ranking —
    which is exactly how Layer 3 uses them (policy-level, never per-customer)."""
    rng = np.random.default_rng(41)
    n = 120_000
    X = _covariates(n, rng)
    treated = (rng.random(n) < 0.5).astype(int)
    p0 = 0.015 + 0.01 * X["x1"]
    tau_true = (0.008 * X["x1"]).to_numpy()          # 0 .. 0.8pp
    y = (rng.random(n) < (p0 + treated * tau_true)).to_numpy().astype(float)

    xl = XLearner().fit(X, y, treated)
    est = xl.predict_tau(X)

    # aggregate recovery within 0.15pp
    assert est.mean() == pytest.approx(tau_true.mean(), abs=0.0015)
    # ranking: higher-x1 halves must show higher estimated incremental risk
    hi, lo = X["x1"] > 0.5, X["x1"] <= 0.5
    assert est[hi].mean() > est[lo].mean()
    # tier-level calibration: top vs bottom x1 quintile ordering with sane scale
    q = pd.qcut(X["x1"], 5, labels=False)
    est_top, est_bot = est[q == 4].mean(), est[q == 0].mean()
    true_top, true_bot = tau_true[q == 4].mean(), tau_true[q == 0].mean()
    assert est_top - est_bot == pytest.approx(true_top - true_bot, abs=0.003)
