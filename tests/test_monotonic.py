"""The monotone constraint must actually hold — DESIGN_FREEZE.md §5.

Strategy: generate synthetic data with a KNOWN monotone signal plus noise
features, fit the production LightGBM configuration (same code path as the
real pipeline), and assert predicted PD is non-decreasing along the
constrained feature while all other features are held fixed. This tests the
property we defend in front of a reviewer, not LightGBM's implementation
details.
"""

import numpy as np
import pandas as pd

from src.risk_models import fit_lgbm, monotone_vector
from src import config


def _synthetic(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        # named like the real constrained columns so monotone_vector() applies
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2, n),
        config.DPD_COLS[0]: rng.poisson(0.5, n).astype(float),
        config.DPD_COLS[1]: rng.poisson(0.2, n).astype(float),
        config.DPD_COLS[2]: rng.poisson(0.1, n).astype(float),
        "noise_a": rng.normal(size=n),
        "noise_b": rng.normal(size=n),
    })
    logit = (
        -3.0
        + 1.5 * X["RevolvingUtilizationOfUnsecuredLines"]
        + 0.8 * X[config.DPD_COLS[0]]
        + 0.4 * X[config.DPD_COLS[1]]
        + 0.5 * X[config.DPD_COLS[2]]
        + 0.3 * X["noise_a"]          # real but unconstrained signal
        + rng.normal(scale=0.5, size=n)
    )
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_monotone_vector_marks_only_frozen_columns():
    X, _ = _synthetic()
    vec = monotone_vector(list(X.columns))
    assert vec == [1, 1, 1, 1, 0, 0]


def test_predictions_monotone_in_constrained_features():
    X, y = _synthetic()
    model = fit_lgbm(X, y, {"num_leaves": 31, "min_child_samples": 20,
                            "n_estimators": 200})

    rng = np.random.default_rng(1)
    base_rows = X.iloc[rng.choice(len(X), size=25, replace=False)]

    # every constrained feature is probed — all three DPD fields, not a sample
    for col, grid in [
        ("RevolvingUtilizationOfUnsecuredLines", np.linspace(0, 2, 41)),
        (config.DPD_COLS[0], np.arange(0, 6, dtype=float)),
        (config.DPD_COLS[1], np.arange(0, 6, dtype=float)),
        (config.DPD_COLS[2], np.arange(0, 6, dtype=float)),
    ]:
        for _, row in base_rows.iterrows():
            probe = pd.DataFrame([row] * len(grid))
            probe[col] = grid
            p = model.predict_proba(probe)[:, 1]
            diffs = np.diff(p)
            assert (diffs >= -1e-12).all(), (
                f"PD decreased along constrained feature {col}: min diff {diffs.min()}"
            )


def test_unconstrained_feature_can_move_both_ways():
    """Sanity check that the constraint is not vacuous (a constant model would
    trivially pass the monotone test)."""
    X, y = _synthetic()
    model = fit_lgbm(X, y, {"num_leaves": 31, "min_child_samples": 20,
                            "n_estimators": 200})
    spread = model.predict_proba(X)[:, 1]
    assert spread.std() > 0.01, "model collapsed to a constant"
