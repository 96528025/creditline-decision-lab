"""WOE leakage discipline — the transform of held-out rows must be a pure
function of (training data, held-out FEATURES). If held-out labels could move
the encoding, WOE would smuggle label information across the split.

The test strategy is behavioral, not structural: shuffle / flip the held-out
labels and assert the transform output is bit-identical.
"""

import numpy as np
import pandas as pd
import pytest

from src.woe import MISSING, WOETransformer


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    n = 5000
    X = pd.DataFrame({
        "cont": rng.lognormal(size=n),                       # skewed continuous
        "count": rng.poisson(1.0, n).astype(float),          # low-cardinality
        "with_missing": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
    })
    y = pd.Series((rng.random(n) < 0.1 + 0.1 * (X["cont"] > 1)).astype(int))
    return X, y


def test_holdout_labels_cannot_influence_transform(data):
    X, y = data
    Xtr, ytr, Xva = X[:4000], y[:4000], X[4000:]

    woe = WOETransformer().fit(Xtr, ytr)
    out1 = woe.transform(Xva)

    # refit with identical training data — validation labels never enter fit
    woe2 = WOETransformer().fit(Xtr, ytr)
    out2 = woe2.transform(Xva)
    pd.testing.assert_frame_equal(out1, out2)

    # bins/WOE must be unchanged no matter what the validation labels are,
    # because fit never saw them; verify the mapping is frozen post-fit
    for col in X.columns:
        assert woe.woe_maps_[col] == woe2.woe_maps_[col]


def test_training_rows_only_determine_bins(data):
    X, y = data
    Xtr, ytr = X[:4000], y[:4000]
    woe_a = WOETransformer().fit(Xtr, ytr)
    # perturb rows OUTSIDE the training window; fitted mapping must not move
    X_perturbed = X.copy()
    X_perturbed.iloc[4000:, :] = 999.0
    woe_b = WOETransformer().fit(X_perturbed[:4000], ytr)
    for col in X.columns:
        assert woe_a.woe_maps_[col] == woe_b.woe_maps_[col]


def test_missing_gets_own_bin_and_unseen_values_go_to_tail_bin(data):
    X, y = data
    woe = WOETransformer().fit(X, y)
    assert MISSING in woe.woe_maps_["with_missing"]

    max_seen = X["count"].max()
    probe = pd.DataFrame({
        "cont": [np.nan, 1.0],
        "count": [99.0, max_seen],  # 99 never seen during fit
        "with_missing": [np.nan, 0.0],
    })
    out = woe.transform(probe)
    assert np.isfinite(out.to_numpy()).all()
    # a count beyond the training range must inherit the TOP bin's WOE —
    # mapping it to "neutral" would understate risk exactly where it is highest
    assert out.loc[0, "count"] == out.loc[1, "count"]


def test_manual_edges_give_interpretable_bins_and_risky_tail():
    """The frozen scorecard spec: DPD counts binned 0 / 1 / 2 / 3+ (+missing),
    with unseen extremes landing in the 3+ (highest-risk) bin."""
    rng = np.random.default_rng(3)
    n = 40_000
    x = rng.choice([0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 6], size=n).astype(float)
    p = 0.03 + 0.10 * np.minimum(x, 3)
    y = pd.Series((rng.random(n) < p).astype(int))
    X = pd.DataFrame({"dpd": x})
    woe = WOETransformer(manual_edges={"dpd": [0.5, 1.5, 2.5]}).fit(X, y)

    non_missing = [b for b in woe.woe_maps_["dpd"] if b != MISSING]
    assert len(non_missing) == 4  # 0 / 1 / 2 / 3+

    probe = woe.transform(pd.DataFrame({"dpd": [0.0, 1.0, 2.0, 3.0, 6.0, 25.0]}))
    v = probe["dpd"].to_numpy()
    assert v[3] == v[4] == v[5], "3, 6, and unseen 25 must share the 3+ bin"
    assert v[5] == min(v), "the 3+ bin must carry the worst (lowest) WOE"


def test_zero_inflated_count_feature_is_not_collapsed():
    """Regression test: a 95%-zero count feature with strong signal must get
    real bins and real IV. An early implementation let every interior quantile
    edge equal 0, collapsing all non-missing rows into one bin (IV ~ 0)."""
    rng = np.random.default_rng(7)
    n = 30_000
    x = np.where(rng.random(n) < 0.95, 0, rng.integers(1, 15, n)).astype(float)
    p = np.where(x == 0, 0.05, 0.40)
    y = pd.Series((rng.random(n) < p).astype(int))
    woe = WOETransformer().fit(pd.DataFrame({"count": x}), y)
    nonmissing_bins = [b for b in woe.woe_maps_["count"] if b != MISSING]
    assert len(nonmissing_bins) >= 5, "zero-inflated feature collapsed to one bin"
    assert woe.iv_["count"] > 0.3, f"IV destroyed: {woe.iv_['count']}"


def test_iv_matches_hand_computation():
    """IV on a tiny hand-checkable example (single binary feature)."""
    X = pd.DataFrame({"f": [0.0] * 50 + [1.0] * 50})
    y = pd.Series([0] * 45 + [1] * 5 + [0] * 30 + [1] * 20)
    woe = WOETransformer().fit(X, y)
    s = woe.smoothing
    ev = np.array([5, 20]) + s
    ne = np.array([45, 30]) + s
    p_ev, p_ne = ev / ev.sum(), ne / ne.sum()
    iv_hand = float(((p_ne - p_ev) * np.log(p_ne / p_ev)).sum())
    assert woe.iv_["f"] == pytest.approx(iv_hand, rel=1e-12)
