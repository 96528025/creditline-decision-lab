"""Policy-value estimator + portfolio guardrail semantics."""

import numpy as np
import pandas as pd
import pytest

from src import policy


def _rct(n=80_000, seed=7, subgroup_effect=20.0, subgroup_def_effect=0.004):
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.5                     # the subgroup a policy targets
    treated = (rng.random(n) < 0.5).astype(int)
    spend = 200 + rng.normal(0, 80, n) + treated * good * subgroup_effect
    p_def = 0.02 + treated * good * subgroup_def_effect
    default = (rng.random(n) < p_def).astype(int)
    df = pd.DataFrame({"treated": treated, "spend_post": spend,
                       "default_post": default})
    return df, pd.Series(good, index=df.index)


def test_policy_value_recovers_share_times_subgroup_effect():
    df, good = _rct()
    pv = policy.policy_value(df, good)
    # per-eligible incremental spend = share (0.5) * subgroup effect (20)
    assert pv["inc_spend_per_eligible"] == pytest.approx(10.0, abs=1.0)
    lo, hi = pv["inc_spend_per_eligible_ci"]
    assert lo < 10.0 < hi
    # per-eligible incremental default = 0.5 * 0.4pp = 0.2pp
    assert pv["inc_default_pp_per_eligible"] == pytest.approx(0.2, abs=0.12)
    assert pv["inc_default_pp_upper_bound"] > pv["inc_default_pp_per_eligible"]


def test_feasibility_is_portfolio_scale_upper_bound():
    df, good = _rct(subgroup_def_effect=0.012)     # unsafe subgroup effect
    pv = policy.policy_value(df, good)
    assert not policy.feasible(pv, delta_pp=0.30)
    df2, good2 = _rct(seed=9, subgroup_def_effect=0.0)
    pv2 = policy.policy_value(df2, good2)
    assert policy.feasible(pv2, delta_pp=0.50)


def test_policy_specs_freeze_absolute_thresholds():
    rng = np.random.default_rng(3)
    feat_tr = pd.DataFrame({"pd_cal": rng.uniform(0, 0.09, 1000)})
    specs = policy.build_policy_specs(feat_tr, rng.normal(10, 5, 1000),
                                      rng.uniform(0, 0.005, 1000), mde=12.0)
    cut = specs["low_risk_3quintiles"]["pd_max"]
    # applying to a DIFFERENT split must use the frozen absolute cutoff,
    # not that split's own quantile
    feat_va = pd.DataFrame({"pd_cal": rng.uniform(0, 0.02, 500)})  # shifted dist
    targeted = policy.apply_policy(specs["low_risk_3quintiles"], feat_va,
                                   np.zeros(500), np.zeros(500))
    assert targeted.mean() > 0.9                    # nearly all below frozen cut
    assert cut == pytest.approx(float(feat_tr["pd_cal"].quantile(0.60)))


def test_tiny_subgroup_refused():
    df, _ = _rct(n=2000)
    tiny = pd.Series(False, index=df.index)
    tiny.iloc[:60] = True
    with pytest.raises(ValueError, match="too small"):
        policy.policy_value(df, tiny)
