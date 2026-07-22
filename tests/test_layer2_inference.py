"""Layer-2 statistical machinery: each estimator is checked against an
independent reference implementation or a known closed form — the inference
that gates the ship decision must not be self-certifying."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import config, inference, power


# ----------------------------------------------------------- non-inferiority
def test_newcombe_upper_bound_matches_statsmodels():
    from statsmodels.stats.proportion import confint_proportions_2indep

    cases = [(40, 3000, 30, 3000), (12, 900, 25, 1100), (300, 30000, 280, 30000)]
    for k_t, n_t, k_c, n_c in cases:
        ours = inference.newcombe_upper_bound(k_t, n_t, k_c, n_c, alpha_one_sided=0.05)
        _, ref_hi = confint_proportions_2indep(
            k_t, n_t, k_c, n_c, method="newcomb", alpha=0.10, compare="diff")
        assert ours == pytest.approx(ref_hi, abs=1e-10)


def test_newcombe_bound_sane_properties():
    ub_small = inference.newcombe_upper_bound(35, 1000, 30, 1000)
    ub_large = inference.newcombe_upper_bound(350, 10000, 300, 10000)
    diff = 35 / 1000 - 30 / 1000
    assert ub_small > diff              # a 95% upper bound sits above the point estimate
    assert ub_large < ub_small          # more data -> tighter bound


def test_noninferiority_verdict_language_and_power_column():
    # not demonstrating non-inferiority must never read as "proven unsafe";
    # power adequacy is a separate, orthogonal column
    tab = inference.noninferiority_verdict(
        k_t=50, n_t=2000, k_c=40, n_c=2000,
        deltas_pp=(0.10, 5.0), powered={0.10: False, 5.0: True})
    r_tight = tab.loc[tab["delta_pp"] == 0.10].iloc[0]
    r_wide = tab.loc[tab["delta_pp"] == 5.0].iloc[0]
    assert r_tight["noninferiority"] == "NOT DEMONSTRATED"
    assert not r_tight["power_adequate"]
    assert r_wide["noninferiority"] == "DEMONSTRATED"
    assert r_wide["power_adequate"]
    assert "FAIL" not in tab.to_string() and "unsafe" not in tab.to_string()


# ------------------------------------------------------------------- power
def test_power_closed_form_spot_check():
    # p=0.04, delta=0.5pp, one-sided 5%, 80%: n = 2*0.04*0.96*(1.645+0.8416)^2/0.005^2
    expect = 2 * 0.04 * 0.96 * (stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) ** 2 / 0.005 ** 2
    got = power.n_per_arm_noninferiority(0.04, 0.005)
    assert got == int(np.ceil(expect))


def test_power_uses_anticipated_difference_not_zero():
    """The audit finding: with a frozen DGP anticipating +0.158pp, the power
    distance at δ=0.30pp is 0.142pp, not 0.30pp — required n explodes and the
    tight margin is UNDERPOWERED even though the zero-diff scenario looks fine."""
    tab = power.power_table(p_base=0.0213, sigma_spend=300, mde=12,
                            deltas_pp=(0.30, 0.50, 0.75),
                            n_available_per_arm=33001,
                            anticipated_diff_pp=0.158)
    t = tab.set_index("delta_pp")
    assert t["n_per_arm_guardrail"][0.30] > t["n_per_arm_guardrail"][0.50] > \
        t["n_per_arm_guardrail"][0.75]
    # anticipated-diff n must exceed the zero-diff sensitivity scenario n
    assert (tab["n_per_arm_guardrail"]
            > tab["n_per_arm_guardrail_zero_diff_scenario"]).all()
    assert not t["power_adequate"][0.30]
    assert t["power_adequate"][0.50] and t["power_adequate"][0.75]
    # closed form at δ=0.50: distance = 0.342pp
    expect = 2 * 0.0213 * 0.9787 * (stats.norm.ppf(0.95)
                                    + stats.norm.ppf(0.80)) ** 2 / 0.00342 ** 2
    assert t["n_per_arm_guardrail"][0.50] == int(np.ceil(expect))


def test_power_refuses_margin_not_exceeding_anticipated_diff():
    with pytest.raises(ValueError, match="undemonstrable"):
        power.n_per_arm_noninferiority(0.02, delta=0.001, anticipated_diff=0.002)


# ------------------------------------------------------------------- CUPED
def test_cuped_preserves_ate_and_reduces_variance():
    rng = np.random.default_rng(5)
    n = 60_000
    pre = rng.lognormal(5, 0.8, n)
    treated = rng.random(n) < 0.5
    post = pre * np.exp(rng.normal(0, 0.3, n)) + treated * 15.0
    adj = inference.cuped_adjust(post, pre)

    raw = inference.mean_diff(post[treated], post[~treated])
    cup = inference.mean_diff(adj[treated], adj[~treated])
    # unbiased: both estimates near the true ATE of 15
    assert cup["diff"] == pytest.approx(15.0, abs=4 * cup["se"])
    assert abs(cup["diff"] - raw["diff"]) < 4 * raw["se"]
    # variance genuinely reduced (pre/post correlation is high by design)
    assert cup["se"] < 0.6 * raw["se"]


def test_cuped_bootstrap_reestimates_theta_and_covers_truth():
    rng = np.random.default_rng(17)
    n = 30_000
    pre = rng.lognormal(5, 0.8, n)
    treated = (rng.random(n) < 0.5).astype(int)
    post = pre * np.exp(rng.normal(0, 0.3, n)) + treated * 15.0
    lo, hi = inference.cuped_bootstrap_ci(post, pre, treated, n_boot=800, seed=1)
    assert lo < 15.0 < hi
    # interval must not be degenerate and must be far tighter than plain
    plain = inference.mean_diff(post[treated == 1], post[treated == 0])
    assert 0 < (hi - lo) < 2 * (plain["ci_hi"] - plain["ci_lo"])


def test_ancova_crosscheck_agrees_with_cuped():
    rng = np.random.default_rng(23)
    n = 40_000
    pre = rng.lognormal(5, 0.8, n)
    treated = (rng.random(n) < 0.5).astype(int)
    post = pre * np.exp(rng.normal(0, 0.3, n)) + treated * 15.0
    adj = inference.cuped_adjust(post, pre)
    cup = inference.mean_diff(adj[treated == 1], adj[treated == 0])
    anc = inference.ancova_crosscheck(post, pre, treated)
    # asymptotically equivalent estimators: point estimates within a fraction
    # of a standard error of each other
    assert abs(anc["diff"] - cup["diff"]) < 0.5 * cup["se"]


# ------------------------------------------------------- spend CATE truth
def test_analytic_cate_matches_oracle_monte_carlo():
    """The analytic E[Y(1)-Y(0)|X] must reproduce a brute-force simulation of
    the frozen DGP branch logic (independent oracle seed) at representative
    covariate points — including negative-tau censoring and the inactive
    branches."""
    from src.experiment_design import DGP_PARAMS, cate_spend_true, tau_spend

    if not (config.ARTIFACTS_DIR / "frozen_thresholds.json").exists():
        pytest.skip("E5 threshold artifact not present")
    p = DGP_PARAMS
    rng = np.random.default_rng(424242)  # oracle seed, disjoint from DGP seeds
    n_mc = 400_000

    for pd_val, s_pre in [(0.005, 200.0), (0.05, 150.0), (0.089, 30.0),
                          (0.089, 3.0), (0.02, 0.0), (0.089, 0.0)]:
        tau = float(tau_spend(np.array([pd_val]), p)[0])
        if s_pre > 0:
            base = s_pre * np.exp(rng.normal(0, p["spend_post_noise_sd"], n_mc))
            mc = (np.maximum(0.0, base + tau) - base).mean()
        else:
            organic = rng.random(n_mc) < p["spend_post_p_activate"]
            org_spend = rng.lognormal(p["spend_pre_mu"], p["spend_pre_sigma"], n_mc)
            y0 = np.where(organic, org_spend, 0.0)
            extra = (rng.random(n_mc) < p["tau_spend_treated_activation"]) & ~organic
            act_spend = rng.lognormal(p["spend_pre_mu"], p["spend_pre_sigma"], n_mc)
            y1 = np.where(organic, np.maximum(0.0, org_spend + tau),
                          np.where(extra, act_spend, np.maximum(0.0, tau)))
            mc = (y1 - y0).mean()
        analytic = float(cate_spend_true(np.array([pd_val]),
                                         np.array([s_pre]), p)[0])
        se_mc = 3.0 if s_pre == 0 else 0.05 + abs(tau) * 0.01
        assert analytic == pytest.approx(mc, abs=4 * se_mc), (
            f"pd={pd_val}, spend_pre={s_pre}: analytic {analytic} vs MC {mc}")


# --------------------------------------------------------------------- BH
def test_bh_matches_scipy_reference():
    rng = np.random.default_rng(9)
    p = pd.Series(rng.random(20) ** 2)
    ours = inference.bh_adjust(p)
    ref = stats.false_discovery_control(p.to_numpy(), method="bh")
    np.testing.assert_allclose(ours.to_numpy(), ref, rtol=1e-12)


# ------------------------------------------------------------ design freeze
def test_design_freeze_refuses_drift(tmp_path, monkeypatch):
    from src import experiment_design as ed

    monkeypatch.setattr(ed, "DESIGN_PATH", tmp_path / "design.json")
    monkeypatch.setattr(config, "OUTCOME_MARKER", tmp_path / "marker.json")
    ed.freeze(eligible_default_rate=0.037, n_eligible=94291,
              linkage_manifest_sha="abc")
    # same inputs: fine (idempotent)
    ed.freeze(eligible_default_rate=0.037, n_eligible=94291,
              linkage_manifest_sha="abc")
    # changed planning baseline: hard error
    with pytest.raises(RuntimeError, match="design mismatch"):
        ed.freeze(eligible_default_rate=0.045, n_eligible=94291,
                  linkage_manifest_sha="abc")


def test_dgp_params_code_drift_detected(tmp_path, monkeypatch):
    from src import experiment_design as ed

    monkeypatch.setattr(ed, "DESIGN_PATH", tmp_path / "design.json")
    monkeypatch.setattr(config, "OUTCOME_MARKER", tmp_path / "marker.json")
    ed.freeze(0.037, 94291, "abc")
    monkeypatch.setitem(ed.DGP_PARAMS, "tau_default_cap", 0.001)  # tamper
    with pytest.raises(RuntimeError, match="no longer match"):
        ed.load_frozen()
