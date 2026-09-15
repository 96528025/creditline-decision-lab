"""DGP properties. generate_outcomes is a pure function of (cohort, design),
so it can be tested without touching any frozen artifact."""

import numpy as np
import pandas as pd
import pytest

from src import config, experiment_design
from src.simulate_experiment import (assign_splits_and_treatment,
                                     generate_outcomes, load_frozen_cohort)

# load_frozen_cohort() re-verifies the manifest (which hashes the raw CSV) and reloads
# the raw data, so the frozen artifacts alone are not enough to run these tests.
pytestmark = pytest.mark.skipif(
    not ((config.ARTIFACTS_DIR / "linkage_manifest.json").exists()
         and config.DATA_RAW.exists()),
    reason="frozen linkage cohort not built or raw data not downloaded",
)


@pytest.fixture(scope="module")
def cohort():
    return load_frozen_cohort()


@pytest.fixture(scope="module")
def design(cohort):
    return {
        "dgp_params": experiment_design.DGP_PARAMS,
        "planning_control_default_rate": float(cohort[config.TARGET].mean()),
    }


def test_splits_and_randomization_are_exact_and_balanced(cohort):
    a = assign_splits_and_treatment(cohort)
    shares = a["policy_split"].value_counts(normalize=True)
    for name, want in zip(("policy_train", "policy_val", "policy_test"),
                          config.POLICY_SPLIT):
        assert shares[name] == pytest.approx(want, abs=0.005)
    for _, g in a.groupby("policy_split"):
        assert abs(g["treated"].sum() - len(g) / 2) <= 1  # exact 1:1
    # randomization balance on a pre-treatment covariate
    t = cohort.loc[a["treated"] == 1, "pd_cal"].mean()
    c = cohort.loc[a["treated"] == 0, "pd_cal"].mean()
    assert abs(t - c) < 0.001


def test_outcomes_deterministic_given_frozen_seed(cohort, design):
    o1, t1 = generate_outcomes(cohort, design)
    o2, t2 = generate_outcomes(cohort, design)
    pd.testing.assert_frame_equal(o1, o2)
    pd.testing.assert_frame_equal(t1, t2)


def test_outcomes_artifact_contains_no_truth_columns(cohort, design):
    outcomes, truth = generate_outcomes(cohort, design)
    leaked = set(outcomes.columns) & set(truth.columns)
    assert not leaked, f"truth columns leaked into outcomes: {leaked}"
    assert set(outcomes.columns) == {"policy_split", "treated", "spend_pre",
                                     "spend_post", "default_post"}


def test_dgp_matches_frozen_planning_assumptions(cohort, design):
    outcomes, truth = generate_outcomes(cohort, design)
    ctrl = outcomes[outcomes["treated"] == 0]
    # control default rate consistent with the frozen planning baseline
    assert ctrl["default_post"].mean() == pytest.approx(
        design["planning_control_default_rate"], abs=0.004)
    # designed heterogeneity: spend effect positive for low risk, negative at
    # the eligibility boundary; incremental default increasing in risk
    lo = cohort["pd_cal"] < cohort["pd_cal"].quantile(0.2)
    hi = cohort["pd_cal"] > cohort["pd_cal"].quantile(0.8)
    assert truth.loc[lo, "tau_spend_true"].mean() > truth.loc[hi, "tau_spend_true"].mean()
    assert truth.loc[hi, "tau_default_true"].mean() > truth.loc[lo, "tau_default_true"].mean()
    # true aggregate incremental default sits below the frozen delta margin
    assert truth["tau_default_true"].mean() * 100 < config.DELTA_NONINF_PP
