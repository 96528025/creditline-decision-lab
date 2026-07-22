"""Frozen eligibility rule E1–E5: unit behavior + SQL/Python parity."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import config, data_prep
from src.eligibility import DPD_3059, DPD_6089, DPD_90, eligibility


def _toy():
    df = pd.DataFrame({
        DPD_90:  [0, 1, 0, 0, np.nan, 0],
        DPD_6089: [0, 0, 1, 0, np.nan, 0],
        DPD_3059: [1, 0, 0, 2, np.nan, 0],
        "dpd_sentinel_flag": [0, 0, 0, 0, 1, 0],
    })
    pd_scores = pd.Series([0.02, 0.01, 0.03, 0.02, 0.50, 0.30])
    return df, pd_scores


def test_rules_fire_individually_and_funnel_is_sequential():
    df, scores = _toy()
    mask, funnel = eligibility(df, scores, e5_threshold=0.25)
    # row0 eligible (one mild 30-59 allowed); row1 E1; row2 E2; row3 E3;
    # row4 E4 (sentinel; NaN counts pass E1-E3); row5 E5 (0.30 > 0.25)
    assert mask.tolist() == [True, False, False, False, False, False]
    assert funnel["newly_excluded"].tolist() == [0, 1, 1, 1, 1, 1]
    assert funnel["remaining"].iloc[-1] == 1


def test_e3_boundary_allows_exactly_one_mild_delinquency():
    df, scores = _toy()
    df.loc[0, DPD_3059] = config.ELIG_MAX_3059DPD      # exactly at the limit
    mask, _ = eligibility(df, scores, e5_threshold=0.25)
    assert bool(mask.iloc[0])
    df.loc[0, DPD_3059] = config.ELIG_MAX_3059DPD + 1  # one past the limit
    mask, _ = eligibility(df, scores, e5_threshold=0.25)
    assert not bool(mask.iloc[0])


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="warehouse not built")
def test_sql_prescore_cohort_matches_python_e1_to_e4():
    """sql/02_cohort_eligibility.sql implements E1-E4; the Python rule with an
    E5 threshold no rule can reach must agree with it row-for-row."""
    con = sqlite3.connect(config.DB_PATH)
    try:
        sql = pd.read_sql(
            "SELECT customer_id, eligible_pre_score FROM v_eligibility_candidates"
            " ORDER BY customer_id", con)
    finally:
        con.close()

    raw = data_prep.load_raw()
    from src.splits import make_or_load_split
    dev_idx, _ = make_or_load_split(raw)
    dev = data_prep.CleaningRules("primary").fit_transform(raw.loc[dev_idx])
    mask, _ = eligibility(dev, pd.Series(0.0, index=dev.index), e5_threshold=1.1)

    # customer_id is 1-based over the raw file; dev row index is 0-based
    py = pd.Series(mask.to_numpy(), index=dev.index + 1)
    sql_s = sql.set_index("customer_id")["eligible_pre_score"].astype(bool)
    assert len(py) == len(sql_s)
    pd.testing.assert_series_equal(py.sort_index(), sql_s.sort_index(),
                                   check_names=False)


def test_frozen_threshold_artifact_is_loaded_not_recomputed(tmp_path, monkeypatch):
    from src import eligibility as em
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    first = em.compute_and_freeze_e5_threshold(pd.Series(np.linspace(0, 1, 101)))
    # a different score distribution must NOT move the frozen threshold
    second = em.compute_and_freeze_e5_threshold(pd.Series(np.linspace(0, 0.1, 101)))
    assert first == second == pytest.approx(0.8)
    assert json.loads((tmp_path / "frozen_thresholds.json").read_text())[
        "e5_abs_pd_threshold"] == pytest.approx(0.8)
