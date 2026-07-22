"""Frozen-split integrity + cleaning-rule agreement between Python and SQL.

These are the boring tests that make the rest of the project defensible:
- the persisted RISK-TEST split is disjoint, complete, stratified, and
  reproducible from the frozen seed;
- the SQL v_clean view and src/data_prep.py implement the SAME primary rules
  (two implementations of one spec must not drift apart silently).
"""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config, data_prep


pytestmark = pytest.mark.skipif(
    not config.DATA_RAW.exists(), reason="raw data not downloaded"
)


def test_split_indices_frozen_and_reproducible():
    path = config.ARTIFACTS_DIR / "risk_split_indices.json"
    if not path.exists():
        pytest.skip("split not yet materialized (run src.run_layer1)")
    saved = json.loads(path.read_text())
    dev, test = np.array(saved["dev"]), np.array(saved["test"])
    df = data_prep.load_raw()

    assert len(np.intersect1d(dev, test)) == 0
    assert len(dev) + len(test) == len(df)
    # stratification: test event rate within 0.3pp of population
    assert abs(df.loc[test, config.TARGET].mean() - df[config.TARGET].mean()) < 0.003
    # reproducible from the frozen seed
    dev2, test2 = train_test_split(
        df.index.to_numpy(), test_size=saved["test_fraction"],
        stratify=df[config.TARGET], random_state=saved["seed"],
    )
    assert set(map(int, test2)) == set(saved["test"])


def test_sql_and_python_cleaning_agree():
    if not config.DB_PATH.exists():
        pytest.skip("SQLite db not built (run src.load_db)")
    con = sqlite3.connect(config.DB_PATH)
    try:
        sql = pd.read_sql(
            "SELECT customer_id, revolving_utilization, util_extreme_flag,"
            "       dpd_sentinel_flag, dpd_30_59, age"
            " FROM v_clean ORDER BY customer_id", con)
    finally:
        con.close()

    py = data_prep.CleaningRules("primary").fit_transform(data_prep.load_raw())
    assert len(sql) == len(py)

    np.testing.assert_allclose(
        sql["revolving_utilization"].to_numpy(),
        py["RevolvingUtilizationOfUnsecuredLines"].to_numpy(), rtol=1e-9)
    np.testing.assert_array_equal(
        sql["util_extreme_flag"].to_numpy(), py["util_extreme_flag"].to_numpy())
    np.testing.assert_array_equal(
        sql["dpd_sentinel_flag"].to_numpy(), py["dpd_sentinel_flag"].to_numpy())
    # NaN alignment for a sentinel-nulled count and invalid age
    np.testing.assert_array_equal(
        sql["dpd_30_59"].isna().to_numpy(),
        py["NumberOfTime30-59DaysPastDueNotWorse"].isna().to_numpy())
    np.testing.assert_array_equal(
        sql["age"].isna().to_numpy(), py["age"].isna().to_numpy())


def test_dev_views_exclude_holdout_rows():
    """Target-aware EDA reads v_clean_dev; it must contain exactly the frozen
    RISK-DEV rows, so holdout labels cannot inform exploratory judgment."""
    path = config.ARTIFACTS_DIR / "risk_split_indices.json"
    if not (config.DB_PATH.exists() and path.exists()):
        pytest.skip("db or split not built")
    saved = json.loads(path.read_text())
    con = sqlite3.connect(config.DB_PATH)
    try:
        (n_dev,) = con.execute("SELECT COUNT(*) FROM v_clean_dev").fetchone()
        ids = pd.read_sql("SELECT customer_id FROM risk_dev_ids", con)["customer_id"]
    finally:
        con.close()
    assert n_dev == len(saved["dev"])
    # customer_id is 1-based; the persisted split indices are 0-based
    assert set(ids) == {i + 1 for i in saved["dev"]}


def _clean_dev():
    """Governance tests fit on RISK-DEV only — fitting on the full file would
    contradict the discipline these very tests exist to enforce."""
    from src.splits import make_or_load_split

    raw = data_prep.load_raw()
    dev_idx, _ = make_or_load_split(raw)
    return data_prep.CleaningRules("primary").fit_transform(raw.loc[dev_idx])


def test_lr_feature_list_drops_duplicated_flags():
    from src.risk_models import WoeLogisticModel

    df = _clean_dev().head(20_000)
    y, X = df[config.TARGET], df.drop(columns=[config.TARGET])
    m = WoeLogisticModel(C=1.0).fit(X, y)
    fitted = set(m.woe.woe_maps_)
    assert fitted.isdisjoint(config.LR_DROP_FLAGS)
    assert "util_extreme_flag" in fitted  # not redundant with the capped value


def test_scorecard_has_no_micro_bins():
    """Governance check on the real fitted scorecard: no WOE bin (except the
    explicitly allowed missing bins) may hold fewer than 100 training rows."""
    df = _clean_dev()
    from src.risk_models import WoeLogisticModel

    m = WoeLogisticModel(C=1.0).fit(df.drop(columns=[config.TARGET]), df[config.TARGET])
    for feat, table in m.woe.bin_tables_.items():
        non_missing = table[table.index != "__MISSING__"]
        assert (non_missing["n"] >= 100).all(), (
            f"{feat} has micro-bins:\n{non_missing[non_missing['n'] < 100]}"
        )


def test_sentinel_rows_all_flagged_and_counts_nulled():
    df = data_prep.load_raw()
    cleaned = data_prep.CleaningRules("primary").fit_transform(df)
    flagged = cleaned["dpd_sentinel_flag"] == 1
    assert flagged.sum() == data_prep.sentinel_mask(df).sum()
    for c in config.DPD_COLS:
        assert cleaned.loc[flagged, c].isna().all()
        assert not cleaned.loc[~flagged, c].isin(config.DPD_SENTINELS).any()
