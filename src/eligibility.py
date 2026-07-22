"""The FROZEN eligibility rule E1–E5 — DESIGN_FREEZE.md §6.

Consumes only pre-treatment variables and the calibrated OOF score. The E5
percentile is a design choice frozen in config; its ABSOLUTE threshold is
computed once from the RISK-DEV OOF distribution and persisted to
reports/artifacts/frozen_thresholds.json BEFORE any simulated outcome exists
(execution-order guarantee §9). Nothing here may change in response to
observed treatment effects.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from . import config

DPD_3059, DPD_6089, DPD_90 = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
)


def compute_and_freeze_e5_threshold(oof_cal: pd.Series) -> float:
    """Absolute E5 threshold = the frozen percentile of the RISK-DEV calibrated
    OOF PD distribution. Persisted once; loaded (never recomputed) afterwards.

    On its own this file cannot prove it belongs to the current score artifact;
    that binding is enforced by the linkage manifest (src/manifest.py), which
    stores the threshold alongside the OOF parquet's content hash and refuses
    mismatched combinations."""
    path = config.ARTIFACTS_DIR / "frozen_thresholds.json"
    if path.exists():
        return float(json.loads(path.read_text())["e5_abs_pd_threshold"])
    thr = float(oof_cal.quantile(config.ELIG_RISK_PERCENTILE))
    path.write_text(json.dumps({
        "e5_percentile": config.ELIG_RISK_PERCENTILE,
        "e5_abs_pd_threshold": thr,
        "n_risk_dev": int(len(oof_cal)),
        "frozen_on": str(date.today()),
        "note": "frozen BEFORE any simulated outcome generation (DESIGN_FREEZE.md §9)",
    }, indent=2))
    return thr


def eligibility(clean_dev: pd.DataFrame, oof_cal: pd.Series,
                e5_threshold: float) -> tuple[pd.Series, pd.DataFrame]:
    """Apply E1–E5. Returns (eligible mask, sequential attrition funnel).

    NaN delinquency counts occur only for sentinel rows; comparisons with NaN
    are False, so E1–E3 pass them through and E4 removes them explicitly.
    """
    e1 = clean_dev[DPD_90] >= 1                       # severe recent delinquency
    e2 = clean_dev[DPD_6089] >= 1
    e3 = clean_dev[DPD_3059] >= config.ELIG_MAX_3059DPD + 1
    e4 = clean_dev["dpd_sentinel_flag"] == 1          # untrustworthy records
    e5 = oof_cal > e5_threshold                       # highest predicted-risk tier

    rules = [("start", pd.Series(False, index=clean_dev.index)),
             ("E1 90+ DPD >= 1", e1), ("E2 60-89 DPD >= 1", e2),
             (f"E3 30-59 DPD >= {config.ELIG_MAX_3059DPD + 1}", e3),
             ("E4 sentinel rows", e4),
             (f"E5 calibrated PD > {e5_threshold:.4f} (p{int(config.ELIG_RISK_PERCENTILE*100)})", e5)]

    excluded = pd.Series(False, index=clean_dev.index)
    funnel_rows = []
    for name, rule in rules:
        newly = rule & ~excluded
        excluded = excluded | rule
        funnel_rows.append({"rule": name,
                            "newly_excluded": int(newly.sum()),
                            "remaining": int((~excluded).sum())})
    funnel = pd.DataFrame(funnel_rows)
    return ~excluded, funnel
