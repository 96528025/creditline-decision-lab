"""Cleaning rules for the Give Me Some Credit data — DESIGN_FREEZE.md §4.

Design of this module
---------------------
Two kinds of rules, kept deliberately separate because they have different
leakage properties:

1. STATIC rules use only frozen constants (sentinel codes 96/98, utilization
   cap 2.0, "age==0 is invalid"). They estimate nothing from data, so applying
   them before splitting cannot leak label or distributional information.
2. FITTED rules (only needed by the sensitivity variants: per-column sentinel
   caps, p99 winsorization) estimate constants from data, so they are fit on
   training folds only, via CleaningRules.fit().

The PRIMARY specification is fully static. Sensitivity variants are evaluated
by cross-validation on RISK-DEV only (never on RISK-TEST).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

UTIL_COL = "RevolvingUtilizationOfUnsecuredLines"
INCOME_COL = "MonthlyIncome"
DEPENDENTS_COL = "NumberOfDependents"

VARIANTS = ("primary", "sentinel_cap", "sentinel_drop", "util_missing", "util_p99")

# Flags added by cleaning. They are legitimate model features: each encodes a
# pre-treatment data condition, not any label information.
FLAG_COLS = [
    "dpd_sentinel_flag",
    "util_extreme_flag",
    "income_missing_flag",
    "dependents_missing_flag",
]


def load_raw(path=None) -> pd.DataFrame:
    """Load cs-training.csv, dropping the unnamed row-index column."""
    path = path or config.DATA_RAW
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.drop(columns=[first])
    assert config.TARGET in df.columns, "target column missing — wrong file?"
    return df


def sentinel_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where any delinquency-count field carries a 96/98 special code.

    EDA fact this rule rests on: the codes co-occur across all three fields in
    the same rows, which is how we know they are codes rather than counts.
    """
    m = pd.Series(False, index=df.index)
    for c in config.DPD_COLS:
        m |= df[c].isin(config.DPD_SENTINELS)
    return m


class CleaningRules:
    """Apply frozen cleaning rules; variant selects primary vs sensitivity spec.

    fit() is a no-op for the primary spec (all constants frozen); the two
    data-dependent variants estimate their constants from the fitted (training)
    data only.
    """

    def __init__(self, variant: str = "primary"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.sentinel_caps_: dict[str, float] | None = None
        self.util_p99_: float | None = None

    def fit(self, df: pd.DataFrame) -> "CleaningRules":
        if self.variant == "sentinel_cap":
            # Cap at the maximum NON-sentinel value observed in training data.
            self.sentinel_caps_ = {
                c: float(df.loc[~df[c].isin(config.DPD_SENTINELS), c].max())
                for c in config.DPD_COLS
            }
        elif self.variant == "util_p99":
            self.util_p99_ = float(df[UTIL_COL].quantile(0.99))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- invalid age (single known row with age == 0) ---
        df.loc[df["age"] == 0, "age"] = np.nan

        # --- DPD sentinel codes ---
        flagged = sentinel_mask(df)
        df["dpd_sentinel_flag"] = flagged.astype(int)
        if self.variant == "sentinel_cap":
            assert self.sentinel_caps_ is not None, "call fit() first"
            for c, cap in self.sentinel_caps_.items():
                df.loc[flagged, c] = np.minimum(df.loc[flagged, c], cap)
        elif self.variant == "sentinel_drop":
            df = df.loc[~flagged].copy()
        else:  # primary and the two utilization variants
            # counts are not trusted for flagged rows; the flag carries the signal
            for c in config.DPD_COLS:
                df.loc[flagged, c] = np.nan

        # --- revolving utilization ---
        # Primary (frozen): cap at 2.0 + flag. (1, 2] retained as plausible
        # over-limit utilization. No mechanism is asserted for values > 2; the
        # magnitude beyond the cap is simply not trusted.
        extreme = df[UTIL_COL] > config.UTILIZATION_CAP
        df["util_extreme_flag"] = extreme.astype(int)
        if self.variant == "util_missing":
            df.loc[extreme, UTIL_COL] = np.nan
        elif self.variant == "util_p99":
            assert self.util_p99_ is not None, "call fit() first"
            df[UTIL_COL] = np.minimum(df[UTIL_COL], self.util_p99_)
        else:
            df[UTIL_COL] = np.minimum(df[UTIL_COL], config.UTILIZATION_CAP)

        # --- missingness flags ---
        # income_missing_flag doubles as the DebtRatio validity flag: DebtRatio
        # is computed against income, so when income is missing its values are
        # implausibly large and mean something different. One flag, two roles —
        # a second identical column would only add perfect collinearity.
        df["income_missing_flag"] = df[INCOME_COL].isna().astype(int)
        df["dependents_missing_flag"] = df[DEPENDENTS_COL].isna().astype(int)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model feature set: all cleaned numerics + flags, minus the target."""
    return [c for c in df.columns if c != config.TARGET]
