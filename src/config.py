"""Single source of truth for seeds, paths, and FROZEN design constants.

Everything here mirrors DESIGN_FREEZE.md. Constants marked FROZEN may not be
changed after the simulated experiment outcomes are generated (DESIGN_FREEZE.md §9);
any pre-outcome change requires an entry in the Amendment Log.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "cs-training.csv"
DB_PATH = ROOT / "data" / "creditline.db"
FIGURES_DIR = ROOT / "reports" / "figures"
ARTIFACTS_DIR = ROOT / "reports" / "artifacts"  # metrics tables, frozen thresholds

TARGET = "SeriousDlqin2yrs"

# ---------------------------------------------------------------------------
# Seeds (FROZEN). One seed per pipeline stage so stages can be re-run
# independently without perturbing each other's randomness.
# ---------------------------------------------------------------------------
SEED_SPLIT = 20260721        # S0 risk-test holdout + S1 outer fold assignment
SEED_MODEL = 20260722        # model fitting / tuning
SEED_EXPERIMENT = 20260723   # S3 policy splits, randomization, DGP outcomes
SEED_BOOTSTRAP = 20260724    # inference resampling

# ---------------------------------------------------------------------------
# Sample-splitting plan (FROZEN — DESIGN_FREEZE.md §2)
# ---------------------------------------------------------------------------
RISK_TEST_FRACTION = 0.20    # S0: stratified by target; used exactly once
N_OUTER_FOLDS = 5            # S1: outer cross-fitting folds on RISK-DEV
N_INNER_FOLDS = 5            # §3: inner folds for tuning + cross-fitted calibration
POLICY_SPLIT = (0.40, 0.30, 0.30)  # S3: policy-train / policy-val / policy-test

# ---------------------------------------------------------------------------
# Data-handling rules (FROZEN — DESIGN_FREEZE.md §4)
# ---------------------------------------------------------------------------
DPD_SENTINELS = (96, 98)     # special codes in the three delinquency-count fields
UTILIZATION_CAP = 2.0        # primary rule: cap; (1, 2] retained as plausible over-limit
# Sensitivity variants are defined in src/data_prep.py, evaluated on RISK-DEV CV only.

DPD_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# ---------------------------------------------------------------------------
# Eligibility rule E1–E5 (FROZEN — DESIGN_FREEZE.md §6). No age-based rule.
# E5's percentile is a design choice; its ABSOLUTE PD threshold is recorded to
# reports/artifacts/frozen_thresholds.json before any outcome generation.
# ---------------------------------------------------------------------------
ELIG_MAX_90DPD = 0           # E1: exclude any 90+ days late
ELIG_MAX_6089DPD = 0         # E2: exclude any 60-89 days late
ELIG_MAX_3059DPD = 1         # E3: allow 0-1 mild delinquencies, exclude >= 2
# E4: exclude dpd_sentinel_flag == 1
ELIG_RISK_PERCENTILE = 0.80  # E5: exclude calibrated OOF PD above this RISK-DEV pctile

# ---------------------------------------------------------------------------
# Guardrail margin (FROZEN — DESIGN_FREEZE.md §7)
# δ is a HYPOTHETICAL STAKEHOLDER-PROVIDED business risk tolerance — an input
# to the design, not a quantity chosen to fit the available sample size.
# ---------------------------------------------------------------------------
DELTA_NONINF_PP = 0.50               # percentage points, absolute
DELTA_SENSITIVITY_PP = (0.30, 0.50, 0.75)  # power reported at all three margins
ALPHA_ONE_SIDED = 0.05
POWER_TARGET = 0.80

# ---------------------------------------------------------------------------
# Monotone constraints (FROZEN — DESIGN_FREEZE.md §5): PD non-decreasing in
# capped utilization and each delinquency count; all else unconstrained.
# ---------------------------------------------------------------------------
MONOTONE_INCREASING = ["RevolvingUtilizationOfUnsecuredLines", *DPD_COLS]

# ---------------------------------------------------------------------------
# Scorecard (WOE + LR) specification — audit amendments, DESIGN_FREEZE.md §5
# ---------------------------------------------------------------------------
# Fixed, interpretable WOE groups for count features: 0 / 1 / 2 / 3+ (+ missing).
# Interior edges are open-ended, so counts above anything seen in training land
# in the highest-risk tail bin, never in a neutral bucket.
WOE_MANUAL_EDGES = {
    "NumberOfTime30-59DaysPastDueNotWorse": [0.5, 1.5, 2.5],
    "NumberOfTime60-89DaysPastDueNotWorse": [0.5, 1.5, 2.5],
    "NumberOfTimes90DaysLate": [0.5, 1.5, 2.5],
    "NumberOfDependents": [0.5, 1.5, 2.5],
    "NumberRealEstateLoansOrLines": [0.5, 1.5, 2.5],
}
# The LR drops flags whose information is fully duplicated by a WOE missing
# bin (income/dependents missing; sentinel == the three DPD-missing bins).
# util_extreme_flag is kept: it is NOT redundant with the capped value.
# LightGBM keeps all flags (no WOE layer there).
LR_DROP_FLAGS = ["income_missing_flag", "dependents_missing_flag", "dpd_sentinel_flag"]
