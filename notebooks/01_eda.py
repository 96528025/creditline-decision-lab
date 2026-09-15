# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — EDA: Give Me Some Credit
#
# **Dataset provenance (honesty statement).** Kaggle "Give Me Some Credit"
# (2011), 150,000 labeled rows, target = serious delinquency / financial
# distress within two years. The originating institution, geography, and
# sampling frame are **not established** in public documentation. No result is attributed
# to a specific lender or population; this is a methodological prototype — see
# `DESIGN_FREEZE.md` §1.
#
# **Holdout discipline.** All **target-aware** statistics in this notebook
# (default rates, risk-by-segment tables) are computed on **RISK-DEV only**
# (the 80% development split), via dev-restricted SQL views. Feature-only
# statistics (distributions, missingness, sentinel co-occurrence) use the full
# file — features are observable at scoring time; only labels carry holdout
# information. An earlier version of this notebook computed target-aware EDA
# on the full population, which contaminated the "untouched holdout" claim;
# that is disclosed in the `DESIGN_FREEZE.md` amendment log, RISK-TEST is now
# framed as a **fixed secondary holdout diagnostic**, and the primary
# generalization evidence is nested outer CV on RISK-DEV.
#
# This notebook reads from the SQLite warehouse built by `src/load_db.py`
# (raw table + cleaning views + dev-only segment views in `sql/`).

# %%
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src import config, data_prep  # noqa: E402

con = sqlite3.connect(config.DB_PATH)
raw = pd.read_sql("SELECT * FROM raw_credit", con)          # feature-only uses
dev = pd.read_sql(
    "SELECT r.* FROM raw_credit r JOIN risk_dev_ids USING (customer_id)", con
)                                                            # target-aware uses
print(f"full file: {raw.shape}   RISK-DEV: {dev.shape}")
dev.head()

# %% [markdown]
# ## Class imbalance and missingness

# %%
target_rate = dev["serious_dlqin2yrs"].mean()
print(f"RISK-DEV default rate: {target_rate:.4%}  ->  PR-AUC baseline (random) = {target_rate:.4f}")
missing = raw.drop(columns=["serious_dlqin2yrs"]).isna().mean().sort_values(ascending=False)
missing[missing > 0].to_frame("missing_fraction (full file, feature-only)")

# %% [markdown]
# ~6.7% positives — imbalanced enough that ROC-AUC alone can flatter a model;
# we therefore also report PR-AUC and KS in Layer 1. Missingness is
# concentrated in `monthly_income` (~20%) and `dependents` (~2.6%); both get
# missing-indicator flags and (for the scorecard) their own WOE bin, never a
# silent imputation.

# %% [markdown]
# ## Quirk 1 — sentinel codes 96/98 in the delinquency-count fields
#
# The evidence that these are special codes rather than genuine counts:
# they co-occur in the **same rows across all three fields** (a feature-only
# fact, shown on the full file). Their default rate is computed on RISK-DEV.

# %%
pd.read_sql("""
SELECT dpd_30_59 IN (96,98) AS s3059, dpd_60_89 IN (96,98) AS s6089,
       dpd_90 IN (96,98) AS s90, COUNT(*) AS n_full_file
FROM raw_credit
WHERE dpd_30_59 IN (96,98) OR dpd_60_89 IN (96,98) OR dpd_90 IN (96,98)
GROUP BY s3059, s6089, s90
""", con)

# %%
pd.read_sql("""
SELECT COUNT(*) AS n_dev, AVG(serious_dlqin2yrs) AS default_rate_dev
FROM raw_credit r JOIN risk_dev_ids USING (customer_id)
WHERE dpd_30_59 IN (96,98) OR dpd_60_89 IN (96,98) OR dpd_90 IN (96,98)
""", con)

# %% [markdown]
# All 269 flagged rows carry the code in *all three* fields simultaneously —
# a count interpretation would mean someone was 30-59, 60-89, **and** 90+ days
# late 96+ times each, which is not a credible payment history. Their RISK-DEV
# default rate (~55%) is 8x the population rate, so the rows are informative
# and must not be dropped.
#
# **Frozen primary rule (`DESIGN_FREEZE.md` §4.1):** `dpd_sentinel_flag` = 1,
# counts set to missing; the flag carries the risk signal. Sensitivity
# analyses re-run the models with (a) counts capped at the max non-sentinel
# value and (b) rows dropped — results in notebook 02.

# %% [markdown]
# ## Quirk 2 — revolving utilization > 1

# %%
util_dev = dev["revolving_utilization"]
summary = pd.DataFrame({
    "n_dev": [len(util_dev), (util_dev > 1).sum(), (util_dev > 2).sum(), (util_dev > 10).sum()],
    "default_rate_dev": [
        dev["serious_dlqin2yrs"].mean(),
        dev.loc[util_dev > 1, "serious_dlqin2yrs"].mean(),
        dev.loc[util_dev > 2, "serious_dlqin2yrs"].mean(),
        dev.loc[util_dev > 10, "serious_dlqin2yrs"].mean(),
    ],
}, index=["all", "util > 1", "util > 2", "util > 10"])
summary

# %%
util = raw["revolving_utilization"]  # distribution shape: feature-only, full file
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
util.clip(upper=3).hist(bins=60, ax=axes[0])
axes[0].axvline(1.0, color="tab:orange", ls="--", lw=1, label="util = 1")
axes[0].axvline(2.0, color="tab:red", ls="--", lw=1, label="cap = 2 (frozen)")
axes[0].set_title("Utilization (clipped at 3 for display)")
axes[0].legend()
np.log10(util[util > 0]).hist(bins=60, ax=axes[1])
axes[1].set_title("log10(utilization), all positive values")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "eda_utilization.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Values in (1, 2] are plausible over-limit utilization. Beyond 2 the tail
# stretches into the tens of thousands; whatever the mechanism (we do **not**
# assert one), the exact magnitudes are not trustworthy as ratios.
#
# **Frozen primary rule (§4.2): cap at 2.0 + `util_extreme_flag`**, with the
# monotone constraint applied to the capped variable. Sensitivity analyses:
# (a) values > 2 set to missing, (b) p99 winsorization.

# %% [markdown]
# ## Quirk 3 (minor) — DebtRatio changes meaning when income is missing

# %%
pd.read_sql("""
SELECT income_missing_flag,
       COUNT(*) AS n,
       AVG(debt_ratio) AS mean_debt_ratio,
       AVG(serious_dlqin2yrs) AS default_rate
FROM v_clean_dev GROUP BY income_missing_flag
""", con)

# %% [markdown]
# With income missing, `debt_ratio` averages in the hundreds — it is not the
# same quantity (a ratio needs a denominator). `income_missing_flag` therefore
# doubles as a DebtRatio validity flag for the models.

# %% [markdown]
# ## Target rate by segment (dev-only SQL views)

# %%
seg_age = pd.read_sql("SELECT * FROM v_default_by_age", con)
seg_util = pd.read_sql("SELECT * FROM v_default_by_util", con)
seg_inc = pd.read_sql("SELECT * FROM v_default_by_income", con)
seg_dpd = pd.read_sql("SELECT * FROM v_default_by_dpd3059", con)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (seg, col, title) in zip(axes.ravel(), [
    (seg_age, "age_band", "Default rate by age band (RISK-DEV)"),
    (seg_util, "util_band", "Default rate by utilization band (RISK-DEV)"),
    (seg_inc, "income_segment", "Default rate by income quintile (RISK-DEV)"),
    (seg_dpd, "dpd_30_59_band", "Default rate by 30-59 DPD count (RISK-DEV)"),
]):
    ax.bar(seg[col], seg["default_rate"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(target_rate, color="k", ls="--", lw=0.8)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "eda_segments.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Patterns are exactly what domain knowledge predicts — risk falls with age
# and income, rises steeply with utilization and delinquency count. The
# monotone directions frozen for the LightGBM constraints (utilization,
# delinquency counts) are visible here in the **development data**; we
# deliberately do not constrain age or income (§5).
#
# One row has `age = 0` — treated as missing (§4.3). The data documents no
# lower age limit, so the youngest segment is labeled "<30".

# %% [markdown]
# ## Feature distributions (full file, feature-only)

# %%
clean = data_prep.CleaningRules("primary").fit_transform(data_prep.load_raw())
num_cols = ["age", "MonthlyIncome", "DebtRatio", "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines", "NumberOfDependents"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, c in zip(axes.ravel(), num_cols):
    x = clean[c].dropna()
    if c in ("MonthlyIncome", "DebtRatio"):
        x = np.log10(x[x > 0])
        c = f"log10({c})"
    ax.hist(x, bins=50)
    ax.set_title(c, fontsize=10)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "eda_distributions.png", dpi=150, bbox_inches="tight")

# %%
con.close()

# %% [markdown]
# ## Frozen handling summary (feeds Layer 1)
#
# | Quirk | Primary rule (frozen) | Sensitivity analyses |
# |---|---|---|
# | 96/98 sentinels (3 DPD fields, 269 rows) | flag + counts → missing | cap at max non-sentinel; drop rows |
# | utilization > 1 | cap at 2.0 + flag; (1,2] kept | >2 → missing; p99 winsorize |
# | income missing (~20%) | flag + own WOE bin / native NaN | — |
# | DebtRatio w/ missing income | validity flag (= income flag) | — |
# | age = 0 (1 row) | → missing | — |
