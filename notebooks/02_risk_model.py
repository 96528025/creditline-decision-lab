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
# # 02 — Layer 1: Default-Risk Models
#
# Exactly **two** models (scope frozen, `DESIGN_FREEZE.md` §5):
#
# 1. **WOE + Logistic Regression** — scorecard-style interpretable baseline.
#    WOE binning handles skew/outliers, gives missing values a learned bin,
#    and yields IV as a per-feature strength summary a validator can audit.
#    Count features use FIXED interpretable groups (0 / 1 / 2 / 3+ / missing);
#    all numeric edges are open-ended so an unseen extreme value (say 25 past
#    90-day delinquencies) lands in the highest-risk tail bin, never in a
#    neutral bucket. The scorecard has its own feature list: flags that merely
#    duplicate a WOE missing bin are excluded (collinearity, not information).
# 2. **Monotonic LightGBM** — challenger. LightGBM over XGBoost because native
#    NaN handling matches our cleaning rules (sentinels/invalids → NaN + flag)
#    and `monotone_constraints` are first-class. PD is constrained
#    non-decreasing in capped utilization and the three delinquency counts:
#    robustness in sparse regions, business trust (a score must not fall when
#    delinquencies rise), and a *testable* validation property
#    (`tests/test_monotonic.py`). This is governance-**aware** — real
#    governance would also need stability monitoring, fairness analysis, and
#    documentation beyond this project's scope; we claim none of that.
#
# **Evaluation discipline.** All selection (WOE bins, imputation constants,
# hyperparameters) is nested inside training data (§3). The **primary
# generalization evidence is the nested outer 5-fold CV on RISK-DEV**;
# RISK-TEST serves as a **fixed secondary holdout diagnostic**. It is not
# pristine — its labels informed early full-population EDA in aggregate, and
# it was re-evaluated once after a binning bug fix — both disclosed in the
# `DESIGN_FREEZE.md` amendment log. Honest accounting of holdout usage beats
# an "untouched test set" narrative that no longer holds. This notebook only
# *reads* artifacts produced by `python -m src.run_layer1`.

# %%
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src import config  # noqa: E402

ART = config.ARTIFACTS_DIR
FIG = config.FIGURES_DIR
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# %% [markdown]
# ## Primary generalization evidence: nested outer CV (RISK-DEV)
#
# Each fold's hyperparameters were re-tuned inside that fold's training data,
# so the table below reports the stability of the *whole procedure*, not of
# one lucky configuration. These numbers — not the holdout — are the basis for
# model selection (including carrying LightGBM into the Linkage phase).

# %%
stab = pd.read_csv(ART / "stability_outer_cv.csv")
stab_summary = (stab.groupby("model")[["roc_auc", "pr_auc", "ks", "brier"]]
                .agg(["mean", "std"]).round(4))
display(stab_summary)
display(stab[["fold", "model", "roc_auc", "ks",
              "chosen_C", "chosen_num_leaves", "chosen_n_estimators"]])

# %% [markdown]
# ## Secondary holdout diagnostic: RISK-TEST (fixed, reused, disclosed)

# %%
final = json.loads((ART / "final_test_metrics.json").read_text())
final_df = pd.DataFrame({k: v for k, v in final.items() if k != "note"}).T
display(final_df[["roc_auc", "pr_auc", "ks", "brier", "n", "event_rate"]])
print(final["note"])

# %%
display(Image(str(FIG / "roc_test.png")))
display(Image(str(FIG / "calibration_test.png")))

# %% [markdown]
# **Leakage / sanity hunt.** Public benchmarks for this dataset put strong
# models at ROC-AUC ≈ 0.86–0.87 (the 2011 winners were ≈ 0.869 on the private
# leaderboard). Our test AUC sits inside that band, and CV vs test metrics
# agree within fold-level noise. Nothing looks "too good" — a test AUC of 0.95
# here would have meant a leak, not a breakthrough.
#
# The sanity hunt also runs in the other direction: in the first Layer-1 run,
# IV(NumberOfTimes90DaysLate) came out at 0.04 — flatly contradicting domain
# knowledge (a 90-day delinquency is one of the strongest default predictors
# anywhere). Root cause: on a 94.6%-zero count feature, every interior
# quantile edge equals 0, so the WOE binning silently collapsed all
# non-missing rows into a single bin. Fixed (per-value bins for
# low-cardinality features + a collapse guard), covered by a regression test
# (`test_zero_inflated_count_feature_is_not_collapsed`), and the pipeline was
# re-run — **including a second RISK-TEST evaluation, disclosed in the
# `DESIGN_FREEZE.md` amendment log** rather than quietly absorbed. A metric
# that looks too *bad* against domain priors deserves the same suspicion as
# one that looks too good.
#
# A repo audit later found a second discipline breach: v1 of the EDA notebook
# computed target-aware statistics (segment default rates, sentinel risk) on
# the FULL population, so holdout labels informed exploratory judgment in
# aggregate. Remedied by dev-only SQL views + the reframing above; also in the
# amendment log. The through-line of both incidents: when a claim ("untouched
# holdout") stops being true, we change the claim, not the story.
#
# **Calibration.** The scorecard LR is naturally close to calibrated; the
# GBDT's raw scores are usable but Layers 2–3 consume *cross-fitted,
# cross-calibrated* OOF scores produced in the Linkage phase — raw test-set
# calibration here is a Layer-1 diagnostic, not the product.

# %% [markdown]
# ## WOE / Information Value

# %%
iv = pd.read_csv(ART / "iv_table.csv", index_col=0)
display(iv)
display(Image(str(FIG / "iv_ranking.png")))

# %%
# Example WOE bin table: utilization (the strongest feature)
woe_bins = pd.read_csv(ART / "woe_bins.csv")
display(woe_bins[woe_bins["feature"] == "RevolvingUtilizationOfUnsecuredLines"])

# %% [markdown]
# Conventional IV reading: > 0.3 = strong, 0.1–0.3 = medium. Utilization and
# the delinquency counts dominate — consistent with the segment EDA and with
# how the monotone constraints were chosen.

# %% [markdown]
# ## Sensitivity analyses (CV on RISK-DEV only — RISK-TEST untouched)
#
# Hyperparameters held at the primary spec's values so differences isolate the
# cleaning rule. `sentinel_drop` trains without sentinel rows but is still
# *evaluated* on everyone (a deployed model cannot drop applicants).

# %%
sens = pd.read_csv(ART / "sensitivity_cv.csv")
sens_summary = (sens.groupby(["variant", "model"])[["roc_auc", "ks"]]
                .agg(["mean", "std"]).round(4))
display(sens_summary)

# %% [markdown]
# ## SHAP explainability (monotonic LightGBM)
#
# **Framing:** these are *illustrative local risk explanations / a candidate
# reason-code mapping*. They are **not** validated adverse-action notices —
# real adverse-action reason codes require validated mappings, regulatory
# review, and consistency testing far beyond a SHAP plot.

# %%
display(Image(str(FIG / "shap_global_bar.png")))
display(Image(str(FIG / "shap_beeswarm_lgbm.png")))

# %%
for tag in ("low", "median", "high"):
    display(Image(str(FIG / f"shap_local_{tag}_risk.png")))

# %% [markdown]
# ### SHAP vs IV ranking

# %%
comp = pd.read_csv(ART / "shap_vs_iv_ranking.csv", index_col=0)
display(comp)

# %% [markdown]
# IV ranks features by *univariate* separation after binning; SHAP ranks
# *multivariate* contribution inside the fitted GBDT. Agreement at the top
# (utilization, delinquency counts) is reassuring. Divergence lower down is
# expected: correlated features share SHAP credit, and a feature with modest
# IV can matter in interactions (or vice versa). Where the two disagree, the
# scorecard view (IV) is the one a model validator would start from — another
# reason to keep the WOE+LR baseline alongside the GBDT.
