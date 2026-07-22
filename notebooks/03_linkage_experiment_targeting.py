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
# # 03 — Linkage → Simulated Experiment → Targeting Decision
#
# **Everything from here on is a SIMULATION built on real pre-treatment
# covariates.** The risk layer (notebooks 01–02) models real outcomes; the
# credit-line experiment and targeting policy below use simulated treatment
# and outcomes from a frozen, fully-documented data-generating process (DGP).
# Because the DGP is known, every inferential *procedure* can be graded
# against ground truth — which is the point: the deliverable is an auditable
# decision pipeline, not the simulated numbers themselves.
#
# This notebook READS frozen artifacts (`python -m src.run_linkage`,
# `run_layer2`, `run_layer3`); it recomputes nothing, so re-executing it
# cannot touch any frozen boundary.

# %%
import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src import config  # noqa: E402

ART, FIG = config.ARTIFACTS_DIR, config.FIGURES_DIR
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# %% [markdown]
# ## 1. Linkage: calibrated OOF scores → frozen eligibility cohort
#
# Every RISK-DEV customer carries a PD produced by machinery that never saw
# their own label: nested cross-fitting (5 outer folds; tuning, WOE,
# preprocessing, and the **isotonic calibrator itself** all fit strictly
# inside outer-training data via inner cross-fits). A behavioral unit test
# flips a fold's labels and requires bit-identical scores for that fold.
# Isotonic calibration is order-preserving but introduces ties, which is why
# calibrated AUC dips microscopically (0.8654 → 0.8651) while Brier improves.

# %%
display(json.loads((ART / "oof_calibration_metrics.json").read_text()))
display(Image(str(FIG / "oof_calibration.png")))

# %% [markdown]
# The eligibility rule E1–E5 (frozen in `DESIGN_FREEZE.md` §6, absolute E5
# threshold recorded before any outcome existed) leaves 94,291 of 120,000
# (78.6%). The whole cohort is bound by `linkage_manifest.json` — content
# hashes over raw data, split, scores, threshold, and eligible IDs; any
# mismatch is a hard error, and the Layer-2 outcome marker forbids cohort
# mutation afterwards. "Frozen" is machine-checked, not promised.

# %%
display(pd.read_csv(ART / "eligibility_funnel.csv"))

# %% [markdown]
# ## 2. Frozen experiment design and power
#
# Design frozen BEFORE outcome generation (`experiment_design_frozen.json`):
# planning control default rate = the eligible cohort's **historical** rate
# (2.13%) — a pre-treatment planning proxy, explicitly not an observed control
# outcome; spend MDE = \$12 (hypothetical stakeholder input, like δ = 0.50pp);
# the pre-registered decision rule; and the full DGP with heterogeneous true
# effects (spend effect positive for low-risk eligible, negative near the risk
# boundary; incremental default increasing in risk).
#
# **Power is computed at the anticipated true difference.** The frozen DGP
# itself anticipates ≈ +0.158pp incremental default, so the distance to the
# non-inferiority boundary is δ − 0.158pp, not δ. This is the difference
# between a design that is honest with itself and one that isn't:

# %%
display(pd.read_csv(ART / "power_analysis.csv"))

# %% [markdown]
# At δ = 0.30pp the design is **UNDERPOWERED** (needs ~128k/arm, has 33k) —
# reported as such, with the zero-difference n kept only as a labeled
# sensitivity scenario. The guardrail is the binding constraint at tight
# margins; spend binds at 0.75pp.

# %% [markdown]
# ## 3. Confirmatory analysis (POLICY-TRAIN ∪ VAL; TEST untouched here)

# %%
res = json.loads((ART / "layer2_confirmatory_results.json").read_text())
display({k: res[k] for k in ("spend_plain", "spend_cuped",
                             "spend_ancova_hc3_crosscheck")})
display(pd.DataFrame(res["guardrail"]))
print("decision rule:", res["decision_rule"])
print("ATE decision:", res["ATE_decision"])
display(Image(str(FIG / "layer2_results.png")))

# %% [markdown]
# Reading this honestly:
# - **CUPED** (θ re-estimated inside every bootstrap replicate) cuts the CI
#   width by more than half; the ANCOVA + HC3 regression cross-check agrees to
#   the cent. The hurdle decomposition and winsorized sensitivity are in the
#   artifact.
# - **Guardrail:** one-sided 95% Newcombe upper bound = 0.316pp. At the frozen
#   0.50pp margin non-inferiority is **DEMONSTRATED** (with adequate power);
#   at 0.30pp it is **NOT DEMONSTRATED** — which, per the power table, is what
#   an underpowered margin was expected to produce regardless of truth. "Not
#   demonstrated" is never read as "proven unsafe."
# - Result: **SHIP at the pre-registered 0.50pp margin, while 0.30pp
#   non-inferiority was not demonstrated under an underpowered
#   tighter-margin design.**

# %% [markdown]
# ### Validation against DGP truth (simulation privilege)

# %%
display(json.loads((ART / "layer2_truth_validation.json").read_text()))

# %% [markdown]
# Two things worth defending in an interview:
# - The guardrail's decision is **correct against truth** (true Δdefault
#   0.158pp < 0.50pp; the 0.316pp bound sits above the truth, as a valid
#   upper bound should).
# - The CUPED CI *misses* the expected ATE (+23.19) by \$0.08 while the plain
#   CI covers it. Diagnosis: randomization drew a slightly spend-poorer
#   treated arm; CUPED's correction was the ex-ante right move, and a 95% CI
#   missing ~5% of the time is what 95% means. We report the miss rather than
#   choosing the estimator that happened to look better.

# %% [markdown]
# ## 4. Targeting (Layer 3): individual effects construct the policy;
# ## safety is validated at the portfolio level
#
# CATE recovery on POLICY-VAL against the **analytic** conditional expected
# effect E[Y(1)−Y(0)|X] (never against realized noisy contrasts):

# %%
cv = json.loads((ART / "cate_validation.json").read_text())
display(pd.DataFrame({k: {"pehe": v["pehe"], "rank_corr": v["rank_corr"]}
                      for k, v in cv.items()}).T)

# %% [markdown]
# - **Spend:** the X-learner recovers real structure (rank corr 0.65; PEHE
#   ~\$10 on an effect ranging ±\$30), and clearly beats the T-learner
#   baseline — the expected regime when effects are smoother than outcomes.
# - **Default: rank corr ≈ 0.03 — individual default CATEs are statistically
#   worthless here, and we say so.** This is the empirical justification for
#   the design's central safety choice: no per-customer default guarantee is
#   ever claimed; the guardrail is the independent randomized-experiment
#   estimate of the POLICY-LEVEL incremental default, with a one-sided upper
#   bound against the portfolio tolerance δ.

# %%
display(pd.read_csv(ART / "policy_candidates_val.csv").set_index("policy")
        [["share_targeted", "inc_spend_per_eligible",
          "inc_default_pp_per_eligible", "inc_default_pp_upper_bound", "feasible"]])
display(Image(str(FIG / "layer3_targeting.png")))

# %% [markdown]
# **What the candidate table actually says (restrained reading):** the frozen
# winner `tau_spend_positive` targets 98% of eligibles and beats `treat_all`
# by **\$0.41 per eligible on VAL — far smaller than the estimation
# uncertainty**. This DGP's spend effect is positive for most eligible
# customers, so near-broad coverage is genuinely close to optimal, and the
# honest description of the outcome is a **near-broad-coverage policy under a
# portfolio-level safety constraint** — not a strong-personalization win. The
# targeting machinery still earns its keep: the risk-tier gradient is real
# (τ̂ falls from ~\$35 in the safest quintile to ~\$12 in the riskiest, where
# coverage drops to 93%), and in a world where τ_default had been larger, this
# same pipeline is what would have caught it.

# %% [markdown]
# ### The frozen policy and its single POLICY-TEST evaluation
#
# The policy is an executable, hash-bound artifact: persisted X-learner
# boosters + a loadable spec + input-state hashes in `policy_freeze.json`.
# Reruns verify-and-load; POLICY-TEST was evaluated exactly once, and the
# stored evaluation records the freeze hash it was produced under.

# %%
te = json.loads((ART / "policy_test_evaluation.json").read_text())
display({k: te[k] for k in ("policy", "policy_value_test", "truth_validation")})

# %%
display(pd.read_csv(ART / "policy_by_risk_tier.csv", index_col=0))

# %% [markdown]
# **TEST reading:** the frozen policy delivers +\$26.54 incremental spend per
# eligible [95% CI 19.5, 33.5] versus **no intervention** — the comparison the
# test was designed for; it does **not** establish superiority over
# `treat_all`. The portfolio guardrail holds (UB 0.232pp < 0.50pp), and the
# DGP truth check confirms the decision was correct (true incremental default
# 0.149pp; true incremental spend \$23.05, covered by the CI).
#
# ## 5. Recommendation
#
# **Ship the credit-line increase broadly to the eligible population, with
# the frozen policy's exclusions at the risky margin, under the 0.50pp
# portfolio guardrail** — and monitor with exactly the machinery used here:
# a randomized holdout and a portfolio-level non-inferiority bound, not
# per-customer risk predictions. At a 0.30pp tolerance the honest answer is
# "this design cannot demonstrate it; a larger experiment would be required."

# %% [markdown]
# ---
# *Methodological prototype on a public dataset of undocumented provenance;
# Layers 2–3 are simulation. See README → Limitations before quoting any
# number.*
