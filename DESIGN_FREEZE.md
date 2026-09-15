# DESIGN FREEZE — CreditLine Decision Lab

**Recorded design date: 2026-07-21.** The amendment log records the project's design and implementation sequence; local dates and content hashes are not independent proof of preregistration.
Everything below binds all later work. Amendments are allowed only *before* the simulated
experiment outcomes are generated, and must be recorded in the Amendment Log at the bottom.
Nothing here may be changed in response to observed treatment effects or policy results.

---

## 0. Central question

> Among customers eligible for a credit-line increase, can we identify a targeting policy
> that raises expected spend while keeping the increase in default risk below a predefined
> tolerance?

One story, three genuinely connected layers: **Risk → Experiment → Decision.**

## 1. Dataset & provenance (honesty statement)

Kaggle **"Give Me Some Credit"** (2011 competition), `cs-training.csv`, ~150,000 labeled
rows, target = `SeriousDlqin2yrs` (serious delinquency / financial distress within two
years). The originating institution, geography, sampling frame, and product context are
**not established** in the public documentation. This project therefore:

- never describes the data as any real lender's data or as representative of the US
  credit-card population;
- treats Layer 1 as a consumer credit-risk **prototype**, and Layers 2–3 as a
  clearly-labeled **simulation** built on these customers' real pre-treatment covariates;
- frames all results as a **methodological prototype**, not evidence about any real
  portfolio.

Raw data is never committed; see `data/download.md`.

## 2. Frozen sample-splitting plan

```
cs-training.csv (~150,000 rows; all seeds fixed in src/config.py)
│
├─[S0] 20% stratified holdout (by target) ──► RISK-TEST (~30k)
│        · FIXED SECONDARY HOLDOUT DIAGNOSTIC for Layer-1 prediction.
│          Primary generalization evidence is the nested outer CV on
│          RISK-DEV; RISK-TEST usage history (it is reused, not pristine)
│          is disclosed in the Amendment Log
│        · does NOT enter the experiment or policy layers
│
└─[S1] 80% RISK-DEV (~120k)
     │
     ├─ 5-fold OUTER cross-fitting ──► one OOF risk score per customer
     │    · ALL fitting for fold k — preprocessing, WOE bins, imputation
     │      statistics, hyperparameter tuning, base model, AND calibrator —
     │      touches only the outer-training data (folds ≠ k). See §3.
     │
     ├─[S2] FROZEN eligibility rule (§6) applied to pre-treatment variables
     │      + calibrated OOF score ──► ELIGIBLE population
     │
     └─ ELIGIBLE
          ├─ 1:1 randomization Treatment / Control; DGP generates outcomes
          │  ONCE, with a fixed seed, only after S2 is frozen and executed
          └─[S3] 40 / 30 / 30 split:
               ├─ POLICY-TRAIN (40%): CATE fitting, candidate-policy construction
               ├─ POLICY-VAL   (30%): policy-value evaluation, threshold selection
               │     └ Layer 2's pre-registered confirmatory analysis (ATE +
               │       guardrail) runs on POLICY-TRAIN ∪ POLICY-VAL
               └─ POLICY-TEST  (30%): the frozen policy is evaluated here
                                      EXACTLY ONCE; untouched by any other analysis
```

Frozen decisions:
- **RISK-TEST is excluded from the experiment layer.** Including it would not leak labels
  into the risk model, but keeping the experiment layer fully disjoint from the risk
  holdout is the cleanest defensible position;
  the cost is a ~20% smaller eligible pool.
- **Layer 2's confirmatory analysis uses POLICY-TRAIN ∪ POLICY-VAL only.** POLICY-TEST is
  invisible to every reported analysis until the single final policy evaluation.

## 3. Nested cross-fitting (calibration and tuning)

For each **outer fold k** (K = 5) on RISK-DEV:

1. **Outer-training data** = folds ≠ k. Nothing below may touch fold k.
2. Preprocessing statistics (imputation values, caps) and **WOE bins** are fit on
   outer-training data only.
3. **Hyperparameter tuning** runs entirely inside the outer-training data via inner
   cross-validation (early stopping likewise uses only inner-validation folds).
4. The **base model** is then fit on the full outer-training data with the tuned settings.
5. **Calibration is itself cross-fitted *within* the outer-training data**: split
   outer-training into inner folds, generate inner-OOF scores (each from a model that
   never saw that inner fold), and fit the calibrator on those inner-OOF (score, label)
   pairs. The calibrator never sees fold k's labels, and — critically — is never fit on
   OOF scores produced by any model whose training data included fold k.
6. Apply base model + calibrator to fold k → fold k's calibrated OOF score.

Rationale: the naive alternative ("fit fold k's calibrator on the other folds' OOF
scores") is subtly leaky, because fold j's OOF score comes from a model trained on data
that *includes fold k*. The nested scheme removes that path. A unit test asserts the
OOF property by retraining with fold k excluded and checking predictions match.

**Frozen before the Linkage phase begins:**
- **Calibrator = isotonic regression**, fit per step 5 above. Rationale: ~77k
  inner-training rows with a ~6.7% event rate are ample for a nonparametric fit;
  isotonic imposes no sigmoid shape on GBDT scores that are already near-calibrated,
  and it is order-preserving — though it may introduce ties (tied calibrated scores
  can slightly reduce AUC relative to the raw scores; observed and accepted).
- **Model carried into Linkage = the monotonic LightGBM**, selected on RISK-DEV
  outer-CV performance (mean AUC 0.865 vs 0.856 for the scorecard) — NOT on
  RISK-TEST results.

## 4. Frozen data-handling rules (known quirks)

Handled explicitly; never fed in as plain numbers, never silently dropped.

### 4.1 Sentinel codes 96/98 in the three delinquency-count fields
These values co-occur across all three fields in the same small set of rows and are
clearly special codes, not counts.
- **Primary rule:** create `dpd_sentinel_flag`; set the three counts to missing.
  LR path: missing forms its own WOE bin. LightGBM path: flag + native NaN.
- **Sensitivity analyses:** (a) cap each count at its maximum non-sentinel value;
  (b) drop the rows. Report stability of AUC / KS / calibration and of the IV ranking.
- These rows are also excluded from eligibility (E4), so they never enter Layers 2–3.

### 4.2 Revolving utilization > 1
- **Primary rule: cap utilization at 2.0** and create `util_extreme_flag` for values > 2.
  Values in (1, 2] are retained as-is (over-limit utilization is plausible). The
  **monotone constraint applies to the capped utilization variable.**
  We do **not** assert a mechanism for values > 2 (e.g., that they are balances recorded
  as ratios); they are simply treated as extreme values whose exact magnitude is not
  trusted beyond the cap.
- **Sensitivity analyses:** (a) set values > 2 to missing + flag; (b) winsorize at the
  99th percentile. Report the same stability metrics.

### 4.3 Other documented handling
- `MonthlyIncome` missing (~20%): missing-indicator; own WOE bin / native NaN.
- `DebtRatio` when income is missing behaves differently (implausibly large values);
  add a `debtratio_income_missing` validity flag.
- `NumberOfDependents` missing (~2.6%): same treatment.
- `age = 0` (single row): set to missing.

All sensitivity analyses are evaluated by cross-validation on RISK-DEV only; RISK-TEST
is spent solely on the primary specification.

## 5. Models (exactly two — scope frozen)

1. **Logistic Regression on WOE-binned features** — governance-friendly scorecard style;
   IV reported per feature. Audit amendments: count features (three DPD counts,
   dependents, real-estate loans) use fixed interpretable bins **0 / 1 / 2 / 3+**
   (+ missing); all numeric bin edges are open-ended so values beyond the training
   range land in a tail bin, never a neutral one; the scorecard's feature list
   excludes flags whose information is fully duplicated by a WOE missing bin
   (`config.LR_DROP_FLAGS`) — the GBDT, which has no WOE layer, keeps all flags.
2. **Monotonic LightGBM** — chosen over XGBoost for native NaN handling (consistent with
   §4's missing-value strategy) and first-class `monotone_constraints`. Constraints:
   predicted default probability **non-decreasing** in capped utilization and in each of
   the three delinquency counts; other features unconstrained (imposing directions on
   age/income would require assumptions we don't want to defend). Framed as
   governance-**aware**, never "regulator-accepted."

No further models. CatBoost, causal forests, survival models, deep learning, and fairness
analysis are Future Work by design.

## 6. Frozen eligibility rule

Applied to pre-treatment variables and the calibrated OOF score only, frozen **before**
any treatment outcome exists:

| # | Rule | Rationale |
|---|------|-----------|
| E1 | Exclude `NumberOfTimes90DaysLate ≥ 1` | severe recent delinquency |
| E2 | Exclude `NumberOfTime60-89DaysPastDueNotWorse ≥ 1` | severe recent delinquency |
| E3 | Exclude `NumberOfTime30-59DaysPastDueNotWorse ≥ 2` | 0–1 mild delinquencies allowed; ≥ 2 excluded |
| E4 | Exclude `dpd_sentinel_flag = 1` | untrustworthy records |
| E5 | Exclude calibrated OOF PD in the top 20% of the RISK-DEV distribution | highest-risk tier excluded; the percentile threshold is a **design choice**; its absolute value is recorded and frozen before outcome generation |

There is **no age-based rule.** (An earlier draft had "exclude age < 21"; it was removed
rather than dressed up as a legal or compliance requirement.) The single `age = 0` row is
handled as a missing feature value per §4.3.

The survivors of E1–E5 constitute the documented low-to-medium-risk eligible range.
The rule lives in `src/eligibility.py`; frozen absolute thresholds live in
`src/config.py` and are echoed in the README.

## 7. Guardrail margin δ and power analysis

- **δ = +0.50 percentage points (absolute)** is the non-inferiority margin for the
  incremental default rate. It is specified here as a **hypothetical
  stakeholder-provided business risk tolerance** — the number a risk-management partner
  would hand the analytics team before the test. It is an input to the design, not a
  quantity chosen to fit the available sample size.
- The prospective power analysis computes required sample sizes for **both** (a) spend
  superiority at the minimum economically meaningful lift and (b) default non-inferiority
  at δ, and takes the larger. It additionally reports a **sensitivity table at
  δ ∈ {0.30, 0.50, 0.75} pp**. At δ = 0.30 pp the design is expected to be underpowered
  given the eligible population; if so, it is reported as **UNDERPOWERED**, plainly —
  assumptions are not adjusted after outcomes are observed.
- One-sided 95% upper confidence bound (score-based / Newcombe-style, or validated
  bootstrap) for the default-rate difference; the guardrail passes only if that bound
  is below δ. "Not statistically significant" is never presented as "proven safe."

## 8. Layer 3 default constraint is POLICY-LEVEL

Individual CATE estimates — especially τ_default(x) for a rare binary outcome — are
noisy. Therefore, frozen procedure:

1. On **POLICY-TRAIN**: fit τ_spend(x), τ_default(x) (T-learner baseline + X-learner)
   and construct candidate policies from them plus the baseline risk score.
2. On **POLICY-VAL**: evaluate each candidate's **aggregate** incremental spend and
   **aggregate** incremental default with a randomized-experiment policy-value estimator.
   A candidate is **feasible** only if the one-sided upper confidence bound on its
   policy-level incremental default is below δ.
3. Freeze the selected policy; evaluate it **exactly once** on POLICY-TEST.

An individual customer's τ_default upper bound is **never** interpreted as a validated
per-customer safety guarantee. *Individual effects construct the policy; safety is
validated at the portfolio level.* If the policy-level guardrail is underpowered, it is
reported as **INCONCLUSIVE** — tied back to §7 — rather than forced to a "safe"
conclusion. Because this is a simulation with known ground truth, the guardrail
*decision* is additionally validated against the true incremental default under the DGP.

## 9. Execution-order guarantees

The intended sequence is below. Runtime checks bind the recorded cohort, design parameters, outcomes, policy models, and test result; they do not freeze every diagnostic or source file:

1. Freeze this document.
2. Split off RISK-TEST; its labels are read only by the Layer-1 holdout diagnostic,
   never by model fitting, selection, or the experiment/policy layers. Target-aware
   EDA reads RISK-DEV only (dev-restricted SQL views).
3. Produce nested cross-fitted OOF scores on RISK-DEV.
4. Record frozen absolute eligibility thresholds (E5 percentile → absolute value).
5. Only then may `simulate_experiment.py` generate treatment assignments and outcomes.
6. Policy work obeys the S3 boundaries; the final policy evaluation on POLICY-TEST is recorded once. Later runs verify and read that saved result without repeating its predictions or outcome comparison.

## Amendment log

| Date | Change | Why | Before outcomes generated? |
|------|--------|-----|---------------------------|
| 2026-07-21 | v1 frozen. Incorporates review amendments: δ framing as stakeholder tolerance with 3-margin sensitivity; nested (not naive) cross-fitted calibration/tuning; utilization primary rule = cap at 2.0 + flag with monotone constraint on the capped variable; age rule removed from eligibility; RISK-TEST excluded from experiment; Layer-2 confirmatory on TRAIN∪VAL; policy-level default constraint. | Reviewer (project owner) sign-off | Yes — no outcomes exist yet |
| 2026-07-21 | **Implementation bug fix + documented second RISK-TEST evaluation.** The first Layer-1 run revealed IV(NumberOfTimes90DaysLate) = 0.04, contradicting domain knowledge; root cause was quantile-edge collapse on zero-inflated counts, which silently put all non-missing rows of the 90+/60-89 DPD features into ONE WOE bin. The binning was fixed (per-value bins for low-cardinality features + collapse guard, with a regression test) and the full pipeline re-run, INCLUDING a second evaluation on RISK-TEST. We record this openly rather than pretending "exactly once" survived: the alternative — shipping a scorecard whose implementation contradicted its spec — is worse. No design constant changed; the frozen split is unchanged; no simulation outcomes exist yet. | Bug fix, discovered by IV sanity review | Yes — no outcomes exist yet |
| 2026-07-21 | **Audit findings (project-owner code review) and remediation.** (1) *Holdout contamination via EDA:* notebook 01 v1 computed target-aware statistics (segment default rates, sentinel-row default rate, utilization risk tables) on the FULL population, so RISK-TEST labels informed exploratory judgment in aggregate before the holdout evaluation. Remediation: all "pristine / untouched / exactly once" claims about RISK-TEST are retired; the **primary generalization evidence is the nested outer CV on RISK-DEV**; RISK-TEST is reframed as a **fixed secondary holdout diagnostic (reused, disclosed)**; target-aware EDA now reads dev-only SQL views (`v_clean_dev` + dev-only segment views); the data is deliberately NOT re-split — a fresh test set would launder, not remove, the contamination. Feature-only statistics (distributions, missingness, sentinel co-occurrence) remain full-population: features are observable at scoring time; only labels carry holdout information. (2) *WOE robustness:* per-value bins produced n=1/n=2 delinquency bins, and unseen high counts mapped to neutral WOE 0. Remediation: count features use fixed interpretable bins 0/1/2/3+ (+ missing); all numeric edges are open-ended so out-of-range values land in a tail bin; the scorecard drops flags duplicated by WOE missing bins (LR has its own feature list; the GBDT keeps all flags). (3) Frozen for Linkage: calibrator = isotonic (§3); linkage model = monotonic LightGBM, selected on RISK-DEV outer-CV, not on RISK-TEST. Full pipeline re-run after these changes — RISK-TEST is a reused fixed holdout and is reported as such. | Project-owner audit | Yes — no outcomes exist yet |
| 2026-07-21 | **Layer 2 truth-bookkeeping amendment (post-outcome; observed data untouched).** After outcome generation, the truth-validation was found to compare estimates against mean(τ_spend) although the frozen DGP's causal effect also includes a treated-activation channel and applies τ to inactive treated customers. The truth artifact was extended to full potential-outcomes accounting (spend_y0/spend_y1/delta per customer, monotone-coupled default potential outcomes); the observed outcomes parquet was verified **value-identical and hash-unchanged** before and after (the marker records the amendment). No design parameter, margin, baseline, or heterogeneity function changed — this is bookkeeping of what the frozen DGP already implied. Known DGP wart kept frozen: organic post-period activators share one scalar lognormal draw. | Truth-accounting fix, discovered during validation | No — outcomes existed; hence hash-verified no-touch procedure |
| 2026-07-21 | **Layer-2 audit round 2 (statistical corrections; observed data untouched).** (1) *Power estimand:* non-inferiority power was computed assuming zero true difference although the frozen DGP anticipates ≈+0.158pp incremental default; corrected to distance = δ − anticipated difference. Revised guardrail n/arm: 128,020 (δ=0.30pp) / 22,056 (0.50) / 7,360 (0.75) vs 33,001 available → **δ=0.30pp is UNDERPOWERED; non-inferiority NOT demonstrated there**; the frozen 0.50pp decision (DEMONSTRATED, powered) and the SHIP verdict are unchanged. Zero-difference n retained as a labeled sensitivity scenario. Verdict language changed from FAIL to NOT DEMONSTRATED with a separate power-adequacy column — failure to demonstrate is never read as proven harm. (2) *Spend CATE truth:* delta_spend_true is a realized finite-sample contrast, not E[Y(1)−Y(0)|X]; added `cate_truth.parquet` with the analytic conditional expected effect (lognormal-censoring closed forms, cross-validated against an oracle-seed Monte Carlo), hash-bound into the marker; PEHE may only be computed against it. Under the corrected estimand the CUPED CI misses the expected ATE (+23.19) by 0.08 — a disclosed ≈2σ event; the plain CI covers. (3) *CUPED bootstrap* now re-estimates θ inside every replicate, with an ANCOVA+HC3 cross-check (agrees to the cent). (4) Planning σ simulator now includes the DGP's 2% organic activation. Limitations to carry into write-up: the additive-τ mechanism activates nearly all inactive treated customers (active rate 85.4%→99.5%) — a strong mechanism, not "3% activation"; organic activators share one scalar lognormal draw. | Project-owner audit (round 2) | No — outcomes existed; analysis/planning code only, outcome hashes unchanged |
| 2026-07-21 | **Layer-3 freeze hardening (audit P1: the policy freeze did not freeze an executable policy).** The v1 policy_freeze.json stored a name and a non-loadable spec (`null` = infinity) while τ̂ came from in-memory refits. Migration to freeze v2: X-learner boosters persisted and content-hashed; loadable finite-constraints spec; input-state hashes (outcome marker, linkage manifest, TRAIN/VAL/TEST id hashes, feature schema); candidate-table hash + selection and test decision rules recorded. Reruns verify-and-load; they never refit-for-decision, re-select, or re-predict POLICY-TEST. **The stored TEST evaluation was NOT re-run:** model identity with the v1 in-memory models was proven by byte-identical reproduction of the VAL candidate table, and the TEST side was verified from targeting counts (covariates + assignment) only — no TEST outcome was read; the TEST artifact gained hash bindings with all numbers unchanged. Also fixed (P2): the risk-tier summary table now uses TRAIN∪VAL rows only, so even covariate-only summaries respect the TEST boundary. Tamper tests cover every vector: same winner name with changed spec, model files, input hashes, candidate table, δ, or freeze hash. | Project-owner audit (round 3) | No — outcomes and TEST evaluation existed; verified no-touch migration |
| 2026-07-21 | **Pre-publication freeze-integrity fixes (audit round 4).** (1) *The POLICY-TEST result's own numbers were unprotected* — it was bound to the freeze, marker, and id hashes (proving the conditions it was produced under) but nothing detected an edit to the evaluation values themselves. Added `result_sha256` over the artifact's payload, anchored in the outcome marker. Because the result records the marker hash while the marker now records the result hash, the marker reference switched to an **identity hash** (marker minus Layer-3 back-references) to break the circularity; the policy freeze's copy of that field was updated accordingly. Pure hash bookkeeping: every evaluation number is byte-identical and POLICY-TEST was **not** re-run (asserted during migration). (2) *The VAL candidate table was written before being verified*, so a disagreeing rerun would have destroyed the frozen-run artifact before erroring. Now hashed in memory, verified, and only then written (`policy.write_artifact_verified`), with a test proving the on-disk file survives a mismatch. (3) Removed the unused `econml` dependency and its incorrect comment — the meta-learners are hand-rolled on LightGBM; the full suite passes with econml uninstalled. (4) Tightened README and executive-summary wording: no blanket "no leakage" claim (the RISK-TEST breach is disclosed), "proof" → a graded check on one design, and tier-level spend figures labeled as model estimates. | Project-owner audit (round 4), pre-publication | No — outcomes and TEST evaluation existed; hash-bookkeeping only |
| 2026-09-15 | **Entry-point preservation fixes.** Refuse Layer-1 refits after downstream freeze; refuse recreation of a missing frozen split, outcome marker, CATE truth file, policy freeze, or already-recorded policy test result. Verify the saved design payload against its digest and verify outcome bytes before Layer-3 work. README metrics now match the retained holdout artifact and reproduction instructions distinguish verification from a new experiment. | Regression checks for missing/corrupted state and documentation consistency | No — statistical specifications, model files, outcomes, figures, and evaluation values unchanged |
