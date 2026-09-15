# CreditLine Decision Lab

[![CI](https://github.com/96528025/creditline-decision-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/96528025/creditline-decision-lab/actions/workflows/ci.yml)

**A Python and SQL decision prototype connecting credit-risk modeling, a simulated randomized experiment, and policy targeting.** It asks whether a credit-line increase could raise spending within a predefined default-risk tolerance.

The risk models use 150,000 anonymized records from Kaggle's *Give Me Some Credit*. The experiment uses real covariates with **simulated treatment, spending, and default outcomes**. Its dollar figures and policy decisions describe that simulation, not a real lending program.

[Executive summary](reports/executive_summary.md) · [Recorded results](reports/artifacts) · [Design and amendments](DESIGN_FREEZE.md) · [Reproduce](#reproduce)

**Stack:** Python, SQL/SQLite, pandas, scikit-learn, LightGBM, SciPy, statsmodels, SHAP, pytest, and Jupyter.

## Engineering highlights

| Problem | Implemented approach | Evidence |
| --- | --- | --- |
| Validation labels can leak into risk scores | Training-only WOE transforms, nested tuning, and calibration entirely within each outer training fold | [`risk_models.py`](src/risk_models.py), [`oof.py`](src/oof.py), label-perturbation tests |
| Experiment results can influence design choices | Recorded eligibility, planning assumptions, margins, and simulation parameters; hashes bind the retained artifacts | [`manifest.py`](src/manifest.py), [`experiment_design.py`](src/experiment_design.py) |
| A non-significant default increase does not establish an acceptable risk bound | Separate power analysis and a one-sided Newcombe confidence bound against a specified margin | [`power.py`](src/power.py), [`inference.py`](src/inference.py) |
| A policy name alone does not preserve the decision | Persisted X-learner models, executable thresholds, candidate-table hashes, and a stored test evaluation | [`cate.py`](src/cate.py), [`run_layer3.py`](src/run_layer3.py), freeze tests |
| A rerun can overwrite evidence or repeat a held-out evaluation | Refuse missing or inconsistent frozen state before regeneration; verify outcome bytes before policy work | [`test_freeze_entrypoints.py`](tests/test_freeze_entrypoints.py) |

There is no hosted application. The committed tables, model files, notebooks, and figures are the inspectable outputs.

## Architecture

```text
150,000 Kaggle credit records
├── RISK-TEST: 30,000 — secondary holdout diagnostic; prior use disclosed
└── RISK-DEV: 120,000
    ├── Layer 1: WOE + logistic regression vs monotonic LightGBM
    │   Five outer folds, inner tuning, nested cross-fitted calibration
    └── Frozen eligibility: 94,291 records
        ├── Layer 2: simulated randomized experiment
        │   Confirmatory analysis on POLICY-TRAIN + POLICY-VAL (66,003)
        └── Layer 3: T-/X-learner comparison and candidate policies
            Train / validation / test split: 40% / 30% / 30%
            Persist policy → evaluate POLICY-TEST once → verify saved result
```

The historical RISK-TEST was touched by early EDA and reused after implementation fixes. Nested outer cross-validation on RISK-DEV is the primary generalization evidence; neither split is described as an independent external validation dataset. The [amendment log](DESIGN_FREEZE.md#amendment-log) preserves this history.

## Recorded results

### 1. Risk models — real data

| Model | Outer-CV ROC-AUC, mean ± SD | Holdout ROC-AUC | Average precision | KS | Brier |
| --- | --- | --- | --- | --- | --- |
| WOE + logistic regression | 0.856 ± 0.006 | 0.851 | 0.387 | 0.548 | 0.0499 |
| Monotonic LightGBM | 0.865 ± 0.005 | 0.859 | 0.405 | 0.569 | 0.0488 |

Sources: [outer-fold results](reports/artifacts/stability_outer_cv.csv) and [holdout metrics](reports/artifacts/final_test_metrics.json). The artifact's `pr_auc` field is computed with scikit-learn's **average precision**, not trapezoidal integration. Compared with each model's primary sensitivity setting, the largest absolute change in mean CV AUC across the recorded cleaning variants is approximately **0.00119** ([sensitivity table](reports/artifacts/sensitivity_cv.csv)).

The LightGBM constraints make predicted risk non-decreasing in capped utilization and the three delinquency inputs, **holding other model inputs fixed**. They do not establish fairness or monotonicity across simultaneous changes in derived flags. SHAP plots are illustrative model explanations.

### 2. Randomized experiment — simulated outcomes

| Measure | Recorded result |
| --- | --- |
| CUPED-adjusted spending difference | +$25.21 per eligible customer; 95% bootstrap interval [$23.27, $27.17] |
| Observed adjustment benefit | About 80% lower outcome variance in this simulation |
| Default difference, one-sided 95% upper bound | 0.316 percentage points |
| Specified 0.50-point margin | Non-inferiority demonstrated; planned sample size adequate |
| Tighter 0.30-point margin | Non-inferiority not demonstrated; planned sample size inadequate |

Sources: [confirmatory results](reports/artifacts/layer2_confirmatory_results.json), [power analysis](reports/artifacts/power_analysis.csv), and [truth validation](reports/artifacts/layer2_truth_validation.json). CUPED re-estimates its adjustment coefficient in each bootstrap replicate, with an ANCOVA/HC3 cross-check. Power uses the anticipated default increase, not a zero-effect assumption.

The simulation's true expected spending effect is $23.19. **The CUPED interval misses it by about $0.08; the unadjusted interval covers it.** The default guardrail decision agrees with the known truth for this design. These are results from one simulation, not a general coverage or safety guarantee.

### 3. Targeting — simulated policy evaluation

The X-learner's spending-effect ranking correlates **0.65** with analytic truth; its default-effect ranking correlates only **0.03**. Individual default estimates do not support per-customer safety claims.

The selected policy targets about **98%** of eligible records. It exceeds treat-all by only **$0.41 per eligible customer on validation**, insufficient evidence for a personalization advantage. On the recorded policy test split it estimates **+$26.54 per eligible customer** [95% interval $19.54–$33.55], with a default upper bound of **0.232 percentage points** against the 0.50-point margin. That test comparison is against **no intervention**, not against treat-all.

Sources: [CATE validation](reports/artifacts/cate_validation.json), [candidate comparison](reports/artifacts/policy_candidates_val.csv), and [stored test evaluation](reports/artifacts/policy_test_evaluation.json).

## Reproduce

### Inspect and test the recorded run

Use Python 3.13. No credentials or raw dataset are needed for the default artifact and synthetic-data checks:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q -rs
```

Without `data/raw/cs-training.csv`, 15 data-dependent tests skip; one full-data refitting test is excluded by default. `requirements.lock` is a pinned environment snapshot that includes macOS-specific packages; `requirements.txt` is the portable installation entry point.

For the data-dependent checks, obtain the file following [data/download.md](data/download.md), then run:

```bash
.venv/bin/python -m src.load_db
.venv/bin/python -m pytest -q -rs
.venv/bin/python -m pytest -m slow -q
.venv/bin/python -m src.simulate_experiment
```

The database loader builds local SQLite views. The slow test refits an outer-fold **RISK-DEV** model and calibrator to compare with the retained OOF scores; it does not repeat POLICY-TEST. With the committed freeze present, the final command verifies the cohort, design, and outcome hashes without redrawing outcomes.

### Generate a separate experiment

For a new experiment, use a separate checkout with an empty `reports/artifacts/` and a separate database. Preserve the recorded run. The generation order is:

```text
src.load_db → src.run_layer1 → src.run_linkage → src.run_layer2 → src.run_layer3
```

Each is a Python module (`python -m <module>`). Layer 1 refuses to refit over downstream frozen state. Linkage and policy reruns verify their bound artifacts; some diagnostic tables and figures are recomputed. Missing freeze markers or a missing already-recorded policy test result require restoration, not automatic regeneration.

Hashes check consistency with the retained records. They are **not** an external timestamp, proof of preregistration, an immutable storage system, or a hash of every source file and diagnostic. A deliberately different experiment needs its own design and outputs.

## Scope and limitations

- The dataset's originating institution, geography, and sampling frame are not established here; no result is attributed to a specific lender or population.
- Experiment treatment and outcomes are simulated. The frozen generator applies additive spending effects to inactive treated customers too, producing an active rate of 85.4% → 99.5%; organic activators also share one scalar draw. These disclosed mechanisms are retained, not silently repaired after seeing results.
- The 0.50-point tolerance and $12 spending threshold are hypothetical design inputs. No real credit-line amounts, dollar-valued default losses, production lending decisions, or realized revenue are measured.
- The policy bound is a portfolio-level estimate under the simulated randomized design. Default-effect predictions are not reliable individual risk estimates; validation-based policy selection is assessed separately on the retained test split.
- There is no fairness validation, production monitoring, lending API, or validated adverse-action explanation system. These are outside the implemented prototype.

Implementation details and historical deviations remain in [DESIGN_FREEZE.md](DESIGN_FREEZE.md); the shorter business explanation is in the [executive summary](reports/executive_summary.md).
