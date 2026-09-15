# CreditLine Decision Lab

> **EN** — One auditable business decision: should we offer credit-line increases, to
> whom, and is the expected spend worth the added default risk? Three genuinely
> connected layers — a default-risk model (WOE scorecard vs monotonic LightGBM), a
> clearly-labeled **simulated** randomized credit-line experiment (spend superiority +
> default non-inferiority), and a policy-level causal targeting decision. The result:
> ship broadly at the pre-registered 0.50pp risk tolerance (non-inferiority
> demonstrated), while at 0.30pp the honest answer is "this design cannot demonstrate
> it." Every freeze in the pipeline is machine-verified by content hash, and every
> deviation found in audit is disclosed in an amendment log.
>
> **中文** — 一个可审计的业务决策:是否提额、给谁提额、增量消费是否值得增量违约风险。
> 三层连贯:违约风险模型(WOE 评分卡 vs 单调 LightGBM)→ 明确标注为**模拟**的随机化
> 提额实验(消费 superiority + 违约 non-inferiority)→ policy 层因果定向决策。结论:在
> 预注册的 0.50pp 风险容忍度下证明非劣、建议广覆盖发放;在 0.30pp 下诚实回答"本设计
> 无法证明"。全流程冻结均由内容哈希机器校验,审计发现的每处偏差都记录在 Amendment Log。

**The hero of this repo is not a model leaderboard — it is the decision procedure:**
unit-tested leakage boundaries (and, where one was breached, disclosure rather than a
quiet repair), pre-registered inference, freezes enforced by content hash, and —
because Layers 2–3 are simulated with a known DGP — a *graded* check that the safety
procedure reached the correct call on this design. One correct call is evidence the
procedure works here, not proof that it always will.

---

## Architecture

```
Kaggle "Give Me Some Credit" (150k, real)          [provenance: undocumented →
│                                                   methodological prototype only]
├─ RISK-TEST 20% ──► fixed secondary holdout diagnostic (usage history disclosed)
└─ RISK-DEV 80%
   ├─ LAYER 1 · RISK        WOE+LR scorecard vs monotonic LightGBM
   │                        nested CV = primary generalization evidence
   ├─ LINKAGE               nested cross-fitted isotonic calibration → OOF PD
   │                        frozen eligibility E1–E5 → 94,291 eligible (78.6%)
   │                        linkage_manifest.json: hash-bound cohort freeze
   ├─ LAYER 2 · EXPERIMENT  [SIMULATION] frozen design + DGP → outcomes ONCE
   │                        power at anticipated diff · CUPED · Newcombe UB
   │                        confirmatory on POLICY-TRAIN∪VAL (66k)
   └─ LAYER 3 · DECISION    X-learner CATEs → candidate policies →
                            portfolio guardrail on POLICY-VAL →
                            frozen executable policy → ONE POLICY-TEST eval
```

The full frozen design — splitting plan, nested cross-fitting rules, data-handling
rules, eligibility, δ, decision rules — is in **[DESIGN_FREEZE.md](DESIGN_FREEZE.md)**,
whose amendment log records every deviation discovered in three audit rounds, with
dates, reasons, and what was (and was not) allowed to change.

## Headline results (all honest, none cherry-picked)

**Layer 1 — risk (real data).** Nested outer CV on RISK-DEV is the primary evidence;
RISK-TEST is a reused, disclosed holdout diagnostic.

| model | CV AUC (mean±sd) | holdout AUC | PR-AUC | KS | Brier |
|---|---|---|---|---|---|
| WOE + LR scorecard | 0.856 ± 0.006 | 0.851 | 0.396 | 0.553 | 0.0496 |
| monotonic LightGBM | 0.865 ± 0.005 | 0.859 | 0.405 | 0.569 | 0.0488 |

Sensitivity analyses over all frozen data-handling rules move CV AUC by ≤ 0.0014.
Both models sit inside the public benchmark band (~0.86–0.87) — nothing "too good."

**Layer 2 — simulated experiment (66k confirmatory).**

| quantity | value |
|---|---|
| spend lift (CUPED, θ re-bootstrapped) | **+$25.21** [95% boot 23.27, 27.17], −80% variance |
| ANCOVA + HC3 cross-check | +$25.21 [23.24, 27.18] |
| default Δ one-sided 95% UB (Newcombe) | **0.316pp** |
| non-inferiority @ δ=0.50pp (powered) | **DEMONSTRATED → SHIP** |
| non-inferiority @ 0.30pp (underpowered: needs 128k/arm, has 33k) | **NOT DEMONSTRATED** |
| truth check (DGP) | true ATE +23.19 (plain CI covers; CUPED CI misses by $0.08 — disclosed ~2σ event); true Δdefault 0.158pp → guardrail decision **correct** |

**Layer 3 — targeting (restrained reading).** Spend CATE recovery is real (X-learner
rank corr 0.65 vs analytic truth; PEHE ~$10 on a ±$30 effect); **individual default
CATEs are statistically worthless (rank corr 0.03) — so no per-customer safety claim
is ever made.** The frozen policy (`τ̂_spend > 0`, 98% coverage) beat treat-all by only
$0.41/eligible on validation — within noise. The accurate story is a
**near-broad-coverage policy under a portfolio-level safety constraint**, not a
personalization win. One-shot POLICY-TEST: **+$26.54/eligible** [19.5, 33.5], default
UB **0.232pp < 0.50pp**; DGP truth ($23.05, 0.149pp) confirms both the CI coverage and
the guardrail decision.

## Key decisions (the "why" table)

| decision | why |
|---|---|
| **WOE + LR** as baseline | scorecard-style governance: auditable bins (fixed 0/1/2/3+ for counts), a learned bin for missing, IV per feature; the interpretable anchor the challenger must beat |
| **monotonic** LightGBM | PD must not fall as delinquency/utilization rises: robustness in sparse regions, business trust, and a *testable* validation property; governance-aware, never "regulator-accepted" |
| **KS** alongside AUC | the classic credit acceptance metric: max separation of good/bad score distributions; PR-AUC because 6.7% positives make ROC alone flattering |
| **nested** cross-fitted calibration | a calibrator fit on other folds' OOF scores is subtly leaky (those scores come from models that saw the target fold); isotonic chosen: nonparametric, order-preserving (ties disclosed) |
| **separate power** for spend & default, larger n wins | the guardrail (rare events) usually binds; and power must be computed at the **anticipated** true difference — using δ alone while the DGP anticipates +0.158pp would be self-deception (audit round 2) |
| **non-inferiority + one-sided bound** | "not significant" ≠ "safe"; the guardrail passes only if the one-sided 95% Newcombe upper bound clears the margin — score-based, not Wald, because events are rare |
| **bootstrap + winsorize + hurdle** for spend | zero-inflated heavy tails: means converge slowly (bootstrap CI), tails dominate (winsorized sensitivity), and activation vs intensification are different businesses (two-part view) |
| **CUPED** | pre-period spend is free variance: −80% here, CI width halved; θ re-estimated inside every bootstrap replicate, ANCOVA+HC3 as an independent check |
| **FDR only for exploratory** | the two co-primary pre-registered decisions use their frozen thresholds; BH applies solely to labeled exploratory segment scans |
| **policy-level (not per-customer) default safety** | measured: individual default CATEs carry ~zero signal (rank corr 0.03); individual effects construct the policy, the randomized portfolio guardrail validates safety |
| **policy value, not Qini** for spend | spend is continuous; cumulative-gain / policy-value curves benchmarked against random targeting; binary Qini would be a category error (acceptable only for default) |
| **hash-bound freezes everywhere** | "frozen" is a machine-checked invariant (linkage manifest, outcome marker, executable policy freeze with persisted models), not a promise in prose |

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# exact environment for the committed results: requirements.lock (pip freeze)
# put cs-training.csv in data/raw/  (see data/download.md)
.venv/bin/python -m src.load_db        # SQLite warehouse + dev-only SQL views
.venv/bin/python -m src.run_layer1     # risk models: nested CV + holdout diagnostic
.venv/bin/python -m src.run_linkage    # calibrated OOF + frozen eligibility (manifest)
.venv/bin/python -m src.run_layer2     # frozen design → power → outcomes ONCE → inference
.venv/bin/python -m src.run_layer3     # CATEs → frozen policy → one-shot POLICY-TEST
.venv/bin/python -m pytest -q          # 66 fast tests; 15 skip until cs-training.csv is in place
.venv/bin/python -m pytest -m slow -q  # full-data artifact-reproducibility test
# notebooks: jupytext --to notebook --execute notebooks/0*.py
```

All seeds are frozen in `src/config.py`. Re-running any stage against existing
artifacts verifies hashes and refuses silent regeneration.

Layout: `sql/` warehouse (cleaning views, cohort CTEs, window-function segments) ·
`src/` pipeline · `notebooks/` 01 EDA, 02 risk, 03 experiment+targeting ·
`reports/artifacts/` frozen state + results · `reports/figures/` · `tests/`.

## Tests worth reading

- `test_oof_no_leakage.py` — flip a fold's labels; its OOF scores must be
  bit-identical (and other folds' must move).
- `test_woe_leakage.py` — shuffled validation labels cannot move the transform; a
  zero-inflated regression test that once caught a real bin-collapse bug.
- `test_monotonic.py` — the constraint holds on synthetic monotone data, probed
  feature by feature.
- `test_linkage_manifest.py` / `test_policy_freeze.py` — every tamper vector
  (swapped scores, thresholds, model files, specs, id sets) must raise.
- `test_layer2_inference.py` — Newcombe bound vs statsmodels, BH vs scipy, CUPED
  unbiasedness, power closed forms, analytic CATE vs oracle Monte Carlo.
- `test_xlearner_recovery.py` — the spec-required recovery of known heterogeneous
  effects, continuous and rare-binary.

## Limitations (read before quoting any number)

1. **Provenance.** The dataset's institution, geography, sampling frame, and product
   context are not documented. Nothing here is Capital One data, and no result
   transfers to any real portfolio or the US credit-card population. This is a
   methodological prototype.
2. **Layers 2–3 are simulation.** Real covariates, simulated treatment and outcomes.
   The DGP is frozen and disclosed, including its warts: the additive spend effect
   activates nearly all inactive treated customers (active rate 85.4% → 99.5% — a
   strong mechanism, not "3% activation"), and organic activators share a single
   scalar draw. Spend levels, effect sizes, and the SHIP conclusion are properties of
   this DGP, not of any market.
3. **RISK-TEST is not pristine.** Early EDA touched full-population target statistics,
   and the holdout was re-evaluated once after a bug fix — both disclosed; nested CV
   is the primary evidence. We chose disclosed imperfection over re-splitting, which
   would have laundered rather than removed the contamination.
4. **δ = 0.30pp was underpowered** (needs ~128k/arm at the anticipated difference);
   non-inferiority there is *not demonstrated* — which is not evidence of harm.
5. **Individual default CATEs are noise** (rank corr 0.03). Any per-customer
   "incremental risk" claim from this pipeline would be fiction; only the portfolio
   guardrail is load-bearing.
6. **The TEST evaluation compares the frozen policy against no intervention** — it
   does not establish superiority over treat-all ($0.41/eligible on VAL is within
   noise).
7. Real credit-line governance would additionally require stability monitoring,
   fairness analysis, adverse-action-compliant reason codes, and validated
   documentation — all explicitly out of scope. SHAP outputs are illustrative, not
   adverse-action notices.

## Future work (deliberately cut — scope control is a feature)

CatBoost / additional challengers · causal forests · survival (time-to-default)
modeling · fairness analysis · deep tabular baselines (disproportionate here given
dataset size and interpretability goals — not categorically worse) · shorter-horizon
outcome windows · cost-based policy optimization with dollar-valued default losses.

---

*Executive summary for non-technical readers:
[reports/executive_summary.md](reports/executive_summary.md).*
