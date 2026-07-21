# CreditLine Decision Lab

> **EN** — One auditable business decision: should we offer credit-line increases, to
> whom, and is the expected spend worth the added default risk? Three connected layers:
> a default-risk model (WOE scorecard vs monotonic LightGBM), a simulated randomized
> credit-line experiment (superiority on spend, non-inferiority on default), and a
> policy-level causal targeting decision.
>
> **中文** — 一个可审计的业务决策:是否提额、给谁提额、增量消费是否值得增量违约风险。
> 三层连贯:违约风险模型(WOE 评分卡 vs 单调 LightGBM)→ 模拟随机化提额实验
> (消费 superiority + 违约 non-inferiority)→ policy 层因果定向决策。

**Status: Layer 1 (risk model) complete and audited; Linkage / Layer 2 / Layer 3 upcoming.**

All binding design decisions — sample-splitting plan, nested cross-fitting rules,
frozen data-handling rules, eligibility rule, guardrail margin δ, policy-level safety
validation — were frozen *before* modeling in [DESIGN_FREEZE.md](DESIGN_FREEZE.md).
Its **amendment log** records, with dates and reasons, every deviation discovered in
audit — including the holdout-usage history of RISK-TEST (a *fixed secondary holdout
diagnostic*; the primary generalization evidence is nested outer CV on RISK-DEV).
This repo prefers disclosed imperfections over unfalsifiable purity claims.

## Honesty statement

The dataset (Kaggle "Give Me Some Credit", 2011) has undocumented institution,
geography, and sampling frame. This project is a **methodological prototype**: it is
not Capital One data, and results do not transfer to any real portfolio. Layers 2–3
are a clearly-labeled **simulation** on real pre-treatment covariates.

## Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
# exact environment used for the committed results: requirements.lock (pip freeze)
# put cs-training.csv in data/raw/ (see data/download.md)
.venv/bin/python -m src.load_db        # SQLite warehouse + SQL views
.venv/bin/python -m src.run_layer1     # Layer 1: nested CV (primary evidence), holdout diagnostic, SHAP
.venv/bin/python -m pytest tests/ -q
# notebooks: jupytext --to notebook --execute notebooks/01_eda.py notebooks/02_risk_model.py
```

Layout: `sql/` warehouse queries · `src/` pipeline · `notebooks/` narrative ·
`reports/` figures + artifacts · `tests/` leakage / monotonicity / split-integrity tests.

*(Full README — architecture diagram, results tables, key-decision rationale, and
Limitations — lands in the write-up phase.)*
