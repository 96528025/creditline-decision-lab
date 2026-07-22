"""Heterogeneous treatment-effect models — DESIGN_FREEZE.md §8, fit on
POLICY-TRAIN only.

Two meta-learners per outcome, hand-rolled on LightGBM (scope frozen;
causal forests are Future Work):

- T-learner (baseline): fit outcome models μ1 on treated, μ0 on control;
  τ̂(x) = μ1(x) − μ0(x). Simple, but each arm's model extrapolates alone —
  regularization biases don't cancel.
- X-learner: impute individual effects against the OPPOSITE arm's outcome
  model (D1 = Y_t − μ0(X_t), D0 = μ1(X_c) − Y_c), regress the imputed effects
  on covariates, and blend. With a randomized 50/50 design the propensity
  weight is 0.5, so the blend is a plain average. Typically beats the
  T-learner when effects are smoother than outcomes — exactly our DGP.

τ_default targets a RARE binary outcome: individual estimates are NOISY by
nature. They are used for ranking and policy construction only; the safety
claim is made at the portfolio level (src/policy.py), never per customer.

Estimation notes: outcome models are LGBMRegressor even for the binary
default outcome (probability-scale regression keeps the meta-learner algebra
additive; a logit link would put imputed differences on the wrong scale).
Two fixed parameter sets, chosen on synthetic recovery tests (never on the
frozen simulation truth): OUTCOME models get ordinary capacity; EFFECT models
(the X-learner's τ stages) get much stronger regularization, because imputed
pseudo-outcomes are dominated by outcome noise while the effect surface
underneath is smooth — an effect model with outcome-model capacity mostly
fits that noise (synthetic PEHE 5.8 vs 2.3 for the regularized version).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import config

_COMMON = dict(
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=config.SEED_MODEL,
    verbose=-1,
)
OUTCOME_PARAMS = dict(_COMMON, num_leaves=31, min_child_samples=200)
EFFECT_PARAMS = dict(_COMMON, num_leaves=15, min_child_samples=2000)


def _fit(X: pd.DataFrame, y: np.ndarray, params: dict) -> LGBMRegressor:
    m = LGBMRegressor(**params)
    m.fit(X, y)
    return m


class TLearner:
    def fit(self, X: pd.DataFrame, y: np.ndarray, treated: np.ndarray) -> "TLearner":
        t = treated == 1
        self.mu1_ = _fit(X[t], y[t], OUTCOME_PARAMS)
        self.mu0_ = _fit(X[~t], y[~t], OUTCOME_PARAMS)
        return self

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class XLearner:
    """Randomized 50/50 design -> propensity 0.5, blend = simple average.

    Persistable: save() writes all four boosters as LightGBM text models and
    returns their content hashes; load() restores a prediction-equivalent
    learner from disk. The frozen policy references these files by hash —
    "the policy" is the persisted artifacts, never a refit in memory.
    """

    PARTS = ("mu1", "mu0", "tau1", "tau0")

    def fit(self, X: pd.DataFrame, y: np.ndarray, treated: np.ndarray) -> "XLearner":
        t = treated == 1
        self.mu1_ = _fit(X[t], y[t], OUTCOME_PARAMS)
        self.mu0_ = _fit(X[~t], y[~t], OUTCOME_PARAMS)
        d1 = y[t] - self.mu0_.predict(X[t])       # treated: observed minus counterfactual
        d0 = self.mu1_.predict(X[~t]) - y[~t]     # control: counterfactual minus observed
        self.tau1_ = _fit(X[t], d1, EFFECT_PARAMS)
        self.tau0_ = _fit(X[~t], d0, EFFECT_PARAMS)
        return self

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 * self.tau1_.predict(X) + 0.5 * self.tau0_.predict(X)

    def save(self, directory) -> dict[str, str]:
        import hashlib
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for part in self.PARTS:
            text = getattr(self, f"{part}_").booster_.model_to_string()
            (directory / f"{part}.txt").write_text(text)
            hashes[part] = hashlib.sha256(text.encode()).hexdigest()
        return hashes

    @classmethod
    def load(cls, directory, expected_hashes: dict[str, str]) -> "XLearner":
        """Restore from disk, refusing any file whose hash differs from the
        frozen record — a silently swapped model is a different policy."""
        import hashlib
        from pathlib import Path

        import lightgbm as lgb

        directory = Path(directory)
        obj = cls.__new__(cls)
        for part in cls.PARTS:
            text = (directory / f"{part}.txt").read_text()
            got = hashlib.sha256(text.encode()).hexdigest()
            if got != expected_hashes[part]:
                raise RuntimeError(
                    f"persisted model {directory.name}/{part}.txt hash {got[:12]}… "
                    f"does not match the frozen policy record "
                    f"{expected_hashes[part][:12]}… — refusing to load")
            setattr(obj, f"{part}_", lgb.Booster(model_str=text))
        return obj


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """Precision in Estimating Heterogeneous Effects: RMSE against the
    CONDITIONAL EXPECTED effect E[Y1-Y0|X] (cate_truth artifact) — never
    against realized noisy contrasts."""
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
