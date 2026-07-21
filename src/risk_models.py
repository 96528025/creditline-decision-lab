"""The two Layer-1 risk models (scope frozen — exactly two, DESIGN_FREEZE.md §5).

1. WOE + Logistic Regression — the interpretable scorecard-style baseline.
2. Monotonic LightGBM — the challenger. LightGBM over XGBoost because its native
   NaN handling matches our missing-value strategy (sentinels and invalid values
   become NaN + flag, no imputation to defend) and monotone_constraints are
   first-class.

Why monotone constraints: they buy robustness in sparse regions, business
trust (a score that can DROP when delinquencies increase is undefendable in
front of a risk committee), and easier validation — the constraint is a testable
property (tests/test_monotonic.py). This is governance-AWARE modeling, not a
claim of regulatory compliance.

Tuning discipline (DESIGN_FREEZE.md §3): every call receives training data only;
hyperparameters are selected by inner CV within that data. Callers are
responsible for never passing held-out rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config
from .woe import WOETransformer

# Small, honest grids. The point is disciplined selection, not an exhaustive
# search that invites overfitting the inner-validation folds.
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)
LGBM_GRID = tuple(
    {"num_leaves": nl, "min_child_samples": mcs}
    for nl in (15, 31, 63)
    for mcs in (20, 100)
)
LGBM_FIXED = dict(
    learning_rate=0.05,
    n_estimators=3000,          # ceiling; actual size set by early stopping
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=config.SEED_MODEL,
    verbose=-1,
)


def monotone_vector(feature_names: list[str]) -> list[int]:
    """+1 for utilization + delinquency counts, 0 elsewhere (frozen §5)."""
    return [1 if f in config.MONOTONE_INCREASING else 0 for f in feature_names]


# --------------------------------------------------------------------------- LR
class WoeLogisticModel:
    """WOE transform + L2 logistic regression, fit as one unit on training data.

    The scorecard has its OWN feature list: flags that merely duplicate a WOE
    missing bin (config.LR_DROP_FLAGS) are excluded — feeding the same
    indicator twice adds collinearity, not information. Count features use the
    fixed interpretable bins in config.WOE_MANUAL_EDGES.
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.woe = WOETransformer(manual_edges=config.WOE_MANUAL_EDGES)
        self.lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")

    @staticmethod
    def _select(X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in config.LR_DROP_FLAGS if c in X.columns])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeLogisticModel":
        Xw = self.woe.fit_transform(self._select(X), y)
        self.lr.fit(Xw, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.woe.transform(self._select(X)))


def tune_lr(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED_MODEL,
            n_folds: int = config.N_INNER_FOLDS) -> tuple[float, pd.DataFrame]:
    """Inner-CV selection of C. WOE is refit inside every inner-training fold —
    fitting WOE once on all rows would leak inner-validation labels into the
    encoding and overstate inner-val AUC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for C in LR_C_GRID:
        aucs = []
        for tr, va in skf.split(X, y):
            m = WoeLogisticModel(C=C).fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        rows.append({"C": C, "mean_auc": np.mean(aucs), "sd_auc": np.std(aucs)})
    table = pd.DataFrame(rows)
    best_C = float(table.loc[table["mean_auc"].idxmax(), "C"])
    return best_C, table


# ------------------------------------------------------------------------- LGBM
def tune_lgbm(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED_MODEL,
              n_folds: int = config.N_INNER_FOLDS) -> tuple[dict, pd.DataFrame]:
    """Inner-CV grid selection with per-fold early stopping.

    Returns the winning params with n_estimators set to the median best
    iteration across inner folds (a standard way to carry an early-stopped size
    to a refit on the full training data, which has no validation set to stop on).
    """
    mono = monotone_vector(list(X.columns))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for grid in LGBM_GRID:
        aucs, best_iters = [], []
        for tr, va in skf.split(X, y):
            m = LGBMClassifier(**LGBM_FIXED, **grid, monotone_constraints=mono)
            m.fit(
                X.iloc[tr], y.iloc[tr],
                eval_X=X.iloc[va], eval_y=y.iloc[va],
                eval_metric="auc",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
            )
            aucs.append(m.best_score_["valid_0"]["auc"])
            best_iters.append(m.best_iteration_)
        rows.append({**grid, "mean_auc": np.mean(aucs), "sd_auc": np.std(aucs),
                     "median_best_iter": int(np.median(best_iters))})
    table = pd.DataFrame(rows)
    best = table.loc[table["mean_auc"].idxmax()]
    params = {
        "num_leaves": int(best["num_leaves"]),
        "min_child_samples": int(best["min_child_samples"]),
        "n_estimators": int(best["median_best_iter"]),
    }
    return params, table


def fit_lgbm(X: pd.DataFrame, y: pd.Series, params: dict) -> LGBMClassifier:
    """Final monotonic LightGBM fit on training data with tuned params."""
    fixed = {**LGBM_FIXED, "n_estimators": params["n_estimators"]}
    m = LGBMClassifier(
        **fixed,
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        monotone_constraints=monotone_vector(list(X.columns)),
    )
    m.fit(X, y)
    return m
