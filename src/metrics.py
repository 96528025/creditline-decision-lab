"""Risk-model evaluation metrics — DESIGN_FREEZE.md §5 evaluation set.

Why these four:
- ROC-AUC: standard rank-ordering power, comparable across papers/teams.
- PR-AUC: with ~6.7% positives, ROC-AUC can look healthy while precision at
  business-relevant recall is poor; PR-AUC is the imbalance-honest complement.
- KS statistic: max separation between cumulative score distributions of goods
  and bads — the classic credit-scoring acceptance metric, and interviewers at
  banks expect it.
- Calibration (reliability curve + Brier): Layers 2-3 consume the score as a
  PROBABILITY (eligibility tiers, DGP inputs), so rank-ordering alone is not
  enough; predicted PDs must track observed default rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def summarize(y_true: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "ks": ks_statistic(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "n": int(len(y_true)),
        "event_rate": float(np.mean(y_true)),
    }


def calibration_table(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile-of-score reliability table (equal-count bins, not equal-width:
    with a right-skewed score distribution, equal-width bins leave most bins
    nearly empty and the curve uninterpretable)."""
    df = pd.DataFrame({"y": y_true, "p": score})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "count")
    )
    return g.reset_index(drop=True)
