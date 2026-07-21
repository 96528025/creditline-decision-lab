"""Shared figure helpers for Layer 1. All figures land in reports/figures/."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import config

FIG_KW = dict(dpi=150, bbox_inches="tight")


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, **FIG_KW)
    plt.close(fig)
    return str(path)


def roc_overlay(y_true, scores: dict[str, np.ndarray], name: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=label, lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def calibration_plot(tables: dict[str, pd.DataFrame], name: str, title: str) -> str:
    """Reliability curves from metrics.calibration_table outputs."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = 0.0
    for label, t in tables.items():
        ax.plot(t["mean_pred"], t["obs_rate"], "o-", ms=4, lw=1.2, label=label)
        lim = max(lim, t["mean_pred"].max(), t["obs_rate"].max())
    lim *= 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted PD (score decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def bar_table(series: pd.Series, name: str, title: str, xlabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.35 * len(series))))
    series.iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, name)


def save_fig(fig, name: str) -> str:
    return _save(fig, name)
