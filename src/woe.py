"""Weight-of-Evidence (WOE) transform + Information Value (IV), scorecard style.

Statistical reasoning
---------------------
WOE maps each bin b of a feature to ln(P(bin b | non-event) / P(bin b | event)).
Binning + WOE gives logistic regression three properties that matter for a
governance-friendly scorecard:

- per-bin log-odds encoding, without enforcing monotonicity across ordered bins (the extreme
  DebtRatio / utilization tails land in a bin instead of dominating a linear term);
- a principled slot for MISSING values: they form their own bin, whose WOE is
  learned from training data instead of an arbitrary imputation;
- IV = Σ (P(b|non-event) − P(b|event)) · WOE(b), a standard univariate
  predictive-strength summary comparable across features.

Leakage discipline: bins and WOE values are estimated in fit() from training
data only; transform() applies the frozen mapping. A unit test verifies that
validation labels can be shuffled without changing the transform output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "__MISSING__"


class WOETransformer:
    """WOE binning with a dedicated missing bin per feature.

    Binning strategy, most-governed first: (1) manual fixed edges where an
    interpretable grouping is specified (count features: 0 / 1 / 2 / 3+);
    (2) per-value-via-midpoint-edges for other low-cardinality numerics;
    (3) quantile bins for continuous features. All numeric edge sets are
    open-ended, so out-of-range values at scoring time land in a tail bin.
    Laplace-style smoothing (+0.5 events/non-events per bin) keeps WOE finite
    in pure bins.
    """

    def __init__(self, n_bins: int = 8, smoothing: float = 0.5, max_card: int = 20,
                 manual_edges: dict[str, list[float]] | None = None):
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.max_card = max_card
        # manual_edges: fixed, interpretable interior edges per feature
        # (e.g. [0.5, 1.5, 2.5] -> bins 0 / 1 / 2 / 3+). Scorecard governance
        # prefers bins a human can read over data-driven micro-bins.
        self.manual_edges = manual_edges or {}
        self.bin_edges_: dict[str, np.ndarray | None] = {}   # None => categorical-by-value
        self.woe_maps_: dict[str, dict] = {}                  # bin label -> WOE
        self.iv_: pd.Series | None = None
        self.bin_tables_: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        y = np.asarray(y)
        iv = {}
        for col in X.columns:
            x = X[col]
            edges = self._make_edges(x)
            self.bin_edges_[col] = edges
            labels = self._assign_bins(x, edges)
            table = self._woe_table(labels, y)
            self.bin_tables_[col] = table
            self.woe_maps_[col] = table["woe"].to_dict()
            iv[col] = float(table["iv_contrib"].sum())
        self.iv_ = pd.Series(iv, name="IV").sort_values(ascending=False)
        return self

    def _make_edges(self, x: pd.Series) -> np.ndarray | None:
        # Numeric edges are ALWAYS open-ended at both extremes, so a value
        # beyond anything seen in training lands in the corresponding TAIL bin
        # (e.g. an unseen delinquency count of 25 falls into "3+"), never in a
        # "neutral" bucket. Only binary features use per-value labels, where
        # unseen values cannot occur.
        if x.name in self.manual_edges:
            return np.asarray(self.manual_edges[x.name], dtype=float)
        nonmiss = x.dropna()
        u = np.unique(nonmiss)
        if len(u) <= 2:
            return None  # binary flag -> per-value bins
        if len(u) <= self.max_card:
            # low-cardinality numeric: one bin per observed value, expressed as
            # midpoint EDGES so out-of-range values map to the nearest tail bin
            return (u[:-1] + u[1:]) / 2.0
        qs = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(nonmiss, qs))
        interior = edges[1:-1]
        # Zero-inflated guard: for heavily skewed variables every interior
        # quantile can equal the modal value, collapsing ALL non-missing rows
        # into one bin and silently destroying the feature (this exact failure
        # put NumberOfTimes90DaysLate's IV at 0.04 in an early run — the fix is
        # a regression test now). Fall back to midpoints of (subsampled)
        # distinct values.
        if len(interior) == 0:
            if len(u) > 2 * self.max_card:
                u = np.unique(np.quantile(u, np.linspace(0, 1, self.max_card)))
            interior = (u[:-1] + u[1:]) / 2.0
        return interior

    def _assign_bins(self, x: pd.Series, edges: np.ndarray | None) -> pd.Series:
        if edges is None:
            lab = x.astype(object).where(x.notna(), MISSING)
        else:
            idx = np.searchsorted(edges, x.to_numpy(dtype=float), side="right")
            lab = pd.Series(idx, index=x.index, dtype=object)
            lab[x.isna()] = MISSING
        return lab

    def _woe_table(self, labels: pd.Series, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"bin": labels.to_numpy(), "y": y})
        g = df.groupby("bin", dropna=False)["y"].agg(events="sum", n="count")
        g["non_events"] = g["n"] - g["events"]
        ev = g["events"] + self.smoothing
        ne = g["non_events"] + self.smoothing
        p_ev = ev / ev.sum()
        p_ne = ne / ne.sum()
        g["woe"] = np.log(p_ne / p_ev)          # higher WOE = safer bin
        g["event_rate"] = g["events"] / g["n"]
        g["iv_contrib"] = (p_ne - p_ev) * g["woe"]
        return g

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            if col not in self.woe_maps_:
                raise KeyError(f"{col} was not seen during fit")
            labels = self._assign_bins(X[col], self.bin_edges_[col])
            m = self.woe_maps_[col]
            # .get(0.0) can only trigger for per-value (binary-flag) features;
            # numeric features always land in an open-ended edge bin
            out[col] = labels.map(lambda b: m.get(b, 0.0)).astype(float)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
