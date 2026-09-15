"""Layer 1 pipeline: risk-model development under the frozen splitting plan.

Run: python -m src.run_layer1

Steps (DESIGN_FREEZE.md §2, §3, §5):
  1. S0: stratified 20% RISK-TEST holdout; indices saved and frozen.
  2. Outer 5-fold nested CV on RISK-DEV: per outer fold, hyperparameters are
     tuned by inner CV *within the outer-training data*, then models are
     evaluated on the outer-validation fold. This yields (a) the stability
     check and (b) uncalibrated OOF scores for every RISK-DEV customer,
     saved for the Linkage phase (which adds nested cross-fitted calibration).
  3. Final models: tuned by inner CV on all of RISK-DEV, fit on all of
     RISK-DEV, evaluated on RISK-TEST — the fixed secondary holdout
     diagnostic (its full usage history is in DESIGN_FREEZE.md's amendment
     log; the primary generalization evidence is step 2's nested CV).
  4. WOE/IV table from the final LR fit.
  5. Sensitivity analyses for the frozen data-handling rules — evaluated by
     CV on RISK-DEV only; RISK-TEST is spent solely on the primary spec.
     Model hyperparameters are held at the primary spec's values so the
     comparison isolates the data-handling choice.
  6. SHAP explainability on the final LightGBM (computed on RISK-DEV data),
     including a SHAP-vs-IV ranking comparison.

Every artifact lands in reports/artifacts/ (tables, json) and
reports/figures/ (png).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

from . import config, data_prep, metrics, plots
from .risk_models import WoeLogisticModel, fit_lgbm, tune_lgbm, tune_lr
from .splits import make_or_load_split


def _art(name: str):
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.ARTIFACTS_DIR / name


# ------------------------------------------------- outer-fold stability + OOF
def outer_cv(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested outer CV on RISK-DEV. Returns (stability table, OOF score frame)."""
    skf = StratifiedKFold(n_splits=config.N_OUTER_FOLDS, shuffle=True,
                          random_state=config.SEED_SPLIT)
    rows, oof = [], pd.DataFrame(
        index=X.index, columns=["fold", "oof_lr", "oof_lgbm"], dtype=float
    )
    for k, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]

        # ALL selection happens inside the outer-training data (§3)
        best_C, _ = tune_lr(Xtr, ytr)
        lr = WoeLogisticModel(C=best_C).fit(Xtr, ytr)
        lgbm_params, _ = tune_lgbm(Xtr, ytr)
        lgbm = fit_lgbm(Xtr, ytr, lgbm_params)

        p_lr = lr.predict_proba(Xva)[:, 1]
        p_gb = lgbm.predict_proba(Xva)[:, 1]
        oof.iloc[va, oof.columns.get_loc("fold")] = k
        oof.iloc[va, oof.columns.get_loc("oof_lr")] = p_lr
        oof.iloc[va, oof.columns.get_loc("oof_lgbm")] = p_gb

        for model, p in (("lr_woe", p_lr), ("lgbm_mono", p_gb)):
            rows.append({"fold": k, "model": model, **metrics.summarize(yva, p)})
        rows[-2]["chosen_C"] = best_C
        rows[-1].update({f"chosen_{k_}": v for k_, v in lgbm_params.items()})
        print(f"  outer fold {k}: {time.time()-t0:.0f}s  "
              f"lr_auc={rows[-2]['roc_auc']:.4f} lgbm_auc={rows[-1]['roc_auc']:.4f}")
    return pd.DataFrame(rows), oof


# ----------------------------------------------------------- sensitivity CV
def sensitivity_cv(raw_dev: pd.DataFrame, lgbm_params: dict, lr_C: float) -> pd.DataFrame:
    """CV each data-handling variant on RISK-DEV with primary-spec
    hyperparameters, so differences isolate the cleaning rule itself."""
    rows = []
    for variant in data_prep.VARIANTS:
        skf = StratifiedKFold(n_splits=config.N_OUTER_FOLDS, shuffle=True,
                              random_state=config.SEED_SPLIT)
        # split on the raw frame; fitted cleaning constants must come from
        # training rows only (matters for sentinel_cap / util_p99)
        y_all = raw_dev[config.TARGET]
        for k, (tr, va) in enumerate(skf.split(raw_dev, y_all)):
            cleaner = data_prep.CleaningRules(variant).fit(raw_dev.iloc[tr])
            dtr, dva = cleaner.transform(raw_dev.iloc[tr]), cleaner.transform(raw_dev.iloc[va])
            # sentinel_drop removes rows — allowed in train, but the validation
            # fold keeps ALL rows (a deployed model cannot drop applicants),
            # so dropped-variant scores are still evaluated on everyone.
            if variant == "sentinel_drop":
                dva = data_prep.CleaningRules("primary").transform(raw_dev.iloc[va])
            ytr, yva = dtr[config.TARGET], dva[config.TARGET]
            Xtr = dtr.drop(columns=[config.TARGET])
            Xva = dva.drop(columns=[config.TARGET])
            lr = WoeLogisticModel(C=lr_C).fit(Xtr, ytr)
            gb = fit_lgbm(Xtr, ytr, lgbm_params)
            for model, p in (("lr_woe", lr.predict_proba(Xva)[:, 1]),
                             ("lgbm_mono", gb.predict_proba(Xva)[:, 1])):
                rows.append({"variant": variant, "fold": k, "model": model,
                             **metrics.summarize(yva, p)})
        print(f"  sensitivity variant done: {variant}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- SHAP
def shap_analysis(lgbm, X_dev: pd.DataFrame, iv: pd.Series) -> pd.DataFrame:
    """Global SHAP importance + illustrative local explanations.

    Labeled as ILLUSTRATIVE local risk explanations / candidate reason-code
    mapping — NOT validated adverse-action notices.
    """
    rng = np.random.default_rng(config.SEED_MODEL)
    sample = X_dev.iloc[rng.choice(len(X_dev), size=5000, replace=False)]
    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):  # binary case may return [class0, class1]
        sv = sv[1]

    import matplotlib.pyplot as plt
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plots.save_fig(plt.gcf(), "shap_beeswarm_lgbm.png")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns,
                         name="mean_abs_shap").sort_values(ascending=False)
    plots.bar_table(mean_abs, "shap_global_bar.png",
                    "Global SHAP importance — monotonic LightGBM (RISK-DEV sample)",
                    "mean |SHAP| (log-odds units)")

    # three illustrative customers: low / median / high predicted risk
    p = lgbm.predict_proba(sample)[:, 1]
    for tag, i in (("low", int(np.argmin(p))),
                   ("median", int(np.argsort(p)[len(p) // 2])),
                   ("high", int(np.argmax(p)))):
        exp = shap.Explanation(values=sv[i], base_values=explainer.expected_value[-1]
                               if np.ndim(explainer.expected_value) else explainer.expected_value,
                               data=sample.iloc[i].to_numpy(), feature_names=list(sample.columns))
        shap.plots.waterfall(exp, show=False, max_display=12)
        plots.save_fig(plt.gcf(), f"shap_local_{tag}_risk.png")

    # rank comparison: univariate IV vs multivariate SHAP
    comp = pd.DataFrame({
        "iv_rank": iv.rank(ascending=False),
        "shap_rank": mean_abs.reindex(iv.index).rank(ascending=False),
        "iv": iv,
        "mean_abs_shap": mean_abs.reindex(iv.index),
    }).sort_values("shap_rank")
    comp.to_csv(_art("shap_vs_iv_ranking.csv"))
    return comp


# ------------------------------------------------------------------- main
def main() -> None:
    anchors = [config.ARTIFACTS_DIR / "linkage_manifest.json",
               config.ARTIFACTS_DIR / "frozen_thresholds.json",
               config.OUTCOME_MARKER]
    if any(path.exists() for path in anchors):
        raise RuntimeError(
            "Layer 1 cannot refit over frozen downstream inputs. Verify the "
            "recorded run with the tests; use a separate artifact directory "
            "for a new experiment. Nothing was overwritten.")
    t0 = time.time()
    raw = data_prep.load_raw()
    dev_idx, test_idx = make_or_load_split(raw)
    raw_dev, raw_test = raw.loc[dev_idx], raw.loc[test_idx]
    print(f"S0 split: dev={len(raw_dev)}, test={len(raw_test)} (frozen)")

    # primary cleaning is fully static -> no fitted constants, no leakage path
    cleaner = data_prep.CleaningRules("primary")
    dev = cleaner.fit_transform(raw_dev)
    test = cleaner.transform(raw_test)
    y_dev, X_dev = dev[config.TARGET], dev.drop(columns=[config.TARGET])
    y_test, X_test = test[config.TARGET], test.drop(columns=[config.TARGET])

    # ---- step 2: nested outer CV (stability + OOF scores for linkage)
    print("outer CV (nested tuning) ...")
    stability, oof = outer_cv(X_dev, y_dev)
    stability.to_csv(_art("stability_outer_cv.csv"), index=False)
    oof.to_parquet(_art("oof_scores_uncalibrated.parquet"))

    # ---- step 3: final models -> secondary holdout diagnostic on RISK-TEST
    print("final tuning on full RISK-DEV ...")
    best_C, lr_table = tune_lr(X_dev, y_dev)
    lgbm_params, lgbm_table = tune_lgbm(X_dev, y_dev)
    lr_table.to_csv(_art("tuning_lr.csv"), index=False)
    lgbm_table.to_csv(_art("tuning_lgbm.csv"), index=False)

    lr = WoeLogisticModel(C=best_C).fit(X_dev, y_dev)
    gb = fit_lgbm(X_dev, y_dev, lgbm_params)

    p_lr, p_gb = lr.predict_proba(X_test)[:, 1], gb.predict_proba(X_test)[:, 1]
    final = {
        "lr_woe": {**metrics.summarize(y_test, p_lr), "chosen_C": best_C},
        "lgbm_mono": {**metrics.summarize(y_test, p_gb), **{f"chosen_{k}": v for k, v in lgbm_params.items()}},
        "note": "RISK-TEST is a fixed secondary holdout diagnostic evaluated on the "
                "primary spec only; its full usage history is disclosed in the "
                "DESIGN_FREEZE.md amendment log. Primary generalization evidence "
                "is the nested outer CV on RISK-DEV.",
    }
    _art("final_test_metrics.json").write_text(json.dumps(final, indent=2))

    plots.roc_overlay(y_test, {"WOE + LR": p_lr, "monotonic LightGBM": p_gb},
                      "roc_test.png", "ROC on RISK-TEST (secondary holdout diagnostic)")
    cal = {m: metrics.calibration_table(y_test, p)
           for m, p in (("WOE + LR", p_lr), ("monotonic LightGBM", p_gb))}
    for m, t in cal.items():
        t.to_csv(_art(f"calibration_test_{'lr' if 'LR' in m else 'lgbm'}.csv"), index=False)
    plots.calibration_plot(cal, "calibration_test.png",
                           "Calibration on RISK-TEST (score deciles)")

    # ---- step 4: WOE / IV
    iv = lr.woe.iv_
    iv.to_csv(_art("iv_table.csv"))
    plots.bar_table(iv, "iv_ranking.png",
                    "Information Value per feature (WOE fit on RISK-DEV)", "IV")
    pd.concat(lr.woe.bin_tables_, names=["feature", "bin"]).to_csv(_art("woe_bins.csv"))

    # ---- step 5: sensitivity analyses (CV on RISK-DEV only)
    print("sensitivity analyses ...")
    sens = sensitivity_cv(raw_dev, lgbm_params, best_C)
    sens.to_csv(_art("sensitivity_cv.csv"), index=False)

    # ---- step 6: SHAP
    print("SHAP ...")
    shap_analysis(gb, X_dev, iv)

    print(f"Layer 1 complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
