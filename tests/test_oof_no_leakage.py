"""OOF property of the linkage scores — DESIGN_FREEZE.md §3.

Two complementary guarantees:

1. BEHAVIORAL: flipping every label in outer fold k must leave fold k's raw
   AND calibrated scores bit-identical. If any model or calibrator that
   touches fold k's scores had seen fold k's labels, the scores would move.
   This covers the subtle path §3 exists to close: a calibrator fit on scores
   from models that trained on fold k.
2. DETERMINISM (artifact-level): fold 0's persisted scores are reproducible
   from scratch by refitting on outer-training data only — proving the
   artifact really was produced out-of-fold, not by a model that saw everyone.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.oof import outer_folds, produce_calibrated_oof


def _synthetic(n=6000, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 2, n),
        config.DPD_COLS[0]: rng.poisson(0.4, n).astype(float),
        config.DPD_COLS[1]: rng.poisson(0.2, n).astype(float),
        config.DPD_COLS[2]: rng.poisson(0.1, n).astype(float),
        "noise": rng.normal(size=n),
    })
    logit = -2.5 + X.iloc[:, 0] + 0.7 * X.iloc[:, 1] + 0.4 * X["noise"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


SMALL_PARAMS = {k: {"num_leaves": 15, "min_child_samples": 20, "n_estimators": 60}
                for k in range(config.N_OUTER_FOLDS)}


def test_flipping_a_folds_labels_does_not_move_its_scores():
    X, y = _synthetic()
    # Hold the fold STRUCTURE fixed across both runs (stratified splitting
    # depends on y, so re-deriving folds from flipped labels would change
    # which rows are outer-training — a confound, not a leak). With identical
    # folds, fold k's labels are used by nothing that produces fold k's
    # scores, so every fold-k score must be bit-identical.
    folds = outer_folds(X, y)
    oof_a = produce_calibrated_oof(X, y, per_fold_params=SMALL_PARAMS,
                                   verbose=False, folds=folds)

    k = 2
    _, va = folds[k]
    y_flipped = y.copy()
    y_flipped.iloc[va] = 1 - y_flipped.iloc[va]
    oof_b = produce_calibrated_oof(X, y_flipped, per_fold_params=SMALL_PARAMS,
                                   verbose=False, folds=folds)

    in_k = oof_a["fold"] == k
    assert in_k.sum() == len(va)
    pd.testing.assert_series_equal(
        oof_a.loc[in_k, "oof_raw"], oof_b.loc[in_k, "oof_raw"], check_exact=True)
    pd.testing.assert_series_equal(
        oof_a.loc[in_k, "oof_cal"], oof_b.loc[in_k, "oof_cal"], check_exact=True)
    # sanity: the flip must move OTHER folds' scores (their models saw fold k)
    assert not oof_a.loc[~in_k, "oof_raw"].equals(oof_b.loc[~in_k, "oof_raw"])


def test_every_row_scored_and_fold_sizes_balanced():
    X, y = _synthetic()
    oof = produce_calibrated_oof(X, y, per_fold_params=SMALL_PARAMS, verbose=False)
    assert oof[["fold", "oof_raw", "oof_cal"]].notna().all().all()
    sizes = oof["fold"].value_counts()
    assert len(sizes) == config.N_OUTER_FOLDS
    assert sizes.max() - sizes.min() <= len(X) * 0.01


@pytest.mark.slow
def test_persisted_artifact_reproducible_out_of_fold():
    """Refit fold 0 from scratch on outer-training data only; predictions on
    fold 0 must match the persisted linkage artifact exactly."""
    art = config.ARTIFACTS_DIR / "oof_scores_calibrated.parquet"
    if not (art.exists() and config.DATA_RAW.exists()):
        pytest.skip("linkage artifact or raw data not present")

    from src import data_prep
    from src.oof import load_per_fold_params, _fit_calibrator
    from src.risk_models import fit_lgbm
    from src.splits import make_or_load_split

    raw = data_prep.load_raw()
    dev_idx, _ = make_or_load_split(raw)
    dev = data_prep.CleaningRules("primary").fit_transform(raw.loc[dev_idx])
    y, X = dev[config.TARGET], dev.drop(columns=[config.TARGET])

    saved = pd.read_parquet(art)
    tr, va = outer_folds(X, y)[0]
    params = load_per_fold_params()[0]
    base = fit_lgbm(X.iloc[tr], y.iloc[tr], params)
    calib = _fit_calibrator(X.iloc[tr], y.iloc[tr], params,
                            seed=config.SEED_MODEL + 0)
    raw_pred = base.predict_proba(X.iloc[va])[:, 1]
    np.testing.assert_allclose(raw_pred, saved.iloc[va]["oof_raw"], rtol=1e-9)
    np.testing.assert_allclose(calib.predict(raw_pred),
                               saved.iloc[va]["oof_cal"], rtol=1e-9)
