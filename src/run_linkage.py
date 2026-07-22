"""Linkage phase: calibrated OOF scores -> frozen eligibility cohort.

Run: python -m src.run_linkage

Freeze semantics (see src/manifest.py): the linkage cohort is bound by a
content-hash manifest. Verified state is a safe no-op; any mismatch is a hard
RuntimeError; partial artifact sets are refused outright. Once the Layer-2
outcome marker exists, this script can only VERIFY — never regenerate.

Outputs:
  reports/artifacts/oof_scores_calibrated.parquet   fold / raw / calibrated PD per dev row
  reports/artifacts/frozen_thresholds.json          absolute E5 threshold (frozen once)
  reports/artifacts/eligibility_funnel.csv          E1-E5 sequential attrition
  reports/artifacts/eligible_ids.json               frozen eligible row indices
  reports/artifacts/linkage_manifest.json           content hashes binding all of the above
  reports/figures/oof_calibration.png               raw vs calibrated reliability (dev OOF)
"""

from __future__ import annotations

import json
import time

import pandas as pd

from . import config, data_prep, manifest, metrics, plots
from .eligibility import compute_and_freeze_e5_threshold, eligibility
from .oof import load_per_fold_params, produce_calibrated_oof
from .splits import make_or_load_split


def _artifact_paths():
    art = config.ARTIFACTS_DIR
    return (art / "oof_scores_calibrated.parquet",
            art / "frozen_thresholds.json",
            art / "eligible_ids.json")


def _verify_eligible_matches_mask(mask: pd.Series, elig_path) -> None:
    saved = json.loads(elig_path.read_text())["eligible_row_indices"]
    current = sorted(int(i) for i in mask.index[mask])
    if saved != current:
        raise RuntimeError(
            "eligible_ids.json does not match the mask recomputed from the "
            f"current OOF scores and threshold ({len(saved)} saved vs "
            f"{len(current)} recomputed). The cohort freeze is broken; do not "
            "proceed. Delete ALL linkage artifacts deliberately to rebuild."
        )


def main() -> None:
    t0 = time.time()
    art = config.ARTIFACTS_DIR
    art.mkdir(parents=True, exist_ok=True)
    oof_path, thr_path, elig_path = _artifact_paths()
    outcomes_frozen = config.OUTCOME_MARKER.exists()

    raw = data_prep.load_raw()
    dev_idx, _ = make_or_load_split(raw)
    dev = data_prep.CleaningRules("primary").fit_transform(raw.loc[dev_idx])
    y_dev, X_dev = dev[config.TARGET], dev.drop(columns=[config.TARGET])

    stored = manifest.load()
    existing = [p for p in (oof_path, thr_path, elig_path) if p.exists()]

    if stored is None and outcomes_frozen:
        raise RuntimeError(
            f"{config.OUTCOME_MARKER.name} exists but there is no linkage "
            "manifest — the cohort the experiment was generated from cannot be "
            "verified. Investigate before touching anything."
        )

    if stored is None and 0 < len(existing) < 3:
        missing = [p.name for p in (oof_path, thr_path, elig_path) if not p.exists()]
        raise RuntimeError(
            f"partial linkage artifacts (missing: {missing}) and no manifest. "
            "Regenerating some artifacts while inheriting others is forbidden; "
            "delete the whole set deliberately, then rerun."
        )

    if stored is None and not existing:
        if outcomes_frozen:  # unreachable given the check above, kept explicit
            raise RuntimeError("outcomes frozen; cohort may not be regenerated")
        print("no linkage artifacts: full generation (nested cross-fit) ...")
        oof = produce_calibrated_oof(X_dev, y_dev,
                                     per_fold_params=load_per_fold_params())
        oof.to_parquet(oof_path)
        thr = compute_and_freeze_e5_threshold(oof["oof_cal"])
        mask, funnel = eligibility(dev, oof["oof_cal"], thr)
        elig_path.write_text(json.dumps(
            {"eligible_row_indices": sorted(int(i) for i in dev.index[mask])}))
        manifest.write(manifest.build(oof, thr, elig_path))
        mode = "generated"
    else:
        if len(existing) < 3:
            missing = [p.name for p in (oof_path, thr_path, elig_path) if not p.exists()]
            raise RuntimeError(
                f"a linkage state exists but artifacts are missing: {missing}. "
                "Nothing will be regenerated piecemeal; restore the files or "
                "(pre-outcome only) delete the entire set + manifest and rerun."
            )
        oof = pd.read_parquet(oof_path)
        thr = float(json.loads(thr_path.read_text())["e5_abs_pd_threshold"])
        if stored is None:
            # one-time adoption of a complete pre-manifest artifact set:
            # only after proving internal consistency
            recomputed_thr = float(oof["oof_cal"].quantile(config.ELIG_RISK_PERCENTILE))
            if abs(recomputed_thr - thr) > 1e-12:
                raise RuntimeError(
                    f"frozen threshold {thr} does not reproduce from the OOF "
                    f"artifact (quantile gives {recomputed_thr}); the threshold "
                    "belongs to a different score artifact. Refusing to adopt."
                )
            mask, funnel = eligibility(dev, oof["oof_cal"], thr)
            _verify_eligible_matches_mask(mask, elig_path)
            manifest.write(manifest.build(oof, thr, elig_path))
            mode = "adopted (internal consistency proven, manifest written)"
        else:
            problems = manifest.diff(stored, manifest.build(oof, thr, elig_path))
            if problems:
                raise RuntimeError(
                    "linkage manifest mismatch — the frozen cohort's inputs or "
                    "artifacts changed:\n  " + "\n  ".join(problems) +
                    "\nNothing was regenerated. Either restore the original "
                    "artifacts or (pre-outcome only) delete the ENTIRE linkage "
                    "artifact set + manifest deliberately and rerun."
                )
            mask, funnel = eligibility(dev, oof["oof_cal"], thr)
            _verify_eligible_matches_mask(mask, elig_path)
            mode = "verified against manifest"

    # diagnostics (side-effect-free with respect to the frozen artifacts)
    diag = {}
    for name, col in (("raw OOF", "oof_raw"), ("calibrated OOF", "oof_cal")):
        s = metrics.summarize(y_dev.to_numpy(), oof[col].to_numpy())
        diag[name] = s
        print(f"  {name}: auc={s['roc_auc']:.4f} brier={s['brier']:.5f}")
    (art / "oof_calibration_metrics.json").write_text(json.dumps(diag, indent=2))
    plots.calibration_plot(
        {n: metrics.calibration_table(y_dev.to_numpy(), oof[c].to_numpy())
         for n, c in (("raw OOF", "oof_raw"), ("calibrated OOF", "oof_cal"))},
        "oof_calibration.png",
        "OOF reliability on RISK-DEV: raw vs isotonic-calibrated",
    )
    funnel.to_csv(art / "eligibility_funnel.csv", index=False)
    print(funnel.to_string(index=False))
    print(f"[{mode}] eligible: {int(mask.sum())} of {len(dev)} "
          f"({mask.mean():.1%}); E5 abs threshold = {thr:.6f}")
    print(f"linkage complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
