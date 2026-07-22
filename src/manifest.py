"""Machine-verifiable freeze for the Linkage cohort.

The linkage manifest binds together everything the eligibility cohort depends
on — raw data, frozen split, per-fold model params, OOF scores, E5 threshold,
eligible IDs — by content hash. The design contract (enforced by
src/run_linkage.py):

- manifest present + everything matches  -> verified, safe no-op;
- manifest present + ANY mismatch        -> RuntimeError naming the mismatches;
- no manifest + no artifacts             -> full generation, then manifest;
- no manifest + complete consistent set  -> one-time adoption after internal
  consistency checks (threshold reproduces from the OOF scores; eligible IDs
  reproduce from the mask);
- no manifest + PARTIAL artifacts        -> RuntimeError; delete the whole set
  deliberately, never regenerate half and inherit the other half.

"Frozen" thereby stops being a documentation promise and becomes an invariant
a machine checks on every run.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config

MANIFEST_PATH = config.ARTIFACTS_DIR / "linkage_manifest.json"


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def index_sha256(index) -> str:
    """Hash of the OOF row-index VALUES (order-sensitive), independent of
    parquet encoding details."""
    return hashlib.sha256(np.asarray(index, dtype=np.int64).tobytes()).hexdigest()


def build(oof: pd.DataFrame, e5_threshold: float, eligible_path) -> dict:
    art = config.ARTIFACTS_DIR
    return {
        "raw_data_md5": file_md5(config.DATA_RAW),
        "split_sha256": file_sha256(art / "risk_split_indices.json"),
        "model_params_sha256": file_sha256(art / "stability_outer_cv.csv"),
        "oof_parquet_sha256": file_sha256(art / "oof_scores_calibrated.parquet"),
        "oof_index_sha256": index_sha256(oof.index),
        "oof_row_count": int(len(oof)),
        "e5_percentile": config.ELIG_RISK_PERCENTILE,
        "e5_abs_threshold": float(e5_threshold),
        "eligible_ids_sha256": file_sha256(eligible_path),
        "eligible_count": int(len(json.loads(eligible_path.read_text())["eligible_row_indices"])),
        "calibrator_type": "isotonic",
    }


def diff(expected: dict, actual: dict) -> list[str]:
    problems = []
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, float):
            ok = got is not None and abs(want - got) < 1e-12
        else:
            ok = want == got
        if not ok:
            problems.append(f"{key}: manifest={want!r} current={got!r}")
    return problems


def write(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load() -> dict | None:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return None
