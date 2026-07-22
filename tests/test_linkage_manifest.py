"""The linkage freeze must be a machine-checked invariant, not a promise.

Covers the contract in src/manifest.py + src/run_linkage.py:
- verified state matches the real artifacts on disk;
- any tampering (threshold, eligible IDs, OOF file) is detected as a diff;
- the outcome marker blocks regeneration through an explicit RuntimeError
  (not an `assert`, which python -O would strip).
"""

import json

import pandas as pd
import pytest

from src import config, manifest


needs_artifacts = pytest.mark.skipif(
    not manifest.MANIFEST_PATH.exists(), reason="linkage manifest not built"
)


@needs_artifacts
def test_manifest_matches_artifacts_on_disk():
    stored = manifest.load()
    oof = pd.read_parquet(config.ARTIFACTS_DIR / "oof_scores_calibrated.parquet")
    thr = json.loads((config.ARTIFACTS_DIR / "frozen_thresholds.json").read_text())
    current = manifest.build(oof, thr["e5_abs_pd_threshold"],
                             config.ARTIFACTS_DIR / "eligible_ids.json")
    assert manifest.diff(stored, current) == []


@needs_artifacts
def test_manifest_detects_each_kind_of_tampering():
    stored = manifest.load()
    oof = pd.read_parquet(config.ARTIFACTS_DIR / "oof_scores_calibrated.parquet")
    thr = json.loads((config.ARTIFACTS_DIR / "frozen_thresholds.json").read_text())
    current = manifest.build(oof, thr["e5_abs_pd_threshold"],
                             config.ARTIFACTS_DIR / "eligible_ids.json")

    for key, bad in [
        ("e5_abs_threshold", current["e5_abs_threshold"] + 1e-3),
        ("oof_parquet_sha256", "0" * 64),
        ("oof_index_sha256", "0" * 64),
        ("eligible_ids_sha256", "0" * 64),
        ("eligible_count", current["eligible_count"] - 1),
        ("oof_row_count", current["oof_row_count"] + 1),
        ("calibrator_type", "platt"),
    ]:
        tampered = {**current, key: bad}
        problems = manifest.diff(stored, tampered)
        assert any(key in p for p in problems), f"tampering with {key} not detected"


def test_outcome_marker_blocks_fresh_generation(tmp_path, monkeypatch):
    """With the outcome marker present and no manifest, run_linkage must raise
    RuntimeError before touching anything."""
    from src import run_linkage

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTCOME_MARKER", tmp_path / "EXPERIMENT_OUTCOMES_FROZEN.json")
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "linkage_manifest.json")
    config.OUTCOME_MARKER.write_text("{}")

    with pytest.raises(RuntimeError, match="cannot be verified|may not be regenerated"):
        run_linkage.main()


def test_partial_artifacts_without_manifest_are_refused(tmp_path, monkeypatch):
    from src import run_linkage

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTCOME_MARKER", tmp_path / "EXPERIMENT_OUTCOMES_FROZEN.json")
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "linkage_manifest.json")
    (tmp_path / "frozen_thresholds.json").write_text(json.dumps({"e5_abs_pd_threshold": 0.1}))

    with pytest.raises(RuntimeError, match="partial linkage artifacts"):
        run_linkage.main()
