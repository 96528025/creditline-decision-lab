"""Entry-point regressions: missing frozen files must not trigger fresh work."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import config, experiment_design, run_layer1, run_layer3, simulate_experiment, splits


def stop_work(*args, **kwargs):
    raise AssertionError("reached data loading or model work before checking frozen state")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTCOME_MARKER", tmp_path / "EXPERIMENT_OUTCOMES_FROZEN.json")
    monkeypatch.setattr(run_layer3, "POLICY_FREEZE_PATH", tmp_path / "policy_freeze.json")
    monkeypatch.setattr(run_layer3, "TEST_EVAL_PATH", tmp_path / "policy_test_evaluation.json")
    monkeypatch.setattr(run_layer3, "MODELS_DIR", tmp_path / "policy_models")
    for attr, name in [("OUTCOMES_PATH", "experiment_outcomes.parquet"), ("TRUTH_PATH", "dgp_truth.parquet"), ("CATE_TRUTH_PATH", "cate_truth.parquet")]:
        monkeypatch.setattr(simulate_experiment, attr, tmp_path / name)
    return tmp_path


@pytest.mark.parametrize("anchor", ["linkage_manifest.json", "EXPERIMENT_OUTCOMES_FROZEN.json"])
def test_layer1_refuses_refit_after_downstream_freeze(isolated, monkeypatch, anchor):
    (isolated / anchor).write_text("{}")
    monkeypatch.setattr(run_layer1.data_prep, "load_raw", stop_work)
    with pytest.raises(RuntimeError, match="Layer 1.*frozen"):
        run_layer1.main()


def test_missing_split_is_not_regenerated_after_freeze(isolated):
    config.OUTCOME_MARKER.write_text("{}")
    df = pd.DataFrame({config.TARGET: [0, 1] * 20})
    with pytest.raises(RuntimeError, match="split.*missing"):
        splits.make_or_load_split(df)
    assert not (isolated / "risk_split_indices.json").exists()


def test_design_payload_must_match_its_recorded_digest(isolated, monkeypatch):
    path = isolated / "experiment_design_frozen.json"
    monkeypatch.setattr(experiment_design, "DESIGN_PATH", path)
    experiment_design.freeze(0.02, 1000, "cohort")
    saved = json.loads(path.read_text())
    saved["dgp_params"]["spend_mde_dollars"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="payload.*hash"):
        experiment_design.load_frozen()
    with pytest.raises(RuntimeError, match="payload.*hash"):
        experiment_design.freeze(0.02, 1000, "cohort")


def test_missing_cate_truth_cannot_replace_its_hash_anchor(isolated, monkeypatch):
    config.OUTCOME_MARKER.write_text(json.dumps({"cate_truth_sha256": "original"}))
    before = config.OUTCOME_MARKER.read_bytes()
    monkeypatch.setattr(simulate_experiment, "load_frozen_cohort", stop_work)
    with pytest.raises(RuntimeError, match="cate_truth.*missing"):
        simulate_experiment.compute_cate_truth()
    assert config.OUTCOME_MARKER.read_bytes() == before
    assert not simulate_experiment.CATE_TRUTH_PATH.exists()


def test_orphaned_outcomes_are_not_redrawn(isolated, monkeypatch):
    simulate_experiment.OUTCOMES_PATH.write_bytes(b"original evidence")
    monkeypatch.setattr(simulate_experiment, "load_frozen_cohort", stop_work)
    with pytest.raises(RuntimeError, match="outcome.*marker"):
        simulate_experiment.main()
    assert simulate_experiment.OUTCOMES_PATH.read_bytes() == b"original evidence"


@pytest.mark.parametrize("missing", ["freeze", "result"])
def test_layer3_refuses_missing_state_before_model_work(isolated, monkeypatch, missing):
    config.OUTCOME_MARKER.write_text(json.dumps({"policy_test_result_sha256": "original"}))
    if missing == "freeze":
        run_layer3.TEST_EVAL_PATH.write_text("{}")
    else:
        run_layer3.POLICY_FREEZE_PATH.write_text("{}")
    monkeypatch.setattr(run_layer3, "load_frozen_cohort", stop_work)
    with pytest.raises(RuntimeError, match="policy.*missing"):
        run_layer3.main()
    assert config.OUTCOME_MARKER.read_text() == json.dumps({"policy_test_result_sha256": "original"})


def test_layer3_verifies_outcome_bytes_before_loading_models(isolated, monkeypatch):
    config.OUTCOME_MARKER.write_text(json.dumps({"outcomes_sha256": "original"}))
    run_layer3.POLICY_FREEZE_PATH.write_text("{}")
    simulate_experiment.OUTCOMES_PATH.write_bytes(b"changed outcomes")
    monkeypatch.setattr(experiment_design, "load_frozen", lambda: {"dgp_params": {"delta_noninf_pp": 0.5}})
    monkeypatch.setattr(run_layer3, "load_frozen_cohort", stop_work)
    with pytest.raises(RuntimeError, match="outcome freeze broken"):
        run_layer3.main()
