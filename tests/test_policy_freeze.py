"""Freeze/one-shot behavior for the Layer-3 policy — the audit gap: a frozen
policy must be an executable, hash-bound object, and every tampering vector
(same winner name, different spec / models / inputs / freeze hash) must be
detected."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, manifest, policy
from src.cate import XLearner
from src.run_layer3 import MODELS_DIR, POLICY_FREEZE_PATH, TEST_EVAL_PATH


# ------------------------------------------------------------ serialization
def test_spec_json_round_trip_including_infinities():
    specs = policy.build_policy_specs(
        pd.DataFrame({"pd_cal": np.linspace(0, 0.09, 100)}),
        np.linspace(-5, 30, 100), np.linspace(0, 0.005, 100), mde=12.0)
    for name, spec in specs.items():
        encoded = policy.spec_to_json(spec)
        assert all(np.isfinite(v) for v in encoded.values()), name
        decoded = policy.spec_from_json(encoded)
        assert decoded == spec, name


def test_spec_from_json_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown spec keys"):
        policy.spec_from_json({"tau_s_min": 0.0, "backdoor": 1.0})


# --------------------------------------------------------- tamper detection
def _fake_frozen():
    return {
        "policy": "tau_spend_positive",
        "spec": {"tau_s_min": 0.0},
        "delta_pp": 0.5,
        "model_hashes": {"spend": {"tau1": "aaa", "tau0": "bbb"},
                         "default": {"tau1": "ccc", "tau0": "ddd"}},
        "candidate_table_sha256": "e" * 64,
        "outcome_marker_sha256": "f" * 64,
        "test_ids_sha256": "1" * 64,
        "feature_schema_sha256": "2" * 64,
    }


@pytest.mark.parametrize("mutate", [
    ("spec", {"tau_s_min": 5.0}),                       # same winner, new spec
    ("model_hashes", {"spend": {"tau1": "XXX", "tau0": "bbb"},
                      "default": {"tau1": "ccc", "tau0": "ddd"}}),
    ("candidate_table_sha256", "0" * 64),
    ("outcome_marker_sha256", "0" * 64),
    ("test_ids_sha256", "0" * 64),
    ("feature_schema_sha256", "0" * 64),
    ("delta_pp", 0.75),
    ("policy", "treat_all"),
])
def test_every_tampering_vector_is_detected(mutate):
    frozen = _fake_frozen()
    key, bad = mutate
    current = {**_fake_frozen(), key: bad}
    problems = policy.freeze_problems(frozen, current)
    assert problems, f"tampering with {key} went undetected"
    with pytest.raises(RuntimeError, match="freeze violated"):
        policy.assert_frozen(problems, "test")


def test_identical_state_passes():
    assert policy.freeze_problems(_fake_frozen(), _fake_frozen()) == []


# ----------------------------------------------------- persisted model I/O
def test_xlearner_save_load_prediction_equivalent(tmp_path):
    rng = np.random.default_rng(51)
    n = 4000
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(size=n)})
    treated = (rng.random(n) < 0.5).astype(int)
    y = 10 * X["a"].to_numpy() + treated * 5.0 + rng.normal(0, 2, n)
    xl = XLearner().fit(X, y, treated)
    hashes = xl.save(tmp_path)
    loaded = XLearner.load(tmp_path, hashes)
    np.testing.assert_allclose(loaded.predict_tau(X), xl.predict_tau(X),
                               rtol=1e-12)


def test_xlearner_load_refuses_tampered_model(tmp_path):
    rng = np.random.default_rng(52)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 2000)})
    treated = (rng.random(2000) < 0.5).astype(int)
    y = X["a"].to_numpy() + treated * 2.0 + rng.normal(0, 1, 2000)
    hashes = XLearner().fit(X, y, treated).save(tmp_path)
    (tmp_path / "tau1.txt").write_text(
        (tmp_path / "tau1.txt").read_text() + "\n")   # 1-byte change
    with pytest.raises(RuntimeError, match="does not match the frozen policy"):
        XLearner.load(tmp_path, hashes)


# ------------------------------------------------- real artifacts, if built
needs_freeze = pytest.mark.skipif(not POLICY_FREEZE_PATH.exists(),
                                  reason="policy freeze not built")


@needs_freeze
def test_real_freeze_is_executable_and_hash_consistent():
    frozen = json.loads(POLICY_FREEZE_PATH.read_text())
    spec = policy.spec_from_json(frozen["spec"])          # must load
    assert spec["tau_s_min"] == 0.0                       # tau_spend_positive
    for outcome in ("spend", "default"):
        XLearner.load(MODELS_DIR / outcome, frozen["model_hashes"][outcome])


@needs_freeze
def test_real_test_artifact_bound_to_freeze():
    frozen_sha = manifest.file_sha256(POLICY_FREEZE_PATH)
    stored = json.loads(TEST_EVAL_PATH.read_text())
    assert stored["policy_freeze_sha256"] == frozen_sha
    assert stored["outcome_marker_sha256"] == manifest.file_sha256(config.OUTCOME_MARKER)
    assert stored["policy"] == json.loads(POLICY_FREEZE_PATH.read_text())["policy"]