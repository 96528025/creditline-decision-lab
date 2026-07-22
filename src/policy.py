"""Candidate targeting policies and the POLICY-LEVEL guardrail —
DESIGN_FREEZE.md §8.

Frozen semantics:
- A policy π maps covariates to treat / don't-treat. Candidates are built on
  POLICY-TRAIN from CATE estimates + the baseline risk score.
- Policy VALUE is expressed PER ELIGIBLE CUSTOMER (portfolio scale):
      incremental_spend(π)   = share(π) · [E(spend | T, π=1) − E(spend | C, π=1)]
      incremental_default(π) = share(π) · [E(default | T, π=1) − E(default | C, π=1)]
  estimated on the RANDOMIZED experiment by comparing arms WITHIN the
  policy's targeted subgroup (valid because treatment is randomized, so the
  targeted subgroup's arms are exchangeable). share(π) is computed on the
  full evaluation split.
- FEASIBILITY: one-sided 95% upper confidence bound on the policy-level
  incremental default per eligible customer must sit below the frozen
  portfolio tolerance δ. Individual τ̂_default upper bounds are NEVER used as
  a per-customer safety guarantee — individual effects construct the policy;
  safety is validated at the portfolio level.
- Selection on POLICY-VAL: the feasible candidate with the highest
  incremental spend. The frozen winner is evaluated EXACTLY ONCE on
  POLICY-TEST (enforced by a written freeze artifact + one-shot marker).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .inference import newcombe_upper_bound


def policy_value(df: pd.DataFrame, targeted: pd.Series,
                 alpha_one_sided: float = 0.05) -> dict:
    """Randomized-experiment policy-value estimate on one evaluation split.

    df needs columns treated / spend_post / default_post; `targeted` is the
    policy's boolean recommendation for every row of df.
    """
    share = float(targeted.mean())
    sub = df[targeted]
    t, c = sub[sub["treated"] == 1], sub[sub["treated"] == 0]
    if len(t) < 50 or len(c) < 50:
        raise ValueError("targeted subgroup too small for arm comparison")

    d_spend = t["spend_post"].mean() - c["spend_post"].mean()
    se_spend = np.sqrt(t["spend_post"].var(ddof=1) / len(t)
                       + c["spend_post"].var(ddof=1) / len(c))

    k_t, k_c = int(t["default_post"].sum()), int(c["default_post"].sum())
    d_def = k_t / len(t) - k_c / len(c)
    ub_def = newcombe_upper_bound(k_t, len(t), k_c, len(c), alpha_one_sided)

    return {
        "share_targeted": share,
        "n_targeted_t": len(t), "n_targeted_c": len(c),
        # per-eligible-customer (portfolio) scale
        "inc_spend_per_eligible": share * d_spend,
        "inc_spend_per_eligible_ci": (share * (d_spend - 1.96 * se_spend),
                                      share * (d_spend + 1.96 * se_spend)),
        "inc_default_pp_per_eligible": share * d_def * 100,
        "inc_default_pp_upper_bound": share * ub_def * 100,
        # subgroup scale (for interpretation)
        "subgroup_spend_diff": d_spend,
        "subgroup_default_diff_pp": d_def * 100,
    }


def feasible(pv: dict, delta_pp: float = config.DELTA_NONINF_PP) -> bool:
    """Portfolio guardrail: one-sided 95% UB of incremental default per
    eligible customer below the frozen tolerance."""
    return pv["inc_default_pp_upper_bound"] < delta_pp


def build_policy_specs(train_feat: pd.DataFrame, tau_s_train: np.ndarray,
                       tau_d_train: np.ndarray, mde: float) -> dict[str, dict]:
    """Candidate rules with ALL distribution-dependent cutoffs resolved to
    absolute values on POLICY-TRAIN. The specs are then applied unchanged on
    later splits — recomputing a quantile on POLICY-VAL/TEST would leak those
    splits into the policy definition."""
    pd_train = train_feat["pd_cal"]
    return {
        "treat_all":            {"tau_s_min": -np.inf, "pd_max": np.inf, "tau_d_max": np.inf},
        "tau_spend_positive":   {"tau_s_min": 0.0, "pd_max": np.inf, "tau_d_max": np.inf},
        "tau_spend_ge_mde":     {"tau_s_min": float(mde), "pd_max": np.inf, "tau_d_max": np.inf},
        "low_risk_3quintiles":  {"tau_s_min": -np.inf, "tau_d_max": np.inf,
                                 "pd_max": float(pd_train.quantile(0.60))},
        "tau_pos_and_low_risk": {"tau_s_min": 0.0, "tau_d_max": np.inf,
                                 "pd_max": float(pd_train.median())},
        "tau_pos_and_tau_def_low": {"tau_s_min": 0.0, "pd_max": np.inf,
                                    "tau_d_max": float(np.median(tau_d_train))},
    }


def apply_policy(spec: dict, feat: pd.DataFrame, tau_s: np.ndarray,
                 tau_d: np.ndarray) -> pd.Series:
    """Evaluate a frozen spec on any split (thresholds are absolute values)."""
    return ((pd.Series(tau_s, index=feat.index) >= spec["tau_s_min"])
            & (feat["pd_cal"] <= spec["pd_max"])
            & (pd.Series(tau_d, index=feat.index) <= spec["tau_d_max"]))


# ------------------------------------------------------- freeze serialization
def spec_to_json(spec: dict) -> dict:
    """Losslessly serializable spec: only FINITE (active) constraints are
    stored. A `null`-for-infinity encoding is not executable; this one
    round-trips exactly (tested)."""
    return {k: float(v) for k, v in spec.items() if np.isfinite(v)}


def spec_from_json(d: dict) -> dict:
    out = {"tau_s_min": -np.inf, "pd_max": np.inf, "tau_d_max": np.inf}
    unknown = set(d) - set(out)
    if unknown:
        raise ValueError(f"unknown spec keys {unknown}")
    out.update({k: float(v) for k, v in d.items()})
    return out


def freeze_problems(frozen: dict, current: dict) -> list[str]:
    """Compare the frozen policy record against the currently-derivable state.
    Nested dicts (model hashes) are compared key-by-key; any difference —
    same winner name with different spec, models, inputs, or table hashes —
    is a freeze violation."""
    problems = []
    for key, want in frozen.items():
        if key in ("frozen_on", "val_results", "amendments", "selection_rule",
                   "test_decision_rule"):
            continue  # descriptive fields
        got = current.get(key)
        if isinstance(want, dict) and isinstance(got, dict):
            for sub, w in want.items():
                g = got.get(sub)
                if w != g:
                    problems.append(f"{key}.{sub}: frozen={w!r} current={g!r}")
        elif want != got:
            problems.append(f"{key}: frozen={want!r} current={got!r}")
    return problems


def write_artifact_verified(path, content: bytes, expected_sha: str | None,
                            what: str) -> str:
    """Hash → verify → (only then) write. Returns the content hash.

    Ordering is the whole point: writing before comparing would destroy the
    frozen-run artifact on any disagreeing rerun, leaving the error message as
    the only surviving evidence of what the frozen state used to be. On
    mismatch this raises with the file on disk untouched.
    """
    import hashlib

    sha = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise RuntimeError(
            f"{what} no longer reproduces the frozen artifact (frozen "
            f"{expected_sha[:12]}…, current {sha[:12]}…). The on-disk file was "
            "NOT overwritten; investigate before doing anything else.")
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
    return sha


def assert_frozen(problems: list[str], context: str) -> None:
    if problems:
        raise RuntimeError(
            f"policy freeze violated ({context}):\n  " + "\n  ".join(problems) +
            "\nNothing was re-selected or re-evaluated. Restore the frozen "
            "artifacts; the frozen choice stands.")
