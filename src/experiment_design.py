"""FROZEN experiment design for the simulated credit-line-increase test.

Everything the Layer-2 analysis could be tempted to bend after seeing outcomes
lives HERE, is content-hashed into reports/artifacts/experiment_design_frozen.json
BEFORE outcome generation, and is verified by every later stage:

- the control-baseline planning assumption. The eligible cohort's HISTORICAL
  default rate is used as the planning proxy for the control arm. It is a
  PRE-TREATMENT PLANNING PROXY, not an observed control outcome — the
  simulated experiment has produced nothing yet when it is frozen;
- the minimum economically meaningful spend lift (hypothetical stakeholder
  input, like δ);
- the full DGP specification: baseline spend model, heterogeneous true-effect
  functions for both outcomes, and noise models.

After outcomes are generated (EXPERIMENT_OUTCOMES_FROZEN.json exists), none of
these numbers — baseline, δ, MDE, heterogeneity functions — may change.
src/simulate_experiment.py enforces this with content hashes and explicit
RuntimeErrors.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from . import config

DESIGN_PATH = config.ARTIFACTS_DIR / "experiment_design_frozen.json"

# ---------------------------------------------------------------------------
# FROZEN DGP + planning parameters. The dict itself is hashed; changing any
# value after freezing is detectable and refused.
# ---------------------------------------------------------------------------
DGP_PARAMS: dict = {
    # --- spend model (fully synthetic: the credit dataset has no spend) ---
    # pre-period 6-month card spend: zero-inflated lognormal, scale linked to
    # (standardized log) income so CUPED has a real covariate to work with
    "spend_pre_p_active": 0.85,
    "spend_pre_mu": 5.2,          # log-dollars; median active spend ~ $181
    "spend_pre_sigma": 0.80,
    "spend_pre_income_coef": 0.30,
    # control post-period spend = pre * exp(noise); small organic activation
    "spend_post_noise_sd": 0.35,
    "spend_post_p_activate": 0.02,
    # --- true heterogeneous treatment effect on spend (dollars, additive) ---
    # tau_spend(x) = a * (1 - pd/pd_e5_threshold) + b  -> positive for
    # low-risk eligible, negative near the eligibility risk boundary
    "tau_spend_slope": 40.0,
    "tau_spend_intercept": -8.0,
    "tau_spend_treated_activation": 0.03,   # extra activation prob if treated
    # --- default model ---
    # control default prob proportional to the calibrated OOF PD, rescaled in
    # aggregate to the frozen planning baseline (keeps DGP consistent with the
    # planning assumption by construction)
    # true incremental default: tau_default(x) = cap * (pd/threshold)^power
    "tau_default_cap": 0.010,
    "tau_default_power": 1.5,
    # --- design constants (mirrored from config for hashing) ---
    "delta_noninf_pp": config.DELTA_NONINF_PP,
    "delta_sensitivity_pp": list(config.DELTA_SENSITIVITY_PP),
    "spend_mde_dollars": 12.0,    # minimum economically meaningful lift per
                                  # customer per 6 months — hypothetical
                                  # stakeholder input, NOT tuned to power
    "alpha_one_sided": config.ALPHA_ONE_SIDED,
    "alpha_superiority_two_sided": 0.05,
    "power_target": config.POWER_TARGET,
    "policy_split": list(config.POLICY_SPLIT),
    "seed_experiment": config.SEED_EXPERIMENT,
    # --- pre-registered decision rule (frozen text, hashed with the rest) ---
    "decision_rule": (
        "SHIP only if BOTH hold on POLICY-TRAIN ∪ POLICY-VAL: "
        "(1) CUPED-adjusted spend lift point estimate >= spend_mde_dollars AND "
        "its 95% bootstrap CI lower bound > 0; "
        "(2) default non-inferiority PASSES at delta_noninf_pp via the "
        "one-sided 95% Newcombe upper bound. "
        "Otherwise DO NOT SHIP at the ATE level (Layer 3 may still identify a "
        "feasible targeted segment policy)."),
}


def params_sha256(params: dict | None = None) -> str:
    blob = json.dumps(params or DGP_PARAMS, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def freeze(eligible_default_rate: float, n_eligible: int,
           linkage_manifest_sha: str) -> dict:
    """Write the frozen design artifact (refuses if outcomes already exist and
    the design would differ)."""
    design = {
        "planning_control_default_rate": float(eligible_default_rate),
        "planning_note": (
            "HISTORICAL 2-year default rate of the frozen eligible cohort, "
            "used as a pre-treatment PLANNING PROXY for the control arm. It is "
            "not an observed control outcome; no simulated outcome existed "
            "when this was frozen."),
        "n_eligible": int(n_eligible),
        "dgp_params": DGP_PARAMS,
        "dgp_params_sha256": params_sha256(),
        "linkage_manifest_sha256": linkage_manifest_sha,
        "frozen_on": str(date.today()),
    }
    if DESIGN_PATH.exists():
        stored = load_frozen()
        for key in ("planning_control_default_rate", "dgp_params_sha256",
                    "linkage_manifest_sha256", "n_eligible"):
            if stored.get(key) != design[key]:
                raise RuntimeError(
                    f"experiment design mismatch on {key!r}: frozen="
                    f"{stored.get(key)!r} vs current={design[key]!r}. The "
                    "design may not drift; restore code/artifacts or delete "
                    "the design + outcomes deliberately (pre-outcome only)."
                )
        return stored
    if config.OUTCOME_MARKER.exists():
        raise RuntimeError(
            "outcomes exist but no frozen design artifact — cannot establish "
            "that the design predates the outcomes. Investigate.")
    DESIGN_PATH.write_text(json.dumps(design, indent=2))
    return design


def load_frozen() -> dict:
    if not DESIGN_PATH.exists():
        raise RuntimeError("experiment design not frozen yet — run freeze() "
                           "(via src.run_layer2) before generating outcomes")
    stored = json.loads(DESIGN_PATH.read_text())
    payload = stored.get("dgp_params")
    if not isinstance(payload, dict) or not payload or params_sha256(payload) != stored.get("dgp_params_sha256"):
        raise RuntimeError("frozen design payload does not match its recorded hash")
    if stored["dgp_params_sha256"] != params_sha256():
        raise RuntimeError(
            "DGP parameters in code no longer match the frozen design artifact "
            "— refusing to proceed. Post-outcome parameter edits are forbidden; "
            "if this is a pre-outcome change, delete the design artifact "
            "deliberately and re-freeze.")
    return stored


# ---------------------------------------------------------------------------
# True-effect functions (shared by the DGP and by Layer-3 ground-truth
# validation; analysis code must never call these on observed data)
# ---------------------------------------------------------------------------
def tau_spend(pd_cal: np.ndarray, p: dict | None = None) -> np.ndarray:
    p = p or DGP_PARAMS
    thr_frac = np.clip(pd_cal, 0, None) / _e5_threshold()
    return p["tau_spend_slope"] * (1.0 - thr_frac) + p["tau_spend_intercept"]


def tau_default(pd_cal: np.ndarray, p: dict | None = None) -> np.ndarray:
    p = p or DGP_PARAMS
    thr_frac = np.clip(pd_cal / _e5_threshold(), 0, 1)
    return p["tau_default_cap"] * thr_frac ** p["tau_default_power"]


def _e5_threshold() -> float:
    return float(json.loads(
        (config.ARTIFACTS_DIR / "frozen_thresholds.json").read_text()
    )["e5_abs_pd_threshold"])


# ---------------------------------------------------------------------------
# Analytic conditional expected spend effect: E[Y(1) - Y(0) | X = x]
# ---------------------------------------------------------------------------
def _censored_additive_effect(tau: np.ndarray, log_scale: np.ndarray,
                              sigma: float) -> np.ndarray:
    """E[max(0, B + tau) - B] for B ~ lognormal(log_scale, sigma).

    For tau >= 0 this is exactly tau. For tau < 0 the effect is censored at
    -B: with L = -tau and a = (ln L - log_scale)/sigma,
        E = tau * (1 - Phi(a)) - exp(log_scale + sigma^2/2) * Phi(a - sigma).
    """
    from scipy.stats import norm

    tau = np.asarray(tau, dtype=float)
    out = np.where(tau >= 0, tau, np.nan)
    neg = tau < 0
    if neg.any():
        L = -tau[neg]
        ls = np.broadcast_to(log_scale, tau.shape)[neg]
        a = (np.log(L) - ls) / sigma
        out[neg] = (tau[neg] * (1 - norm.cdf(a))
                    - np.exp(ls + sigma ** 2 / 2) * norm.cdf(a - sigma))
    return out


def cate_spend_true(pd_cal: np.ndarray, spend_pre: np.ndarray,
                    p: dict | None = None) -> np.ndarray:
    """TRUE conditional expected spend effect under the frozen DGP,
    E[Y(1) - Y(0) | pd_cal, spend_pre] — integrating over post-period noise,
    organic activation, and treated-activation draws; independent of the
    realized treatment assignment.

    This — not the realized finite-sample contrast delta_spend_true — is the
    estimand CATE models are scored against (PEHE). Branches mirror the frozen
    generator exactly:
    - spend_pre > 0: B = spend_pre * exp(N(0, sd_noise)); effect is the
      censored additive tau.
    - spend_pre = 0: with prob 0.02 the customer organically activates in both
      arms (effect = censored tau against a fresh lognormal base); otherwise
      with prob 0.03 treatment activates them (effect = mean activation
      spend); otherwise the additive tau floors at zero (effect = max(tau,0)).
    Cross-validated against an oracle-seed Monte Carlo in tests.
    """
    p = p or DGP_PARAMS
    tau = tau_spend(pd_cal, p)
    active = np.asarray(spend_pre) > 0

    eff_active = _censored_additive_effect(
        tau, np.log(np.where(active, spend_pre, 1.0)), p["spend_post_noise_sd"])

    mean_act_spend = float(np.exp(p["spend_pre_mu"] + p["spend_pre_sigma"] ** 2 / 2))
    eff_organic = _censored_additive_effect(
        tau, np.full_like(tau, p["spend_pre_mu"]), p["spend_pre_sigma"])
    p_org = p["spend_post_p_activate"]
    p_act = p["tau_spend_treated_activation"]
    eff_inactive = (p_org * eff_organic
                    + (1 - p_org) * (p_act * mean_act_spend
                                     + (1 - p_act) * np.maximum(tau, 0.0)))

    return np.where(active, eff_active, eff_inactive)
