"""Simulated randomized credit-line-increase experiment — DESIGN_FREEZE.md §2 S3.

THIS IS A SIMULATION, clearly labeled as such: real pre-treatment covariates
and calibrated OOF scores from the frozen eligible cohort, with outcomes drawn
from the FROZEN DGP in src/experiment_design.py. Because the DGP is known,
every downstream inferential claim can be validated against ground truth.

Execution-order guarantees:
- refuses to run unless the linkage manifest verifies (frozen cohort intact);
- refuses to run unless the experiment design is frozen (design predates data);
- generates outcomes AT MOST ONCE: on success it writes the fixed-name marker
  EXPERIMENT_OUTCOMES_FROZEN.json (content hashes of outcomes + truth + design);
  if the marker exists, it can only VERIFY, never regenerate.
- ground-truth per-customer effects are stored in a SEPARATE artifact
  (dgp_truth.parquet). Analysis code reads experiment_outcomes.parquet only;
  the truth file is reserved for validation sections.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from . import config, data_prep, experiment_design, manifest
from .splits import make_or_load_split

OUTCOMES_PATH = config.ARTIFACTS_DIR / "experiment_outcomes.parquet"
TRUTH_PATH = config.ARTIFACTS_DIR / "dgp_truth.parquet"
CATE_TRUTH_PATH = config.ARTIFACTS_DIR / "cate_truth.parquet"


def compute_cate_truth() -> pd.DataFrame:
    """Conditional-expected-effect ground truth, E[Y(1)-Y(0) | X]:
    - cate_spend_true: analytic integral over the frozen DGP's noise,
      activation, and organic channels (experiment_design.cate_spend_true) —
      the ONLY valid reference for spend PEHE. delta_spend_true remains
      available but is a realized finite-sample contrast, not the estimand.
    - cate_default_true: p1 - p0 on the probability scale (exact).

    Depends only on pre-treatment quantities (pd_cal, pre-period spend);
    never on the realized assignment. Written once and hash-bound into the
    outcome marker; verified on later calls.
    """
    design = experiment_design.load_frozen()
    marker = json.loads(config.OUTCOME_MARKER.read_text())
    if CATE_TRUTH_PATH.exists():
        if _sha(CATE_TRUTH_PATH) != marker.get("cate_truth_sha256"):
            raise RuntimeError("cate_truth.parquet does not match the marker "
                               "hash — refusing to trust or regenerate it silently")
        return pd.read_parquet(CATE_TRUTH_PATH)

    if marker.get("cate_truth_sha256"):
        raise RuntimeError("frozen cate_truth.parquet is missing; restore it "
                           "without replacing its recorded hash")
    cohort = load_frozen_cohort()
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    pd_cal = cohort["pd_cal"].to_numpy()
    spend_pre = outcomes["spend_pre"].to_numpy()   # pre-treatment covariate
    p = design["dgp_params"]
    p0 = pd_cal * (design["planning_control_default_rate"] / pd_cal.mean())
    p1 = np.clip(p0 + experiment_design.tau_default(pd_cal, p), 0, 1)
    cate = pd.DataFrame({
        "cate_spend_true": experiment_design.cate_spend_true(pd_cal, spend_pre, p),
        "cate_default_true": p1 - p0,
    }, index=cohort.index)
    cate.to_parquet(CATE_TRUTH_PATH)
    marker["cate_truth_sha256"] = _sha(CATE_TRUTH_PATH)
    marker.setdefault("amendments", []).append(
        "2026-07-21: cate_truth.parquet added — analytic conditional expected "
        "effects E[Y(1)-Y(0)|X] for both outcomes (audit round 2: the realized "
        "contrast delta_spend_true is not a valid PEHE reference). Derived "
        "entirely from the frozen DGP + pre-treatment covariates; observed "
        "outcomes untouched.")
    config.OUTCOME_MARKER.write_text(json.dumps(marker, indent=2))
    return cate


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_cohort() -> pd.DataFrame:
    """Eligible cohort with covariates + calibrated OOF PD, after verifying the
    linkage manifest end-to-end (never trust unverified artifacts)."""
    stored = manifest.load()
    if stored is None:
        raise RuntimeError("no linkage manifest — run src.run_linkage first")
    art = config.ARTIFACTS_DIR
    oof = pd.read_parquet(art / "oof_scores_calibrated.parquet")
    thr = json.loads((art / "frozen_thresholds.json").read_text())["e5_abs_pd_threshold"]
    problems = manifest.diff(stored, manifest.build(oof, thr, art / "eligible_ids.json"))
    if problems:
        raise RuntimeError("linkage manifest mismatch:\n  " + "\n  ".join(problems))

    raw = data_prep.load_raw()
    dev_idx, _ = make_or_load_split(raw)
    dev = data_prep.CleaningRules("primary").fit_transform(raw.loc[dev_idx])
    elig = json.loads((art / "eligible_ids.json").read_text())["eligible_row_indices"]
    cohort = dev.loc[elig].copy()
    cohort["pd_cal"] = oof.loc[elig, "oof_cal"]
    return cohort


def assign_splits_and_treatment(cohort: pd.DataFrame) -> pd.DataFrame:
    """Policy-train/val/test split (stratified by PD quintile for balance) and
    exact 1:1 randomization within each split."""
    rng = np.random.default_rng(config.SEED_EXPERIMENT)
    out = pd.DataFrame(index=cohort.index)
    pd_q = pd.qcut(cohort["pd_cal"], q=5, labels=False, duplicates="drop")

    f_train, f_val, _ = config.POLICY_SPLIT
    split = pd.Series("", index=cohort.index, dtype=object)
    for _, idx in cohort.groupby(pd_q, observed=True).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n = len(idx)
        n_tr, n_va = int(round(n * f_train)), int(round(n * f_val))
        split.loc[idx[:n_tr]] = "policy_train"
        split.loc[idx[n_tr:n_tr + n_va]] = "policy_val"
        split.loc[idx[n_tr + n_va:]] = "policy_test"
    out["policy_split"] = split

    treated = pd.Series(0, index=cohort.index, dtype=int)
    for _, idx in out.groupby("policy_split").groups.items():
        idx = rng.permutation(np.asarray(idx))
        treated.loc[idx[: len(idx) // 2]] = 1   # exact 1:1 within split
    out["treated"] = treated
    return out


def generate_outcomes(cohort: pd.DataFrame, design: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw pre-period spend, post-period spend, and default outcomes from the
    frozen DGP. Returns (outcomes, truth)."""
    p = design["dgp_params"]
    rng = np.random.default_rng(p["seed_experiment"] + 1)
    n = len(cohort)
    pd_cal = cohort["pd_cal"].to_numpy()

    # standardized log income (missing -> 0 == "average"), links spend to a
    # real covariate so CUPED and CATE features have genuine structure
    inc = np.log1p(cohort["MonthlyIncome"].to_numpy(dtype=float))
    mu_i, sd_i = np.nanmean(inc), np.nanstd(inc)
    z_inc = np.where(np.isnan(inc), 0.0, (inc - mu_i) / sd_i)

    active_pre = rng.random(n) < p["spend_pre_p_active"]
    spend_pre = np.where(
        active_pre,
        rng.lognormal(p["spend_pre_mu"] + p["spend_pre_income_coef"] * z_inc,
                      p["spend_pre_sigma"]),
        0.0,
    )

    assign = assign_splits_and_treatment(cohort)
    treated = assign["treated"].to_numpy()

    # control post-period spend: persistence * multiplicative noise + organic activation
    post_noise = rng.normal(0, p["spend_post_noise_sd"], n)
    organic = rng.random(n) < p["spend_post_p_activate"]
    base_post = np.where(
        spend_pre > 0, spend_pre * np.exp(post_noise),
        np.where(organic, rng.lognormal(p["spend_pre_mu"], p["spend_pre_sigma"]), 0.0),
    )

    # true heterogeneous spend effect (additive dollars) + treated activation.
    # NOTE: the rng draw ORDER below is frozen — the potential-outcomes
    # bookkeeping further down uses a SEPARATE stream precisely so that the
    # observed outcomes remain bit-identical to the originally generated ones.
    t_spend = experiment_design.tau_spend(pd_cal, p)
    extra_act = (rng.random(n) < p["tau_spend_treated_activation"]) & (base_post == 0)
    spend_post = base_post.copy()
    spend_post[treated == 1] = np.maximum(
        0.0, base_post[treated == 1] + t_spend[treated == 1])
    treated_new = (treated == 1) & extra_act
    spend_post[treated_new] = rng.lognormal(
        p["spend_pre_mu"], p["spend_pre_sigma"], int(treated_new.sum()))

    # default: control prob proportional to calibrated PD, rescaled in
    # aggregate to the FROZEN planning baseline; treated adds tau_default(x).
    # A single uniform per customer drives BOTH potential outcomes (monotone
    # coupling), so realized outcomes are exactly `treated ? y1 : y0`.
    p0 = pd_cal * (design["planning_control_default_rate"] / pd_cal.mean())
    t_def = experiment_design.tau_default(pd_cal, p)
    p1 = np.clip(p0 + t_def, 0, 1)
    u = rng.random(n)
    default_y0 = (u < p0).astype(int)
    default_y1 = (u < p1).astype(int)
    default_post = np.where(treated == 1, default_y1, default_y0)

    # ---- potential-outcomes bookkeeping (truth artifact only) ----
    # spend_y0 is the no-treatment path; spend_y1 applies tau to spenders and
    # the activation channel to non-spenders. Activation spends realized in
    # the experiment reuse the realized draws; counterfactual activations
    # (untreated customers who WOULD have activated) come from an independent
    # stream — same distribution, and provably no effect on observed data.
    rng_cf = np.random.default_rng(p["seed_experiment"] + 2)
    cf_act_spend = rng_cf.lognormal(p["spend_pre_mu"], p["spend_pre_sigma"], n)
    spend_y0 = base_post
    # frozen behavior: additive tau applies to EVERY treated customer (an
    # inactive customer with positive tau starts spending tau dollars), and
    # the activation channel then overrides for activators
    spend_y1 = np.maximum(0.0, base_post + t_spend)
    spend_y1[treated_new] = spend_post[treated_new]        # realized activation draws
    cf_act = extra_act & (treated == 0)                    # would-have-activated controls
    spend_y1[cf_act] = cf_act_spend[cf_act]
    if not np.allclose(np.where(treated == 1, spend_y1, spend_y0), spend_post):
        raise RuntimeError("potential-outcome bookkeeping inconsistent with "
                           "realized outcomes — refusing to write truth")

    outcomes = pd.DataFrame({
        "policy_split": assign["policy_split"],
        "treated": treated,
        "spend_pre": spend_pre,
        "spend_post": spend_post,
        "default_post": default_post,
    }, index=cohort.index)
    truth = pd.DataFrame({
        "tau_spend_true": t_spend,           # the frozen τ function (no activation channel)
        "tau_default_true": t_def,
        "p_default_control_true": p0,
        "spend_y0_true": spend_y0,           # full potential outcomes incl. activation
        "spend_y1_true": spend_y1,
        "delta_spend_true": spend_y1 - spend_y0,
        "delta_default_prob_true": p1 - p0,
    }, index=cohort.index)
    return outcomes, truth


def verify_frozen_outcomes() -> None:
    """Verify recorded inputs and outcomes without drawing or writing data."""
    marker = json.loads(config.OUTCOME_MARKER.read_text())
    problems = []
    checks = [("outcomes_sha256", OUTCOMES_PATH), ("truth_sha256", TRUTH_PATH),
              ("linkage_manifest_sha256", manifest.MANIFEST_PATH)]
    if "cate_truth_sha256" in marker:
        checks.append(("cate_truth_sha256", CATE_TRUTH_PATH))
    for key, path in checks:
        if not path.exists():
            problems.append(f"{path.name} missing")
        elif _sha(path) != marker.get(key):
            problems.append(f"{path.name} hash mismatch")
    if marker.get("design_sha256") != experiment_design.params_sha256():
        problems.append("DGP params changed after outcome generation")
    if problems:
        raise RuntimeError(
            "outcome freeze broken:\n  " + "\n  ".join(problems) +
            "\nOutcomes are generated at most once; nothing will be redrawn.")


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if not config.OUTCOME_MARKER.exists() and any(
            path.exists() for path in (OUTCOMES_PATH, TRUTH_PATH, CATE_TRUTH_PATH)):
        raise RuntimeError("existing outcome artifacts have no freeze marker; "
                           "restore the marker instead of redrawing outcomes")
    cohort = load_frozen_cohort()
    design = experiment_design.load_frozen()

    if config.OUTCOME_MARKER.exists():
        verify_frozen_outcomes()
        print("outcomes already frozen and verified — no-op")
        return

    outcomes, truth = generate_outcomes(cohort, design)
    outcomes.to_parquet(OUTCOMES_PATH)
    truth.to_parquet(TRUTH_PATH)
    config.OUTCOME_MARKER.write_text(json.dumps({
        "generated_on": str(date.today()),
        "seed": design["dgp_params"]["seed_experiment"],
        "outcomes_sha256": _sha(OUTCOMES_PATH),
        "truth_sha256": _sha(TRUTH_PATH),
        "design_sha256": experiment_design.params_sha256(),
        "linkage_manifest_sha256": _sha(manifest.MANIFEST_PATH),
        "note": "cohort, split, eligibility, and design are IMMUTABLE from this point",
    }, indent=2))
    n_by = outcomes.groupby(["policy_split", "treated"]).size()
    print(n_by.to_string())
    print(f"outcomes generated ONCE and frozen ({len(outcomes)} customers)")


if __name__ == "__main__":
    main()
