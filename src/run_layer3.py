"""Layer 3 pipeline: CATE models -> candidate policies -> policy-level
guardrail -> frozen executable policy -> single POLICY-TEST evaluation.

Run: python -m src.run_layer3

Freeze semantics (audit-hardened): "the policy" is the PERSISTED artifacts —
X-learner boosters on disk plus an executable spec — bound by content hash in
policy_freeze.json together with the input state (outcome marker, split ID
hashes, feature schema) and the candidate table + selection rule that chose
it. Reruns verify and load; they never refit-for-decision, re-select, or
re-predict POLICY-TEST. The one-shot TEST artifact records the freeze hash it
was produced under and is only ever verified and read back afterwards.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from . import config, experiment_design, manifest, policy
from .cate import TLearner, XLearner, pehe
from .simulate_experiment import (OUTCOMES_PATH, compute_cate_truth,
                                  load_frozen_cohort, verify_frozen_outcomes)

POLICY_FREEZE_PATH = config.ARTIFACTS_DIR / "policy_freeze.json"
TEST_EVAL_PATH = config.ARTIFACTS_DIR / "policy_test_evaluation.json"
MODELS_DIR = config.ARTIFACTS_DIR / "policy_models"

SELECTION_RULE = ("among candidates with one-sided 95% UB of incremental "
                  "default per eligible < delta_pp, choose max incremental "
                  "spend per eligible on POLICY-VAL")
TEST_DECISION_RULE = ("evaluate the frozen policy exactly once on POLICY-TEST: "
                      "per-eligible incremental spend with 95% CI and "
                      "incremental default with one-sided 95% Newcombe UB; "
                      "feasible iff UB < delta_pp; decisions compared to DGP "
                      "truth (simulation privilege)")


def _art(name: str):
    return config.ARTIFACTS_DIR / name


def _ids_sha(index) -> str:
    return hashlib.sha256(
        np.sort(np.asarray(index, dtype=np.int64)).tobytes()).hexdigest()


LAYER3_MARKER_KEYS = ("policy_test_result_sha256",)


def marker_identity_sha() -> str:
    """Hash of the experiment-outcome marker EXCLUDING Layer-3 back-references.

    The marker anchors the POLICY-TEST result hash, while the TEST result
    records the marker hash — hashing the whole file in both directions would
    be circular (writing either one would invalidate the other). Excluding the
    Layer-3 keys keeps the *experiment identity* — seed, outcome/truth/cate
    hashes, design hash — stable and verifiable in both directions.
    """
    m = json.loads(config.OUTCOME_MARKER.read_text())
    payload = {k: v for k, v in m.items() if k not in LAYER3_MARKER_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


def result_payload_sha(result: dict) -> str:
    """Content hash of the POLICY-TEST evaluation ITSELF (every number in it),
    excluding the self-referential hash field. Binding the result to the
    freeze/marker/id hashes proves it was produced under the right conditions;
    only this proves the numbers were not edited afterwards. The value is
    mirrored into the outcome marker, so the check has an external anchor."""
    payload = {k: v for k, v in result.items() if k != "result_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()


def build_features(cohort: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment features only: cleaned covariates + calibrated OOF PD +
    simulated pre-period spend. Post-treatment variables never enter."""
    feat = cohort.drop(columns=[config.TARGET]).copy()
    feat["spend_pre"] = outcomes["spend_pre"]
    return feat


def current_env_state(feat, idx_tr, idx_va, idx_te) -> dict:
    return {
        "outcome_marker_sha256": marker_identity_sha(),
        "linkage_manifest_sha256": manifest.file_sha256(manifest.MANIFEST_PATH),
        "train_ids_sha256": _ids_sha(feat.index[idx_tr]),
        "val_ids_sha256": _ids_sha(feat.index[idx_va]),
        "test_ids_sha256": _ids_sha(feat.index[idx_te]),
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(list(feat.columns)).encode()).hexdigest(),
    }


def main() -> None:
    t0 = time.time()
    if not config.OUTCOME_MARKER.exists():
        raise RuntimeError("no frozen experiment outcomes — run Layer 2 first")

    marker = json.loads(config.OUTCOME_MARKER.read_text())
    if not POLICY_FREEZE_PATH.exists() and (
            TEST_EVAL_PATH.exists() or marker.get("policy_test_result_sha256")
            or MODELS_DIR.exists()
            or _art("policy_candidates_val.csv").exists()):
        raise RuntimeError("frozen policy state is missing; restore the policy "
                           "freeze before loading data or fitting models")
    if marker.get("policy_test_result_sha256") and not TEST_EVAL_PATH.exists():
        raise RuntimeError("frozen policy test result is missing; restore it "
                           "instead of evaluating POLICY-TEST again")

    design = experiment_design.load_frozen()
    verify_frozen_outcomes()
    p = design["dgp_params"]
    delta_pp = p["delta_noninf_pp"]
    cohort = load_frozen_cohort()
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    cate_truth = compute_cate_truth()
    feat = build_features(cohort, outcomes)

    split = outcomes["policy_split"]
    idx_tr, idx_va = split == "policy_train", split == "policy_val"
    idx_te = split == "policy_test"
    y_spend = outcomes["spend_post"].to_numpy()
    y_def = outcomes["default_post"].to_numpy()
    tr_flag = outcomes["treated"].to_numpy()
    env = current_env_state(feat, idx_tr, idx_va, idx_te)

    frozen = json.loads(POLICY_FREEZE_PATH.read_text()) \
        if POLICY_FREEZE_PATH.exists() else None

    # ---- policy models: fit-and-persist once, verify-and-load afterwards
    xlearners = {}
    if frozen is None:
        print("fitting X-learners (spend + default) on POLICY-TRAIN ...")
        model_hashes = {}
        for outcome, y in (("spend", y_spend), ("default", y_def)):
            xl = XLearner().fit(feat[idx_tr], y[idx_tr.to_numpy()],
                                tr_flag[idx_tr.to_numpy()])
            model_hashes[outcome] = xl.save(MODELS_DIR / outcome)
            xlearners[outcome] = xl
    else:
        policy.assert_frozen(policy.freeze_problems(
            {k: frozen[k] for k in env}, env), "input state")
        for outcome in ("spend", "default"):
            xlearners[outcome] = XLearner.load(
                MODELS_DIR / outcome, frozen["model_hashes"][outcome])
        model_hashes = frozen["model_hashes"]
        print("frozen policy models verified and loaded from disk")

    def tau(outcome, mask):
        return xlearners[outcome].predict_tau(feat[mask])

    # ---- CATE validation on POLICY-VAL (truth is validation-only; T-learner
    #      baselines are refit here — they are comparisons, not the policy)
    print("CATE validation ...")
    tlearners = {o: TLearner().fit(feat[idx_tr], y[idx_tr.to_numpy()],
                                   tr_flag[idx_tr.to_numpy()])
                 for o, y in (("spend", y_spend), ("default", y_def))}
    truth_va = cate_truth[idx_va]
    tiers_va = pd.qcut(feat.loc[idx_va, "pd_cal"], q=5,
                       labels=[f"riskQ{i}" for i in range(1, 6)])
    cate_val = {}
    for outcome, truth_col, scale in (("spend", "cate_spend_true", 1.0),
                                      ("default", "cate_default_true", 100.0)):
        preds = {"X": tau(outcome, idx_va) * scale,
                 "T": tlearners[outcome].predict_tau(feat[idx_va]) * scale}
        tru = truth_va[truth_col].to_numpy() * scale
        for learner, est in preds.items():
            cate_val[f"{outcome}_{learner}"] = {
                "pehe": pehe(est, tru),
                "rank_corr": float(stats.spearmanr(est, tru).statistic),
                "pehe_by_risk_tier": {
                    str(t): pehe(est[(tiers_va == t).to_numpy()],
                                 tru[(tiers_va == t).to_numpy()])
                    for t in tiers_va.cat.categories},
            }
    _art("cate_validation.json").write_text(json.dumps(cate_val, indent=2))
    for k, v in cate_val.items():
        print(f"  {k}: PEHE={v['pehe']:.3f} rank_corr={v['rank_corr']:.3f}")

    # ---- candidates on VAL (thresholds frozen on TRAIN estimates)
    tau_s_tr, tau_d_tr = tau("spend", idx_tr), tau("default", idx_tr)
    specs = policy.build_policy_specs(feat[idx_tr], tau_s_tr, tau_d_tr,
                                      mde=p["spend_mde_dollars"])
    tau_s_va, tau_d_va = tau("spend", idx_va), tau("default", idx_va)
    val_df = outcomes[idx_va]
    rows = []
    for name, spec in specs.items():
        targeted = policy.apply_policy(spec, feat[idx_va], tau_s_va, tau_d_va)
        pv = policy.policy_value(val_df, targeted)
        pv["policy"] = name
        pv["feasible"] = policy.feasible(pv, delta_pp)
        rows.append(pv)
    cand = pd.DataFrame(rows).set_index("policy")
    # Hash IN MEMORY and verify BEFORE writing: writing first would destroy the
    # frozen-run artifact whenever a rerun disagrees with it, leaving the error
    # message as the only surviving evidence. The file is (re)written only when
    # no freeze exists yet, or when the bytes are provably identical.
    cand_sha = policy.write_artifact_verified(
        _art("policy_candidates_val.csv"), cand.to_csv().encode(),
        None if frozen is None else frozen["candidate_table_sha256"],
        "POLICY-VAL candidate table")
    print(cand[["share_targeted", "inc_spend_per_eligible",
                "inc_default_pp_per_eligible", "inc_default_pp_upper_bound",
                "feasible"]].round(4).to_string())

    feasible_cand = cand[cand["feasible"]]
    if feasible_cand.empty:
        raise RuntimeError("no feasible candidate policy on POLICY-VAL — "
                           "report INCONCLUSIVE rather than forcing a winner")
    winner = feasible_cand["inc_spend_per_eligible"].idxmax()

    if frozen is None:
        frozen = {
            "policy": winner,
            "spec": policy.spec_to_json(specs[winner]),
            "selection_rule": SELECTION_RULE,
            "test_decision_rule": TEST_DECISION_RULE,
            "delta_pp": delta_pp,
            "model_hashes": model_hashes,
            "candidate_table_sha256": cand_sha,
            **env,
            "val_results": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in cand.loc[winner].items()},
            "frozen_on": str(date.today()),
        }
        POLICY_FREEZE_PATH.write_text(json.dumps(frozen, indent=2, default=float))
        print(f"policy frozen: {winner}")
    else:
        current = {"policy": winner, "spec": policy.spec_to_json(specs[winner]),
                   "delta_pp": delta_pp, "model_hashes": model_hashes,
                   "candidate_table_sha256": cand_sha, **env}
        policy.assert_frozen(policy.freeze_problems(
            {k: frozen[k] for k in current}, current), "policy re-derivation")
        print(f"frozen policy verified: {winner}")

    # ---- trade-off + uplift curves (VAL only; deterministic diagnostics)
    ks = np.arange(10, 100, 10)
    cuts = {int(k): float(np.quantile(tau_s_tr, 1 - k / 100)) for k in ks}
    trade = []
    for k, cut in cuts.items():
        targeted = pd.Series(tau_s_va >= cut, index=val_df.index)
        if targeted.sum() < 200:
            continue
        pv = policy.policy_value(val_df, targeted)
        trade.append({"top_k_pct": k, "tau_cutoff": cut,
                      "inc_spend_per_eligible": pv["inc_spend_per_eligible"],
                      "inc_default_pp_per_eligible": pv["inc_default_pp_per_eligible"],
                      "inc_default_pp_upper_bound": pv["inc_default_pp_upper_bound"]})
    trade = pd.DataFrame(trade)
    trade.to_csv(_art("policy_tradeoff_curve.csv"), index=False)

    # ---- ONE-SHOT POLICY-TEST evaluation (verify-and-read afterwards)
    freeze_sha = manifest.file_sha256(POLICY_FREEZE_PATH)
    marker = json.loads(config.OUTCOME_MARKER.read_text())
    if TEST_EVAL_PATH.exists():
        test_result = json.loads(TEST_EVAL_PATH.read_text())
        # conditions it was produced under ...
        policy.assert_frozen(policy.freeze_problems(
            {"policy_freeze_sha256": test_result.get("policy_freeze_sha256"),
             "test_ids_sha256": test_result.get("test_ids_sha256"),
             "outcome_marker_sha256": test_result.get("outcome_marker_sha256")},
            {"policy_freeze_sha256": freeze_sha,
             "test_ids_sha256": env["test_ids_sha256"],
             "outcome_marker_sha256": env["outcome_marker_sha256"]}),
            "stored POLICY-TEST evaluation")
        # ... AND the result content itself, anchored in the outcome marker
        policy.assert_frozen(policy.freeze_problems(
            {"result_sha256": marker.get("policy_test_result_sha256"),
             "self_recorded_sha256": test_result.get("result_sha256")},
            {"result_sha256": result_payload_sha(test_result),
             "self_recorded_sha256": result_payload_sha(test_result)}),
            "POLICY-TEST result content")
        print("POLICY-TEST already evaluated once — result content and "
              "conditions verified, read back (no re-prediction)")
    else:
        tau_s_te, tau_d_te = tau("spend", idx_te), tau("default", idx_te)
        spec = policy.spec_from_json(frozen["spec"])
        targeted_te = policy.apply_policy(spec, feat[idx_te], tau_s_te, tau_d_te)
        pv_te = policy.policy_value(outcomes[idx_te], targeted_te)
        pv_te["feasible"] = policy.feasible(pv_te, delta_pp)

        truth_te = cate_truth[idx_te]
        true_inc_def_pp = float(targeted_te.mean()
                                * truth_te.loc[targeted_te, "cate_default_true"].mean() * 100)
        true_inc_spend = float(targeted_te.mean()
                               * truth_te.loc[targeted_te, "cate_spend_true"].mean())
        test_result = {
            "policy": frozen["policy"],
            "policy_freeze_sha256": freeze_sha,
            "test_ids_sha256": env["test_ids_sha256"],
            "outcome_marker_sha256": env["outcome_marker_sha256"],
            "policy_value_test": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in pv_te.items()},
            "truth_validation": {
                "true_inc_default_pp_per_eligible": true_inc_def_pp,
                "true_inc_spend_per_eligible": true_inc_spend,
                "upper_bound_above_truth": bool(
                    pv_te["inc_default_pp_upper_bound"] >= true_inc_def_pp),
                "true_below_delta": bool(true_inc_def_pp < delta_pp),
                "guardrail_decision_correct": bool(
                    pv_te["feasible"] == (true_inc_def_pp < delta_pp)),
            },
            "evaluated_on": str(date.today()),
        }
        test_result["result_sha256"] = result_payload_sha(test_result)
        TEST_EVAL_PATH.write_text(json.dumps(test_result, indent=2, default=float))
        marker["policy_test_result_sha256"] = test_result["result_sha256"]
        config.OUTCOME_MARKER.write_text(json.dumps(marker, indent=2))
        print("POLICY-TEST evaluated (once); result content hash anchored "
              "in the outcome marker")
    print(json.dumps(test_result["policy_value_test"], indent=2)[:600])
    print(json.dumps(test_result["truth_validation"], indent=2))

    # ---- figures (VAL diagnostics)
    order = np.argsort(-tau_s_va)
    va_sorted = val_df.iloc[order]
    fracs, gains = [], []
    for k in np.linspace(0.05, 1.0, 20):
        top = va_sorted.iloc[: int(len(va_sorted) * k)]
        tt, cc = top[top["treated"] == 1], top[top["treated"] == 0]
        gains.append((tt["spend_post"].mean() - cc["spend_post"].mean()) * k)
        fracs.append(k)
    ate_val = (val_df[val_df["treated"] == 1]["spend_post"].mean()
               - val_df[val_df["treated"] == 0]["spend_post"].mean())

    import matplotlib.pyplot as plt

    from . import plots as _plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fracs, gains, "o-", label="X-learner ranking")
    axes[0].plot([0, 1], [0, ate_val], "k--", lw=1, label="random targeting")
    axes[0].set_xlabel("fraction targeted (ranked by τ̂_spend)")
    axes[0].set_ylabel("incremental spend per eligible ($)")
    axes[0].set_title("Cumulative gain — POLICY-VAL")
    axes[0].legend()
    axes[1].plot(trade["inc_default_pp_upper_bound"],
                 trade["inc_spend_per_eligible"], "o-")
    for _, r in trade.iterrows():
        axes[1].annotate(f"{int(r['top_k_pct'])}%",
                         (r["inc_default_pp_upper_bound"],
                          r["inc_spend_per_eligible"]), fontsize=7)
    axes[1].axvline(delta_pp, color="tab:red", ls="--", lw=1,
                    label=f"δ = {delta_pp}pp")
    axes[1].set_xlabel("incremental default UB per eligible (pp)")
    axes[1].set_ylabel("incremental spend per eligible ($)")
    axes[1].set_title("Spend / default trade-off — POLICY-VAL")
    axes[1].legend()
    fig.tight_layout()
    _plots.save_fig(fig, "layer3_targeting.png")

    # ---- tier-level recommendation table: TRAIN ∪ VAL ONLY. Strict boundary:
    #      TEST rows are excluded even from covariate-only summaries.
    trva = (idx_tr | idx_va)
    tiers = pd.qcut(feat.loc[trva, "pd_cal"], q=5,
                    labels=[f"riskQ{i}" for i in range(1, 6)])
    tau_s_all = tau("spend", trva)
    tau_d_all = tau("default", trva)
    targeted_all = policy.apply_policy(policy.spec_from_json(frozen["spec"]),
                                       feat[trva], tau_s_all, tau_d_all)
    tier_table = pd.DataFrame({
        "n": tiers.value_counts().sort_index(),
        "share_targeted": targeted_all.groupby(tiers, observed=True).mean(),
        "mean_tau_spend_hat": pd.Series(tau_s_all, index=feat.index[trva])
                                .groupby(tiers, observed=True).mean(),
        "mean_tau_default_hat_pp": pd.Series(tau_d_all * 100, index=feat.index[trva])
                                     .groupby(tiers, observed=True).mean(),
    })
    tier_table.to_csv(_art("policy_by_risk_tier.csv"))
    print(tier_table.round(3).to_string())
    print(f"layer 3 complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
