"""Layer 2 pipeline: freeze design -> power -> generate outcomes -> analyze.

Run: python -m src.run_layer2

Strict order (each step refuses to run out of sequence):
  1. verify frozen linkage cohort;
  2. FREEZE the experiment design (planning baseline from the eligible
     cohort's HISTORICAL default rate — a pre-treatment planning proxy, not an
     observed control outcome — plus the full DGP spec and decision rule);
  3. prospective power analysis from frozen planning inputs only;
  4. generate outcomes ONCE (writes EXPERIMENT_OUTCOMES_FROZEN.json);
  5. pre-registered confirmatory analysis on POLICY-TRAIN ∪ POLICY-VAL
     (POLICY-TEST stays untouched for Layer 3's single final evaluation);
  6. exploratory segment analyses (BH-corrected, labeled exploratory);
  7. validation against DGP ground truth (simulation privilege, labeled).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, experiment_design, inference, manifest, power
from .simulate_experiment import (OUTCOMES_PATH, TRUTH_PATH, compute_cate_truth,
                                  load_frozen_cohort)
from . import simulate_experiment


def _art(name: str):
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.ARTIFACTS_DIR / name


def planning_sigma_spend(cohort: pd.DataFrame, params: dict) -> float:
    """σ of control-arm post spend under the FROZEN baseline spend model,
    simulated with a planning seed disjoint from the outcome seed. Uses no
    outcome data — the DGP spec itself is the (frozen) planning knowledge."""
    rng = np.random.default_rng(params["seed_experiment"] + 777_000)
    n = len(cohort)
    inc = np.log1p(cohort["MonthlyIncome"].to_numpy(dtype=float))
    mu_i, sd_i = np.nanmean(inc), np.nanstd(inc)
    z_inc = np.where(np.isnan(inc), 0.0, (inc - mu_i) / sd_i)
    active = rng.random(n) < params["spend_pre_p_active"]
    pre = np.where(active, rng.lognormal(
        params["spend_pre_mu"] + params["spend_pre_income_coef"] * z_inc,
        params["spend_pre_sigma"]), 0.0)
    # mirror the full frozen control-arm DGP, including the 2% organic
    # activation of pre-period non-spenders (planning uses vector draws; the
    # frozen generator's shared-scalar organic draw is a documented wart)
    organic = rng.random(n) < params["spend_post_p_activate"]
    post = np.where(
        pre > 0, pre * np.exp(rng.normal(0, params["spend_post_noise_sd"], n)),
        np.where(organic, rng.lognormal(params["spend_pre_mu"],
                                        params["spend_pre_sigma"], n), 0.0))
    return float(post.std(ddof=1))


def main() -> None:
    t0 = time.time()

    # ---- 1-2: frozen cohort + frozen design
    cohort = load_frozen_cohort()
    elig_hist_rate = float(cohort[config.TARGET].mean())
    design = experiment_design.freeze(
        eligible_default_rate=elig_hist_rate,
        n_eligible=len(cohort),
        linkage_manifest_sha=manifest.file_sha256(manifest.MANIFEST_PATH),
    )
    p = design["dgp_params"]
    print(f"design frozen: planning control default rate = {elig_hist_rate:.4%} "
          "(historical planning proxy, NOT an observed control outcome)")

    # ---- 3: prospective power (pre-outcome, frozen inputs only)
    f_tr, f_va, _ = p["policy_split"]
    n_avail = int(len(cohort) * (f_tr + f_va) / 2)
    sigma = planning_sigma_spend(cohort, p)
    # The frozen DGP itself anticipates a positive incremental default; power
    # must be computed against (δ - anticipated), never against δ alone.
    anticipated_pp = float(experiment_design.tau_default(
        cohort["pd_cal"].to_numpy(), p).mean() * 100)
    ptab = power.power_table(
        p_base=design["planning_control_default_rate"],
        sigma_spend=sigma, mde=p["spend_mde_dollars"],
        deltas_pp=tuple(p["delta_sensitivity_pp"]),
        n_available_per_arm=n_avail,
        anticipated_diff_pp=anticipated_pp,
        alpha_one_sided=p["alpha_one_sided"],
        alpha_two_sided=p["alpha_superiority_two_sided"],
        power=p["power_target"])
    ptab.to_csv(_art("power_analysis.csv"), index=False)
    print(ptab.to_string(index=False))

    # ---- 4: outcomes (at most once; marker-guarded)
    simulate_experiment.main()
    outcomes = pd.read_parquet(OUTCOMES_PATH)

    # ---- 5: pre-registered confirmatory analysis on TRAIN ∪ VAL
    conf = outcomes[outcomes["policy_split"].isin(["policy_train", "policy_val"])]
    t = conf[conf["treated"] == 1]
    c = conf[conf["treated"] == 0]
    y_t, y_c = t["spend_post"].to_numpy(), c["spend_post"].to_numpy()

    plain = inference.mean_diff(y_t, y_c)
    plain["boot_ci"] = inference.bootstrap_diff_ci(y_t, y_c)
    wins = inference.winsorized_diff(y_t, y_c)
    hurdle = inference.hurdle_decomposition(y_t, y_c)

    y_all = conf["spend_post"].to_numpy()
    x_pre = conf["spend_pre"].to_numpy()
    tr = conf["treated"].to_numpy()
    y_adj = inference.cuped_adjust(y_all, x_pre)
    a_t, a_c = y_adj[tr == 1], y_adj[tr == 0]
    cuped = inference.mean_diff(a_t, a_c)
    # bootstrap re-estimates θ inside every replicate (estimated θ is not a
    # known constant); ANCOVA + HC3 is an independent cross-check
    cuped["boot_ci"] = inference.cuped_bootstrap_ci(y_all, x_pre, tr)
    cuped["var_reduction"] = 1 - (np.var(y_adj, ddof=1)
                                  / np.var(conf["spend_post"], ddof=1))
    ancova = inference.ancova_crosscheck(y_all, x_pre, tr)

    powered = {row["delta_pp"]: bool(row["power_adequate"])
               for _, row in ptab.iterrows()}
    guard = inference.noninferiority_verdict(
        k_t=int(t["default_post"].sum()), n_t=len(t),
        k_c=int(c["default_post"].sum()), n_c=len(c),
        deltas_pp=tuple(p["delta_sensitivity_pp"]), powered=powered,
        alpha_one_sided=p["alpha_one_sided"])
    guard.to_csv(_art("guardrail_noninferiority.csv"), index=False)

    frozen_delta = p["delta_noninf_pp"]
    guard_row = guard.loc[guard["delta_pp"] == frozen_delta].iloc[0]
    guard_ok = guard_row["noninferiority"] == "DEMONSTRATED"
    spend_ok = (cuped["diff"] >= p["spend_mde_dollars"]) and (cuped["boot_ci"][0] > 0)
    ship = spend_ok and guard_ok

    results = {
        "population": "policy_train ∪ policy_val (POLICY-TEST untouched)",
        "spend_plain": plain, "spend_winsorized_p99": wins,
        "spend_hurdle": hurdle, "spend_cuped": cuped,
        "spend_ancova_hc3_crosscheck": ancova,
        "guardrail": guard.to_dict(orient="records"),
        "decision_rule": p["decision_rule"],
        "spend_condition_met": bool(spend_ok),
        "guardrail_condition_met": bool(guard_ok),
        "ATE_decision": "SHIP" if ship else "DO NOT SHIP (at ATE level)",
    }
    _art("layer2_confirmatory_results.json").write_text(
        json.dumps(results, indent=2, default=float))

    print(f"\nspend  plain: {plain['diff']:+.2f} "
          f"[boot {plain['boot_ci'][0]:+.2f}, {plain['boot_ci'][1]:+.2f}]")
    print(f"spend  CUPED: {cuped['diff']:+.2f} "
          f"[full-pipeline boot {cuped['boot_ci'][0]:+.2f}, {cuped['boot_ci'][1]:+.2f}] "
          f"(variance -{cuped['var_reduction']:.1%})")
    print(f"spend ANCOVA+HC3 crosscheck: {ancova['diff']:+.2f} "
          f"[{ancova['ci_lo']:+.2f}, {ancova['ci_hi']:+.2f}]")
    print(guard.to_string(index=False))
    print(f"ATE decision: {results['ATE_decision']}")

    # ---- 6: exploratory segments (BH; hypothesis-generating only)
    conf_cov = load_frozen_cohort().loc[conf.index]
    seg = pd.qcut(conf_cov["pd_cal"], q=5, labels=[f"riskQ{i}" for i in range(1, 6)])
    rows = []
    from scipy import stats as st
    for name, idx in conf.groupby(seg, observed=True).groups.items():
        s = conf.loc[idx]
        st_t, st_c = s[s["treated"] == 1], s[s["treated"] == 0]
        tt = st.ttest_ind(st_t["spend_post"], st_c["spend_post"], equal_var=False)
        rows.append({"segment": str(name), "n": len(s),
                     "spend_diff": st_t["spend_post"].mean() - st_c["spend_post"].mean(),
                     "default_diff_pp": (st_t["default_post"].mean()
                                         - st_c["default_post"].mean()) * 100,
                     "p_raw": float(tt.pvalue)})
    expl = pd.DataFrame(rows)
    expl["p_bh"] = inference.bh_adjust(expl["p_raw"])
    expl["label"] = "EXPLORATORY (BH-adjusted; hypothesis-generating only)"
    expl.to_csv(_art("layer2_exploratory_segments.csv"), index=False)

    # ---- 7: validation against DGP ground truth (labeled simulation privilege)
    truth = pd.read_parquet(TRUTH_PATH).loc[conf.index]
    cate = compute_cate_truth().loc[conf.index]
    # Two truth benchmarks, both reported:
    # - expected ATE = mean conditional expected effect E[Y1-Y0|X] over the
    #   confirmatory population (the estimand-level truth);
    # - realized SATE = mean realized potential-outcome contrast (finite-sample
    #   truth for THIS draw of the noise).
    true_ate_expected = float(cate["cate_spend_true"].mean())
    true_sate_realized = float(truth["delta_spend_true"].mean())
    true_ate_default_pp = float(cate["cate_default_true"].mean() * 100)
    validation = {
        "note": "possible ONLY because this is a simulation with known DGP",
        "true_ATE_spend_expected": true_ate_expected,
        "true_SATE_spend_realized_contrast": true_sate_realized,
        "true_ATE_spend_tau_only": float(truth["tau_spend_true"].mean()),
        "est_ATE_spend_cuped": cuped["diff"],
        "est_ATE_spend_plain": plain["diff"],
        "cuped_ci_covers_expected_ate": bool(
            cuped["boot_ci"][0] <= true_ate_expected <= cuped["boot_ci"][1]),
        "cuped_ci_covers_realized_sate": bool(
            cuped["boot_ci"][0] <= true_sate_realized <= cuped["boot_ci"][1]),
        "plain_ci_covers_expected_ate": bool(
            plain["boot_ci"][0] <= true_ate_expected <= plain["boot_ci"][1]),
        "true_incremental_default_pp": true_ate_default_pp,
        "guardrail_upper_bound_pp": float(guard["one_sided_95_upper_pp"].iloc[0]),
        "guardrail_bound_above_truth": bool(
            guard["one_sided_95_upper_pp"].iloc[0] >= true_ate_default_pp),
        "true_effect_below_frozen_delta": bool(true_ate_default_pp < frozen_delta),
        "guardrail_decision_correct_vs_truth": bool(
            guard_ok == (true_ate_default_pp < frozen_delta)),
    }
    _art("layer2_truth_validation.json").write_text(
        json.dumps(validation, indent=2))
    print(f"truth check: expected ATE {true_ate_expected:+.2f} / realized SATE "
          f"{true_sate_realized:+.2f} (est CUPED {cuped['diff']:+.2f}), "
          f"true Δdefault {true_ate_default_pp:.3f}pp vs UB "
          f"{guard['one_sided_95_upper_pp'].iloc[0]:.3f}pp")

    # ---- figures
    import matplotlib.pyplot as plt

    from . import plots as _plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    bins = np.linspace(0, np.quantile(conf["spend_post"], 0.99), 60)
    ax.hist(y_c, bins=bins, alpha=0.6, label="control", density=True)
    ax.hist(y_t, bins=bins, alpha=0.6, label="treated", density=True)
    ax.set_title("Post-period spend (clipped at p99)")
    ax.legend()

    ax = axes[1]
    for i, (label, est) in enumerate((("plain", plain), ("CUPED", cuped))):
        lo, hi = est["boot_ci"]
        ax.errorbar([i], [est["diff"]], yerr=[[est["diff"] - lo], [hi - est["diff"]]],
                    fmt="o", capsize=6, label=label)
    ax.axhline(p["spend_mde_dollars"], color="tab:red", ls="--", lw=1,
               label=f"MDE ${p['spend_mde_dollars']:.0f}")
    ax.axhline(true_ate_expected, color="k", ls=":", lw=1,
               label="true expected ATE (DGP)")
    ax.set_xticks([0, 1], ["plain", "CUPED"])
    ax.set_title("Spend lift: 95% bootstrap CIs")
    ax.legend(fontsize=8)

    ax = axes[2]
    x = np.arange(len(guard))
    ax.bar(x, guard["one_sided_95_upper_pp"], width=0.5,
           label="one-sided 95% upper bound")
    ax.scatter(x, [guard["observed_diff_pp"].iloc[0]] * len(x), color="k",
               zorder=3, label="observed diff")
    ax.scatter(x, [true_ate_default_pp] * len(x), color="tab:green", marker="^",
               zorder=3, label="true Δdefault (DGP)")
    for i, d in enumerate(guard["delta_pp"]):
        ax.hlines(d, i - 0.35, i + 0.35, color="tab:red", ls="--", lw=1.2)
    ax.set_xticks(x, [f"δ={d}pp" for d in guard["delta_pp"]])
    ax.set_ylabel("percentage points")
    ax.set_title("Default guardrail vs margins (red = δ)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _plots.save_fig(fig, "layer2_results.png")

    print(f"layer 2 complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
