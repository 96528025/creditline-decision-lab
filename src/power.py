"""Prospective power analysis — DESIGN_FREEZE.md §7.

TWO analyses, larger n wins (both computed from FROZEN planning inputs only):

(a) spend superiority at the minimum economically meaningful lift (MDE).
    Planning σ comes from a planning-seed simulation of the FROZEN baseline
    spend model — legitimate pre-outcome knowledge, since the DGP spec is
    frozen before any outcome exists (and uses a seed disjoint from the
    outcome seed).
(b) default-rate non-inferiority at margin δ, one-sided α, assuming equal true
    rates at the planning baseline. Reported at δ ∈ {0.30, 0.50, 0.75} pp; if
    the eligible population cannot support a margin, the design is reported
    UNDERPOWERED at that margin — assumptions are never adjusted afterwards.

Default is low-frequency, so the guardrail is expected to drive sample size at
tight margins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def n_per_arm_superiority(sigma: float, mde: float, alpha_two_sided: float = 0.05,
                          power: float = 0.80) -> int:
    """Two-sample difference in means, equal allocation."""
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * (sigma ** 2) * (z_a + z_b) ** 2 / mde ** 2))


def n_per_arm_noninferiority(p_base: float, delta: float,
                             anticipated_diff: float = 0.0,
                             alpha_one_sided: float = 0.05,
                             power: float = 0.80) -> int:
    """Difference in proportions, H0: p_t - p_c >= delta.

    Power is computed at the ANTICIPATED true difference: the distance to the
    null boundary is (delta - anticipated_diff), not delta. Setting
    anticipated_diff = 0 while the frozen DGP specifies a positive expected
    incremental default would overstate power — the design must be powered
    against what it itself anticipates. anticipated_diff = 0 remains available
    as an explicitly-labeled planning sensitivity scenario.
    """
    distance = delta - anticipated_diff
    if distance <= 0:
        raise ValueError(
            f"margin {delta} does not exceed anticipated difference "
            f"{anticipated_diff}; non-inferiority is undemonstrable by design")
    z_a = stats.norm.ppf(1 - alpha_one_sided)
    z_b = stats.norm.ppf(power)
    var = 2 * p_base * (1 - p_base)
    return int(np.ceil(var * (z_a + z_b) ** 2 / distance ** 2))


def power_table(p_base: float, sigma_spend: float, mde: float,
                deltas_pp: tuple, n_available_per_arm: int,
                anticipated_diff_pp: float,
                alpha_one_sided: float = 0.05,
                alpha_two_sided: float = 0.05,
                power: float = 0.80) -> pd.DataFrame:
    """One row per δ margin. Primary guardrail n uses the frozen DGP's
    anticipated incremental default; the zero-difference n is reported as a
    labeled sensitivity scenario only."""
    n_spend = n_per_arm_superiority(sigma_spend, mde, alpha_two_sided, power)
    rows = []
    for d_pp in deltas_pp:
        n_guard = n_per_arm_noninferiority(
            p_base, d_pp / 100, anticipated_diff_pp / 100, alpha_one_sided, power)
        n_guard_zero = n_per_arm_noninferiority(
            p_base, d_pp / 100, 0.0, alpha_one_sided, power)
        n_req = max(n_spend, n_guard)
        rows.append({
            "delta_pp": d_pp,
            "anticipated_true_diff_pp": anticipated_diff_pp,
            "n_per_arm_spend": n_spend,
            "n_per_arm_guardrail": n_guard,
            "n_per_arm_guardrail_zero_diff_scenario": n_guard_zero,
            "n_per_arm_required": n_req,
            "binding_constraint": "guardrail" if n_guard >= n_spend else "spend",
            "n_per_arm_available": n_available_per_arm,
            "power_adequate": bool(n_available_per_arm >= n_req),
        })
    return pd.DataFrame(rows)
