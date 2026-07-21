-- 02_cohort_eligibility.sql
-- Pre-score eligibility candidate cohort: rules E1–E4 of DESIGN_FREEZE.md §6.
-- E5 (calibrated OOF risk-score tier) cannot live in SQL until the Linkage
-- phase produces scores; src/eligibility.py applies the full E1–E5 rule and is
-- the authoritative implementation. This view documents the deterministic part
-- of the cohort in the warehouse layer, which is where a real eligibility
-- cohort would be built.

DROP VIEW IF EXISTS v_eligibility_candidates;

CREATE VIEW v_eligibility_candidates AS
WITH flags AS (
    SELECT
        *,
        CASE WHEN dpd_90 >= 1 THEN 1 ELSE 0 END        AS excl_e1_severe_90,
        CASE WHEN dpd_60_89 >= 1 THEN 1 ELSE 0 END     AS excl_e2_severe_6089,
        CASE WHEN dpd_30_59 >= 2 THEN 1 ELSE 0 END     AS excl_e3_repeat_3059,
        dpd_sentinel_flag                              AS excl_e4_sentinel
    FROM v_clean_dev  -- eligibility exists only within RISK-DEV (§2)
)
SELECT
    *,
    CASE WHEN COALESCE(excl_e1_severe_90, 0) = 0
          AND COALESCE(excl_e2_severe_6089, 0) = 0
          AND COALESCE(excl_e3_repeat_3059, 0) = 0
          AND excl_e4_sentinel = 0
         THEN 1 ELSE 0 END AS eligible_pre_score
FROM flags;
