-- 01_load_and_clean.sql
-- Cleaning VIEW over the raw table, mirroring the PRIMARY rules frozen in
-- DESIGN_FREEZE.md §4. The authoritative implementation for modeling is
-- src/data_prep.py (a view cannot hold fitted constants for the sensitivity
-- variants); this view exists so EDA and cohort queries read cleaned,
-- flag-annotated rows straight from SQL. tests/ verifies the two agree.
--
-- Raw table `raw_credit` is loaded by src/load_db.py (columns renamed to
-- snake_case; SQLite dislikes '-' in identifiers).

DROP VIEW IF EXISTS v_clean;

CREATE VIEW v_clean AS
SELECT
    customer_id,
    serious_dlqin2yrs,

    -- utilization: cap at 2.0, flag beyond (primary rule; (1,2] retained)
    CASE WHEN revolving_utilization > 2.0 THEN 2.0
         ELSE revolving_utilization END                       AS revolving_utilization,
    CASE WHEN revolving_utilization > 2.0 THEN 1 ELSE 0 END   AS util_extreme_flag,

    -- age: 0 is invalid -> NULL
    CASE WHEN age = 0 THEN NULL ELSE age END                  AS age,

    -- delinquency counts: 96/98 are special codes, not counts -> NULL + flag
    CASE WHEN dpd_30_59 IN (96, 98) OR dpd_60_89 IN (96, 98)
              OR dpd_90 IN (96, 98)
         THEN 1 ELSE 0 END                                    AS dpd_sentinel_flag,
    CASE WHEN dpd_30_59 IN (96, 98) OR dpd_60_89 IN (96, 98) OR dpd_90 IN (96, 98)
         THEN NULL ELSE dpd_30_59 END                         AS dpd_30_59,
    CASE WHEN dpd_30_59 IN (96, 98) OR dpd_60_89 IN (96, 98) OR dpd_90 IN (96, 98)
         THEN NULL ELSE dpd_60_89 END                         AS dpd_60_89,
    CASE WHEN dpd_30_59 IN (96, 98) OR dpd_60_89 IN (96, 98) OR dpd_90 IN (96, 98)
         THEN NULL ELSE dpd_90 END                            AS dpd_90,

    debt_ratio,
    monthly_income,
    CASE WHEN monthly_income IS NULL THEN 1 ELSE 0 END        AS income_missing_flag,
    open_credit_lines,
    real_estate_loans,
    dependents,
    CASE WHEN dependents IS NULL THEN 1 ELSE 0 END            AS dependents_missing_flag
FROM raw_credit;

-- RISK-DEV-only counterpart. ALL target-aware EDA must read from this view:
-- the RISK-TEST holdout's labels may not inform exploratory judgment.
-- risk_dev_ids is materialized by src/load_db.py from the frozen S0 split.
DROP VIEW IF EXISTS v_clean_dev;

CREATE VIEW v_clean_dev AS
SELECT c.* FROM v_clean c
JOIN risk_dev_ids d USING (customer_id);
