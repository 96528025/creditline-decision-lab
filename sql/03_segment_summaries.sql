-- 03_segment_summaries.sql
-- Segment-level default rates and distribution summaries consumed by the EDA
-- notebook. ALL of these are target-aware, so they read v_clean_dev
-- (RISK-DEV only) — holdout labels stay out of exploratory judgment.
-- Window functions are used where the task genuinely needs within-population
-- ranking (income quintiles); plain GROUP BY elsewhere.

-- (a) default rate by age band. The data has no documented lower age limit,
--     so the youngest band is labeled "<30", not "21-29".
DROP VIEW IF EXISTS v_default_by_age;
CREATE VIEW v_default_by_age AS
SELECT
    CASE
        WHEN age < 30 THEN '<30'
        WHEN age < 40 THEN '30-39'
        WHEN age < 50 THEN '40-49'
        WHEN age < 60 THEN '50-59'
        WHEN age < 70 THEN '60-69'
        ELSE '70+'
    END                                   AS age_band,
    COUNT(*)                              AS n,
    AVG(serious_dlqin2yrs)                AS default_rate
FROM v_clean_dev
WHERE age IS NOT NULL
GROUP BY age_band
ORDER BY MIN(age);

-- (b) default rate by utilization band (post-cap, so bands are interpretable)
DROP VIEW IF EXISTS v_default_by_util;
CREATE VIEW v_default_by_util AS
SELECT
    CASE
        WHEN revolving_utilization < 0.10 THEN 'a: <0.10'
        WHEN revolving_utilization < 0.30 THEN 'b: 0.10-0.30'
        WHEN revolving_utilization < 0.60 THEN 'c: 0.30-0.60'
        WHEN revolving_utilization < 1.00 THEN 'd: 0.60-1.00'
        WHEN revolving_utilization < 2.00 THEN 'e: 1.00-2.00 (over-limit)'
        ELSE 'f: capped at 2.00'
    END                                   AS util_band,
    COUNT(*)                              AS n,
    AVG(serious_dlqin2yrs)                AS default_rate
FROM v_clean_dev
GROUP BY util_band
ORDER BY util_band;

-- (c) default rate by income quintile (NTILE over non-missing incomes) with
--     the missing-income group shown as its own segment — hiding it would
--     misrepresent ~20% of the population.
DROP VIEW IF EXISTS v_default_by_income;
CREATE VIEW v_default_by_income AS
WITH ranked AS (
    SELECT serious_dlqin2yrs,
           monthly_income,
           NTILE(5) OVER (ORDER BY monthly_income) AS income_q
    FROM v_clean_dev
    WHERE monthly_income IS NOT NULL
)
SELECT 'Q' || income_q                    AS income_segment,
       COUNT(*)                           AS n,
       MIN(monthly_income)                AS seg_min_income,
       MAX(monthly_income)                AS seg_max_income,
       AVG(serious_dlqin2yrs)             AS default_rate
FROM ranked
GROUP BY income_q
UNION ALL
SELECT 'missing', COUNT(*), NULL, NULL, AVG(serious_dlqin2yrs)
FROM v_clean_dev WHERE monthly_income IS NULL
ORDER BY income_segment;

-- (d) delinquency profile: default rate by (capped display of) 30-59 DPD count
DROP VIEW IF EXISTS v_default_by_dpd3059;
CREATE VIEW v_default_by_dpd3059 AS
SELECT
    CASE WHEN dpd_30_59 IS NULL THEN 'sentinel/NULL'
         WHEN dpd_30_59 >= 4 THEN '4+'
         ELSE CAST(dpd_30_59 AS TEXT) END AS dpd_30_59_band,
    COUNT(*)                              AS n,
    AVG(serious_dlqin2yrs)                AS default_rate
FROM v_clean_dev
GROUP BY dpd_30_59_band
ORDER BY dpd_30_59_band;
