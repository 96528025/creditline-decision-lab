"""Load cs-training.csv into SQLite and build the SQL views.

Run: python -m src.load_db
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from . import config

# raw Kaggle column -> snake_case SQL column
RENAME = {
    "SeriousDlqin2yrs": "serious_dlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "dpd_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd_60_89",
    "NumberOfDependents": "dependents",
}


def main() -> None:
    df = pd.read_csv(config.DATA_RAW)
    # Materialize the frozen S0 split BEFORE renaming (splits.py reads the
    # original target column). The warehouse gets a risk_dev_ids table so that
    # every target-aware EDA query can be restricted to RISK-DEV — holdout
    # labels must not inform exploratory judgment (DESIGN_FREEZE.md amendment log).
    from .splits import make_or_load_split
    dev_idx, _ = make_or_load_split(df)

    first = df.columns[0]
    if first.startswith("Unnamed"):
        df = df.rename(columns={first: "customer_id"})
    else:
        df.insert(0, "customer_id", range(1, len(df) + 1))
    df = df.rename(columns=RENAME)
    dev_ids = df.loc[dev_idx, "customer_id"].to_frame()

    con = sqlite3.connect(config.DB_PATH)
    try:
        df.to_sql("raw_credit", con, if_exists="replace", index=False)
        dev_ids.to_sql("risk_dev_ids", con, if_exists="replace", index=False)
        for script in sorted((config.ROOT / "sql").glob("*.sql")):
            con.executescript(script.read_text())
        n, rate = con.execute(
            "SELECT COUNT(*), AVG(serious_dlqin2yrs) FROM v_clean_dev"
        ).fetchone()
        print(f"loaded raw_credit + views: RISK-DEV {n} rows, dev default rate {rate:.4%}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
