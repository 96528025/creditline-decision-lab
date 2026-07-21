# Getting the data

This project uses the Kaggle **"Give Me Some Credit"** dataset (2011 competition).
Raw data is **not** committed to this repository.

## Provenance note

The dataset is an anonymized consumer-credit dataset from a 2011 Kaggle competition.
Its originating institution, geography, and population representativeness are **not
established** in the public documentation. Results built on it are a methodological
prototype only — see the Limitations section of the README.

## Option A — kagglehub (used by this repo's scripts)

```bash
pip install kagglehub
python -c "
import kagglehub, shutil, pathlib
path = kagglehub.competition_download('GiveMeSomeCredit')
print('Downloaded to:', path)
"
```

Requires a Kaggle account with the competition rules accepted, and either
`~/.kaggle/kaggle.json` or `~/.kaggle/access_token`. If the competition download is
unavailable to your account, any faithful mirror of `cs-training.csv` works — verify
it has 150,000 rows and the 11 columns listed below.

## Option B — manual

1. Visit https://www.kaggle.com/c/GiveMeSomeCredit/data
2. Download and unzip `cs-training.csv`.

## Mirror actually used in this build

The original competition download requires per-account rules acceptance. This build
used the Kaggle mirror `brycecf/give-me-some-credit-dataset` (v1) via
`kagglehub.dataset_download`, then verified it against known ground-truth facts of the
original file: 150,000 rows, default rate 6.684%, 269 rows with co-occurring 96/98
sentinels in all three delinquency fields, 19.82% missing `MonthlyIncome`, one
`age == 0` row. `md5(cs-training.csv) = cce525a1d41f234f58d3bb52f3d2516b`.

## Where to put it

```
data/raw/cs-training.csv
```

## Expected schema (11 columns + index)

| Column | Type |
|---|---|
| SeriousDlqin2yrs (target) | 0/1 |
| RevolvingUtilizationOfUnsecuredLines | float |
| age | int |
| NumberOfTime30-59DaysPastDueNotWorse | int (96/98 sentinels) |
| DebtRatio | float |
| MonthlyIncome | float (~20% missing) |
| NumberOfOpenCreditLinesAndLoans | int |
| NumberOfTimes90DaysLate | int (96/98 sentinels) |
| NumberRealEstateLoansOrLines | int |
| NumberOfTime60-89DaysPastDueNotWorse | int (96/98 sentinels) |
| NumberOfDependents | float (~2.6% missing) |

Sanity check: 150,000 rows; overall default rate ≈ 6.7%.
