"""The frozen S0 split, factored out so BOTH the pipeline and the SQL loader
consume the same persisted indices (the EDA warehouse needs to know which rows
are RISK-DEV so that target-aware queries never touch holdout labels)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def make_or_load_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stratified RISK-TEST holdout. Indices are persisted the first time and
    loaded afterwards, so the split can never silently drift."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / "risk_split_indices.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return np.array(saved["dev"]), np.array(saved["test"])
    dev_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=config.RISK_TEST_FRACTION,
        stratify=df[config.TARGET],
        random_state=config.SEED_SPLIT,
    )
    path.write_text(json.dumps({
        "seed": config.SEED_SPLIT,
        "test_fraction": config.RISK_TEST_FRACTION,
        "dev": sorted(int(i) for i in dev_idx),
        "test": sorted(int(i) for i in test_idx),
    }))
    return np.sort(dev_idx), np.sort(test_idx)
