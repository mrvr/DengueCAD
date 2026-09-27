"""
MissForest-style imputation (Stekhoven & Bühlmann) via iterative random forests.

Used when comparing against NMI on the canonical Iris demo dataset from the
missForest R package documentation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def missforest_impute(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    categorical_cols: Optional[Sequence[str]] = None,
    max_iter: int = 6,
    n_estimators: int = 40,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Iteratively impute ``feature_cols`` with random forests (MissForest).

    Starts with mean/mode fill, then repeatedly predicts each column's missing
    entries from the others until ``max_iter`` or convergence.
    """
    cat = set(categorical_cols or [])
    out = data.copy()
    X = out[list(feature_cols)].astype(float)
    # Initial fill
    for col in feature_cols:
        if X[col].isna().all():
            X[col] = 0.0
            continue
        if col in cat:
            mode = X[col].mode(dropna=True)
            fill = float(mode.iloc[0]) if len(mode) else 0.0
        else:
            fill = float(X[col].mean())
        X[col] = X[col].fillna(fill)

    mask = data[list(feature_cols)].isna()
    if not mask.any().any():
        out[list(feature_cols)] = X
        return out

    rng = random_state
    prev = None
    for it in range(max_iter):
        # Impute columns with fewest missing first
        order = mask.sum().sort_values().index.tolist()
        for col in order:
            miss_idx = mask.index[mask[col]].tolist()
            if not miss_idx:
                continue
            obs_idx = mask.index[~mask[col]].tolist()
            if len(obs_idx) < 5:
                continue
            others = [c for c in feature_cols if c != col]
            X_obs = X.loc[obs_idx, others]
            y_obs = X.loc[obs_idx, col]
            X_miss = X.loc[miss_idx, others]
            if col in cat:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=rng + it,
                    n_jobs=-1,
                )
                # integer class codes
                y_fit = y_obs.round().astype(int)
                clf.fit(X_obs, y_fit)
                pred = clf.predict(X_miss).astype(float)
            else:
                rgr = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=rng + it,
                    n_jobs=-1,
                )
                rgr.fit(X_obs, y_obs)
                pred = rgr.predict(X_miss)
            X.loc[miss_idx, col] = pred

        if prev is not None:
            delta = float(np.nanmean((X.to_numpy() - prev) ** 2))
            if delta < 1e-6:
                break
        prev = X.to_numpy().copy()

    out[list(feature_cols)] = X
    return out
