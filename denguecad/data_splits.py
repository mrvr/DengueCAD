"""Train / validation / test splits for the DengueCAD NM experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURE_COLS = ["Fever", "Headache", "JointPain", "Bleeding"]
DECISION_COL = "Dengue"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "dengue.csv"
TRAIN_DATASET_NAME = "dengue"


@dataclass
class ExperimentSplits:
    """
    70k rows → one training dataset for NMI / NMPrediction;
    20k complete-case rows → test (performance);
    10k complete-case rows → validation.
    """

    datasets: dict[str, pd.DataFrame]
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    decision_col: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in self.datasets.items():
            rows.append(
                {
                    "split": name,
                    "n_rows": len(df),
                    "n_features": len(self.feature_cols),
                    "n_positive": int((df[self.decision_col] == 1).sum()),
                    "n_missing_cells": int(
                        df[self.feature_cols + [self.decision_col]].isna().sum().sum()
                    ),
                }
            )
        for label, df in (("validation", self.validation), ("test", self.test)):
            rows.append(
                {
                    "split": label,
                    "n_rows": len(df),
                    "n_features": len(self.feature_cols),
                    "n_positive": int((df[self.decision_col] == 1).sum()),
                    "n_missing_cells": int(
                        df[self.feature_cols + [self.decision_col]].isna().sum().sum()
                    ),
                }
            )
        return pd.DataFrame(rows)


def _drop_incomplete(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove records with any missing value in the given columns."""
    return df.dropna(subset=cols).reset_index(drop=True)


def prepare_experiment_splits(
    data_path: Optional[str | Path] = None,
    *,
    train_rows: int = 70_000,
    test_rows: int = 20_000,
    validation_rows: int = 10_000,
    feature_cols: Optional[list[str]] = None,
    decision_col: str = DECISION_COL,
    train_name: str = TRAIN_DATASET_NAME,
) -> ExperimentSplits:
    """
    Load ``dengue.csv`` and build experimental partitions.

    - Rows ``[0, train_rows)`` → one dataset for NMI / NMPrediction.
    - From the remaining rows, take complete-case records (no MVs in modelling
      columns): first ``validation_rows`` → validation, next ``test_rows`` → test.
    """
    path = Path(data_path) if data_path else DEFAULT_DATA
    if not path.is_file():
        raise FileNotFoundError(path)

    feature_cols = list(feature_cols or FEATURE_COLS)
    use_cols = ["Name", *feature_cols, decision_col]
    df = pd.read_csv(path, na_values=["", "?"], usecols=lambda c: c in use_cols)
    keep = [c for c in use_cols if c in df.columns]
    df = df[keep]

    need = train_rows + test_rows + validation_rows
    if len(df) < need:
        raise ValueError(f"Need at least {need} rows; found {len(df)}")

    train_pool = df.iloc[:train_rows].copy().reset_index(drop=True)
    remainder = df.iloc[train_rows:].copy()

    model_cols = feature_cols + [decision_col]
    complete = _drop_incomplete(remainder, model_cols)
    need_holdout = validation_rows + test_rows
    if len(complete) < need_holdout:
        raise ValueError(
            f"Need {need_holdout} complete-case rows after the training block; "
            f"found {len(complete)}."
        )

    validation = complete.iloc[:validation_rows].reset_index(drop=True)
    test = complete.iloc[
        validation_rows : validation_rows + test_rows
    ].reset_index(drop=True)

    assert validation[model_cols].isna().sum().sum() == 0
    assert test[model_cols].isna().sum().sum() == 0

    return ExperimentSplits(
        datasets={train_name: train_pool},
        validation=validation,
        test=test,
        feature_cols=feature_cols,
        decision_col=decision_col,
    )
