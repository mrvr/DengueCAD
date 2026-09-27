"""System tests for experiment splits on vendored dengue.csv."""

from __future__ import annotations

import pytest

from denguecad.data_splits import prepare_experiment_splits
from denguecad.feature_selection import GAWrapperConfig
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig

pytestmark = pytest.mark.system


def test_prepare_splits_shapes(data_dir):
    splits = prepare_experiment_splits(data_dir / "dengue.csv")
    assert list(splits.datasets) == ["dengue"]
    assert len(splits.datasets["dengue"]) == 70_000
    assert len(splits.validation) == 10_000
    assert len(splits.test) == 20_000
    # holdouts must be complete-case
    cols = splits.feature_cols + [splits.decision_col]
    assert splits.validation[cols].isna().sum().sum() == 0
    assert splits.test[cols].isna().sum().sum() == 0


def test_nmprediction_smoke_on_ds_sample(data_dir):
    splits = prepare_experiment_splits(data_dir / "dengue.csv")
    sample = splits.datasets["dengue"].head(800)
    cfg = NMPredictionConfig(
        n_folds=4,
        ga=GAWrapperConfig(
            population_size=6,
            n_generations=3,
            n_folds=3,
            classifier="adt",
            impute_missing=False,
            random_state=0,
        ),
        adt_estimators=15,
        random_state=0,
        verbose=False,
    )
    result = NMPrediction(cfg).fit_predict_cv(
        sample,
        decision_col=splits.decision_col,
        feature_cols=splits.feature_cols,
    )
    assert 1 <= len(result.selected_features) <= 4
    assert 0.0 <= result.performance.auc <= 1.0
