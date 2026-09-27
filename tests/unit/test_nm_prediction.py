"""Unit / system tests for NMPrediction and comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from denguecad.baselines import make_c45, make_lor, make_svm
from denguecad.comparison import wilcoxon_nm_vs
from denguecad.data_splits import prepare_experiment_splits
from denguecad.feature_selection import GAWrapperConfig
from denguecad.metrics import compute_performance
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig

pytestmark = pytest.mark.unit


def test_baselines_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(80, 4))
    y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(int)
    for factory in (make_c45, make_lor, make_svm):
        clf = factory(0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (80,)


def test_compute_performance_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    perf = compute_performance(y, s)
    assert perf.auc == pytest.approx(1.0)
    assert perf.accuracy == pytest.approx(1.0)


def test_wilcoxon_nm_better():
    nm = [0.9] * 10
    other = [0.7] * 10
    w = wilcoxon_nm_vs(nm, other)
    assert w["rank_sum_pos"] > w["rank_sum_neg"]
    assert w["p_value"] < 0.05


def test_nmprediction_on_synthetic():
    rng = np.random.default_rng(1)
    n = 200
    f1 = rng.integers(0, 2, size=n)
    f2 = rng.integers(0, 2, size=n)
    f3 = rng.integers(0, 2, size=n)
    y = ((f1 == 1) & (f2 == 1)).astype(int)
    df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "dec": y})
    cfg = NMPredictionConfig(
        n_folds=4,
        ga=GAWrapperConfig(
            population_size=8,
            n_generations=4,
            n_folds=3,
            classifier="adt",
            impute_missing=False,
            subset_size_penalty=0.05,
            random_state=1,
        ),
        adt_estimators=20,
        random_state=1,
        verbose=False,
    )
    result = NMPrediction(cfg).fit_predict_cv(df, decision_col="dec")
    assert set(result.selected_features).issubset({"f1", "f2", "f3"})
    assert result.performance.accuracy >= 0.7
    assert len(result.fold_accuracies) == 4
