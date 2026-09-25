"""Unit tests: GA operators and wrapper selection on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from denguecad.feature_selection import (
    GAWrapperConfig,
    _crossover,
    _mutate,
    select_influential_features,
)

pytestmark = pytest.mark.unit


def test_paper_ga_default_probabilities():
    cfg = GAWrapperConfig()
    assert cfg.crossover_prob == 1.0
    assert cfg.mutation_prob == 0.001
    assert cfg.n_folds == 10


def test_crossover_and_mutate_shapes():
    rng = np.random.default_rng(1)
    a = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    b = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    c1, c2 = _crossover(a, b, rng, pc=1.0)
    assert c1.shape == a.shape and c2.shape == b.shape
    m = _mutate(a, rng, pm=1.0)
    assert m.shape == a.shape
    assert set(np.unique(m)).issubset({0, 1})


def test_select_influential_features_recovers_signal(synthetic_signal_df):
    cfg = GAWrapperConfig(
        population_size=16,
        n_generations=12,
        crossover_prob=1.0,
        mutation_prob=0.001,
        n_folds=4,
        classifier="tree",
        scoring="accuracy",
        impute_missing=False,
        subset_size_penalty=0.05,
        random_state=7,
        verbose=False,
    )
    result = select_influential_features(
        synthetic_signal_df, decision_col="dec", config=cfg
    )
    selected = set(result.selected_features)
    assert "f1" in selected
    assert "f2" in selected
    assert result.subset_size <= 3
    assert result.cv_score >= 0.75
    assert len(result.history) == cfg.n_generations + 1


def test_empty_features_raises():
    df = pd.DataFrame({"dec": [0, 1, 0, 1]})
    with pytest.raises(ValueError):
        select_influential_features(
            df,
            decision_col="dec",
            feature_cols=[],
            config=GAWrapperConfig(
                impute_missing=False, n_generations=1, population_size=4
            ),
        )


def test_result_as_dict_keys(synthetic_signal_df):
    cfg = GAWrapperConfig(
        population_size=8,
        n_generations=3,
        n_folds=3,
        classifier="tree",
        impute_missing=False,
        random_state=1,
    )
    result = select_influential_features(
        synthetic_signal_df, decision_col="dec", config=cfg
    )
    payload = result.as_dict()
    assert "selected_features" in payload
    assert "cv_score" in payload
    assert payload["subset_size"] == len(payload["selected_features"])
