"""Unit tests for the imputation + prediction hold-out experiment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from denguecad.feature_selection import GAWrapperConfig
from denguecad.holdout_experiment import encode_ordinal, imputation_accuracy, inject_record_mcar
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig

pytestmark = pytest.mark.unit


def test_encode_ordinal_keeps_numeric_order_and_missing():
    raw = pd.DataFrame({
        "age": np.arange(100, dtype=float),
        "sex": ["M", "F"] * 50,
        "label": ["pos", "neg"] * 50,
    })
    raw.loc[3, "age"] = np.nan
    ds = encode_ordinal(raw, name="t", decision_col="label", n_bins=10)
    assert ds.feature_cols == ["age", "sex"]
    age = ds.frame["age"]
    assert pd.isna(age[3])
    assert age.dropna().is_monotonic_increasing
    assert age.max() == 9
    assert set(ds.frame["label"].dropna()) == {0, 1}


def test_inject_record_mcar_masks_requested_rows_only():
    frame = pd.DataFrame(np.ones((50, 5), dtype=int), columns=list("abcde")).astype("Int64")
    frame["y"] = [0, 1] * 25
    masked, mask = inject_record_mcar(frame, list("abcde"), frame.index, row_frac=0.2, max_attrs=3, seed=1)
    rows_hit = mask.any(axis=1)
    assert rows_hit.sum() == 10
    assert mask.sum(axis=1)[rows_hit].between(1, 3).all()
    assert masked["y"].notna().all()
    assert int(masked[list("abcde")].isna().sum().sum()) == int(mask.to_numpy().sum())


def test_imputation_accuracy_counts_masked_cells():
    truth = pd.DataFrame({"a": [0, 1, 2], "b": [1, 1, 0]})
    mask = pd.DataFrame({"a": [True, False, True], "b": [False, True, False]})
    imputed = pd.DataFrame({"a": [0, 9, 1], "b": [9, 1, 9]})
    assert imputation_accuracy(truth, imputed, mask) == pytest.approx(2 / 3)


def test_nmprediction_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(120, 4))
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["y"] = df["f1"]
    cfg = NMPredictionConfig(
        ga=GAWrapperConfig(population_size=4, n_generations=2, n_folds=3, impute_missing=False, random_state=0),
        adt_estimators=10,
        random_state=0,
    )
    nm = NMPrediction(cfg).fit(df, decision_col="y", feature_cols=["f1", "f2", "f3", "f4"])
    pred = nm.predict(df)
    assert "f1" in nm.selected_features_
    assert pred.shape == (120,)
    assert np.mean(pred == df["y"].to_numpy()) > 0.95
