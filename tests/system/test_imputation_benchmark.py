"""Tests for MV imputation benchmark on dataset.csv."""

from __future__ import annotations

from pathlib import Path

import pytest

from denguecad.imputation_benchmark import (
    impute_natural_missing,
    load_and_encode,
    run_benchmark,
)

pytestmark = pytest.mark.system

DATA = Path(__file__).resolve().parents[2] / "data" / "dataset.csv"


@pytest.mark.skipif(not DATA.is_file(), reason="data/dataset.csv missing")
def test_load_and_encode_dataset_csv():
    enc = load_and_encode(DATA)
    assert enc.decision_col == "Outcome"
    assert "Joint_Pain" in enc.feature_cols
    assert enc.frame["Outcome"].notna().all()


@pytest.mark.skipif(not DATA.is_file(), reason="data/dataset.csv missing")
def test_benchmark_includes_nmi_and_baselines():
    table = run_benchmark(DATA, mv_frac=0.1, seed=0, max_complete_rows=80)
    methods = set(table["Method"])
    assert "NM (NMI)" in methods
    assert "kNN" in methods
    assert "MICE" in methods
    assert (table["n_compared"] > 0).all()
    assert (table["Accuracy (%)"] >= 0).all()


@pytest.mark.skipif(not DATA.is_file(), reason="data/dataset.csv missing")
def test_nmi_fills_natural_joint_pain(tmp_path):
    out = tmp_path / "imputed.csv"
    filled = impute_natural_missing(DATA, method="nmi", out_path=out)
    assert out.is_file()
    assert filled["Joint_Pain"].isna().sum() == 0
