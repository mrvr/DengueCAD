"""System tests against vendored NMI datasets under ``data/``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from denguecad.nmi_support import import_nmilib

pytestmark = pytest.mark.system

# Small example / synthetic files copied from NMI for offline testing
SMALL_DATASETS = (
    "data.txt",
    "data1.txt",
    "data2.txt",
    "data3.txt",
    "data4.txt",
    "data5.txt",
)


def test_vendored_data_files_present(data_dir: Path):
    missing = [name for name in SMALL_DATASETS if not (data_dir / name).is_file()]
    assert not missing, f"Missing vendored datasets: {missing}"
    assert (data_dir / "dengue.csv").is_file()


@pytest.mark.parametrize("filename", SMALL_DATASETS)
def test_nmi_impute_small_datasets(data_dir: Path, nmi_root: Path, filename: str):
    nmilib = import_nmilib(nmi_root)
    path = data_dir / filename
    df = pd.read_csv(path, na_values=["", "?"])
    assert "dec" in df.columns or df.columns[-1]
    decision_col = "dec" if "dec" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != decision_col]
    # Only rows with known decision are imputed by NMI
    work = df.dropna(subset=[decision_col])
    if work.empty:
        pytest.skip(f"{filename}: no rows with observed decision")
    # Need at least one complete donor
    if work.dropna().empty:
        pytest.skip(f"{filename}: no complete donor rows")
    imputed = nmilib.non_parametric_imputation(
        work,
        decision_col=decision_col,
        feature_cols=feature_cols,
    )
    # Attribute MVs on known-decision rows should be filled
    known = imputed.dropna(subset=[decision_col])
    assert known[feature_cols].isna().sum().sum() == 0


def test_paper_example_loads(paper_example_df: pd.DataFrame):
    assert len(paper_example_df) >= 1
    assert paper_example_df.shape[1] >= 2
