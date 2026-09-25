"""Shared pytest fixtures for DengueCAD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from denguecad.nmi_support import resolve_nmi_root

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@pytest.fixture
def synthetic_signal_df() -> pd.DataFrame:
    """y depends on f1∧f2; f3/f4 are noise."""
    rng = np.random.default_rng(0)
    n = 120
    f1 = rng.integers(0, 2, size=n)
    f2 = rng.integers(0, 2, size=n)
    f3 = rng.integers(0, 2, size=n)
    f4 = rng.integers(0, 2, size=n)
    y = ((f1 == 1) & (f2 == 1)).astype(int)
    flip = rng.random(n) < 0.05
    y[flip] = 1 - y[flip]
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "dec": y})


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not DATA_DIR.is_dir():
        pytest.skip(f"Local data directory missing: {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session")
def nmi_root() -> Path:
    try:
        return resolve_nmi_root()
    except FileNotFoundError:
        pytest.skip("NMI library not available (set NMI_ROOT or clone sibling NMI)")


@pytest.fixture(scope="session")
def dengue_sample(data_dir: Path) -> pd.DataFrame:
    """Prefer vendored ``data/dengue.csv`` (copied from NMI for local/CI testing)."""
    path = data_dir / "dengue.csv"
    if not path.is_file():
        pytest.skip(f"dengue.csv missing under {data_dir}")
    df = pd.read_csv(path, na_values=["", "?"], nrows=400)
    if df.empty:
        pytest.skip("dengue sample is empty")
    return df


@pytest.fixture(scope="session")
def paper_example_df(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "data.txt"
    if not path.is_file():
        pytest.skip("data/data.txt missing")
    return pd.read_csv(path, na_values=["", "?"])
