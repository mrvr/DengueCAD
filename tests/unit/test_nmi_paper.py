"""Checks for the 2012 and 2013 NMI implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from denguecad.nmi_paper import column_kind, impute_nmi

pytestmark = pytest.mark.unit


def test_column_kind_separates_whole_numbers_from_fractions():
    assert column_kind(pd.Series(["F", "M"])) == "categorical"
    assert column_kind(pd.Series([4, 5, 7])) == "integer"
    assert column_kind(pd.Series([2200.0, 3000.0, np.nan])) == "integer"
    assert column_kind(pd.Series([12.6, 13.2, 11.0])) == "real"


def test_2012_integer_column_uses_mode_of_neighbours():
    """Whole-number columns follow the categorical rule and are filled by the mode."""
    frame = pd.DataFrame(
        {
            "symptom": ["a", "a", "b", "b"],
            "lab": [1, 2, 3, 10],
            "decision": [0, 0, 0, 1],
        }
    )
    masked = frame.copy()
    masked.loc[0, "symptom"] = pd.NA
    filled = impute_nmi(masked, "decision", year=2012)
    # The target's lab value is absent from the complete donors, so every
    # index is 0, every donor is a neighbour, and the symptom mode is "b".
    assert filled.loc[0, "symptom"] == "b"
    assert filled.loc[1:, "symptom"].tolist() == ["a", "b", "b"]


def test_2012_real_fill_is_the_neighbour_mean():
    frame = pd.DataFrame(
        {
            "symptom": ["a", "a", "a", "b"],
            "lab": [1.5, 3.5, 5.5, 100.5],
            "decision": [0, 0, 0, 1],
        }
    )
    masked = frame.copy()
    masked.loc[0, "lab"] = np.nan
    filled = impute_nmi(masked, "decision", year=2012)
    # Same-class donors are rows 1 and 2. The other class is farther (z > 0).
    assert filled.loc[0, "lab"] == pytest.approx((3.5 + 5.5) / 2.0)


def test_2013_imputes_without_dropping_incomplete_donors():
    frame = pd.DataFrame(
        {
            "symptom": ["a", "a", None, "b"],
            "lab": [1.0, 2.0, 2.5, 9.0],
            "decision": [0, 0, 0, 1],
        }
    )
    masked = frame.copy()
    masked.loc[0, "symptom"] = np.nan
    filled = impute_nmi(masked, "decision", year=2013)
    assert pd.notna(filled.loc[0, "symptom"])
    assert filled.loc[0, "lab"] == 1.0
