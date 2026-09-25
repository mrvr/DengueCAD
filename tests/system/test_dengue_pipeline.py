"""
System tests: NMI + DengueCAD end-to-end on dengue surveillance-style data.

These exercise the real sibling library and a dengue sample. They are slower
than unit tests and require ``NMI_ROOT`` or a sibling ``../NMI`` checkout.
"""

from __future__ import annotations

import pandas as pd
import pytest

from denguecad.feature_selection import GAWrapperConfig, select_influential_features
from denguecad.nmi_support import import_nmilib

pytestmark = pytest.mark.system

FEATURES = ["Fever", "Headache", "JointPain", "Bleeding"]
DECISION = "Dengue"


def test_nmi_imputation_roundtrip(dengue_sample, nmi_root):
    nmilib = import_nmilib(nmi_root)
    df = dengue_sample.copy()
    # Inject a few missing attribute values with known decisions
    work = df.dropna(subset=[DECISION]).head(200).copy()
    assert not work.empty
    work.loc[work.index[0], "Fever"] = pd.NA
    work.loc[work.index[1], "Bleeding"] = pd.NA
    imputed = nmilib.non_parametric_imputation(
        work[[*FEATURES, DECISION]],
        decision_col=DECISION,
        feature_cols=FEATURES,
    )
    assert imputed["Fever"].isna().sum() == 0
    assert imputed["Bleeding"].isna().sum() == 0
    assert imputed[DECISION].isna().sum() == 0


def test_ga_wrapper_on_dengue_sample(dengue_sample, nmi_root):
    cfg = GAWrapperConfig(
        population_size=10,
        n_generations=6,
        crossover_prob=1.0,
        mutation_prob=0.001,
        n_folds=4,
        classifier="tree",
        scoring="accuracy",
        impute_missing=True,
        nmi_root=str(nmi_root),
        random_state=42,
        verbose=False,
    )
    result = select_influential_features(
        dengue_sample,
        decision_col=DECISION,
        feature_cols=FEATURES,
        config=cfg,
    )
    assert 1 <= result.subset_size <= len(FEATURES)
    assert set(result.selected_features).issubset(set(FEATURES))
    assert 0.0 <= result.cv_score <= 1.0
    assert result.fitness <= result.cv_score + 1e-9
    assert len(result.history) == cfg.n_generations + 1


def test_pipeline_impute_then_select_is_deterministic_enough(dengue_sample, nmi_root):
    """Same seed should yield the same selected subset."""
    cfg = GAWrapperConfig(
        population_size=8,
        n_generations=4,
        n_folds=3,
        classifier="tree",
        impute_missing=True,
        nmi_root=str(nmi_root),
        random_state=11,
        verbose=False,
    )
    a = select_influential_features(
        dengue_sample, decision_col=DECISION, feature_cols=FEATURES, config=cfg
    )
    b = select_influential_features(
        dengue_sample, decision_col=DECISION, feature_cols=FEATURES, config=cfg
    )
    assert a.selected_features == b.selected_features
    assert a.cv_score == pytest.approx(b.cv_score)
