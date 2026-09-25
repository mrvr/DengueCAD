"""Unit tests: classifiers and NMI path bridge."""

from __future__ import annotations

import pytest

from denguecad import __version__
from denguecad.classifiers import get_classifier_factory, make_decision_tree, make_svm_rbf
from denguecad.nmi_support import ensure_nmi_on_path, resolve_nmi_root

pytestmark = pytest.mark.unit


def test_package_version_semver_shape():
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_classifier_factories():
    svm = make_svm_rbf()
    tree = make_decision_tree()
    assert svm.__class__.__name__ == "SVC"
    assert tree.__class__.__name__ == "DecisionTreeClassifier"
    assert get_classifier_factory("svm")().kernel == "rbf"
    assert get_classifier_factory("tree")() is not None


def test_unknown_classifier_raises():
    with pytest.raises(ValueError):
        get_classifier_factory("not-a-model")


def test_resolve_nmi_root_or_skip():
    try:
        root = resolve_nmi_root()
    except FileNotFoundError:
        pytest.skip("NMI not present locally")
    assert (root / "nmilib.py").is_file()
    ensure_nmi_on_path(root)
