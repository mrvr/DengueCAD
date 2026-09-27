"""
Default classifiers used as the *inner* learner in the wrapper model.

Paper Section C / V: classification performance measures feature-subset
importance. Experiments used SVM-RBF (LibSVM); we also expose a fast
decision-tree option for demos and unit tests.
"""

from __future__ import annotations

from typing import Any, Callable

from denguecad.adt import AlternatingDecisionTree
from sklearn.base import BaseEstimator, clone
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ClassifierFactory = Callable[[], BaseEstimator]


def make_svm_rbf(random_state: int = 42) -> SVC:
    """SVM with RBF kernel (paper Section V / LibSVM)."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=random_state,
    )


def make_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """Fast CART-style tree for quick wrapper evaluations / tests."""
    return DecisionTreeClassifier(random_state=random_state)


def make_adt_classifier(random_state: int = 42) -> AlternatingDecisionTree:
    """ADT for Algorithm 1 wrapper evaluation / NMPrediction."""
    return AlternatingDecisionTree(n_estimators=30, random_state=random_state)


def get_classifier_factory(name: str = "svm", **kwargs: Any) -> ClassifierFactory:
    """
    Return a zero-arg factory that builds a fresh classifier clone.

    Parameters
    ----------
    name :
        ``\"svm\"``, ``\"tree\"``, or ``\"adt\"`` (Algorithm 1 wrapper model).
    """
    key = name.lower().strip()
    if key in {"svm", "svc", "rbf"}:
        def factory() -> BaseEstimator:
            return make_svm_rbf(**kwargs)

        return factory
    if key in {"tree", "dt", "cart", "c4.5"}:
        def factory() -> BaseEstimator:
            return make_decision_tree(**kwargs)

        return factory
    if key in {"adt", "alternating"}:
        def factory() -> BaseEstimator:
            return make_adt_classifier(**kwargs)

        return factory
    raise ValueError(f"Unknown classifier name: {name!r} (use 'svm', 'tree', or 'adt')")


def clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    return clone(estimator)
