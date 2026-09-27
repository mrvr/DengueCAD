"""Baseline classifiers used in the paper comparison (Table I)."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from denguecad.adt import AlternatingDecisionTree


def make_c45(random_state: int = 42) -> DecisionTreeClassifier:
    """
    C4.5-style decision tree (information-gain / entropy splits).

    scikit-learn implements CART; ``criterion='entropy'`` is the standard
    open-source stand-in for C4.5 used in comparative studies.
    """
    return DecisionTreeClassifier(
        criterion="entropy",
        random_state=random_state,
    )


def make_svm(random_state: int = 42) -> Pipeline:
    """SVM with RBF kernel (paper Section V / LibSVM)."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_lor(random_state: int = 42) -> Pipeline:
    """Logistic regression (LOR)."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "lor",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_adt(random_state: int = 42) -> AlternatingDecisionTree:
    """Alternating Decision Tree used by NMPrediction."""
    return AlternatingDecisionTree(n_estimators=40, random_state=random_state)


BASELINE_FACTORIES = {
    "C4.5": make_c45,
    "SVM": make_svm,
    "LOR": make_lor,
    "ADT": make_adt,
}
