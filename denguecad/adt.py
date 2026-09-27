"""
Alternating Decision Tree (ADT) classifier.

Freund & Mason style ADT approximated via AdaBoost of decision stumps
(decision nodes) with additive prediction scores (prediction nodes), as used
in Rao & Kumar IEEE TITB 2012 Section D / Algorithm 1.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class AlternatingDecisionTree(ClassifierMixin, BaseEstimator):
    """
    Alternating Decision Tree via boosted decision stumps.

    An instance is scored by summing prediction-node contributions along
    active decision paths; the sign of the sum indicates class membership
    (paper Section D). ``predict_proba`` maps the decision score to [0, 1]
    via a logistic transform of the AdaBoost decision function.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
        self.model_ = AdaBoostClassifier(
            estimator=stump,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        # Prefer native AdaBoost probabilities when available
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        import numpy as np

        scores = np.asarray(self.decision_function(X), dtype=float)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - p1, p1])
        # multiclass fallback
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
