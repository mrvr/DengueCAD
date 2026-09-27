"""Performance metrics from paper Section IV (AUC, SE, SP, accuracy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)


@dataclass
class PerformanceResult:
    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    threshold: float

    def as_row(self) -> dict[str, float]:
        return {
            "Accuracy (%)": round(100.0 * self.accuracy, 2),
            "SE": round(100.0 * self.sensitivity, 2),
            "SP": round(100.0 * self.specificity, 2),
            "AUC": round(self.auc, 2),
        }


def _positive_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim > 1:
            scores = scores[:, -1]
        # map to (0,1)-ish via logistic for threshold search stability
        return 1.0 / (1.0 + np.exp(-scores))
    pred = np.asarray(estimator.predict(X), dtype=float)
    return pred


def compute_performance(y_true, y_score) -> PerformanceResult:
    """
    Compute Accuracy / SE / SP / AUC.

    Operating point = ROC threshold closest to (0, 1) (equal-error style),
    matching the paper's Section IV description.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    # Closest point to (0, 1): minimise sqrt(FPR^2 + (1-TPR)^2)
    dist = np.sqrt(fpr**2 + (1.0 - tpr) ** 2)
    best = int(np.argmin(dist))
    thr = float(thresholds[best]) if best < len(thresholds) else 0.5

    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    return PerformanceResult(
        accuracy=acc,
        sensitivity=float(se),
        specificity=float(sp),
        auc=roc_auc,
        threshold=thr,
    )


def scores_from_estimator(estimator, X) -> np.ndarray:
    return _positive_scores(estimator, X)
