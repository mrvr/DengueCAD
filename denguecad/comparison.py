"""
Compare NMPrediction with C4.5, SVM and LOR; build paper-style tables.

Table I  — performance comparison
Table II — Wilcoxon matched-pairs rank sum test (fold accuracies)
Table III — influential feature subsets identified by NM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold

from denguecad.baselines import BASELINE_FACTORIES
from denguecad.data_splits import ExperimentSplits
from denguecad.metrics import compute_performance, scores_from_estimator
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig, NMPredictionResult


@dataclass
class MethodCVResult:
    name: str
    performance_row: dict[str, float]
    fold_accuracies: list[float]


def _cv_baseline(
    name: str,
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    decision_col: str,
    *,
    n_folds: int = 10,
    random_state: int = 42,
    svm_max_train: int = 8_000,
) -> MethodCVResult:
    factory = BASELINE_FACTORIES[name]
    use = data[[*feature_cols, decision_col]].dropna()
    X = use[list(feature_cols)].to_numpy(dtype=float)
    y = use[decision_col].to_numpy().astype(int)
    n_folds = min(n_folds, max(2, int(pd.Series(y).value_counts().min())))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    fold_acc: list[float] = []
    for tr, te in skf.split(X, y):
        fit_idx = tr
        if name == "SVM" and len(tr) > svm_max_train:
            # Stratified subsample keeps RBF-SVM tractable on 20k-row folds
            rng = np.random.default_rng(random_state)
            y_tr = y[tr]
            take = []
            for cls in np.unique(y_tr):
                cls_idx = tr[y_tr == cls]
                n_take = max(1, int(round(svm_max_train * (len(cls_idx) / len(tr)))))
                n_take = min(n_take, len(cls_idx))
                take.append(rng.choice(cls_idx, size=n_take, replace=False))
            fit_idx = np.concatenate(take)
        clf = factory(random_state)
        clf.fit(X[fit_idx], y[fit_idx])
        scores = scores_from_estimator(clf, X[te])
        labels = y[te]
        scores_all.append(scores)
        labels_all.append(labels)
        pred = (scores >= 0.5).astype(int)
        fold_acc.append(float(np.mean(pred == labels)))

    perf = compute_performance(np.concatenate(labels_all), np.concatenate(scores_all))
    return MethodCVResult(name=name, performance_row=perf.as_row(), fold_accuracies=fold_acc)


def wilcoxon_nm_vs(
    nm_folds: Sequence[float],
    other_folds: Sequence[float],
) -> dict[str, Any]:
    """
    Wilcoxon matched-pairs signed-rank test on fold accuracies (NM − other).

    Returns positive/negative rank sums and two-sided p-value (paper Table II).
    """
    a = np.asarray(nm_folds, dtype=float)
    b = np.asarray(other_folds, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    # If all differences zero, Wilcoxon is undefined
    if np.allclose(diff, 0):
        return {"rank_sum_pos": 0.0, "rank_sum_neg": 0.0, "p_value": 1.0}

    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {"rank_sum_pos": float("nan"), "rank_sum_neg": float("nan"), "p_value": 1.0}

    # Reconstruct rank sums (+, −) in the spirit of the paper table
    abs_diff = np.abs(diff)
    nonzero = abs_diff > 0
    ranks = np.empty_like(abs_diff)
    ranks[nonzero] = pd.Series(abs_diff[nonzero]).rank(method="average").to_numpy()
    ranks[~nonzero] = 0.0
    pos = float(ranks[diff > 0].sum())
    neg = float(ranks[diff < 0].sum())
    return {
        "rank_sum_pos": pos,
        "rank_sum_neg": neg,
        "p_value": float(p),
        "statistic": float(stat),
    }


def run_dataset_comparison(
    ds_name: str,
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    decision_col: str,
    *,
    nm_config: Optional[NMPredictionConfig] = None,
    n_folds: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run NM + C4.5 + SVM + LOR on one dataset; return table rows + fold stats."""
    nm = NMPrediction(nm_config or NMPredictionConfig(n_folds=n_folds, random_state=random_state, verbose=verbose))
    nm_result: NMPredictionResult = nm.fit_predict_cv(
        data, decision_col=decision_col, feature_cols=feature_cols
    )

    methods = {
        "NM": MethodCVResult(
            name="NM",
            performance_row=nm_result.performance.as_row(),
            fold_accuracies=nm_result.fold_accuracies,
        )
    }
    for baseline in ("C4.5", "LOR", "SVM"):
        if verbose:
            print(f"  baseline {baseline} on {ds_name}…")
        methods[baseline] = _cv_baseline(
            baseline,
            data,
            feature_cols,
            decision_col,
            n_folds=n_folds,
            random_state=random_state,
        )

    table1_rows = []
    for name in ("NM", "C4.5", "LOR", "SVM"):
        row = {"Dataset": ds_name, "Method": name, **methods[name].performance_row}
        table1_rows.append(row)

    table2_rows = []
    for other in ("C4.5", "LOR", "SVM"):
        w = wilcoxon_nm_vs(methods["NM"].fold_accuracies, methods[other].fold_accuracies)
        table2_rows.append(
            {
                "Dataset": ds_name,
                "Method": other,
                "Rank sum (+, −)": f"{w['rank_sum_pos']:.1f}, {w['rank_sum_neg']:.1f}",
                "p-value": round(w["p_value"], 3),
            }
        )

    table3_row = {"Dataset": ds_name, **nm_result.table3_row()}
    return {
        "table1": table1_rows,
        "table2": table2_rows,
        "table3": table3_row,
        "nm_result": nm_result,
        "methods": methods,
    }


def run_full_comparison(
    splits: ExperimentSplits,
    *,
    nm_config: Optional[NMPredictionConfig] = None,
    n_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare all methods on the training dataset(s); return Table I / II / III DataFrames."""
    t1: list[dict] = []
    t2: list[dict] = []
    t3: list[dict] = []
    holdout_rows: list[dict] = []

    for ds_name, df in splits.datasets.items():
        if verbose:
            print(f"=== {ds_name} (n={len(df)}) ===")
        out = run_dataset_comparison(
            ds_name,
            df,
            splits.feature_cols,
            splits.decision_col,
            nm_config=nm_config,
            n_folds=n_folds,
            random_state=random_state,
            verbose=verbose,
        )
        t1.extend(out["table1"])
        t2.extend(out["table2"])
        t3.append(out["table3"])

        # Hold-out validation / test using NM selected features + ADT
        nm: NMPrediction = NMPrediction(nm_config or NMPredictionConfig())
        selected = out["nm_result"].selected_features
        for label, hold in (("validation", splits.validation), ("test", splits.test)):
            perf = nm.evaluate_holdout(
                df,
                hold,
                decision_col=splits.decision_col,
                feature_cols=splits.feature_cols,
                selected_features=selected,
            )
            holdout_rows.append(
                {
                    "Dataset": ds_name,
                    "Holdout": label,
                    "n": len(hold),
                    **perf.as_row(),
                    "features": ", ".join(selected),
                }
            )

    return {
        "table1": pd.DataFrame(t1),
        "table2": pd.DataFrame(t2),
        "table3": pd.DataFrame(t3),
        "holdout": pd.DataFrame(holdout_rows),
    }
