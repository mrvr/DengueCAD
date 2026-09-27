"""
NMPrediction — Algorithm 1 (The NM Methodology).

Rao & Kumar, IEEE TITB 2012:
  1) collect S
  2) impute MV (NMI / Section III-B)
  3) wrapper feature selection with genetic search + ADT evaluation
  4–6) stratified k-fold ADT; accumulate scores P and labels L
  7–8) compute AUC, SE, SP (and accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from denguecad.baselines import make_adt
from denguecad.feature_selection import GAWrapperConfig, select_influential_features
from denguecad.metrics import PerformanceResult, compute_performance, scores_from_estimator
from denguecad.nmi_support import import_nmilib


@dataclass
class NMPredictionConfig:
    n_folds: int = 10
    ga: GAWrapperConfig = field(
        default_factory=lambda: GAWrapperConfig(
            population_size=12,
            n_generations=8,
            crossover_prob=1.0,
            mutation_prob=0.001,
            n_folds=3,
            classifier="adt",
            scoring="accuracy",
            impute_missing=True,
            subset_size_penalty=0.01,
            verbose=False,
        )
    )
    adt_estimators: int = 40
    random_state: int = 42
    nmi_root: Optional[str] = None
    verbose: bool = False


@dataclass
class NMPredictionResult:
    performance: PerformanceResult
    selected_features: list[str]
    n_original_features: int
    fold_accuracies: list[float]
    labels: np.ndarray
    scores: np.ndarray
    config: NMPredictionConfig

    def table3_row(self) -> dict[str, Any]:
        return {
            "# Original features": self.n_original_features,
            "# influential features": len(self.selected_features),
            "Accuracy (%)": round(100.0 * self.performance.accuracy, 2),
            "features identified": ", ".join(self.selected_features),
        }


class NMPrediction:
    """
    End-to-end NM methodology (Algorithm 1).

    Parameters
    ----------
    config :
        Fold / GA / ADT settings (paper defaults where applicable).
    """

    def __init__(self, config: Optional[NMPredictionConfig] = None):
        self.config = config or NMPredictionConfig()

    def _impute(
        self,
        data: pd.DataFrame,
        decision_col: str,
        feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        work = data.copy()
        subset = work[[*feature_cols, decision_col]]
        if not subset.isna().any().any():
            return work
        nmilib = import_nmilib(self.config.nmi_root or self.config.ga.nmi_root)
        imputed = nmilib.non_parametric_imputation(
            subset,
            decision_col=decision_col,
            feature_cols=list(feature_cols),
        )
        out = work.copy()
        out[list(feature_cols)] = imputed[list(feature_cols)]
        return out

    def fit_predict_cv(
        self,
        data: pd.DataFrame,
        *,
        decision_col: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> NMPredictionResult:
        """
        Run Algorithm 1 on dataset ``S`` and return CV performance + features.
        """
        cfg = self.config
        df = pd.DataFrame(data).copy()
        if feature_cols is None:
            feature_cols = [
                c
                for c in df.columns
                if c != decision_col and str(c).lower() not in {"name", "id"}
            ]
        feature_cols = list(feature_cols)
        n_original = len(feature_cols)

        # Steps 1–2: collect + impute
        df = df.dropna(subset=[decision_col]).reset_index(drop=True)
        df = self._impute(df, decision_col, feature_cols)

        # Step 3: wrapper GA + ADT evaluation model
        ga_cfg = cfg.ga
        ga_cfg.classifier = "adt"
        ga_cfg.impute_missing = False  # already imputed
        ga_cfg.nmi_root = cfg.nmi_root or ga_cfg.nmi_root
        ga_cfg.random_state = cfg.random_state
        if cfg.verbose:
            ga_cfg.verbose = True

        wrap = select_influential_features(
            df,
            decision_col=decision_col,
            feature_cols=feature_cols,
            config=ga_cfg,
            classifier_factory=lambda: make_adt(cfg.random_state),
        )
        selected = wrap.selected_features or feature_cols
        if cfg.verbose:
            print(f"NMPrediction influential features: {selected}")

        # Steps 4–6: stratified k-fold ADT on selected features
        use = df[[*selected, decision_col]].dropna()
        X = use[selected].to_numpy(dtype=float)
        y = use[decision_col].to_numpy().astype(int)
        n_folds = min(cfg.n_folds, max(2, int(pd.Series(y).value_counts().min())))
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=cfg.random_state
        )

        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        fold_acc: list[float] = []

        for fold_i, (tr, te) in enumerate(skf.split(X, y), start=1):
            adt = make_adt(cfg.random_state)
            adt.set_params(n_estimators=cfg.adt_estimators)
            adt.fit(X[tr], y[tr])
            scores = scores_from_estimator(adt, X[te])
            labels = y[te]
            all_scores.append(scores)
            all_labels.append(labels)
            # fold accuracy at 0.5 for Wilcoxon matched pairs
            pred = (scores >= 0.5).astype(int)
            fold_acc.append(float(np.mean(pred == labels)))
            if cfg.verbose:
                print(f"  fold {fold_i}/{n_folds} acc={fold_acc[-1]:.4f}")

        P = np.concatenate(all_scores)
        L = np.concatenate(all_labels)
        perf = compute_performance(L, P)
        return NMPredictionResult(
            performance=perf,
            selected_features=list(selected),
            n_original_features=n_original,
            fold_accuracies=fold_acc,
            labels=L,
            scores=P,
            config=cfg,
        )

    def evaluate_holdout(
        self,
        train: pd.DataFrame,
        holdout: pd.DataFrame,
        *,
        decision_col: str,
        feature_cols: Sequence[str],
        selected_features: Sequence[str],
    ) -> PerformanceResult:
        """Train ADT on ``train`` (selected features) and score a clean holdout set."""
        train_i = self._impute(train, decision_col, feature_cols)
        cols = list(selected_features)
        tr = train_i[[*cols, decision_col]].dropna()
        ho = holdout[[*cols, decision_col]].dropna()
        adt = make_adt(self.config.random_state)
        adt.set_params(n_estimators=self.config.adt_estimators)
        adt.fit(tr[cols].to_numpy(dtype=float), tr[decision_col].to_numpy().astype(int))
        scores = scores_from_estimator(adt, ho[cols].to_numpy(dtype=float))
        return compute_performance(ho[decision_col].to_numpy().astype(int), scores)
