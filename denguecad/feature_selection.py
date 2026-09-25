"""
Wrapper subset feature selection with genetic search.

Implements Section C of Rao & Kumar (IEEE TITB 2012): a *wrapper*
feature-evaluation model [Kohavi & John, Artificial Intelligence 97,
1997] where a classifier's cross-validated performance is the fitness
of a binary feature chromosome evolved by a genetic algorithm
[Goldberg, 1989].

Paper GA parameters (Section V):
  crossover probability Pc = 1.0
  mutation probability   Pm = 0.001
Evaluation uses stratified k-fold CV (paper: k = 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from denguecad.classifiers import ClassifierFactory, get_classifier_factory
from denguecad.nmi_support import import_nmilib


@dataclass
class GAWrapperConfig:
    """Hyper-parameters for the GA wrapper search."""

    population_size: int = 30
    n_generations: int = 40
    crossover_prob: float = 1.0  # paper Section V
    mutation_prob: float = 0.001  # paper Section V
    tournament_size: int = 3
    elitism: int = 2
    n_folds: int = 10  # stratified k-fold (paper Section V)
    scoring: str = "accuracy"  # "accuracy" | "auc"
    # Soft preference for compact subsets (Kohavi–John Occam bias)
    subset_size_penalty: float = 0.001
    classifier: str = "svm"  # "svm" | "tree"
    random_state: int = 42
    min_features: int = 1
    verbose: bool = False
    # If True, run NMI imputation before selection when MVs are present
    impute_missing: bool = True
    nmi_root: Optional[str] = None


@dataclass
class WrapperGAResult:
    """Outcome of ``select_influential_features``."""

    selected_features: list[str]
    chromosome: np.ndarray
    fitness: float
    cv_score: float
    subset_size: int
    history: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    config: GAWrapperConfig = field(default_factory=GAWrapperConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_features": list(self.selected_features),
            "fitness": float(self.fitness),
            "cv_score": float(self.cv_score),
            "subset_size": int(self.subset_size),
            "best_fitness_per_generation": list(self.history),
            "n_generations": len(self.history),
        }


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ensure_min_features(chrom: np.ndarray, rng: np.random.Generator, min_features: int) -> np.ndarray:
    out = chrom.copy()
    if out.sum() >= min_features:
        return out
    off = np.flatnonzero(out == 0)
    if len(off) == 0:
        return out
    n_flip = min(min_features - int(out.sum()), len(off))
    pick = rng.choice(off, size=n_flip, replace=False)
    out[pick] = 1
    return out


def _init_population(
    n_features: int,
    pop_size: int,
    rng: np.random.Generator,
    min_features: int,
) -> np.ndarray:
    """Random binary chromosomes; bias toward sparse subsets (~30% on)."""
    pop = (rng.random((pop_size, n_features)) < 0.3).astype(np.int8)
    for i in range(pop_size):
        pop[i] = _ensure_min_features(pop[i], rng, min_features)
    # Guarantee at least one full-feature individual and one sparse seed
    pop[0] = np.ones(n_features, dtype=np.int8)
    if pop_size > 1:
        sparse = np.zeros(n_features, dtype=np.int8)
        sparse[rng.choice(n_features, size=min(min_features, n_features), replace=False)] = 1
        pop[1] = sparse
    return pop


def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    k: int,
) -> np.ndarray:
    idx = rng.integers(0, len(population), size=k)
    winner = idx[np.argmax(fitness[idx])]
    return population[winner].copy()


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    pc: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > pc or len(parent_a) < 2:
        return parent_a.copy(), parent_b.copy()
    point = int(rng.integers(1, len(parent_a)))
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a.astype(np.int8), child_b.astype(np.int8)


def _mutate(chrom: np.ndarray, rng: np.random.Generator, pm: float) -> np.ndarray:
    flips = rng.random(len(chrom)) < pm
    out = chrom.copy()
    out[flips] = 1 - out[flips]
    return out.astype(np.int8)


def _evaluate_chromosome(
    chrom: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier_factory: ClassifierFactory,
    n_folds: int,
    scoring: str,
    subset_size_penalty: float,
    random_state: int,
    min_features: int,
) -> tuple[float, float]:
    """
    Return (fitness, raw_cv_score).

    Fitness = CV score − penalty × (subset_size / n_features).
    Empty / undersized subsets score 0.
    """
    mask = chrom.astype(bool)
    n_sel = int(mask.sum())
    if n_sel < min_features:
        return 0.0, 0.0

    X_sub = X[:, mask]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores: list[float] = []

    for train_idx, test_idx in skf.split(X_sub, y):
        clf: BaseEstimator = classifier_factory()
        # AUC needs predict_proba or decision_function
        if scoring == "auc":
            if hasattr(clf, "probability"):
                clf.set_params(probability=True)
            clf.fit(X_sub[train_idx], y[train_idx])
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_sub[test_idx])
                if proba.shape[1] == 2:
                    score = roc_auc_score(y[test_idx], proba[:, 1])
                else:
                    score = roc_auc_score(y[test_idx], proba, multi_class="ovr")
            else:
                dec = clf.decision_function(X_sub[test_idx])
                score = roc_auc_score(y[test_idx], dec)
        else:
            clf.fit(X_sub[train_idx], y[train_idx])
            pred = clf.predict(X_sub[test_idx])
            score = accuracy_score(y[test_idx], pred)
        scores.append(float(score))

    cv_score = float(np.mean(scores))
    penalty = subset_size_penalty * (n_sel / len(chrom))
    fitness = cv_score - penalty
    return fitness, cv_score


def _maybe_impute(
    data: pd.DataFrame,
    decision_col: str,
    feature_cols: Sequence[str],
    *,
    enabled: bool,
    nmi_root: Optional[str],
) -> pd.DataFrame:
    work = data.copy()
    if not enabled:
        return work
    subset = work[[*feature_cols, decision_col]]
    if not subset.isna().any().any():
        return work
    nmilib = import_nmilib(nmi_root)
    imputed = nmilib.non_parametric_imputation(
        subset,
        decision_col=decision_col,
        feature_cols=list(feature_cols),
    )
    out = work.copy()
    out[list(feature_cols)] = imputed[list(feature_cols)]
    return out


def select_influential_features(
    data: pd.DataFrame,
    *,
    decision_col: Optional[str] = None,
    feature_cols: Optional[Sequence[str]] = None,
    config: Optional[GAWrapperConfig] = None,
    classifier_factory: Optional[ClassifierFactory] = None,
) -> WrapperGAResult:
    """
    Identify an influential feature subset via GA wrapper evaluation.

    Parameters
    ----------
    data :
        Tabular dataset. Missing attribute values may be imputed with NMI
        when ``config.impute_missing`` is True.
    decision_col :
        Class / diagnosis column (defaults to last column).
    feature_cols :
        Candidate feature names (defaults to all columns except decision;
        ID-like ``Name`` is dropped when present).
    config :
        GA / CV hyper-parameters (paper defaults for Pc, Pm, k-fold).
    classifier_factory :
        Optional zero-arg callable returning a fresh sklearn estimator.
        Overrides ``config.classifier`` when provided.

    Returns
    -------
    WrapperGAResult
        Best chromosome, selected feature names, fitness history.
    """
    cfg = config or GAWrapperConfig()
    df = pd.DataFrame(data).copy()

    if decision_col is None:
        decision_col = str(df.columns[-1])
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c != decision_col and str(c).lower() not in {"name", "id", "patient_id"}
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No feature columns available for selection.")

    # Drop rows with missing decision (same convention as NMI)
    df = df.dropna(subset=[decision_col]).reset_index(drop=True)
    df = _maybe_impute(
        df,
        decision_col,
        feature_cols,
        enabled=cfg.impute_missing,
        nmi_root=cfg.nmi_root,
    )

    # Remaining MVs (e.g. held-out) — drop incomplete feature rows for CV
    use = df[[*feature_cols, decision_col]].dropna()
    if use.empty:
        raise ValueError("No complete rows available after imputation / filtering.")

    X = use[feature_cols].to_numpy(dtype=float)
    y = use[decision_col].to_numpy()
    # Encode non-numeric labels if needed
    if y.dtype.kind in {"U", "O", "S"}:
        _, y = np.unique(y, return_inverse=True)

    n_features = len(feature_cols)
    n_folds = min(cfg.n_folds, max(2, int(pd.Series(y).value_counts().min())))
    if n_folds < 2:
        raise ValueError("Need at least 2 samples per class for stratified CV.")

    factory = classifier_factory or get_classifier_factory(
        cfg.classifier, random_state=cfg.random_state
    )
    rng = _rng(cfg.random_state)

    population = _init_population(n_features, cfg.population_size, rng, cfg.min_features)
    fitness = np.zeros(cfg.population_size, dtype=float)
    cv_scores = np.zeros(cfg.population_size, dtype=float)

    def eval_pop(pop: np.ndarray) -> None:
        for i in range(len(pop)):
            fit, cv = _evaluate_chromosome(
                pop[i],
                X,
                y,
                classifier_factory=factory,
                n_folds=n_folds,
                scoring=cfg.scoring,
                subset_size_penalty=cfg.subset_size_penalty,
                random_state=cfg.random_state,
                min_features=cfg.min_features,
            )
            fitness[i] = fit
            cv_scores[i] = cv

    eval_pop(population)
    history: list[float] = [float(fitness.max())]
    best_idx = int(np.argmax(fitness))
    best_chrom = population[best_idx].copy()
    best_fit = float(fitness[best_idx])
    best_cv = float(cv_scores[best_idx])

    if cfg.verbose:
        print(
            f"[gen 0] best_fitness={best_fit:.4f} cv={best_cv:.4f} "
            f"size={int(best_chrom.sum())}/{n_features}"
        )

    for gen in range(1, cfg.n_generations + 1):
        # Elitism
        elite_idx = np.argsort(fitness)[-cfg.elitism :]
        elites = population[elite_idx].copy()
        elite_fit = fitness[elite_idx].copy()
        elite_cv = cv_scores[elite_idx].copy()

        children: list[np.ndarray] = []
        while len(children) < cfg.population_size - cfg.elitism:
            p1 = _tournament_select(population, fitness, rng, cfg.tournament_size)
            p2 = _tournament_select(population, fitness, rng, cfg.tournament_size)
            c1, c2 = _crossover(p1, p2, rng, cfg.crossover_prob)
            c1 = _ensure_min_features(_mutate(c1, rng, cfg.mutation_prob), rng, cfg.min_features)
            c2 = _ensure_min_features(_mutate(c2, rng, cfg.mutation_prob), rng, cfg.min_features)
            children.append(c1)
            if len(children) < cfg.population_size - cfg.elitism:
                children.append(c2)

        population = np.vstack([elites, np.stack(children[: cfg.population_size - cfg.elitism])])
        fitness = np.zeros(cfg.population_size, dtype=float)
        cv_scores = np.zeros(cfg.population_size, dtype=float)
        # Reuse elite fitness
        fitness[: cfg.elitism] = elite_fit
        cv_scores[: cfg.elitism] = elite_cv
        for i in range(cfg.elitism, cfg.population_size):
            fit, cv = _evaluate_chromosome(
                population[i],
                X,
                y,
                classifier_factory=factory,
                n_folds=n_folds,
                scoring=cfg.scoring,
                subset_size_penalty=cfg.subset_size_penalty,
                random_state=cfg.random_state,
                min_features=cfg.min_features,
            )
            fitness[i] = fit
            cv_scores[i] = cv

        gen_best = int(np.argmax(fitness))
        history.append(float(fitness[gen_best]))
        if fitness[gen_best] > best_fit + 1e-12 or (
            abs(fitness[gen_best] - best_fit) <= 1e-12
            and population[gen_best].sum() < best_chrom.sum()
        ):
            best_chrom = population[gen_best].copy()
            best_fit = float(fitness[gen_best])
            best_cv = float(cv_scores[gen_best])

        if cfg.verbose and (gen % 5 == 0 or gen == cfg.n_generations):
            print(
                f"[gen {gen}] best_fitness={best_fit:.4f} cv={best_cv:.4f} "
                f"size={int(best_chrom.sum())}/{n_features}"
            )

    selected = [f for f, bit in zip(feature_cols, best_chrom) if bit]
    return WrapperGAResult(
        selected_features=selected,
        chromosome=best_chrom,
        fitness=best_fit,
        cv_score=best_cv,
        subset_size=int(best_chrom.sum()),
        history=history,
        feature_names=list(feature_cols),
        config=cfg,
    )
