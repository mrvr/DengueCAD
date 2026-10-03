"""
Imputation + prediction hold-out experiment.

For each dataset S and each repeat:

1. ``valid_set`` = complete-case records of S (ground truth, kept aside).
2. MVs are injected into 20% of the valid records (1–3 random attributes each,
   decision untouched); naturally missing cells in S are kept.
3. The resulting set is imputed by NMI (the *imputed dataset*) and by the
   baseline imputers; imputation accuracy is scored on the injected cells.
4. 30% of valid records (stratified) are held out for testing; removing them
   from the NMI-imputed dataset gives ``train_set``.
5. The test dataset is those held-out records taken from ``valid_set`` with the
   decision removed on 20% of them; those decisions are predicted by
   NMPrediction and the baselines (all trained on ``train_set``) and compared
   with the true ``valid_set`` decisions.

Every column is encoded as integer codes: nominal attributes by sorted label,
numeric attributes by ordered value / quantile bin, so imputation accuracy is
exact recovery of the code (the decile bin for continuous attributes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from denguecad.baselines import make_c45, make_lor, make_svm
from denguecad.feature_selection import GAWrapperConfig
from denguecad.imputation_benchmark import impute_knn, impute_mice, impute_nmi, impute_simple
from denguecad.missforest_impute import missforest_impute
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig

IMPUTERS = ("NMI", "MICE", "kNN", "MissForest", "Mean/Mode")
PREDICTORS = ("NMPrediction", "C4.5", "LOR", "SVM")


@dataclass
class EncodedDataset:
    name: str
    frame: pd.DataFrame  # Int64 codes, NA for missing
    feature_cols: list[str]
    decision_col: str


def encode_ordinal(
    raw: pd.DataFrame,
    *,
    name: str,
    decision_col: Optional[str] = None,
    drop: Sequence[str] = (),
    n_bins: int = 10,
) -> EncodedDataset:
    """Encode every column as integer codes, keeping numeric order."""
    decision_col = decision_col or str(raw.columns[-1])
    feature_cols = [c for c in raw.columns if c != decision_col and c not in set(drop)]
    out = pd.DataFrame(index=raw.index)
    for col in [*feature_cols, decision_col]:
        s = raw[col]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype("Int64")
        if s.dtype == object or pd.api.types.is_string_dtype(s) or col == decision_col:
            levels = sorted(s.dropna().astype(str).unique())
            codes = s.astype(str).map({v: i for i, v in enumerate(levels)})
        else:
            num = pd.to_numeric(s, errors="coerce")
            uniq = np.sort(num.dropna().unique())
            if len(uniq) <= 2 * n_bins:
                codes = num.map({v: i for i, v in enumerate(uniq)})
            else:
                codes = pd.qcut(num, q=n_bins, labels=False, duplicates="drop")
        codes = pd.Series(codes, index=raw.index).astype("Float64").astype("Int64")
        codes[s.isna()] = pd.NA
        out[col] = codes
    return EncodedDataset(name, out[[*feature_cols, decision_col]], feature_cols, decision_col)


def inject_record_mcar(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    candidate_rows: Sequence,
    *,
    row_frac: float = 0.2,
    max_attrs: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask 1–``max_attrs`` random attributes in ``row_frac`` of ``candidate_rows``."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    mask = pd.DataFrame(False, index=frame.index, columns=list(feature_cols))
    n_rows = max(1, int(round(row_frac * len(candidate_rows))))
    chosen = rng.choice(np.asarray(candidate_rows), size=n_rows, replace=False)
    upper = min(max_attrs, len(feature_cols))
    for row in chosen:
        k = int(rng.integers(1, upper + 1))
        cols = rng.choice(len(feature_cols), size=k, replace=False)
        for j in cols:
            mask.at[row, feature_cols[j]] = True
    for col in feature_cols:
        out.loc[mask[col], col] = pd.NA
    return out, mask


def _round_to_codes(imputed: pd.DataFrame, reference: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Round numeric imputations to the nearest valid integer code."""
    out = imputed.copy()
    for col in cols:
        lo, hi = reference[col].min(), reference[col].max()
        vals = pd.to_numeric(out[col], errors="coerce").astype(float)
        out[col] = vals.round().clip(lo, hi)
    return out


def run_imputers(
    masked: pd.DataFrame,
    feature_cols: Sequence[str],
    decision_col: str,
    *,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Impute ``masked`` with NMI and each baseline; outputs are integer codes."""
    as_float = masked.astype(float)
    runners: dict[str, Callable[[], pd.DataFrame]] = {
        "NMI": lambda: impute_nmi(
            masked, feature_cols, decision_col, categorical_cols=feature_cols
        ),
        "MICE": lambda: impute_mice(as_float, feature_cols, random_state=seed),
        "kNN": lambda: impute_knn(as_float, feature_cols),
        "MissForest": lambda: missforest_impute(
            as_float, feature_cols, categorical_cols=feature_cols,
            max_iter=5, n_estimators=40, random_state=seed,
        ),
        "Mean/Mode": lambda: impute_simple(
            as_float, feature_cols, feature_cols, strategy="mode"
        ),
    }
    return {
        name: _round_to_codes(fn().astype(float), as_float, feature_cols)
        for name, fn in runners.items()
    }


def imputation_accuracy(truth: pd.DataFrame, imputed: pd.DataFrame, mask: pd.DataFrame) -> float:
    hits = total = 0
    for col in mask.columns:
        rows = mask.index[mask[col]]
        if len(rows) == 0:
            continue
        t = truth.loc[rows, col].astype(float).to_numpy()
        p = imputed.loc[rows, col].astype(float).to_numpy()
        hits += int(np.sum(t == p))
        total += len(rows)
    return hits / total if total else float("nan")


def _stratified_pick(index: pd.Index, labels: pd.Series, frac: float, seed: int) -> pd.Index:
    n = max(1, int(round(frac * len(index))))
    try:
        _, picked = train_test_split(index, test_size=n, stratify=labels.loc[index], random_state=seed)
    except ValueError:
        _, picked = train_test_split(index, test_size=n, random_state=seed)
    return pd.Index(picked)


def _subsample(X: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= limit:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=limit, stratify=y, random_state=seed)
    return X[idx], y[idx]


def run_predictors(
    train: pd.DataFrame,
    test_X: pd.DataFrame,
    feature_cols: Sequence[str],
    decision_col: str,
    *,
    seed: int,
    nm_config: NMPredictionConfig,
    svm_max_train: int = 8_000,
) -> dict[str, np.ndarray]:
    """Train each predictor on ``train`` and predict the decision for ``test_X``."""
    nm = NMPrediction(nm_config).fit(train, decision_col=decision_col, feature_cols=feature_cols)
    preds = {"NMPrediction": nm.predict(test_X)}

    X_tr = train[list(feature_cols)].to_numpy(dtype=float)
    y_tr = train[decision_col].to_numpy().astype(int)
    X_te = test_X[list(feature_cols)].to_numpy(dtype=float)
    for name, factory in (("C4.5", make_c45), ("LOR", make_lor), ("SVM", make_svm)):
        Xf, yf = (X_tr, y_tr) if name != "SVM" else _subsample(X_tr, y_tr, svm_max_train, seed)
        clf = factory(seed).fit(Xf, yf)
        preds[name] = np.asarray(clf.predict(X_te)).astype(int)
    return preds


def default_nm_config(seed: int) -> NMPredictionConfig:
    return NMPredictionConfig(
        ga=GAWrapperConfig(
            population_size=8,
            n_generations=5,
            crossover_prob=1.0,
            mutation_prob=0.001,
            n_folds=3,
            classifier="adt",
            scoring="accuracy",
            impute_missing=False,
            subset_size_penalty=0.01,
            random_state=seed,
        ),
        adt_estimators=40,
        random_state=seed,
    )


def run_repeat(
    ds: EncodedDataset,
    seed: int,
    *,
    mv_row_frac: float = 0.2,
    test_frac: float = 0.3,
    decision_mask_frac: float = 0.2,
) -> dict:
    """One repeat of the protocol; returns imputation and prediction accuracies."""
    S = ds.frame.reset_index(drop=True)
    feats, dec = ds.feature_cols, ds.decision_col

    valid = S.dropna()
    masked, mask = inject_record_mcar(S, feats, valid.index, row_frac=mv_row_frac, seed=seed)

    imputed = run_imputers(masked, feats, dec, seed=seed)
    imp_acc = {name: imputation_accuracy(valid, df, mask.loc[valid.index]) for name, df in imputed.items()}

    test_idx = _stratified_pick(valid.index, valid[dec], test_frac, seed)
    train_set = imputed["NMI"].drop(index=test_idx).dropna()
    train_set[dec] = train_set[dec].astype(int)

    test_set = valid.loc[test_idx].copy()
    hidden_idx = _stratified_pick(test_set.index, test_set[dec], decision_mask_frac, seed)
    truth = test_set.loc[hidden_idx, dec].astype(int).to_numpy()
    test_set.loc[hidden_idx, dec] = pd.NA

    preds = run_predictors(
        train_set, test_set.loc[hidden_idx], feats, dec,
        seed=seed, nm_config=default_nm_config(seed),
    )
    pred_acc = {name: float(np.mean(p == truth)) for name, p in preds.items()}

    return {
        "n_rows": len(S),
        "n_valid": len(valid),
        "n_masked_cells": int(mask.to_numpy().sum()),
        "n_train": len(train_set),
        "n_test": len(test_idx),
        "n_predicted": len(hidden_idx),
        "imputation": imp_acc,
        "prediction": pred_acc,
    }
