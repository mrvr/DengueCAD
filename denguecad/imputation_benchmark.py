"""
Missing-value imputation benchmark: NMI vs common baselines.

Compares, on a complete-case master with controlled masking:
  - NM  — non-parametric imputation (sibling NMI / nmilib)
  - Mode — column-wise most frequent (categoricals) / mean (numeric)
  - Mean — column-wise mean (encoded categoricals treated as numeric)
  - Median
  - kNN  — sklearn KNNImputer
  - MICE — sklearn IterativeImputer

Also can fill the dataset's natural MVs and write an imputed CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

from denguecad.nmi_support import import_nmilib

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
)
DECISION_COL = "Outcome"
# Drop non-features / ultra-high-cardinality raw lab counts for NMI proximity speed.
# (Counts are still available in the raw CSV; we keep Fever_Duration, symptoms, serology.)
DROP_FEATURES = {"Area", "District", "Platelet_Count", "WBC_Count"}


@dataclass
class MethodScore:
    method: str
    accuracy: float
    n_compared: int
    n_match: int
    seconds: float
    by_column: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "Method": self.method,
            "Accuracy (%)": round(100.0 * self.accuracy, 2),
            "n_compared": self.n_compared,
            "n_match": self.n_match,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class EncodedFrame:
    """Numeric frame + encoders so categoricals round-trip for NMI/baselines."""

    frame: pd.DataFrame
    feature_cols: list[str]
    decision_col: str
    categorical_cols: list[str]
    encoders: dict[str, LabelEncoder]
    raw_columns: list[str]


def _is_categorical_series(s: pd.Series) -> bool:
    if s.dtype == object or str(s.dtype) == "string" or pd.api.types.is_string_dtype(s):
        return True
    # Paper: nominal / integer attributes use counting indices (not mean)
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    return False


def load_and_encode(
    path: str | Path = DEFAULT_DATA,
    *,
    decision_col: str = DECISION_COL,
    drop_features: Optional[set[str]] = None,
) -> EncodedFrame:
    drop_features = set(drop_features or DROP_FEATURES)
    raw = pd.read_csv(path, na_values=["", "?", "NA", "N/A", "null", "Null"])
    if decision_col not in raw.columns:
        raise ValueError(f"{decision_col} not in {list(raw.columns)}")

    feature_cols = [
        c
        for c in raw.columns
        if c != decision_col and c not in drop_features
    ]
    work = raw[feature_cols + [decision_col]].copy()

    encoders: dict[str, LabelEncoder] = {}
    categorical_cols: list[str] = []
    encoded = work.copy()

    for col in feature_cols + [decision_col]:
        if not _is_categorical_series(work[col]) and col == decision_col:
            # Outcome is int binary — keep numeric, still "categorical" for NMI
            categorical_cols.append(col)
            continue
        if _is_categorical_series(work[col]) or col == decision_col:
            # High-cardinality integers (labs, age): quantile-bin for NMI proximity
            if (
                col != decision_col
                and pd.api.types.is_integer_dtype(work[col])
                and work[col].nunique(dropna=True) > 20
            ):
                categorical_cols.append(col)
                num = pd.to_numeric(work[col], errors="coerce")
                try:
                    binned = pd.qcut(num, q=10, duplicates="drop")
                except ValueError:
                    binned = pd.cut(num, bins=10)
                enc = LabelEncoder()
                mask = binned.notna()
                enc.fit(binned.loc[mask].astype(str))
                out = pd.Series(pd.NA, index=work.index, dtype="Int64")
                out.loc[mask] = enc.transform(binned.loc[mask].astype(str))
                encoded[col] = out
                encoders[col] = enc
                continue
            categorical_cols.append(col)
            enc = LabelEncoder()
            mask = work[col].notna()
            if mask.any():
                enc.fit(work.loc[mask, col].astype(str))
                out = pd.Series(pd.NA, index=work.index, dtype="Int64")
                out.loc[mask] = enc.transform(work.loc[mask, col].astype(str))
                # Int64 keeps NaN and is categorical for NMI (not float/fractional)
                encoded[col] = out
                encoders[col] = enc
            else:
                encoded[col] = work[col]
        else:
            # Real-valued: round lightly and keep as float (fractional for NMI)
            # Unless low-cardinality after 1-decimal rounding → categorical codes
            num = pd.to_numeric(work[col], errors="coerce")
            rounded = num.round(1)
            if rounded.dropna().nunique() <= 80:
                categorical_cols.append(col)
                enc = LabelEncoder()
                mask = rounded.notna()
                enc.fit(rounded.loc[mask].astype(str))
                out = pd.Series(pd.NA, index=work.index, dtype="Int64")
                out.loc[mask] = enc.transform(rounded.loc[mask].astype(str))
                encoded[col] = out
                encoders[col] = enc
            else:
                encoded[col] = num

    return EncodedFrame(
        frame=encoded,
        feature_cols=feature_cols,
        decision_col=decision_col,
        categorical_cols=[c for c in categorical_cols if c != decision_col],
        encoders=encoders,
        raw_columns=list(raw.columns),
    )


def mask_features(
    master: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    mv_frac: float = 0.12,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Randomly mask attribute cells (never the decision column).

    Returns (mv_frame, boolean mask of shape n×n_features aligned to feature_cols).
    """
    rng = np.random.default_rng(seed)
    out = master.copy()
    n, f = len(out), len(feature_cols)
    n_mask = max(1, int(round(n * f * mv_frac)))
    flat = rng.choice(n * f, size=min(n_mask, n * f), replace=False)
    mask = np.zeros(n * f, dtype=bool)
    mask[flat] = True
    mask = mask.reshape(n, f)
    for j, col in enumerate(feature_cols):
        rows = np.flatnonzero(mask[:, j])
        out.iloc[rows, out.columns.get_loc(col)] = np.nan
    return out, mask


def _score_imputed(
    master: pd.DataFrame,
    imputed: pd.DataFrame,
    feature_cols: Sequence[str],
    mask: np.ndarray,
    categorical_cols: Sequence[str],
) -> tuple[float, int, int, dict[str, float]]:
    matches = 0
    compared = 0
    by_col: dict[str, list[bool]] = {c: [] for c in feature_cols}
    cat = set(categorical_cols)

    for j, col in enumerate(feature_cols):
        rows = np.flatnonzero(mask[:, j])
        if len(rows) == 0:
            continue
        truth = master.iloc[rows][col].to_numpy()
        pred = imputed.iloc[rows][col].to_numpy()
        for t, p in zip(truth, pred):
            if pd.isna(p):
                ok = False
            elif col in cat or (
                isinstance(t, (int, np.integer)) and float(t).is_integer()
            ):
                # categorical / integer: exact match after round
                try:
                    ok = int(round(float(p))) == int(round(float(t)))
                except (TypeError, ValueError):
                    ok = str(p) == str(t)
            else:
                # real-valued: relative tolerance
                ok = bool(np.isclose(float(p), float(t), rtol=0.05, atol=1e-3))
            by_col[col].append(ok)
            matches += int(ok)
            compared += 1

    acc = matches / compared if compared else 0.0
    by_col_acc = {
        c: (sum(v) / len(v) if v else float("nan")) for c, v in by_col.items()
    }
    return acc, compared, matches, by_col_acc


def impute_nmi(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    decision_col: str,
    *,
    nmi_root: Optional[str] = None,
    categorical_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    nmilib = import_nmilib(nmi_root)
    subset = data[[*feature_cols, decision_col]].copy()
    # Keep categoricals / integer codes as int so NMI uses counting (not mean)
    cat = set(categorical_cols or [])
    for col in list(feature_cols) + [decision_col]:
        if col in cat or col == decision_col:
            # nullable int → object/float NaN safe for nmilib after float nan
            s = subset[col]
            mask = s.notna()
            out = s.astype(float)
            out.loc[mask] = s.loc[mask].astype(int).astype(float)
            subset[col] = out
            # Force integer dtype where complete for kind detection on donors
    # Cast fully-observed integer-like columns to Int64-friendly float that
    # only contain whole numbers — nmilib treats non-{0,1} floats as fractional.
    # Re-encode those columns as pandas Int64 with NaN for NMI prepare path:
    for col in feature_cols:
        if col in cat:
            vals = subset[col]
            as_int = pd.Series(pd.NA, index=vals.index, dtype="Int64")
            m = vals.notna()
            as_int.loc[m] = vals.loc[m].round().astype(int)
            subset[col] = as_int
    if decision_col in subset.columns:
        subset[decision_col] = subset[decision_col].round().astype(int)

    imputed = nmilib.non_parametric_imputation(
        subset,
        decision_col=decision_col,
        feature_cols=list(feature_cols),
    )
    out = data.copy()
    out[list(feature_cols)] = imputed[list(feature_cols)]
    return out


def impute_simple(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    strategy: str,
) -> pd.DataFrame:
    """
    strategy: 'mode' (most_frequent for cats, mean for nums),
              'mean', 'median'
    """
    out = data.copy()
    X = out[list(feature_cols)]
    cat = set(categorical_cols)
    if strategy == "mode":
        for col in feature_cols:
            col_strategy = "most_frequent" if col in cat else "mean"
            imp = SimpleImputer(strategy=col_strategy)
            out[col] = imp.fit_transform(X[[col]]).ravel()
    else:
        imp = SimpleImputer(strategy=strategy)
        out[list(feature_cols)] = imp.fit_transform(X)
    return out


def impute_knn(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    out = data.copy()
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    out[list(feature_cols)] = imp.fit_transform(out[list(feature_cols)])
    return out


def impute_mice(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    out = data.copy()
    imp = IterativeImputer(
        random_state=random_state,
        max_iter=20,
        sample_posterior=False,
    )
    out[list(feature_cols)] = imp.fit_transform(out[list(feature_cols)])
    return out


def run_benchmark(
    path: str | Path = DEFAULT_DATA,
    *,
    mv_frac: float = 0.12,
    seed: int = 42,
    nmi_root: Optional[str] = None,
    max_complete_rows: int = 300,
) -> pd.DataFrame:
    """
    Complete-case master → mask → impute with each method → accuracy table.
    """
    import time

    enc = load_and_encode(path)
    complete = enc.frame.dropna().reset_index(drop=True)
    if len(complete) > max_complete_rows:
        complete = complete.sample(n=max_complete_rows, random_state=seed).reset_index(
            drop=True
        )
    if len(complete) < 20:
        raise ValueError(f"Too few complete rows ({len(complete)}) for a benchmark")

    mv, mask = mask_features(
        complete, enc.feature_cols, mv_frac=mv_frac, seed=seed
    )
    # Float view for sklearn imputers (np.nan)
    mv_float = mv.astype(float)
    complete_float = complete.astype(float)

    methods: list[tuple[str, Any]] = [
        (
            "NM (NMI)",
            lambda d: impute_nmi(
                d,
                enc.feature_cols,
                enc.decision_col,
                nmi_root=nmi_root,
                categorical_cols=enc.categorical_cols,
            ),
        ),
        (
            "Mode/Mean",
            lambda d: impute_simple(
                d, enc.feature_cols, enc.categorical_cols, strategy="mode"
            ),
        ),
        (
            "Mean",
            lambda d: impute_simple(
                d, enc.feature_cols, enc.categorical_cols, strategy="mean"
            ),
        ),
        (
            "Median",
            lambda d: impute_simple(
                d, enc.feature_cols, enc.categorical_cols, strategy="median"
            ),
        ),
        ("kNN", lambda d: impute_knn(d, enc.feature_cols)),
        ("MICE", lambda d: impute_mice(d, enc.feature_cols, random_state=seed)),
    ]

    rows: list[MethodScore] = []
    for name, fn in methods:
        t0 = time.perf_counter()
        src = mv if name.startswith("NM") else mv_float
        imputed = fn(src.copy())
        elapsed = time.perf_counter() - t0
        master_ref = complete if name.startswith("NM") else complete_float
        # Normalize imputed to float for scoring
        imp_score = imputed.astype(float)
        acc, n_cmp, n_match, by_col = _score_imputed(
            master_ref.astype(float),
            imp_score,
            enc.feature_cols,
            mask,
            enc.categorical_cols,
        )
        rows.append(
            MethodScore(
                method=name,
                accuracy=acc,
                n_compared=n_cmp,
                n_match=n_match,
                seconds=elapsed,
                by_column=by_col,
            )
        )

    return pd.DataFrame([r.as_row() for r in rows])


def impute_natural_missing(
    path: str | Path = DEFAULT_DATA,
    *,
    method: str = "nmi",
    nmi_root: Optional[str] = None,
    out_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Fill natural MVs on the full dataset and optionally write CSV
    (decoded back to original labels where encoders exist).
    """
    enc = load_and_encode(path)
    data = enc.frame.copy()

    if method == "nmi":
        filled = impute_nmi(
            data, enc.feature_cols, enc.decision_col,
            nmi_root=nmi_root, categorical_cols=enc.categorical_cols,
        )
    elif method == "mode":
        filled = impute_simple(data, enc.feature_cols, enc.categorical_cols, strategy="mode")
    elif method == "knn":
        filled = impute_knn(data, enc.feature_cols)
    elif method == "mice":
        filled = impute_mice(data, enc.feature_cols)
    else:
        raise ValueError(method)

    # Decode categoricals back to original labels
    decoded = filled.copy()
    for col, enc_fit in enc.encoders.items():
        if col not in decoded.columns:
            continue
        vals = decoded[col]
        # round to nearest class code
        codes = vals.round().astype("Int64")
        out = pd.Series(pd.NA, index=decoded.index, dtype=object)
        known = codes.notna()
        # clip to valid classes
        classes = np.arange(len(enc_fit.classes_))
        c = codes.loc[known].clip(lower=0, upper=len(enc_fit.classes_) - 1).astype(int)
        out.loc[known] = enc_fit.inverse_transform(c.to_numpy())
        decoded[col] = out

    # Restore dropped columns from raw file
    raw = pd.read_csv(path, na_values=["", "?", "NA", "N/A"])
    for col in raw.columns:
        if col not in decoded.columns:
            decoded[col] = raw[col].values

    decoded = decoded[raw.columns]
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        decoded.to_csv(out_path, index=False)
    return decoded
