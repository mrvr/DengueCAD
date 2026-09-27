"""
Load Weka / KEEL-style ARFF files (including ``NUMERIC [min,max]`` attributes).

Works for disease datasets under ``experiments/datasets/*.arff``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ATTR_RE = re.compile(
    r"^@attribute\s+(['\"]?)(.+?)\1\s+(.+)$",
    re.IGNORECASE,
)
_NOMINAL_RE = re.compile(r"\{([^}]*)\}")


def _strip_range(type_part: str) -> str:
    """Turn ``NUMERIC [0,1]`` / ``INTEGER [1,10]`` into ``NUMERIC`` / ``INTEGER``."""
    return re.sub(r"\s*\[[^\]]*\]\s*", " ", type_part).strip()


def load_arff(path: str | Path) -> pd.DataFrame:
    """
    Parse an ARFF file into a DataFrame.

    - Last attribute is treated as the decision/class column by convention.
    - ``?`` becomes NaN.
    - Nominal labels are kept as strings (caller may encode).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]

    attrs: list[tuple[str, str]] = []  # (name, kind) kind in {numeric, nominal}
    data_start = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("@data"):
            data_start = i + 1
            break
        if low.startswith("@attribute"):
            m = _ATTR_RE.match(ln)
            if not m:
                raise ValueError(f"Cannot parse attribute line: {ln!r}")
            name = m.group(2).strip()
            type_raw = _strip_range(m.group(3))
            nom = _NOMINAL_RE.search(type_raw)
            if nom:
                attrs.append((name, "nominal"))
            else:
                attrs.append((name, "numeric"))

    if data_start is None or not attrs:
        raise ValueError(f"No @attribute/@data section in {path}")

    rows = []
    for ln in lines[data_start:]:
        if not ln or ln.startswith("%"):
            continue
        # split CSV respecting possible spaces
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(attrs):
            # try tighter split
            parts = [p.strip() for p in re.split(r"\s*,\s*", ln)]
        if len(parts) != len(attrs):
            raise ValueError(
                f"{path.name}: expected {len(attrs)} fields, got {len(parts)} in {ln!r}"
            )
        row = {}
        for (name, kind), raw in zip(attrs, parts):
            if raw == "?" or raw == "":
                row[name] = np.nan
            elif kind == "numeric":
                row[name] = float(raw)
            else:
                row[name] = raw
        rows.append(row)

    df = pd.DataFrame(rows, columns=[a[0] for a in attrs])
    return df


def prepare_for_nm(
    df: pd.DataFrame,
    *,
    decision_col: Optional[str] = None,
    max_numeric_bins: int = 12,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    Encode a disease ARFF frame for NMI / NMPrediction.

    Returns (encoded_frame, feature_cols, decision_col).
    Continuous columns are quantile-binned so NMI uses the fast categorical path.
    """
    work = df.copy()
    if decision_col is None:
        decision_col = str(work.columns[-1])
    feature_cols = [c for c in work.columns if c != decision_col]

    out = pd.DataFrame(index=work.index)
    # Decision
    dec = work[decision_col]
    if dec.dtype == object or str(dec.dtype).startswith("string"):
        codes, _ = pd.factorize(dec.astype(str), sort=True)
        out[decision_col] = pd.Series(codes, index=work.index).astype("Int64")
        out.loc[dec.isna(), decision_col] = pd.NA
    else:
        # map to 0..K-1
        vals = sorted(dec.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(vals)}
        out[decision_col] = dec.map(mapping).astype("Int64")

    for col in feature_cols:
        s = work[col]
        if s.dtype == object or str(s.dtype).startswith("string"):
            codes, _ = pd.factorize(s.astype(str), sort=True)
            series = pd.Series(codes, index=work.index).astype("Int64")
            series.loc[s.isna()] = pd.NA
            out[col] = series
            continue
        num = pd.to_numeric(s, errors="coerce")
        nun = int(num.nunique(dropna=True))
        if nun <= max_numeric_bins * 2:
            # already low-cardinality (often integer flags)
            codes, _ = pd.factorize(num.round(6), sort=True)
            series = pd.Series(codes, index=work.index).astype("Int64")
            series.loc[num.isna()] = pd.NA
            out[col] = series
        else:
            try:
                binned = pd.qcut(num, q=max_numeric_bins, duplicates="drop")
            except ValueError:
                binned = pd.cut(num, bins=min(max_numeric_bins, max(nun, 2)))
            codes, _ = pd.factorize(binned.astype(str), sort=True)
            series = pd.Series(codes, index=work.index).astype("Int64")
            series.loc[num.isna()] = pd.NA
            out[col] = series

    return out[[*feature_cols, decision_col]], feature_cols, decision_col
