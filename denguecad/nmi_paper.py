"""
NMI imputation from the two Rao & Kumar papers, implemented from the text.

2012 — IEEE TITB 16(1), Section III-A/B:
  same-class index uses min of the two frequency ratios (categorical / integer)
  or min of value/mean (real); different-class index uses max of those ratios.
  Neighbours are donors with z-score <= 0. Categorical and integer cells are
  filled with the mode; real cells with the arithmetic mean.
  Only fully complete records are donors. Case II "first column" is read as
  the column being scored (the l-th column), matching Case I.

2013 — IEEE JBHI 17(1), Section III-B, which cites the 2012 paper and changes
  the index, the neighbour score, and the real-valued fill:
  same-class categorical/integer index is min(count/γ, count/γ);
  different-class categorical index is max(β/(β+δ), δ/(β+δ));
  real columns use a skewness-tail count instead of a mean ratio.
  Neighbours have median-based score α <= 0.
  Real cells are an inverse-distance weighted mean of the neighbour values.
  The 2013 text writes that weighted sum as Σ P(i) W(i) but does not define
  P(i); P(i) is taken here as the neighbour's observed value and the weights
  as inverse distance among the selected neighbours.

Rows whose decision is missing are not imputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR_2012 = 2012
YEAR_2013 = 2013


@dataclass(frozen=True)
class _Column:
    name: str
    kind: str  # "categorical" | "integer" | "real"
    values: np.ndarray  # object for categorical, float otherwise
    observed: np.ndarray  # bool


def column_kind(series: pd.Series) -> str:
    """Paper type: categorical, integer (whole numbers), or real (fractional)."""
    if (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return "categorical"
    num = pd.to_numeric(series, errors="coerce")
    observed = num.dropna()
    if observed.empty:
        return "categorical"
    if np.all(np.isclose(observed.to_numpy(dtype=float) % 1.0, 0.0)):
        return "integer"
    return "real"


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _skewness(values: np.ndarray) -> float:
    """Moment skewness m3 / m2^(3/2). Undefined or constant columns return 0."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    centered = v - v.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0.0:
        return 0.0
    m3 = float(np.mean(centered**3))
    return m3 / (m2 ** 1.5)


def _mode(values: np.ndarray):
    vals = values[pd.notna(values)] if values.dtype == object else values[np.isfinite(values.astype(float))]
    if vals.size == 0:
        return np.nan
    uniq, counts = np.unique(vals, return_counts=True)
    return uniq[int(np.argmax(counts))]


def _z_neighbours(distances: np.ndarray) -> np.ndarray:
    """Donors with z(d) <= 0 (2012). All donors if the spread is zero."""
    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return np.array([], dtype=int)
    if d.size == 1:
        return np.array([0], dtype=int)
    mean = float(d.mean())
    var = float(np.sum((d - mean) ** 2) / (d.size - 1))
    if var <= 0.0:
        return np.arange(d.size, dtype=int)
    z = (d - mean) / np.sqrt(var)
    nn = np.flatnonzero(z <= 0.0)
    if nn.size == 0:
        return np.array([int(np.argmin(d))], dtype=int)
    return nn


def _median_neighbours(distances: np.ndarray) -> np.ndarray:
    """Donors with α(d) <= 0 (2013 median score)."""
    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return np.array([], dtype=int)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad <= 0.0:
        nn = np.flatnonzero(d <= med)
        return nn if nn.size else np.array([int(np.argmin(d))], dtype=int)
    alpha = (d - med) / mad
    nn = np.flatnonzero(alpha <= 0.0)
    if nn.size == 0:
        return np.array([int(np.argmin(d))], dtype=int)
    return nn


def _inv_distance_mean(values: np.ndarray, distances: np.ndarray) -> float:
    """Inverse-distance weighted mean. Zero-distance neighbours are averaged."""
    v = np.asarray(values, dtype=float)
    dist = np.asarray(distances, dtype=float)
    ok = np.isfinite(v) & np.isfinite(dist)
    v, dist = v[ok], dist[ok]
    if v.size == 0:
        return float("nan")
    zeros = dist <= 0.0
    if np.any(zeros):
        return float(np.mean(v[zeros]))
    weights = 1.0 / dist
    return float(np.sum(weights * v) / np.sum(weights))


def _safe_min_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.zeros(np.broadcast(left, right).shape, dtype=float)
    ok = (left > 0.0) & (right > 0.0)
    ratio_lr = np.divide(left, right, out=np.zeros_like(out), where=ok)
    ratio_rl = np.divide(right, left, out=np.zeros_like(out), where=ok)
    out[ok] = np.minimum(ratio_lr, ratio_rl)[ok]
    return out


def _safe_max_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.zeros(np.broadcast(left, right).shape, dtype=float)
    ok = (left > 0.0) & (right > 0.0)
    ratio_lr = np.divide(left, right, out=np.zeros_like(out), where=ok)
    ratio_rl = np.divide(right, left, out=np.zeros_like(out), where=ok)
    out[ok] = np.maximum(ratio_lr, ratio_rl)[ok]
    return out


def _tail_counts(sorted_values: np.ndarray, points: np.ndarray, skew: float) -> np.ndarray:
    """Vectorised skewed-side counts. ``sorted_values`` must be sorted."""
    points = np.asarray(points, dtype=float)
    if sorted_values.size == 0:
        return np.zeros(points.shape, dtype=float)
    n_le = np.searchsorted(sorted_values, points, side="right").astype(float)
    if skew < 0.0:
        return n_le
    return float(sorted_values.size) - n_le


def _prepare(data: pd.DataFrame, feature_cols: list[str], decision_col: str):
    decision = data[decision_col]
    dec_observed = decision.notna().to_numpy()
    dec_codes, _ = pd.factorize(decision.astype("string"), use_na_sentinel=True)
    columns: list[_Column] = []
    for name in feature_cols:
        series = data[name]
        kind = column_kind(series)
        if kind == "categorical":
            values = series.astype("object").where(series.notna(), np.nan).to_numpy()
            observed = series.notna().to_numpy()
        else:
            values = _as_float(series)
            observed = np.isfinite(values)
        columns.append(_Column(name, kind, values, observed))
    complete = dec_observed.copy()
    for col in columns:
        complete &= col.observed
    return dec_codes, dec_observed, columns, complete


def _index_matrix(
    columns: list[_Column],
    dec_codes: np.ndarray,
    donor_idx: np.ndarray,
    *,
    year: int,
) -> dict[str, dict]:
    """Precompute the counts and means the pairwise index reads."""
    stats: dict[str, dict] = {}
    n_class = int(dec_codes.max()) + 1 if dec_codes.size and dec_codes.max() >= 0 else 0
    for col in columns:
        if col.kind == "real" and year == YEAR_2013:
            observed_vals = col.values[col.observed]
            by_class_sorted = {}
            by_class_skew = {}
            for c in range(n_class):
                mask = col.observed & (dec_codes == c)
                vals = np.sort(col.values[mask])
                by_class_sorted[c] = vals
                by_class_skew[c] = _skewness(vals)
            stats[col.name] = {
                "kind": "real",
                "skew_all": _skewness(observed_vals),
                "sorted_all": np.sort(observed_vals),
                "gamma_all": int(observed_vals.size),
                "by_class_sorted": by_class_sorted,
                "by_class_skew": by_class_skew,
            }
            continue

        if col.kind == "real":
            means = np.full(max(n_class, 1), np.nan, dtype=float)
            donor = np.zeros(len(dec_codes), dtype=bool)
            donor[donor_idx] = True
            for c in range(n_class):
                mask = donor & col.observed & (dec_codes == c)
                vals = col.values[mask]
                if vals.size:
                    means[c] = float(vals.mean())
            stats[col.name] = {"kind": "real", "mean": means}
            continue

        # Codes cover every observed row. Counts use complete donors (2012)
        # or every observed row (2013).
        observed = col.observed & (dec_codes >= 0)
        codes = np.full(len(col.values), -1, dtype=int)
        if np.any(observed):
            if col.kind == "integer":
                tokens = np.rint(col.values[observed]).astype(int).astype(str)
            else:
                tokens = col.values[observed].astype(str)
            _uniq, inv = np.unique(tokens, return_inverse=True)
            codes[observed] = inv
            n_val = int(_uniq.size)
        else:
            n_val = 0
        if year == YEAR_2012:
            pool = np.zeros(len(dec_codes), dtype=bool)
            pool[donor_idx] = True
            pool &= observed
        else:
            pool = observed
        counts = np.zeros((max(n_class, 1), max(n_val, 1)), dtype=float)
        gamma = np.zeros(max(n_class, 1), dtype=float)
        if np.any(pool) and n_val:
            np.add.at(counts, (dec_codes[pool], codes[pool]), 1.0)
            np.add.at(gamma, dec_codes[pool], 1.0)
        stats[col.name] = {
            "kind": col.kind,
            "codes": codes,
            "counts": counts,
            "gamma": gamma,
        }
    return stats


def _pair_index(
    col: _Column,
    stat: dict,
    t_idx: int,
    donor_idx: np.ndarray,
    dec_codes: np.ndarray,
    *,
    year: int,
) -> np.ndarray:
    """I_l(target, donor) for one column. 0 where either side is unobserved."""
    n = donor_idx.size
    out = np.zeros(n, dtype=float)
    if not col.observed[t_idx]:
        return out
    d_obs = col.observed[donor_idx]
    if not np.any(d_obs):
        return out
    t_class = int(dec_codes[t_idx])
    d_class = dec_codes[donor_idx]
    same = d_class == t_class

    if col.kind == "real" and year == YEAR_2012:
        t_val = float(col.values[t_idx])
        d_val = col.values[donor_idx]
        means = stat["mean"]
        a_hash = float(means[t_class]) if t_class >= 0 else float("nan")
        same_obs = d_obs & same
        if same_obs.any() and np.isfinite(a_hash) and a_hash != 0.0:
            out[same_obs] = np.minimum(t_val / a_hash, d_val[same_obs] / a_hash)
        diff = np.flatnonzero(d_obs & ~same & (d_class >= 0))
        if diff.size and np.isfinite(a_hash):
            q_hash = means[d_class[diff]]
            lam = np.minimum(a_hash, q_hash)
            ok = np.isfinite(lam) & (lam != 0.0) & np.isfinite(d_val[diff])
            use = diff[ok]
            out[use] = np.maximum(t_val / lam[ok], d_val[use] / lam[ok])
        return out

    if col.kind == "real" and year == YEAR_2013:
        t_val = float(col.values[t_idx])
        d_val = col.values[donor_idx]
        gamma = float(stat["gamma_all"])
        if gamma <= 0.0:
            return out
        tau = float(_tail_counts(stat["sorted_all"], np.array([t_val]), stat["skew_all"])[0])
        rho = _tail_counts(stat["sorted_all"], d_val, stat["skew_all"])
        same_obs = d_obs & same & np.isfinite(d_val)
        out[same_obs] = np.minimum(tau / gamma, rho[same_obs] / gamma)
        t_sorted = stat["by_class_sorted"].get(t_class, np.array([]))
        t_skew = stat["by_class_skew"].get(t_class, 0.0)
        beta_t = float(_tail_counts(t_sorted, np.array([t_val]), t_skew)[0])
        for c_j in np.unique(d_class[d_obs & ~same]):
            if c_j < 0:
                continue
            members = np.flatnonzero(d_obs & ~same & (d_class == c_j) & np.isfinite(d_val))
            q_sorted = stat["by_class_sorted"].get(int(c_j), np.array([]))
            q_skew = stat["by_class_skew"].get(int(c_j), 0.0)
            lam = float(t_sorted.size + q_sorted.size)
            if lam <= 0.0 or members.size == 0:
                continue
            delta = _tail_counts(q_sorted, d_val[members], q_skew)
            out[members] = np.minimum(beta_t / lam, delta / lam)
        return out

    # categorical / integer
    codes = stat["codes"]
    t_code = int(codes[t_idx])
    if t_code < 0 or t_class < 0:
        return out
    d_codes = codes[donor_idx]
    counts = stat["counts"]
    valid = d_obs & (d_codes >= 0) & (d_class >= 0)
    c_k = np.zeros(n, dtype=float)
    c_k[valid] = counts[d_class[valid], d_codes[valid]]
    c_i = float(counts[t_class, t_code])
    if year == YEAR_2012:
        out[valid & same] = _safe_min_ratio(c_i, c_k[valid & same])
        out[valid & ~same] = _safe_max_ratio(c_i, c_k[valid & ~same])
        return out
    g = float(stat["gamma"][t_class])
    same_obs = valid & same
    if g > 0.0 and np.any(same_obs):
        out[same_obs] = np.minimum(c_i / g, c_k[same_obs] / g)
    diff = valid & ~same
    if np.any(diff):
        denom = c_i + c_k[diff]
        ok = denom > 0.0
        chosen = np.zeros(diff.sum(), dtype=float)
        chosen[ok] = np.maximum(c_i / denom[ok], c_k[diff][ok] / denom[ok])
        out[diff] = chosen
    return out


def impute_nmi(
    data: pd.DataFrame,
    decision_col: str,
    *,
    year: int = YEAR_2012,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fill missing feature values using the 2012 or 2013 NMI procedure."""
    if year not in (YEAR_2012, YEAR_2013):
        raise ValueError(f"year must be 2012 or 2013, got {year}")
    feature_cols = list(feature_cols or [c for c in data.columns if c != decision_col])
    out = data.copy()
    dec_codes, dec_observed, columns, complete = _prepare(out, feature_cols, decision_col)
    target_idx = np.flatnonzero(dec_observed & ~complete)
    if year == YEAR_2012:
        donor_idx = np.flatnonzero(complete)
    else:
        donor_idx = np.flatnonzero(dec_observed)
    if target_idx.size == 0 or donor_idx.size == 0:
        return out

    stats = _index_matrix(columns, dec_codes, donor_idx, year=year)
    col_by_name = {c.name: c for c in columns}

    for t_idx in target_idx:
        donors = donor_idx[donor_idx != t_idx]
        if donors.size == 0:
            continue
        dist2 = np.zeros(donors.size, dtype=float)
        usable = np.zeros(donors.size, dtype=bool)
        for col in columns:
            both = col.observed[t_idx] & col.observed[donors]
            if not np.any(both):
                continue
            idx = _pair_index(col, stats[col.name], int(t_idx), donors, dec_codes, year=year)
            dist2 += np.where(both, idx**2, 0.0)
            usable |= both
        if not np.any(usable):
            continue
        distances = np.sqrt(dist2)
        distances = np.where(usable, distances, np.inf)
        finite = np.isfinite(distances)
        if not np.any(finite):
            continue
        # Neighbour scores use only donors that share at least one observed column.
        local = distances[finite]
        pick = _z_neighbours(local) if year == YEAR_2012 else _median_neighbours(local)
        nn = donors[np.flatnonzero(finite)[pick]]
        nn_dist = distances[finite][pick]

        for col in columns:
            if col.observed[t_idx]:
                continue
            nn_obs = nn[col.observed[nn]]
            if nn_obs.size == 0:
                continue
            if col.kind in ("categorical", "integer"):
                filled = _mode(col.values[nn_obs])
                if col.kind == "integer" and pd.notna(filled):
                    filled = int(np.rint(float(filled)))
                out.at[out.index[t_idx], col.name] = filled
            elif year == YEAR_2012:
                out.at[out.index[t_idx], col.name] = float(np.nanmean(col.values[nn_obs]))
            else:
                nn_local = np.flatnonzero(col.observed[nn])
                out.at[out.index[t_idx], col.name] = _inv_distance_mean(
                    col.values[nn[nn_local]], nn_dist[nn_local]
                )
    return out
