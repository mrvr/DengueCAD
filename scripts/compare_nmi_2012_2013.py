#!/usr/bin/env python3
"""
Compare the 2012 and 2013 NMI procedures on the two attached dengue datasets.

Protocol (same design as the hematological check):
  * draw 20% of the rows that still have a ground-truth value (seed 42);
  * hide 1–3 random feature cells in each drawn row;
  * impute with both paper methods;
  * score only the hidden cells.

Categorical and whole-number columns are correct only on an exact match.
Fractional columns are also reported as "within 10%" relative error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from denguecad.holdout_experiment import inject_record_mcar
from denguecad.nmi_paper import YEAR_2012, YEAR_2013, column_kind, impute_nmi

ROOT = Path(__file__).resolve().parents[1]
ATTACH = Path(
    "/home/raghuvrdhan/.cursor/projects/"
    "home-raghuvrdhan-Documents-Raghuvardhan-research-New-Method-Imputation-DengueCAD/"
    "attachments/15b930f6-326d-4ce6-a78b-8ce11bf6abbc"
)
OUT = ROOT / "results" / "nmi_2012_vs_2013"
SEED = 42
ROW_FRAC = 0.20


def _load(name: str, decision: str) -> tuple[pd.DataFrame, str]:
    path = ATTACH / name
    if not path.is_file():
        raise SystemExit(f"Missing dataset: {path}")
    frame = pd.read_csv(path)
    if decision not in frame.columns:
        raise SystemExit(f"{name} has no decision column {decision}")
    return frame, decision


def _mask(frame: pd.DataFrame, decision: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    features = [c for c in frame.columns if c != decision]
    complete = frame[features].notna().all(axis=1) & frame[decision].notna()
    candidates = frame.index[complete]
    n_select = max(1, int(round(ROW_FRAC * len(frame))))
    n_select = min(n_select, len(candidates))
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(candidates.to_numpy(), size=n_select, replace=False))
    masked, mask = inject_record_mcar(
        frame, features, selected, row_frac=1.0, max_attrs=3, seed=SEED
    )
    return masked, mask, pd.Index(selected)


def _cell_rows(truth: pd.DataFrame, imputed: pd.DataFrame, mask: pd.DataFrame, decision: str) -> pd.DataFrame:
    rows = []
    for col in mask.columns:
        kind = column_kind(truth[col])
        for idx in mask.index[mask[col]]:
            original = truth.at[idx, col]
            predicted = imputed.at[idx, col]
            if kind == "categorical":
                exact = pd.notna(predicted) and str(predicted) == str(original)
                rel = 0.0 if exact else np.nan
                abs_err = np.nan
            else:
                original_f = float(original)
                predicted_f = float(predicted) if pd.notna(predicted) else np.nan
                exact = bool(np.isfinite(predicted_f) and predicted_f == original_f)
                abs_err = abs(predicted_f - original_f) if np.isfinite(predicted_f) else np.nan
                if kind == "integer":
                    rel = 0.0 if exact else (
                        abs_err / abs(original_f) if original_f != 0 and np.isfinite(abs_err) else np.nan
                    )
                else:
                    rel = (
                        abs_err / abs(original_f)
                        if original_f != 0 and np.isfinite(abs_err)
                        else (0.0 if exact else np.nan)
                    )
            close = exact if kind != "real" else bool(np.isfinite(rel) and rel <= 0.10)
            rows.append({
                "record_id": int(idx) if not isinstance(idx, str) else idx,
                "feature": col,
                "kind": kind,
                "original_value": original,
                "imputed_value": predicted,
                "exact_match": int(exact),
                "within_10pct": int(close) if kind == "real" else int(exact),
                "abs_error": abs_err,
                "rel_error": rel,
            })
    return pd.DataFrame(rows)


def _summarise(cells: pd.DataFrame) -> dict:
    n = len(cells)
    discrete = cells[cells["kind"] != "real"]
    real = cells[cells["kind"] == "real"]
    scored = int(discrete["exact_match"].sum() + real["within_10pct"].sum()) if n else 0
    return {
        "masked_cells": n,
        "discrete_cells": int(len(discrete)),
        "discrete_exact": int(discrete["exact_match"].sum()) if len(discrete) else 0,
        "discrete_exact_pct": round(100 * discrete["exact_match"].mean(), 2) if len(discrete) else None,
        "real_cells": int(len(real)),
        "real_exact": int(real["exact_match"].sum()) if len(real) else 0,
        "real_within_10pct": int(real["within_10pct"].sum()) if len(real) else 0,
        "real_within_10pct_pct": round(100 * real["within_10pct"].mean(), 2) if len(real) else None,
        "real_mae": round(float(real["abs_error"].mean()), 4) if len(real) else None,
        "combined_hits": scored,
        "combined_pct": round(100 * scored / n, 2) if n else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("hematological", "Dengue_Fever_Hematological_Dataset.csv", "Result"),
        ("diseases", "Dengue_diseases_dataset_modified__1_.csv", "dengue_label"),
    ]
    summaries = []
    feature_rows = []
    for label, filename, decision in datasets:
        truth, decision = _load(filename, decision)
        masked, mask, selected = _mask(truth, decision)
        kinds = {c: column_kind(truth[c]) for c in mask.columns}
        print(f"\n=== {label}: {truth.shape[0]} rows, {len(selected)} masked rows, "
              f"{int(mask.to_numpy().sum())} cells ===")
        print("kinds:", kinds)
        for year in (YEAR_2012, YEAR_2013):
            imputed = impute_nmi(masked, decision, year=year)
            cells = _cell_rows(truth, imputed, mask, decision)
            cells.insert(0, "dataset", label)
            cells.insert(1, "method", year)
            cells.to_csv(OUT / f"{label}_{year}_cells.csv", index=False)
            summary = _summarise(cells)
            summary.update({
                "dataset": label,
                "method": year,
                "rows": int(len(truth)),
                "masked_rows": int(len(selected)),
            })
            summaries.append(summary)
            print(f"  {year}: combined {summary['combined_pct']}%  "
                  f"discrete exact {summary['discrete_exact_pct']}%  "
                  f"real within 10% {summary['real_within_10pct_pct']}%")
            for feature, group in cells.groupby("feature"):
                feature_rows.append({
                    "dataset": label,
                    "method": year,
                    "feature": feature,
                    "kind": group["kind"].iloc[0],
                    "n": int(len(group)),
                    "exact": int(group["exact_match"].sum()),
                    "exact_pct": round(100 * group["exact_match"].mean(), 2),
                    "within_10pct": int(group["within_10pct"].sum()),
                    "within_10pct_pct": round(100 * group["within_10pct"].mean(), 2),
                    "mae": round(float(group["abs_error"].mean()), 4) if group["kind"].iloc[0] != "categorical" else None,
                })
    summary_df = pd.DataFrame(summaries)
    feature_df = pd.DataFrame(feature_rows)
    summary_df.to_csv(OUT / "summary.csv", index=False)
    feature_df.to_csv(OUT / "per_feature.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
