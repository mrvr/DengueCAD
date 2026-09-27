#!/usr/bin/env python3
"""
NMPrediction performance on data/dataset.csv (Bangladesh clinical set).

Pipeline (Algorithm 1):
  NMI impute → GA+ADT influential features → stratified k-fold ADT
  Compare with C4.5, SVM, LOR (paper Table I style).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.comparison import run_dataset_comparison
from denguecad.feature_selection import GAWrapperConfig
from denguecad.imputation_benchmark import DEFAULT_DATA, load_and_encode
from denguecad.nm_prediction import NMPredictionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="NMPrediction on dataset.csv")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--ga-pop", type=int, default=12)
    parser.add_argument("--ga-gen", type=int, default=8)
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "nmprediction_dataset_csv"),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated feature names to drop (e.g. IgG,NS1 for label-leak checks)",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    enc = load_and_encode(args.data)
    # Keep Int64 categoricals for fast NMI; sklearn casts via dtype=float in CV
    df = enc.frame.copy()
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    feature_cols = [c for c in enc.feature_cols if c not in exclude]
    decision_col = enc.decision_col
    if not feature_cols:
        raise SystemExit("No features left after --exclude")

    print("=== dataset.csv for NMPrediction ===", flush=True)
    print(f"rows={len(df)} features={len(feature_cols)} decision={decision_col}", flush=True)
    print(f"feature_cols={feature_cols}", flush=True)
    print(f"Outcome counts:\n{df[decision_col].value_counts().to_string()}", flush=True)
    print(f"NA cells before NM: {int(df[feature_cols].isna().sum().sum())}", flush=True)

    nm_cfg = NMPredictionConfig(
        n_folds=args.folds,
        ga=GAWrapperConfig(
            population_size=args.ga_pop,
            n_generations=args.ga_gen,
            crossover_prob=1.0,
            mutation_prob=0.001,
            n_folds=3,
            classifier="adt",
            scoring="accuracy",
            impute_missing=True,
            subset_size_penalty=0.01,
            verbose=not args.quiet,
            random_state=42,
        ),
        adt_estimators=40,
        random_state=42,
        verbose=not args.quiet,
    )

    print("\n=== Running NM + C4.5 + LOR + SVM ===", flush=True)
    out = run_dataset_comparison(
        "dataset.csv",
        df,
        feature_cols,
        decision_col,
        nm_config=nm_cfg,
        n_folds=args.folds,
        random_state=42,
        verbose=not args.quiet,
    )

    table1 = pd.DataFrame(out["table1"])
    table2 = pd.DataFrame(out["table2"])
    table3 = pd.DataFrame([out["table3"]])

    # Decode selected feature names already are encoded col names (same strings)
    print("\n=== TABLE I — Performance (NMPrediction vs baselines) ===")
    print(table1.to_string(index=False))
    print("\n=== TABLE II — Wilcoxon (NM vs others) ===")
    print(table2.to_string(index=False))
    print("\n=== TABLE III — Influential features (NM) ===")
    print(table3.to_string(index=False))

    table1.to_csv(outdir / "table1_performance.csv", index=False)
    table2.to_csv(outdir / "table2_wilcoxon.csv", index=False)
    table3.to_csv(outdir / "table3_influential_features.csv", index=False)

    nm = out["nm_result"]
    summary = {
        "dataset": "dataset.csv",
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "selected_features": ", ".join(nm.selected_features),
        "Accuracy (%)": round(100.0 * nm.performance.accuracy, 2),
        "SE": round(100.0 * nm.performance.sensitivity, 2),
        "SP": round(100.0 * nm.performance.specificity, 2),
        "AUC": round(nm.performance.auc, 2),
        "threshold": nm.performance.threshold,
    }
    pd.DataFrame([summary]).to_csv(outdir / "nmprediction_summary.csv", index=False)
    print(f"\nWrote results under {outdir}")


if __name__ == "__main__":
    main()
