#!/usr/bin/env python3
"""
NMPrediction + C4.5 / SVM / LOR comparison on Dengue_clinical_dataset.csv.
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
from denguecad.imputation_benchmark import load_and_encode
from denguecad.nm_prediction import NMPredictionConfig

DEFAULT = ROOT / "data" / "Dengue_clinical_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--ga-pop", type=int, default=12)
    parser.add_argument("--ga-gen", type=int, default=8)
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "nmprediction_clinical"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    enc = load_and_encode(
        args.data,
        decision_col="Outcome",
        drop_features={"Id"},
    )
    df = enc.frame.copy()
    feature_cols = list(enc.feature_cols)
    decision_col = enc.decision_col

    print("=== Dengue_clinical_dataset.csv ===", flush=True)
    print(f"rows={len(df)} features={len(feature_cols)}", flush=True)
    print(f"features={feature_cols}", flush=True)
    print(f"Outcome:\n{df[decision_col].value_counts().to_string()}", flush=True)
    print(f"NA cells: {int(df[feature_cols].isna().sum().sum())}", flush=True)

    # Quick leak check
    for c in feature_cols:
        try:
            pred = df.groupby(c)[decision_col].transform(lambda s: s.mode().iloc[0])
            acc = float((pred == df[decision_col]).mean())
            if acc >= 0.95:
                print(f"NOTE: {c} alone majority-acc={acc:.3f} vs Outcome", flush=True)
        except Exception:
            pass

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

    print("\n=== NMPrediction vs C4.5 / LOR / SVM ===", flush=True)
    out = run_dataset_comparison(
        "Dengue_clinical",
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

    print("\n=== TABLE I — Performance comparison ===")
    print(table1.to_string(index=False))
    print("\n=== TABLE II — Wilcoxon (NM vs others) ===")
    print(table2.to_string(index=False))
    print("\n=== TABLE III — Influential features (NM) ===")
    print(table3.to_string(index=False))

    table1.to_csv(outdir / "table1_performance.csv", index=False)
    table2.to_csv(outdir / "table2_wilcoxon.csv", index=False)
    table3.to_csv(outdir / "table3_influential_features.csv", index=False)

    nm = out["nm_result"]
    pd.DataFrame(
        [
            {
                "dataset": "Dengue_clinical_dataset.csv",
                "n_rows": len(df),
                "n_features": len(feature_cols),
                "selected_features": ", ".join(nm.selected_features),
                "Accuracy (%)": round(100.0 * nm.performance.accuracy, 2),
                "SE": round(100.0 * nm.performance.sensitivity, 2),
                "SP": round(100.0 * nm.performance.specificity, 2),
                "AUC": round(nm.performance.auc, 2),
            }
        ]
    ).to_csv(outdir / "nmprediction_summary.csv", index=False)
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
