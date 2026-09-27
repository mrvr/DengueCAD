#!/usr/bin/env python3
"""
Run NMPrediction vs C4.5 / SVM / LOR on dengue.csv and emit paper-style tables.

Data protocol:
  - 70_000 rows → one training dataset for NMI / NMPrediction
  - 10_000 complete-case rows → validation
  - 20_000 complete-case rows → test (performance)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.comparison import run_full_comparison
from denguecad.data_splits import prepare_experiment_splits
from denguecad.feature_selection import GAWrapperConfig
from denguecad.nm_prediction import NMPredictionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="NMPrediction comparison experiment")
    parser.add_argument("--data", default=str(ROOT / "data" / "dengue.csv"))
    parser.add_argument("--train-rows", type=int, default=70_000)
    parser.add_argument("--test-rows", type=int, default=20_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--ga-pop", type=int, default=10)
    parser.add_argument("--ga-gen", type=int, default=6)
    parser.add_argument("--outdir", default=str(ROOT / "results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    splits = prepare_experiment_splits(
        args.data,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        validation_rows=args.validation_rows,
    )
    print("=== Split summary ===", flush=True)
    print(splits.summary().to_string(index=False), flush=True)

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
        ),
        verbose=not args.quiet,
    )

    tables = run_full_comparison(
        splits,
        nm_config=nm_cfg,
        n_folds=args.folds,
        verbose=not args.quiet,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n=== TABLE I — Performance comparison (NM vs C4.5, LOR, SVM) ===", flush=True)
    print(tables["table1"].to_string(index=False), flush=True)
    tables["table1"].to_csv(outdir / "table1_performance.csv", index=False)

    print("\n=== TABLE II — Wilcoxon matched-pairs rank sum test (NM vs others) ===", flush=True)
    print(tables["table2"].to_string(index=False), flush=True)
    tables["table2"].to_csv(outdir / "table2_wilcoxon.csv", index=False)

    print("\n=== TABLE III — Influential feature subsets identified by NM ===", flush=True)
    print(tables["table3"].to_string(index=False), flush=True)
    tables["table3"].to_csv(outdir / "table3_influential_features.csv", index=False)

    print("\n=== Hold-out validation (10k) / test (20k) ===", flush=True)
    print(tables["holdout"].to_string(index=False), flush=True)
    tables["holdout"].to_csv(outdir / "holdout_metrics.csv", index=False)

    splits.summary().to_csv(outdir / "split_summary.csv", index=False)
    print(f"\nCSV tables written under {outdir}", flush=True)


if __name__ == "__main__":
    main()
