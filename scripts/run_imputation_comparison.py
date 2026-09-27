#!/usr/bin/env python3
"""
Apply NMI missing-value imputation on data/dataset.csv and compare with
Mode/Mean, Mean, Median, kNN, and MICE.

1) Controlled benchmark on complete cases (mask → impute → accuracy vs truth)
2) Fill natural MVs (Joint_Pain) with NMI and write results CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.imputation_benchmark import (
    DEFAULT_DATA,
    impute_natural_missing,
    load_and_encode,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="NMI vs other MV imputation methods")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--mv-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=250, help="Complete-case subsample for timed benchmark")
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "imputation_dataset_csv"),
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import sys as _sys
    print("=== data/dataset.csv missing values (raw) ===", flush=True)
    enc = load_and_encode(args.data)
    raw_na = (
        __import__("pandas")
        .read_csv(args.data, na_values=["", "?", "NA", "N/A"])
        .isna()
        .sum()
    )
    print(raw_na[raw_na > 0].to_string() if raw_na.sum() else "  (none)", flush=True)
    print(f"features used: {enc.feature_cols}", flush=True)
    print(f"decision: {enc.decision_col}", flush=True)
    print(f"complete-case rows: {enc.frame.dropna().shape[0]} / {len(enc.frame)}", flush=True)

    print("\n=== Imputation accuracy comparison (masked complete cases) ===", flush=True)
    table = run_benchmark(
        args.data, mv_frac=args.mv_frac, seed=args.seed, max_complete_rows=args.max_rows
    )
    print(table.to_string(index=False))
    table_path = outdir / "imputation_comparison.csv"
    table.to_csv(table_path, index=False)

    print("\n=== Fill natural MVs with NMI ===")
    imputed_path = outdir / "dataset_nmi_imputed.csv"
    filled = impute_natural_missing(
        args.data, method="nmi", out_path=imputed_path
    )
    still = filled.isna().sum().sum()
    print(f"Wrote {imputed_path}")
    print(f"Remaining NA cells after NMI: {int(still)}")
    if "Joint_Pain" in filled.columns:
        print("Joint_Pain value counts after NMI:")
        print(filled["Joint_Pain"].value_counts(dropna=False).to_string())

    print(f"\nResults under {outdir}")


if __name__ == "__main__":
    main()
