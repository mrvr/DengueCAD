#!/usr/bin/env python3
"""
Run NMPrediction on multiple disease ARFF datasets (not dengue-only).

Compares NM vs C4.5 / LOR / SVM for each file and writes aggregate tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.arff_io import load_arff, prepare_for_nm
from denguecad.comparison import run_dataset_comparison
from denguecad.feature_selection import GAWrapperConfig
from denguecad.nm_prediction import NMPredictionConfig

DEFAULT_DIR = Path(
    "/home/raghuvrdhan/Documents/Raghuvardhan/research/New Method Imputation/experiments/datasets"
)
DEFAULT_FILES = [
    "appendicitis.arff",
    "bupa.arff",
    "contractions.arff",
    "heart.arff",
    "hepatitis.arff",
    "laryngeal1.arff",
    "lplaudiob.arff",
    "pima.arff",
    "rds.arff",
    "weaning.arff",
    "wisconsin.arff",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-disease NMPrediction ARFF benchmark")
    parser.add_argument("--datadir", default=str(DEFAULT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ga-pop", type=int, default=8)
    parser.add_argument("--ga-gen", type=int, default=5)
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "arff_disease_comparison"),
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    args = parser.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_t1: list[pd.DataFrame] = []
    all_t2: list[pd.DataFrame] = []
    all_t3: list[dict] = []
    summary_rows: list[dict] = []

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
            verbose=False,
            random_state=42,
        ),
        adt_estimators=30,
        random_state=42,
        verbose=False,
    )

    for name in DEFAULT_FILES:
        path = datadir / name
        if not path.is_file():
            print(f"SKIP missing {path}", flush=True)
            continue
        ds_name = path.stem
        print(f"\n===== {ds_name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            raw = load_arff(path)
            encoded, feature_cols, decision_col = prepare_for_nm(raw)
            print(
                f"n={len(encoded)} p={len(feature_cols)} na={int(raw.isna().sum().sum())} "
                f"class={decision_col} "
                f"dist={encoded[decision_col].value_counts().to_dict()}",
                flush=True,
            )
            out = run_dataset_comparison(
                ds_name,
                encoded,
                feature_cols,
                decision_col,
                nm_config=nm_cfg,
                n_folds=args.folds,
                random_state=42,
                verbose=False,
            )
        except Exception as e:
            print(f"FAILED {ds_name}: {e}", flush=True)
            summary_rows.append({"Dataset": ds_name, "status": f"error: {e}"})
            continue

        elapsed = time.perf_counter() - t0
        t1 = pd.DataFrame(out["table1"])
        t2 = pd.DataFrame(out["table2"])
        t3 = out["table3"]
        all_t1.append(t1)
        all_t2.append(t2)
        all_t3.append(t3)

        nm_row = t1[t1["Method"] == "NM"].iloc[0]
        print(
            f"NM Acc={nm_row['Accuracy (%)']} SE={nm_row['SE']} "
            f"SP={nm_row['SP']} AUC={nm_row['AUC']} "
            f"features={t3['features identified']} ({elapsed:.1f}s)",
            flush=True,
        )
        # per-method wide summary
        wide = {"Dataset": ds_name, "n": len(encoded), "p": len(feature_cols), "seconds": round(elapsed, 2)}
        for _, r in t1.iterrows():
            m = r["Method"]
            wide[f"{m}_Acc"] = r["Accuracy (%)"]
            wide[f"{m}_AUC"] = r["AUC"]
        wide["NM_features"] = t3["features identified"]
        wide["status"] = "ok"
        summary_rows.append(wide)

        t1.to_csv(outdir / f"{ds_name}_table1.csv", index=False)
        t2.to_csv(outdir / f"{ds_name}_table2.csv", index=False)
        pd.DataFrame([t3]).to_csv(outdir / f"{ds_name}_table3.csv", index=False)

    if all_t1:
        table1 = pd.concat(all_t1, ignore_index=True)
        table2 = pd.concat(all_t2, ignore_index=True)
        table3 = pd.DataFrame(all_t3)
        table1.to_csv(outdir / "ALL_table1_performance.csv", index=False)
        table2.to_csv(outdir / "ALL_table2_wilcoxon.csv", index=False)
        table3.to_csv(outdir / "ALL_table3_influential_features.csv", index=False)

        print("\n======== ALL TABLE I (NM vs C4.5 / LOR / SVM) ========", flush=True)
        print(table1.to_string(index=False), flush=True)
        print("\n======== ALL TABLE III (influential features) ========", flush=True)
        print(table3.to_string(index=False), flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary_wide.csv", index=False)
    print(f"\nWrote results under {outdir}", flush=True)


if __name__ == "__main__":
    main()
