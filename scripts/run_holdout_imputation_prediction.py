#!/usr/bin/env python3
"""
Imputation accuracy + NMPrediction hold-out accuracy on dengue and multi-disease data.

Protocol (per dataset, per repeat; see denguecad/holdout_experiment.py):
  valid_set = complete-case records (kept aside)
  20% of valid records get 1–3 attributes masked → impute with NMI and baselines
  imputation accuracy = exact recovery of the masked cells vs valid_set
  30% of valid records held out; NMI-imputed dataset minus those = train_set
  decision removed on 20% of the held-out records → predicted by NMPrediction,
  C4.5, LOR, SVM (trained on train_set) → compared with valid_set decisions

Writes results/holdout_imputation_prediction/{per_repeat,summary}.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.arff_io import load_arff
from denguecad.holdout_experiment import IMPUTERS, PREDICTORS, encode_ordinal, run_repeat

DATA = ROOT / "data"
ARFF_DIR = Path(
    "/home/raghuvrdhan/Documents/Raghuvardhan/research/New Method Imputation/experiments/datasets"
)
ARFF_FILES = [
    "appendicitis", "bupa", "contractions", "heart", "hepatitis", "laryngeal1",
    "lplaudiob", "pima", "rds", "weaning", "wisconsin",
]


def dengue_datasets(dengue_rows: int):
    hema = glob.glob(str(DATA / "Dengue Fever Hematology" / "**" / "Dengue-Dataset.csv"), recursive=True)
    specs = [
        (f"dengue.csv ({dengue_rows // 1000}k)",
         lambda: pd.read_csv(DATA / "dengue.csv", nrows=dengue_rows, na_values=["", "?"]),
         "Dengue", ["Name"]),
        ("dataset.csv (Bangladesh)",
         lambda: pd.read_csv(DATA / "dataset.csv", na_values=["", "?"]),
         "Outcome", ["Area", "District"]),
        ("Dengue_clinical",
         lambda: pd.read_csv(DATA / "Dengue_clinical_dataset.csv", na_values=["", "?"]),
         "Outcome", ["Id"]),
        ("Dengue_diseases (CBC)",
         lambda: pd.read_csv(DATA / "archive" / "Dengue_diseases_dataset_modified.csv", na_values=["", "?"]),
         "dengue_label", []),
    ]
    if hema:
        specs.append(("Dengue hematology",
                      lambda: pd.read_csv(hema[0], na_values=["", "?"]), "Result", []))
    for name, loader, dec, drop in specs:
        yield encode_ordinal(loader(), name=name, decision_col=dec, drop=drop)


def multi_disease_datasets():
    for f in ARFF_FILES:
        path = ARFF_DIR / f"{f}.arff"
        if path.is_file():
            yield encode_ordinal(load_arff(path), name=f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dengue-rows", type=int, default=70_000)
    parser.add_argument("--dengue-csv-repeats", type=int, default=3,
                        help="Repeats for the large dengue.csv sample")
    parser.add_argument("--only", choices=["dengue", "multi"], default=None)
    parser.add_argument("--datasets", nargs="*", help="Run only these dataset names")
    parser.add_argument("--outdir", default=str(ROOT / "results" / "holdout_imputation_prediction"))
    args = parser.parse_args()

    groups = []
    if args.only in (None, "dengue"):
        groups.append(("Dengue", dengue_datasets(args.dengue_rows)))
    if args.only in (None, "multi"):
        groups.append(("Multi-disease", multi_disease_datasets()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_repeat_path = outdir / "per_repeat.csv"
    rows: list[dict] = []
    if per_repeat_path.is_file() and args.datasets:
        prev = pd.read_csv(per_repeat_path)
        rows = prev[~prev["Dataset"].isin(args.datasets)].to_dict("records")

    for group, datasets in groups:
        for ds in datasets:
            if args.datasets and ds.name not in args.datasets:
                continue
            repeats = args.dengue_csv_repeats if ds.name.startswith("dengue.csv") else args.repeats
            print(f"=== [{group}] {ds.name}: {len(ds.frame)} rows, {len(ds.feature_cols)} features, "
                  f"{repeats} repeats ===", flush=True)
            for r in range(repeats):
                t0 = time.perf_counter()
                res = run_repeat(ds, seed=r)
                row = {
                    "Group": group, "Dataset": ds.name, "Repeat": r,
                    **{k: res[k] for k in ("n_rows", "n_valid", "n_masked_cells", "n_train", "n_test", "n_predicted")},
                    **{f"imp_{k}": v for k, v in res["imputation"].items()},
                    **{f"pred_{k}": v for k, v in res["prediction"].items()},
                }
                rows.append(row)
                print(f"  repeat {r}: imp NMI={res['imputation']['NMI']:.3f} "
                      f"pred NM={res['prediction']['NMPrediction']:.3f} "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            pd.DataFrame(rows).to_csv(per_repeat_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(per_repeat_path, index=False)

    agg = {c: ["mean", "std"] for c in df.columns if c.startswith(("imp_", "pred_"))}
    sizes = {c: "first" for c in ("n_rows", "n_valid", "n_train", "n_test", "n_predicted")}
    sizes["n_masked_cells"] = "mean"
    summary = df.groupby(["Group", "Dataset"], sort=False).agg({**sizes, **agg, "Repeat": "count"})
    summary.columns = [c if isinstance(c, str) else (c[0] if c[1] in ("first", "count") else f"{c[0]}_{c[1]}")
                       for c in summary.columns]
    summary = summary.rename(columns={"n_masked_cells_mean": "n_masked_cells", "Repeat": "repeats"})
    summary.reset_index().to_csv(outdir / "summary.csv", index=False)
    print(f"\nWrote {per_repeat_path} and {outdir / 'summary.csv'}")
    print("Imputers:", ", ".join(IMPUTERS), "| Predictors:", ", ".join(PREDICTORS))


if __name__ == "__main__":
    main()
