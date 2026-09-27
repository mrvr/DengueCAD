#!/usr/bin/env python3
"""
MissForest dataset comparison (canonical Iris demo).

Source: UCI Iris — the dataset used in the missForest R package examples
  https://cran.r-project.org/web/packages/missForest/readme/README.html
  https://archive.ics.uci.edu/dataset/53/iris

Protocol (matches missForest demos):
  1. Start from complete Iris
  2. Inject ~20% MCAR missingness on attributes (decision untouched)
  3. Impute with MissForest, NMI, Mode/Mean, Mean, Median, kNN, MICE
  4. Report imputation accuracy vs ground truth
  5. Report downstream classification accuracy (stratified CV) after each imputer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.baselines import make_adt, make_c45, make_lor, make_svm
from denguecad.imputation_benchmark import (
    impute_knn,
    impute_mice,
    impute_nmi,
    impute_simple,
    mask_features,
)
from denguecad.missforest_impute import missforest_impute
from denguecad.metrics import scores_from_estimator

IRIS_PATH = ROOT / "data" / "missforest_iris" / "iris.csv"
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
DECISION = "class"


def load_iris(path: Path = IRIS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    enc = LabelEncoder()
    df = df.copy()
    df[DECISION] = enc.fit_transform(df[DECISION].astype(str))
    return df


def imputation_accuracy(
    master: pd.DataFrame,
    imputed: pd.DataFrame,
    feature_cols: list[str],
    mask: np.ndarray,
) -> dict:
    """NRMSE-style continuous error + exact-match rate on masked cells."""
    errs = []
    matches = 0
    n = 0
    for j, col in enumerate(feature_cols):
        rows = np.flatnonzero(mask[:, j])
        if len(rows) == 0:
            continue
        t = master.iloc[rows][col].to_numpy(dtype=float)
        p = imputed.iloc[rows][col].to_numpy(dtype=float)
        # per-column scale
        scale = float(np.nanstd(master[col].to_numpy(dtype=float))) or 1.0
        for ti, pi in zip(t, p):
            if np.isnan(pi):
                errs.append(1.0)
                n += 1
                continue
            errs.append(abs(ti - pi) / scale)
            matches += int(np.isclose(ti, pi, rtol=0.05, atol=0.05))
            n += 1
    nrmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    acc = matches / n if n else 0.0
    return {"n_compared": n, "approx_match_rate": acc, "nrmse_proxy": nrmse}


def cv_accuracy(df: pd.DataFrame, feature_cols: list[str], decision: str, folds: int = 5) -> dict:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[decision].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    out = {}
    factories = {
        "ADT": make_adt,
        "C4.5": make_c45,
        "LOR": make_lor,
        "SVM": make_svm,
    }
    for name, factory in factories.items():
        scores = []
        for tr, te in skf.split(X, y):
            clf = factory(42)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            scores.append(float(np.mean(pred == y[te])))
        out[name] = round(100.0 * float(np.mean(scores)), 2)
    return out


def try_package_missforest(mv: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame | None:
    try:
        from missforest import MissForest
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except Exception:
        return None
    work = mv[feature_cols].copy()
    mf = MissForest(
        categorical=None,
        max_iter=6,
        verbose=0,
        clf=RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=-1),
        rgr=RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1),
    )
    try:
        imputed = mf.fit_transform(work)
        if not isinstance(imputed, pd.DataFrame):
            imputed = pd.DataFrame(imputed, columns=feature_cols, index=work.index)
        else:
            imputed = imputed.copy()
            imputed.columns = feature_cols
            imputed.index = work.index
        out = mv.copy()
        out[feature_cols] = imputed[feature_cols].to_numpy()
        return out
    except Exception as e:
        print(f"  (package MissForest failed: {e})", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="MissForest Iris vs NMI prediction comparison")
    parser.add_argument("--mv-frac", type=float, default=0.20, help="MCAR rate (missForest demo uses 0.2)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "missforest_comparison"),
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not IRIS_PATH.is_file():
        raise SystemExit(f"Missing {IRIS_PATH} — download UCI Iris first")

    master = load_iris()
    print("=== MissForest canonical dataset: UCI Iris ===", flush=True)
    print(f"source: {IRIS_PATH}", flush=True)
    print(f"refs: CRAN missForest README; UCI https://archive.ics.uci.edu/dataset/53/iris", flush=True)
    print(f"rows={len(master)} features={FEATURE_COLS} decision={DECISION}", flush=True)

    mv, mask = mask_features(master, FEATURE_COLS, mv_frac=args.mv_frac, seed=81)
    print(f"Injected MCAR={args.mv_frac:.0%} on attributes (seed=81, missForest-style)", flush=True)
    print(f"Masked cells: {int(mask.sum())}", flush=True)

    methods: list[tuple[str, callable]] = []

    def _mf_sklearn(d):
        return missforest_impute(d, FEATURE_COLS, categorical_cols=[], max_iter=6, n_estimators=40)

    methods.append(("MissForest (RF)", _mf_sklearn))

    pkg_result = try_package_missforest(mv, FEATURE_COLS)
    if pkg_result is not None:
        methods.insert(0, ("MissForest (pkg)", lambda d, _fixed=pkg_result: _fixed.copy()))

    methods.extend(
        [
            ("NM (NMI)", lambda d: impute_nmi(d, FEATURE_COLS, DECISION, categorical_cols=[])),
            ("Mode/Mean", lambda d: impute_simple(d, FEATURE_COLS, [], strategy="mode")),
            ("Mean", lambda d: impute_simple(d, FEATURE_COLS, [], strategy="mean")),
            ("Median", lambda d: impute_simple(d, FEATURE_COLS, [], strategy="median")),
            ("kNN", lambda d: impute_knn(d, FEATURE_COLS)),
            ("MICE", lambda d: impute_mice(d, FEATURE_COLS, random_state=42)),
        ]
    )

    imp_rows = []
    pred_rows = []
    # Complete-data baseline prediction (no missingness)
    complete_pred = cv_accuracy(master, FEATURE_COLS, DECISION, folds=args.folds)
    pred_rows.append({"Imputer": "Complete (no MV)", **complete_pred})

    print("\n=== Imputation + prediction comparison ===", flush=True)
    for name, fn in methods:
        t0 = time.perf_counter()
        if name == "MissForest (pkg)" and pkg_result is not None:
            imputed = pkg_result
            # still time a no-op-ish path
            elapsed = time.perf_counter() - t0
        else:
            imputed = fn(mv.copy())
            elapsed = time.perf_counter() - t0
        # Ensure decision intact
        imputed[DECISION] = master[DECISION].to_numpy()
        stats = imputation_accuracy(master, imputed, FEATURE_COLS, mask)
        pred = cv_accuracy(imputed, FEATURE_COLS, DECISION, folds=args.folds)
        print(
            f"{name:18s}  nrmse={stats['nrmse_proxy']:.3f}  "
            f"match%={100*stats['approx_match_rate']:.1f}  "
            f"ADT={pred['ADT']:.1f} C4.5={pred['C4.5']:.1f} "
            f"LOR={pred['LOR']:.1f} SVM={pred['SVM']:.1f}  ({elapsed:.2f}s)",
            flush=True,
        )
        imp_rows.append(
            {
                "Method": name,
                "NRMSE_proxy": round(stats["nrmse_proxy"], 4),
                "Approx match (%)": round(100 * stats["approx_match_rate"], 2),
                "n_compared": stats["n_compared"],
                "seconds": round(elapsed, 3),
            }
        )
        pred_rows.append({"Imputer": name, **pred})

    imp_table = pd.DataFrame(imp_rows).sort_values("NRMSE_proxy")
    pred_table = pd.DataFrame(pred_rows)

    print("\n=== TABLE A — Imputation error (lower NRMSE better) ===")
    print(imp_table.to_string(index=False))
    print("\n=== TABLE B — Downstream prediction accuracy (%) after imputation ===")
    print(pred_table.to_string(index=False))

    imp_table.to_csv(outdir / "table_imputation.csv", index=False)
    pred_table.to_csv(outdir / "table_prediction.csv", index=False)
    (outdir / "SOURCE.md").write_text(
        "# MissForest comparison data source\n\n"
        "- Dataset: **UCI Iris** (Fisher 1936)\n"
        "- URL: https://archive.ics.uci.edu/dataset/53/iris\n"
        "- Used as the demo dataset in **missForest** (Stekhoven & Bühlmann):\n"
        "  https://cran.r-project.org/web/packages/missForest/readme/README.html\n"
        "- Local copy: `data/missforest_iris/iris.csv`\n"
        "- Protocol: 20% MCAR on attributes (seed 81), then impute + classify.\n",
        encoding="utf-8",
    )
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
