"""
Demo: GA wrapper influential-feature selection on NMI dengue (or synthetic) data.

Usage:
  python scripts/run_influential_features.py
  python scripts/run_influential_features.py --rows 500 --classifier tree --generations 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denguecad.feature_selection import GAWrapperConfig, select_influential_features
from denguecad.nmi_support import resolve_nmi_root


def load_dengue(nmi_root: Path, rows: int | None) -> pd.DataFrame:
    """Prefer vendored ``data/dengue.csv``; fall back to NMI checkout."""
    local = ROOT / "data" / "dengue.csv"
    path = local if local.is_file() else nmi_root / "dengue.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Expected dengue.csv at {local} or {nmi_root}")
    df = pd.read_csv(path, na_values=["", "?"])
    if rows is not None:
        df = df.head(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="GA wrapper influential feature selection")
    parser.add_argument("--nmi-root", default=None, help="Path to local NMI checkout")
    parser.add_argument("--rows", type=int, default=400, help="Row subsample (None = all)")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--classifier", choices=("svm", "tree"), default="tree")
    parser.add_argument("--scoring", choices=("accuracy", "auc"), default="accuracy")
    parser.add_argument("--no-impute", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    nmi_root = resolve_nmi_root(args.nmi_root)
    df = load_dengue(nmi_root, args.rows)
    print(f"Loaded {len(df)} rows (vendored data/ or NMI checkout)")
    print(f"Columns: {list(df.columns)}")

    cfg = GAWrapperConfig(
        population_size=args.population,
        n_generations=args.generations,
        crossover_prob=1.0,
        mutation_prob=0.001,
        n_folds=args.folds,
        scoring=args.scoring,
        classifier=args.classifier,
        impute_missing=not args.no_impute,
        nmi_root=str(nmi_root),
        verbose=args.verbose,
    )
    result = select_influential_features(
        df,
        decision_col="Dengue",
        feature_cols=["Fever", "Headache", "JointPain", "Bleeding"],
        config=cfg,
    )
    print("\n=== Influential feature subset (GA wrapper) ===")
    print(f"Selected: {result.selected_features}")
    print(f"Subset size: {result.subset_size} / {len(result.feature_names)}")
    print(f"CV {cfg.scoring}: {result.cv_score:.4f}")
    print(f"Fitness:      {result.fitness:.4f}")
    print(f"Best fitness trajectory: {[round(x, 4) for x in result.history]}")


if __name__ == "__main__":
    main()
