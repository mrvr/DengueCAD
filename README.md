# DengueCAD

Based on IEEE paper IEEEITBM 2012 *"A New Intelligence Based Approach for Computer-Aided Diagnosis of Dengue Fever"* (Rao & Kumar, IEEE TITB / JBHI).

This project builds on the sibling **NMI** imputation library ([mrvr/NMI](https://github.com/mrvr/NMI.git)) and implements **Section C — Identification of influential features**: a *wrapper* subset evaluation model driven by a **genetic algorithm**.

## Method (Section C)

Influential features are often unknown a priori. Sequential greedy search (forward / backward / bidirectional) can miss good subsets. Stochastic search with **genetic algorithms** explores large spaces more effectively ([Goldberg, 1989][17]).

DengueCAD adopts a **wrapper subset-based feature evaluation** model ([Kohavi & John, 1997][18]): the **classifier itself** scores each candidate feature subset. A binary GA chromosome marks features in/out; fitness is stratified *k*-fold CV performance (paper: *k* = 10), with a small size penalty favoring compact subsets.

Paper GA settings (Section V):

| Parameter | Value |
|-----------|-------|
| Crossover probability \(P_c\) | 1.0 |
| Mutation probability \(P_m\) | 0.001 |
| Inner learner (experiments) | SVM-RBF (LibSVM) |
| CV | Stratified 10-fold |

Missing attribute values can be imputed first with **NMI** (`non_parametric_imputation`) before the wrapper run.

## Layout

```
denguecad/                 # library code
scripts/
  run_influential_features.py
  inspect_tests.py         # CI test-validity gate
  next_version.py          # semver bump for releases
  ci_local.sh              # run the same gates as CI
tests/
  unit/                    # fast unit tests
  system/                  # NMI + dengue end-to-end tests
data/                      # vendored dengue.csv + data*.txt (from NMI)
.github/workflows/ci.yml   # GitHub Actions CI + release (no Jenkins required)
docs/TESTING.md
VERSION                    # semver baseline (e.g. 0.1.0)
```

## Setup (project virtual environment)

All DengueCAD Python work runs in a **local `.venv`** (isolated from system Python and from NMI’s own venv).

```bash
cd DengueCAD

# Create / refresh the environment (once)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Always activate before running code in a new shell
source .venv/bin/activate

# Optional: point at a non-default NMI checkout
# export NMI_ROOT="/path/to/NMI"
```

Without activating, use the venv interpreter explicitly:

```bash
.venv/bin/python scripts/run_influential_features.py
.venv/bin/python -m pytest -q
```

Cursor / VS Code is configured (`.vscode/settings.json`) to use `.venv/bin/python` automatically.

## CI/CD (option 1) — GitHub Actions (no Jenkins)

**Yes — GitHub can do this.** You do not need Jenkins. [GitHub Actions](https://docs.github.com/en/actions) runs on every push/PR using `.github/workflows/ci.yml` (hosted runners). Locally we still run the same gates **before** push.

On every git check-in / pull request to `main`:

1. Inspect unit + system tests for validity (update/remove stale tests)
2. Run **all unit tests**
3. Run **all system tests** (NMI library cloned; datasets from this repo’s `data/`)
4. If the push is to `main` and every gate is green → **create a release** (`vMAJOR.MINOR.PATCH`)

Details: [`docs/TESTING.md`](docs/TESTING.md)

```bash
# Same gates locally (do this before every push)
bash scripts/ci_local.sh
```
## Quick start

```python
import pandas as pd
from denguecad import GAWrapperConfig, select_influential_features

df = pd.read_csv("data/dengue.csv")
cfg = GAWrapperConfig(
    population_size=20,
    n_generations=20,
    crossover_prob=1.0,
    mutation_prob=0.001,
    n_folds=5,
    classifier="tree",   # or "svm" (paper)
    impute_missing=True,
)
result = select_influential_features(
    df,
    decision_col="Dengue",
    feature_cols=["Fever", "Headache", "JointPain", "Bleeding"],
    config=cfg,
)
print(result.selected_features, result.cv_score)
```

Demo CLI:

```bash
python scripts/run_influential_features.py --rows 400 --classifier tree --generations 15
```

Tests:

```bash
python -m pytest -q -m unit
python -m pytest -q -m system
bash scripts/ci_local.sh
```

## References

1. V. Sree Hari Rao & M. Naresh Kumar, “A New Intelligence-Based Approach for Computer-Aided Diagnosis of Dengue Fever,” *IEEE Trans. Inf. Technol. Biomed.*, 16(1), 2012.
2. K. Ron (Kohavi) & H. J. George (John), “Wrappers for feature subset selection,” *Artificial Intelligence*, vol. 97, pp. 273–324, 1997.
3. D. E. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*, 1989.
4. NMI library: https://github.com/mrvr/NMI.git
