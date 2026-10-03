# DengueCAD

Based on IEEE paper IEEEITBM 2012 *"A New Intelligence Based Approach for Computer-Aided Diagnosis of Dengue Fever"* (Rao & Kumar, IEEE TITB / JBHI).

This project builds on the sibling **NMI** imputation library ([mrvr/NMI](https://github.com/mrvr/NMI.git)) and implements:

1. **Section C** — GA wrapper influential-feature selection  
2. **Algorithm 1 (`NMPrediction`)** — NMI imputation → GA+ADT feature selection → stratified *k*-fold ADT → AUC / SE / SP  
3. **Baselines** — C4.5, SVM (RBF), LOR — compared via Table I / Wilcoxon Table II / influential features Table III  

## NMPrediction (Algorithm 1)

```text
Input:  S(m,n), attribute types
Output: Accuracy, AUC, SE, SP

1. Collect records in S
2. Impute missing values (NMI / Section III-B)
3. Extract influential features (wrapper + genetic search; ADT evaluates subsets)
4. Stratified k-fold into T_k / R_k
5. For each fold: train ADT on T_k; score R_k → P; collect labels → L
6. Repeat for all folds
7–8. Compute AUC, SE, SP from (L, P); return
```

## Experiment protocol (`data/dengue.csv`)

| Split | Rows | Role |
|-------|------|------|
| dengue (train) | first 70 000 (single dataset) | Apply NMI / NMPrediction (+ baselines) |
| validation | next complete-case block → 10 000 | Hold-out validation (no MVs) |
| test | next complete-case block → 20 000 | Hold-out test / performance (no MVs) |

```bash
source .venv/bin/activate
python scripts/run_nm_comparison.py
# writes results/table1_performance.csv, table2_wilcoxon.csv, table3_influential_features.csv
```

> Note: vendored `dengue.csv` has four binary symptoms only (not the paper’s 16 clinical/lab attributes), so numeric Table I values will not reproduce the IEEE tables; the **pipeline and report format** match Algorithm 1 / Tables I–III.

## Method (Section C)

Influential features are often unknown a priori. Sequential greedy search (forward / backward / bidirectional) can miss good subsets. Stochastic search with **genetic algorithms** explores large spaces more effectively ([Goldberg, 1989][17]).

DengueCAD adopts a **wrapper subset-based feature evaluation** model ([Kohavi & John, 1997][18]): the **classifier itself** scores each candidate feature subset. A binary GA chromosome marks features in/out; fitness is stratified *k*-fold CV performance (paper: *k* = 10), with a small size penalty favoring compact subsets.

Paper GA settings (Section V):

| Parameter | Value |
|-----------|-------|
| Crossover probability \(P_c\) | 1.0 |
| Mutation probability \(P_m\) | 0.001 |
| Wrapper / final learner (Algorithm 1) | Alternating Decision Tree (ADT) |
| CV | Stratified 10-fold |

Missing attribute values can be imputed first with **NMI** (`non_parametric_imputation`) before the wrapper run.

## Layout

Architecture diagrams (system overview, NMPrediction pipeline, GA wrapper, module dependencies, data protocol, evaluation, imputation benchmark, CI/CD): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), rendered images in [`docs/architecture/`](docs/architecture/), and slides in [`docs/DengueCAD_Architecture_Slides.pptx`](docs/DengueCAD_Architecture_Slides.pptx) ([PDF](docs/DengueCAD_Architecture_Slides.pdf)).

```
denguecad/
  nm_prediction.py         # Algorithm 1 — NMPrediction
  adt.py                   # Alternating Decision Tree
  baselines.py             # C4.5, SVM, LOR
  comparison.py            # Tables I–III + Wilcoxon
  data_splits.py           # 70k single train / 10k val / 20k test
  feature_selection.py     # GA wrapper
  …
scripts/run_nm_comparison.py
results/                   # generated comparison CSVs
data/                      # dengue.csv + data*.txt
.github/workflows/ci.yml
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

External dengue datasets, portals, and challenges: [`docs/REFERENCES.md`](docs/REFERENCES.md).

Performance comparison PDF (dengue + non-dengue tables): [`docs/DengueCAD_Performance_Comparisons.pdf`](docs/DengueCAD_Performance_Comparisons.pdf).

Imputation and prediction hold-out comparison (valid_set / NMI-imputed train_set / 20% hidden decisions; NMI vs MICE, kNN, MissForest, Mean/Mode and NMPrediction vs C4.5, LOR, SVM on dengue and multi-disease data): [`docs/DengueCAD_Imputation_Prediction_Holdout.pdf`](docs/DengueCAD_Imputation_Prediction_Holdout.pdf). Regenerate with:

```bash
python scripts/run_holdout_imputation_prediction.py --only dengue --outdir results/holdout_imputation_prediction/dengue
python scripts/run_holdout_imputation_prediction.py --only multi --outdir results/holdout_imputation_prediction/multi
python scripts/build_holdout_comparison_pdf.py
```
