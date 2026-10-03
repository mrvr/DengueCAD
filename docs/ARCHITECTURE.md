# DengueCAD Architecture

DengueCAD is a computer-aided diagnosis (CAD) stack built on the sibling **NMI** imputation library. It implements the IEEE TITB 2012 **NMPrediction** method (Algorithm 1): missing-value imputation → GA wrapper feature selection → stratified *k*-fold Alternating Decision Tree (ADT) → Accuracy / AUC / SE / SP. The core library is disease-agnostic; dengue is the primary use case, and the same pipeline runs on multi-disease ARFF benchmarks.

Rendered images of every diagram are in [`architecture/`](architecture/) and the slide deck is [`DengueCAD_Architecture_Slides.pptx`](DengueCAD_Architecture_Slides.pptx) ([PDF](DengueCAD_Architecture_Slides.pdf)). The Mermaid blocks below are the source of truth; regenerate images and slides with:

```bash
.venv/bin/python scripts/render_architecture_diagrams.py
.venv/bin/python scripts/build_architecture_slides.py --pdf
```

## 1. System overview

<!-- diagram: 01-system-overview -->
```mermaid
flowchart TB
    subgraph Inputs["Data sources (data/)"]
        direction TB
        D1["dengue.csv<br/>1M rows, 4 binary symptoms"]
        D2["dataset.csv<br/>Bangladesh dengue"]
        D3["Dengue_clinical_dataset.csv<br/>clinical + hematology"]
        D4["KEEL / Weka ARFF<br/>11 disease datasets"]
        D5["missforest_iris/iris.csv"]
    end

    subgraph Scripts["Experiment runners (scripts/)"]
        direction TB
        S1["run_nm_comparison"]
        S2["run_nm_on_dataset_csv<br/>run_nm_on_clinical"]
        S3["run_arff_disease_benchmark<br/>run_holdout_imputation_prediction"]
        S4["run_imputation_comparison<br/>run_missforest_comparison"]
    end

    subgraph Core["denguecad package"]
        direction LR
        IO["Loading & splits<br/>data_splits · arff_io<br/>imputation_benchmark"]
        IMP["Imputation<br/>NMI · MICE · kNN<br/>Mean/Mode · MissForest"]
        NM["NMPrediction<br/>(Algorithm 1)<br/>nm_prediction"]
        FS["GA wrapper<br/>feature_selection"]
        CLF["Learners<br/>adt · baselines<br/>classifiers"]
        EVAL["Evaluation<br/>metrics · comparison"]
        IO --> IMP --> NM --> FS --> CLF --> EVAL
    end

    subgraph NMI["NMI library (sibling repo)"]
        N1["nmilib.non_parametric_imputation"]
    end

    subgraph Outputs["Outputs"]
        direction LR
        R["results/*.csv<br/>(git-ignored)"]
        P["PDF reports<br/>Performance_Comparisons ·<br/>Imputation_Prediction_Holdout"]
        R --> P
    end

    Inputs --> Scripts --> Core
    Core --> Outputs
    Core -.->|"nmi_support"| NMI
```

## 2. NMPrediction pipeline (Algorithm 1)

<!-- diagram: 02-nmprediction-pipeline -->
```mermaid
flowchart LR
    subgraph S1["Steps 1–2: Impute"]
        direction TB
        A["Collect records S(m, n)<br/>+ decision column"] --> B{"Missing<br/>values?"}
        B -- yes --> C["NMI imputation<br/>non_parametric_imputation"]
        B -- no --> C2["Use data as-is"]
    end

    subgraph S2["Step 3: Influential features"]
        direction TB
        D["GA wrapper search<br/>chromosome = feature subset<br/>Pc = 1.0 · Pm = 0.001"] -- candidate --> E["Fitness = 3-fold CV accuracy<br/>of ADT − size penalty"]
        E -- next generation --> D
        E -- best subset --> F["Influential features"]
    end

    subgraph S3["Steps 4–8: Evaluate"]
        direction TB
        G["Stratified 10-fold<br/>train ADT on T_k, score R_k"] --> H["Pool labels L<br/>and scores P"]
        H --> I["Accuracy · AUC · SE · SP"]
        J["Hold-out: fit ADT on train,<br/>score validation / test"]
    end

    S1 --> S2 --> S3
```

## 3. GA wrapper feature selection (Section C)

The wrapper follows Kohavi & John: the classifier itself scores each candidate subset.

<!-- diagram: 03-ga-wrapper -->
```mermaid
flowchart TB
    subgraph Init["Initialise"]
        I1["Random chromosomes<br/>~30% bits on"]
        I2["Seed: all features<br/>+ one sparse subset"]
    end

    subgraph Eval["Evaluate (wrapper)"]
        direction LR
        EV["Mask X to<br/>selected columns"] --> CV["Stratified k-fold CV<br/>ADT / SVM / tree"]
        CV --> FIT["fitness = CV accuracy (or AUC)<br/>− penalty × subset size / n"]
        FIT --> BEST["Track best chromosome<br/>+ per-generation history"]
    end

    subgraph Evolve["Evolve next generation"]
        direction LR
        EL["Elitism:<br/>keep top 2"] --> TS["Tournament<br/>selection, size 3"]
        TS --> CX["One-point crossover<br/>Pc = 1.0"]
        CX --> MU["Bit-flip mutation<br/>Pm = 0.001"]
        MU --> MIN["Enforce ≥<br/>min_features bits"]
    end

    subgraph Out["Result"]
        OUT["WrapperGAResult<br/>selected_features · cv_score · history"]
    end

    Init --> Eval
    Eval -- "more generations" --> Evolve
    Evolve --> Eval
    Eval -- "generations exhausted" --> Out
```

## 4. Module dependencies

Arrows point from a module to the modules it imports.

<!-- diagram: 04-module-dependencies -->
```mermaid
flowchart LR
    scripts["scripts/run_*.py"] --> comparison
    scripts --> data_splits
    scripts --> arff_io
    scripts --> imputation_benchmark
    scripts --> missforest_impute
    scripts --> holdout_experiment

    holdout_experiment --> nm_prediction
    holdout_experiment --> imputation_benchmark
    holdout_experiment --> missforest_impute
    holdout_experiment --> baselines

    comparison --> nm_prediction
    comparison --> baselines
    comparison --> data_splits
    comparison --> metrics

    nm_prediction --> feature_selection
    nm_prediction --> baselines
    nm_prediction --> metrics
    nm_prediction --> nmi_support

    feature_selection --> classifiers
    feature_selection --> nmi_support

    baselines --> adt
    classifiers --> adt

    imputation_benchmark --> nmi_support
    nmi_support -.->|"NMI_ROOT or ../NMI"| nmilib[("NMI: nmilib.py")]
```

| Module | Responsibility |
|--------|----------------|
| `nm_prediction.py` | `NMPrediction`: Algorithm 1 end to end, plus hold-out evaluation |
| `feature_selection.py` | GA wrapper (`select_influential_features`, `GAWrapperConfig`) |
| `adt.py` | Alternating Decision Tree (AdaBoost over decision stumps) |
| `baselines.py` | C4.5 (entropy decision tree), SVM (RBF), LOR (logistic regression), ADT factory |
| `classifiers.py` | Classifier factory used inside the GA wrapper |
| `metrics.py` | Accuracy, AUC, sensitivity (SE), specificity (SP) |
| `comparison.py` | NM vs baselines: Table I (performance), Table II (Wilcoxon), Table III (features) |
| `data_splits.py` | `dengue.csv` protocol: 70k train, 10k validation, 20k test |
| `imputation_benchmark.py` | CSV loading/encoding, MCAR masking, NMI / MICE / kNN / Mean-Mode imputers |
| `missforest_impute.py` | Random-forest iterative imputation (MissForest) |
| `holdout_experiment.py` | valid_set / imputed train_set / hidden-decision protocol; imputation and prediction accuracy |
| `arff_io.py` | ARFF loader tolerant of KEEL `NUMERIC [min, max]` ranges |
| `nmi_support.py` | Locates and imports `nmilib` from the NMI checkout |

## 5. Data protocol for `dengue.csv`

<!-- diagram: 05-dengue-data-protocol -->
```mermaid
flowchart LR
    F["dengue.csv<br/>1,000,000 rows"] --> T["Rows 0 – 69,999<br/>single training set 'dengue'<br/>NMI + NMPrediction + baselines<br/>(10-fold CV)"]
    F --> Rm["Rows 70,000 onward"]
    Rm --> CC["Drop records with any<br/>missing value (complete-case)"]
    CC --> V["First 10,000<br/>validation"]
    CC --> Te["Next 20,000<br/>test (reported performance)"]
```

The other datasets are evaluated with stratified 10-fold CV on the full file, after NMI imputation where values are missing.

## 6. NM vs baselines evaluation

<!-- diagram: 06-evaluation-flow -->
```mermaid
flowchart LR
    DS["Training dataset"] --> BL["Baselines, same 10 folds<br/>C4.5 · LOR · SVM (RBF)"]
    DS --> NM["NMPrediction<br/>NMI → GA+ADT → 10-fold ADT"]
    BL --> T1["Table I<br/>Accuracy · SE · SP · AUC"]
    NM --> T1
    BL --> FA["Per-fold accuracies"]
    NM --> FA
    FA --> T2["Table II<br/>Wilcoxon signed-rank<br/>NM vs each baseline"]
    NM --> T3["Table III<br/>influential features"]
    NM --> HO["Hold-out metrics<br/>validation · test"]
    T1 --> PDF["Comparison PDF<br/>red = best<br/>green = NM when not best"]
    T2 --> PDF
    T3 --> PDF
    HO --> PDF
```

## 7. Imputation benchmark

<!-- diagram: 07-imputation-benchmark -->
```mermaid
flowchart LR
    M["Complete-case master<br/>dataset.csv / Iris"] --> MASK["Inject MCAR missingness<br/>on attributes only"]
    MASK --> NMI["NM (NMI)"]
    MASK --> MF["MissForest"]
    MASK --> MICE["MICE<br/>IterativeImputer"]
    MASK --> KNN["kNN"]
    MASK --> SIMPLE["Mean · Median · Mode"]
    NMI --> SC["Imputation accuracy<br/>vs ground truth<br/>(exact for categorical,<br/>5% tolerance for real)"]
    MF --> SC
    MICE --> SC
    KNN --> SC
    SIMPLE --> SC
    SC --> DOWN["Downstream CV accuracy<br/>ADT · C4.5 · LOR · SVM"]
```

## 8. Imputation + prediction hold-out experiment

Runs on five dengue datasets and eleven disease ARFFs, repeated with different seeds (`holdout_experiment.py`, report `DengueCAD_Imputation_Prediction_Holdout.pdf`).

<!-- diagram: 09-holdout-protocol -->
```mermaid
flowchart TB
    V["valid_set<br/>complete records of dataset S<br/>(ground truth)"]
    subgraph IPH["Imputation phase"]
        direction LR
        M["Mask 20% of records<br/>1–3 attributes each"] --> IMP["Impute<br/>NMI · MICE · kNN<br/>MissForest · Mean/Mode"]
        IMP --> IA["1.1 / 2.1<br/>Imputation accuracy<br/>vs valid_set"]
    end
    subgraph PPH["Prediction phase"]
        direction LR
        HO["30% hold-out<br/>from valid_set"] --> HD["Hide decision on<br/>20% of hold-out"]
        TR["train_set<br/>NMI-imputed minus<br/>30% hold-out"] --> P["Predict hidden decisions<br/>NMPrediction · C4.5<br/>LOR · SVM"]
        HD --> P
        P --> PA["1.2 / 2.2<br/>Prediction accuracy<br/>vs valid_set"]
    end
    V --> IPH
    IPH -- "NMI-imputed dataset" --> PPH
```

## 9. Experiment runners and outputs

| Script | Dataset | Output (under `results/`) |
|--------|---------|---------------------------|
| `run_nm_comparison.py` | `dengue.csv` (70k / 10k / 20k) | `table1_performance.csv`, `table2_wilcoxon.csv`, `table3_influential_features.csv`, `holdout_metrics.csv`, `split_summary.csv` |
| `run_nm_on_dataset_csv.py` | `dataset.csv` | `nmprediction_dataset_csv/` |
| `run_nm_on_clinical.py` | `Dengue_clinical_dataset.csv` | `nmprediction_clinical/` |
| `run_arff_disease_benchmark.py` | 11 disease ARFFs | `arff_disease_comparison/` |
| `run_imputation_comparison.py` | `dataset.csv` + MCAR | `imputation_dataset_csv/` |
| `run_missforest_comparison.py` | Iris + 20% MCAR | `missforest_comparison/` |
| `run_holdout_imputation_prediction.py` | 5 dengue sets + 11 ARFFs (20% records masked, 30% hold-out, 20% hidden decisions) | `holdout_imputation_prediction/{dengue,multi}/` |
| `build_comparison_pdf.py` | all of the above | `docs/DengueCAD_Performance_Comparisons.pdf` |
| `build_holdout_comparison_pdf.py` | hold-out results | `docs/DengueCAD_Imputation_Prediction_Holdout.pdf` |

## 10. CI/CD

<!-- diagram: 08-ci-cd -->
```mermaid
flowchart LR
    P["git push / pull request<br/>to main"] --> I["Inspect tests<br/>scripts/inspect_tests.py"]
    I --> U["Unit tests<br/>pytest -m unit"]
    I --> S["System tests<br/>clone NMI, pytest -m system"]
    U --> G{"All green and<br/>push to main?"}
    S --> G
    G -- yes --> Rel["GitHub Release<br/>semver patch bump"]
```

Run the same gates locally before pushing with `bash scripts/ci_local.sh`. Details: [`TESTING.md`](TESTING.md).
