# Methods guide flowcharts

Mermaid sources for the flowcharts in `docs/DengueCAD_Methods_Guide.pdf`.
Render with:

```bash
python scripts/render_architecture_diagrams.py --source docs/methods/FLOWCHARTS.md --outdir docs/methods
```

Layout note: each chart is a vertical stack of horizontal rows (subgraphs with
`direction LR` linked subgraph-to-subgraph) so it fits an A4 page at a readable
size. Mermaid ignores a subgraph's direction when an edge links one of its
nodes to the outside, so loops stay inside a row.

## Where the methods fit

<!-- diagram: m00-overview -->
```mermaid
flowchart LR
    D["Dataset with<br/>missing values"] --> IMP
    subgraph IMP["Step 1 · Imputation"]
        direction LR
        I1["NMI (NM's imputer)"]
        I2["MICE"]
        I3["kNN"]
        I4["MissForest"]
        I5["Mean / Mode"]
    end
    IMP --> C["Complete<br/>dataset"] --> PRED
    subgraph PRED["Step 2 · Prediction"]
        direction LR
        P1["NMPrediction<br/>(NMI + GA wrapper + ADT)"]
        P2["C4.5 decision tree"]
        P3["Logistic regression (LOR)"]
        P4["SVM (RBF kernel)"]
    end
    PRED --> O["Diagnosis<br/>(decision column)"]
```

## NMI — New Method Imputation

<!-- diagram: m01-nmi -->
```mermaid
flowchart TB
    subgraph R1["1 · Prepare"]
        direction LR
        A["Dataset S<br/>(last column = decision)"] --> B["Complete rows = donors<br/>Incomplete rows = targets"]
        B --> C["Take a target row R<sub>i</sub><br/>(its decision must be known)"]
    end
    subgraph R2["2 · Measure closeness to every donor R<sub>k</sub>"]
        direction LR
        D["Score each attribute<br/>I<sub>Cl</sub>(R<sub>i</sub>, R<sub>k</sub>)<br/>same class → Case I<br/>other class → Case II"] --> E["Distance<br/>d<sub>ik</sub> = √ Σ I<sub>Cl</sub>²"]
        E --> F["z-score the distances<br/>z = (d − mean) / sd"]
        F --> G["Neighbours =<br/>donors with z ≤ 0"]
    end
    subgraph R3["3 · Fill the gaps (repeat for every target row)"]
        direction LR
        H{"Type of the<br/>missing attribute?"} -- "categorical / integer" --> I["MODE of the<br/>neighbours' values"]
        H -- "fractional / real" --> J["MEAN of the<br/>neighbours' values"]
        I --> K["Imputed dataset"]
        J --> K
    end
    R1 --> R2 --> R3
```

## MICE — Multivariate Imputation by Chained Equations

<!-- diagram: m02-mice -->
```mermaid
flowchart TB
    subgraph R1["1 · Start"]
        direction LR
        A["Dataset with<br/>missing values"] --> B["Temporary fill:<br/>column mean"]
    end
    subgraph R2["2 · One round over the incomplete columns (repeat up to 20 rounds)"]
        direction LR
        C["Next column X<sub>j</sub>"] --> D["Fit a regression<br/>X<sub>j</sub> ~ other columns<br/>(rows where X<sub>j</sub> is observed)"]
        D --> E["Predict X<sub>j</sub> where<br/>it was missing"]
        E --> F{"All columns<br/>done?"}
        F -- "no" --> C
    end
    subgraph R3["3 · Finish"]
        direction LR
        G{"Values stable?"} -- "yes" --> H["Imputed dataset"]
    end
    R1 --> R2 --> R3
```

## kNN imputation

<!-- diagram: m03-knn -->
```mermaid
flowchart TB
    subgraph R1["1 · Find neighbours (for each missing cell in column X)"]
        direction LR
        A["Row with X<br/>missing"] --> B["Distance to all rows<br/>(nan-Euclidean: only<br/>columns both observe)"]
        B --> C["Keep the k = 5<br/>closest rows that<br/>have X observed"]
    end
    subgraph R2["2 · Fill"]
        direction LR
        D["Weight each neighbour<br/>by 1 / distance"] --> E["X = weighted average<br/>of the neighbours"]
        E --> G["Imputed dataset"]
    end
    R1 --> R2
```

## MissForest

<!-- diagram: m04-missforest -->
```mermaid
flowchart TB
    subgraph R1["1 · Start"]
        direction LR
        A["Dataset with<br/>missing values"] --> B["Temporary fill:<br/>mean (numeric)<br/>mode (categorical)"]
        B --> C["Order columns by<br/>number missing<br/>(fewest first)"]
    end
    subgraph R2["2 · One round (repeat until the change is tiny, max 5–6 rounds)"]
        direction LR
        D["Next column X<sub>j</sub>"] --> E["Train a random forest<br/>X<sub>j</sub> ~ other columns<br/>(rows with X<sub>j</sub> observed)"]
        E --> F["Replace missing X<sub>j</sub><br/>with forest predictions"]
        F --> G{"All columns<br/>done?"}
        G -- "no" --> D
    end
    R1 --> R2 --> H["Imputed dataset"]
```

## Mean / Median / Mode imputation

<!-- diagram: m05-mean-mode -->
```mermaid
flowchart LR
    A["Column with<br/>missing values"] --> B{"Column<br/>type?"}
    B -- "numeric" --> C["Column MEAN<br/>(MEDIAN if skewed)"]
    B -- "categorical" --> D["Column MODE<br/>(most frequent value)"]
    C --> E["Write that single value<br/>into every missing cell"]
    D --> E
    E --> G["Imputed dataset<br/>(repeat per column)"]
```

## NMPrediction (Algorithm 1)

<!-- diagram: m06-nmprediction -->
```mermaid
flowchart TB
    subgraph R1["Training · steps 1–2"]
        direction LR
        A["Training data<br/>with missing values"] --> B["1 · Impute<br/>with NMI"]
        B --> C["2 · GA wrapper picks<br/>influential features"]
    end
    subgraph R2["Training · steps 3–4"]
        direction LR
        D["3 · Train an Alternating<br/>Decision Tree (ADT)"] --> E["4 · Stratified 10-fold CV<br/>Accuracy · AUC · SE · SP"]
    end
    subgraph R3["Using the model"]
        direction LR
        F["New patient"] --> G["Keep the<br/>selected features"]
        G --> H["ADT score"] --> I["Diagnosis"]
    end
    R1 --> R2 --> R3
```

## GA wrapper feature selection

<!-- diagram: m07-ga-wrapper -->
```mermaid
flowchart TB
    subgraph R1["1 · Start"]
        direction LR
        A["Random population<br/>of bit masks<br/>1 = keep · 0 = drop"] --> B["Fitness = 3-fold CV<br/>accuracy on kept features<br/>− small size penalty"]
    end
    subgraph R2["2 · Next generation (repeat for the set number of generations)"]
        direction LR
        C["Keep the 2 best<br/>(elitism)"] --> D["Pick parents<br/>(tournament of 3)"]
        D --> E["Crossover<br/>Pc = 1.0"]
        E --> F["Mutation<br/>Pm = 0.001"]
        F --> G["Score the<br/>new masks"]
    end
    R1 --> R2 --> H["Best mask =<br/>influential features"]
```

## Alternating Decision Tree (ADT)

<!-- diagram: m08-adt -->
```mermaid
flowchart TB
    subgraph R1["1 · Training (boosting, 40 rounds in DengueCAD)"]
        direction LR
        A["Training rows,<br/>equal weights"] --> B["Find the best simple rule<br/>(stump, e.g. platelets &lt; 100k?)"]
        B --> C["Attach rule with a score<br/>for YES and for NO"]
        C --> D["Up-weight rows the<br/>model gets wrong"]
        D -- "next round" --> B
    end
    subgraph R2["2 · Classify a patient"]
        direction LR
        F["Add the scores of every<br/>rule the patient meets"] --> G{"Total<br/>&gt; 0?"}
        G -- "yes" --> H["Positive<br/>(dengue)"]
        G -- "no" --> I["Negative"]
    end
    R1 --> R2
```

## C4.5 decision tree

<!-- diagram: m09-c45 -->
```mermaid
flowchart TB
    subgraph R1["1 · Grow the tree (start at the root with all rows)"]
        direction LR
        B{"All rows in this<br/>node same class?"} -- "no" --> C["Information gain (ratio)<br/>for every attribute"]
        C --> D["Split on the best<br/>attribute / threshold"]
        D -- "repeat for each child node" --> B
        B -- "yes" --> L["Leaf = that class"]
    end
    subgraph R2["2 · Predict"]
        direction LR
        P["New record"] --> Q["Follow the tests from<br/>root to a leaf"] --> S["Leaf class"]
    end
    R1 --> R2
```

## Logistic regression (LOR)

<!-- diagram: m10-lor -->
```mermaid
flowchart TB
    subgraph R1["1 · Fit (repeat until the weights stop changing)"]
        direction LR
        B["Standardise features<br/>(mean 0, sd 1)"] --> C["Score<br/>z = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + … + b<sub>p</sub>x<sub>p</sub>"]
        C --> D["Probability<br/>p = 1 / (1 + e<sup>−z</sup>)"]
        D --> E["Update weights b to fit<br/>the true labels better<br/>(L-BFGS, L2 penalty)"]
        E -- "iterate" --> C
    end
    subgraph R2["2 · Predict"]
        direction LR
        F["New record"] --> G["p ≥ 0.5 → positive<br/>p &lt; 0.5 → negative"]
    end
    R1 --> R2
```

## Support Vector Machine (SVM, RBF kernel)

<!-- diagram: m11-svm -->
```mermaid
flowchart TB
    subgraph R1["1 · Train"]
        direction LR
        A["Standardise<br/>features"] --> B["RBF kernel similarity<br/>K(x, x') = exp(−γ ‖x − x'‖²)"]
        B --> C["Find the widest-margin<br/>boundary between classes<br/>(C = 1 balances margin vs errors)"]
        C --> D["Rows on the margin =<br/>support vectors"]
    end
    subgraph R2["2 · Predict"]
        direction LR
        E["New record"] --> F["Weighted similarity to<br/>the support vectors"] --> G["Sign → class"]
    end
    R1 --> R2
```
