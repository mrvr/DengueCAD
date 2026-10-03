#!/usr/bin/env python3
"""
PDF report for the imputation + NMPrediction hold-out experiment.

Reads results/holdout_imputation_prediction/{dengue,multi}/summary.csv and
writes docs/DengueCAD_Imputation_Prediction_Holdout.pdf with four tables:
  1.1 Dengue imputation accuracy      1.2 Dengue prediction accuracy
  2.1 Multi-disease imputation        2.2 Multi-disease prediction
Best result per row: light green. NM method (NMI / NMPrediction): light yellow.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "holdout_imputation_prediction"
OUT = ROOT / "docs" / "DengueCAD_Imputation_Prediction_Holdout.pdf"

IMPUTERS = ["NMI", "MICE", "kNN", "MissForest", "Mean/Mode"]
PREDICTORS = ["NMPrediction", "C4.5", "LOR", "SVM"]

BEST = colors.HexColor("#C6EFCE")  # light green
NM = colors.HexColor("#FFF2CC")  # light yellow
HEADER = colors.HexColor("#DDE5F0")  # light blue-grey
ZEBRA = colors.HexColor("#F7F9FC")
TEXT = colors.HexColor("#1A202C")
GRID = colors.HexColor("#B8C4D6")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, textColor=TEXT, spaceAfter=6),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=12, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#4A5568"), spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, textColor=colors.HexColor("#2C5282"),
                             spaceBefore=6, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=colors.HexColor("#2C5282"),
                             spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=11, leading=15, alignment=TA_JUSTIFY,
                               textColor=TEXT, spaceAfter=6),
        "note": ParagraphStyle("n", parent=base["Normal"], fontSize=10, leading=13,
                               textColor=colors.HexColor("#4A5568"), spaceAfter=6),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=10, leading=12, alignment=TA_CENTER,
                               textColor=TEXT),
        "cell_left": ParagraphStyle("cl", parent=base["Normal"], fontSize=10, leading=12, textColor=TEXT),
        "head": ParagraphStyle("h", parent=base["Normal"], fontSize=10, leading=12, alignment=TA_CENTER,
                               fontName="Helvetica-Bold", textColor=TEXT),
    }


def _load(group: str) -> pd.DataFrame | None:
    path = RESULTS / group / "summary.csv"
    return pd.read_csv(path) if path.is_file() else None


def _result_table(df: pd.DataFrame, prefix: str, methods: list[str], nm: str,
                  size_cols: list[tuple[str, str]], styles) -> Table:
    """Methods as columns; best mean per row green, NM column yellow."""
    means = pd.DataFrame({m: 100 * df[f"{prefix}{m}_mean"] for m in methods}).round(1)
    sds = pd.DataFrame({m: 100 * df[f"{prefix}{m}_std"].fillna(0) for m in methods}).round(1)

    header = ["Dataset", *[label for _, label in size_cols], *methods]
    data = [[Paragraph(h, styles["head"]) for h in header]]
    cell_styles: list[tuple] = []
    first_method_col = 1 + len(size_cols)
    nm_col = first_method_col + methods.index(nm)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        best = means.iloc[i - 1].max()
        cells = [Paragraph(str(row["Dataset"]), styles["cell_left"])]
        cells += [Paragraph(f"{int(round(row[c])):,}", styles["cell"]) for c, _ in size_cols]
        for j, m in enumerate(methods):
            mean, sd = means.iloc[i - 1][m], sds.iloc[i - 1][m]
            is_best = mean == best
            text = f"{mean:.1f} ± {sd:.1f}"
            if is_best and m == nm:
                text = f"<b>{text}</b>"
            cells.append(Paragraph(text, styles["cell"]))
            col = first_method_col + j
            if is_best:
                cell_styles.append(("BACKGROUND", (col, i), (col, i), BEST))
            elif m == nm:
                cell_styles.append(("BACKGROUND", (col, i), (col, i), NM))
        data.append(cells)

    # Average and win-count rows
    avg = means.mean().round(1)
    wins = (means.eq(means.max(axis=1), axis=0)).sum()
    for label, values, fmt in (("Average", avg, "{:.1f}"), ("Datasets best (incl. ties)", wins, "{:d}")):
        r = len(data)
        best = values.max()
        cells = [Paragraph(f"<b>{label}</b>", styles["cell_left"])] + [Paragraph("", styles["cell"])] * len(size_cols)
        for j, m in enumerate(methods):
            v = values[m]
            text = fmt.format(int(v) if fmt == "{:d}" else v)
            cells.append(Paragraph(f"<b>{text}</b>", styles["cell"]))
            col = first_method_col + j
            if v == best:
                cell_styles.append(("BACKGROUND", (col, r), (col, r), BEST))
            elif m == nm:
                cell_styles.append(("BACKGROUND", (col, r), (col, r), NM))
        data.append(cells)

    widths = [5.0 * cm, *[2.3 * cm] * len(size_cols), *[(26.5 - 5.0 - 2.3 * len(size_cols)) / len(methods) * cm] * len(methods)]
    t = Table(data, colWidths=widths, repeatRows=1)
    n = len(data)
    base = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER),
        ("BACKGROUND", (nm_col, 0), (nm_col, 0), NM),
        ("ROWBACKGROUNDS", (0, 1), (first_method_col - 1, n - 1), [colors.white, ZEBRA]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("LINEABOVE", (0, n - 2), (-1, n - 2), 1.2, colors.HexColor("#718096")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(base + cell_styles))
    return t


def _legend(styles) -> Table:
    data = [[
        Paragraph("Best result in the row (ties all marked)", styles["cell_left"]), "",
        Paragraph("New Method: NMI (imputation) / NMPrediction (prediction)", styles["cell_left"]), "",
    ]]
    t = Table(data, colWidths=[8 * cm, 0.3 * cm, 9.5 * cm, 0.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BEST),
        ("BACKGROUND", (2, 0), (2, 0), NM),
        ("BOX", (0, 0), (0, 0), 0.5, GRID),
        ("BOX", (2, 0), (2, 0), 0.5, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build() -> Path:
    styles = _styles()
    dengue, multi = _load("dengue"), _load("multi")
    if dengue is None and multi is None:
        raise SystemExit(f"No summary.csv found under {RESULTS}; run run_holdout_imputation_prediction.py")

    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="DengueCAD — Imputation and NMPrediction hold-out comparison")
    s = []
    s.append(Paragraph("Imputation and Prediction Accuracy: New Method vs Baselines", styles["title"]))
    s.append(Paragraph(f"DengueCAD hold-out experiment · NMI and NMPrediction · {date.today():%d %B %Y}",
                       styles["subtitle"]))

    s.append(Paragraph("Protocol", styles["h2"]))
    for step in (
        "<b>1. valid_set.</b> Records without missing values (MVs) are copied aside as ground truth.",
        "<b>2. Missing values.</b> 20% of the valid records get 1–3 randomly chosen attributes removed "
        "(missing completely at random; the decision column is never touched). Natural MVs already in the "
        "data are kept.",
        "<b>3. Imputed dataset.</b> The incomplete set is imputed with the New Method Imputation (NMI) and, "
        "for comparison, MICE, kNN, MissForest and Mean/Mode. <i>Imputation accuracy</i> is the percentage "
        "of removed cells restored to their true value from valid_set.",
        "<b>4. train_set.</b> 30% of valid_set (stratified by class) is held out for testing; removing those "
        "records from the NMI-imputed dataset gives train_set.",
        "<b>5. Test dataset.</b> The held-out records are taken from valid_set and the decision (last column) "
        "is removed for 20% of them. NMPrediction, C4.5, LOR and SVM, all trained on train_set, predict those "
        "decisions. <i>Prediction accuracy</i> is the percentage matching the true valid_set decision.",
        "<b>6. Repeats.</b> Steps 2–5 are repeated with different random seeds (10 repeats; 3 for the 70k "
        "dengue.csv sample). Tables show mean ± standard deviation in %.",
    ):
        s.append(Paragraph(step, styles["body"]))
    s.append(Spacer(1, 4))
    s.append(_legend(styles))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Attributes are encoded as integer codes: nominal values by label, numeric values by their ordered "
        "value or decile bin when there are more than 20 distinct values. For continuous attributes an imputed "
        "cell therefore counts as correct when it falls in the true decile bin. Imputations from MICE and kNN "
        "are rounded to the nearest valid code.", styles["note"]))

    def section(title: str, df: pd.DataFrame | None, prefix: str, methods, nm, size_cols, note: str):
        s.append(Paragraph(title, styles["h2"]))
        if df is None:
            s.append(Paragraph("Results not available.", styles["note"]))
            return
        s.append(_result_table(df, prefix, methods, nm, size_cols, styles))
        s.append(Spacer(1, 4))
        s.append(Paragraph(note, styles["note"]))

    imp_sizes = [("n_rows", "Records"), ("n_masked_cells", "Removed cells / repeat")]
    pred_sizes = [("n_train", "train_set"), ("n_predicted", "Predicted / repeat")]

    s.append(PageBreak())
    s.append(Paragraph("1. Dengue datasets", styles["h1"]))
    section("1.1 Dengue datasets — imputation accuracy (%)", dengue, "imp_", IMPUTERS, "NMI", imp_sizes,
            "dengue.csv uses its first 70,000 records. Dengue_diseases (CBC) and dataset.csv also contain natural "
            "MVs, which are imputed but not scored (they have no ground truth).")
    s.append(PageBreak())
    section("1.2 Dengue datasets — prediction accuracy (%)", dengue, "pred_", PREDICTORS, "NMPrediction",
            pred_sizes,
            "All four predictors are trained on the same NMI-imputed train_set. In dataset.csv the IgG test "
            "result equals the outcome, so every method can reach 100%. dengue.csv has only four binary "
            "symptoms with almost no class signal, so all methods sit near 50%.")

    s.append(PageBreak())
    s.append(Paragraph("2. Multi-disease datasets (KEEL / Weka ARFF)", styles["h1"]))
    section("2.1 Multi-disease — imputation accuracy (%)", multi, "imp_", IMPUTERS, "NMI", imp_sizes,
            "None of the ARFF files has natural MVs, so all removed cells come from step 2.")
    s.append(PageBreak())
    section("2.2 Multi-disease — prediction accuracy (%)", multi, "pred_", PREDICTORS, "NMPrediction",
            pred_sizes,
            "Small datasets have only a handful of predicted records per repeat (for example 5 for rds), so "
            "single-repeat accuracies move in large steps; the mean over repeats is the figure to compare.")

    s.append(PageBreak())
    s.append(Paragraph("Observations", styles["h1"]))
    frames = [d for d in (dengue, multi) if d is not None]
    both = pd.concat(frames, ignore_index=True)
    n_ds = len(both)
    imp = pd.DataFrame({m: (100 * both[f"imp_{m}_mean"]).round(1) for m in IMPUTERS})
    pred = pd.DataFrame({m: (100 * both[f"pred_{m}_mean"]).round(1) for m in PREDICTORS})
    imp_wins = imp.eq(imp.max(axis=1), axis=0).sum()
    pred_wins = pred.eq(pred.max(axis=1), axis=0).sum()
    nmi_best = ", ".join(both.loc[imp["NMI"] == imp.max(axis=1), "Dataset"]) or "none"
    nm_best = ", ".join(both.loc[pred["NMPrediction"] == pred.max(axis=1), "Dataset"]) or "none"
    nmi_vs_mode = int((imp["NMI"] > imp["Mean/Mode"]).sum())
    nm_vs_c45 = int((pred["NMPrediction"] > pred["C4.5"]).sum())
    gap = (pred.max(axis=1) - pred["NMPrediction"]).mean()

    s.append(Paragraph(
        f"<b>Imputation.</b> Across all {n_ds} datasets the best imputer counts are: "
        + ", ".join(f"{m} {int(imp_wins[m])}" for m in IMPUTERS)
        + f" (ties counted for each). NMI is the best imputer on {nmi_best}, and is more accurate than "
        f"Mean/Mode on {nmi_vs_mode} of {n_ds} datasets.", styles["body"]))
    s.append(Paragraph(
        "<b>Where NMI falls behind.</b> For each incomplete record NMI treats every complete record whose "
        "distance z-score is ≤ 0 as a neighbour, i.e. all donors closer than the average distance (roughly "
        "half of them), and imputes their most frequent value. On data dominated by continuous attributes "
        "(for example appendicitis and contractions, where values are decile bins) that wide neighbourhood "
        "pulls imputations towards the overall mode, while MICE, kNN and MissForest exploit the numeric "
        "relationships between attributes. A tighter neighbour rule (z ≤ −1, or the k closest donors) would "
        "make NMI more local on such data.", styles["body"]))
    s.append(Paragraph(
        f"<b>Prediction.</b> Best predictor counts: "
        + ", ".join(f"{m} {int(pred_wins[m])}" for m in PREDICTORS)
        + f". NMPrediction is best (or tied best) on {nm_best}; it beats C4.5 on {nm_vs_c45} of {n_ds} "
        f"datasets and is on average {gap:.1f} percentage points below the best method in each row. "
        "NMPrediction uses only the compact attribute subset chosen by the GA wrapper, while the baselines use "
        "all attributes. On small datasets one record changes accuracy by 5–20 points per repeat, so "
        "differences should be read together with the standard deviations.", styles["body"]))
    s.append(Paragraph(
        "<b>Abbreviations.</b> NMI: New Method (non-parametric) Imputation. NMPrediction: NM methodology, "
        "Algorithm 1 (NMI → GA wrapper feature selection → ADT). MICE: Multiple Imputation by Chained "
        "Equations. kNN: k-nearest-neighbour imputation (k = 5). MissForest: random-forest iterative "
        "imputation. C4.5: entropy decision tree. LOR: logistic regression. SVM: support vector machine "
        "(RBF kernel). ADT: Alternating Decision Tree. MCAR: missing completely at random.", styles["note"]))
    s.append(Paragraph(
        "Source data: results/holdout_imputation_prediction/{dengue,multi}/summary.csv and per_repeat.csv. "
        "Regenerate with scripts/run_holdout_imputation_prediction.py and "
        "scripts/build_holdout_comparison_pdf.py.", styles["note"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(s)
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
