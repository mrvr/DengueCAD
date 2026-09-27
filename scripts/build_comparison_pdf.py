#!/usr/bin/env python3
"""
Build a PDF report of DengueCAD performance comparisons.

Writes: docs/DengueCAD_Performance_Comparisons.pdf
  Part I  — dengue-related experiments
  Part II — non-dengue / multi-disease / MissForest experiments
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "DengueCAD_Performance_Comparisons.pdf"


def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontSize=18,
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1a365d"),
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2c5282"),
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["Normal"],
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "note": ParagraphStyle(
            "NoteCustom",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#744210"),
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4a5568"),
            spaceBefore=2,
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "FooterCustom",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }
    return styles


NM_LABELS = {
    "NM",
    "NM (NMI)",
    "NMI",
    "NMPrediction",
}

BEST_BG = colors.HexColor("#fc8181")  # red — best performer
NM_BG = colors.HexColor("#c6f6d5")  # green — NM-related when not best
HEADER_BG = colors.HexColor("#2c5282")


def _is_nm_related(label: object) -> bool:
    s = str(label).strip()
    if s in NM_LABELS:
        return True
    return s.startswith("NM") or ("NMI" in s)


def _best_row_indices(df: pd.DataFrame) -> set[int]:
    """
    Return positional (0-based) row indices that are best performers.

    - ``Dataset`` + ``Accuracy (%)`` → max Accuracy within each Dataset
    - ``Accuracy (%)`` only → global max
    - ``NRMSE_proxy`` / ``NRMSE`` → global min (lower is better)
    - ``Approx match (%)`` → global max
    - ``Imputer``/``Method`` + ADT/C4.5/LOR/SVM → max mean of those columns
    """
    best: set[int] = set()
    cols = list(df.columns)
    index_list = list(df.index)

    def _pos(i) -> int:
        return index_list.index(i)

    if "Accuracy (%)" in cols:
        acc = pd.to_numeric(df["Accuracy (%)"], errors="coerce")
        if "Dataset" in cols:
            for _, grp in df.groupby(df["Dataset"].astype(str), sort=False):
                vals = acc.loc[grp.index]
                if not vals.notna().any():
                    continue
                m = float(vals.max())
                for i in grp.index:
                    v = acc.loc[i]
                    if pd.notna(v) and abs(float(v) - m) < 1e-9:
                        best.add(_pos(i))
        else:
            if acc.notna().any():
                m = float(acc.max())
                for i, v in acc.items():
                    if pd.notna(v) and abs(float(v) - m) < 1e-9:
                        best.add(_pos(i))
        return best

    for nrmse_col in ("NRMSE_proxy", "NRMSE"):
        if nrmse_col in cols:
            err = pd.to_numeric(df[nrmse_col], errors="coerce")
            if err.notna().any():
                m = float(err.min())
                for i, v in err.items():
                    if pd.notna(v) and abs(float(v) - m) < 1e-9:
                        best.add(_pos(i))
            return best

    if "Approx match (%)" in cols:
        mcol = pd.to_numeric(df["Approx match (%)"], errors="coerce")
        if mcol.notna().any():
            m = float(mcol.max())
            for i, v in mcol.items():
                if pd.notna(v) and abs(float(v) - m) < 1e-9:
                    best.add(_pos(i))
        return best

    clf_cols = [c for c in ("ADT", "C4.5", "LOR", "SVM") if c in cols]
    if clf_cols and (("Imputer" in cols) or ("Method" in cols)):
        means = df[clf_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if means.notna().any():
            m = float(means.max())
            for i, v in means.items():
                if pd.notna(v) and abs(float(v) - m) < 1e-9:
                    best.add(_pos(i))
        return best

    return best


def _best_cells_in_row(df: pd.DataFrame) -> dict[int, set[int]]:
    """Wide pivots: positional row → column indices tying for best score."""
    score_cols = [
        c
        for c in df.columns
        if str(c).endswith("Acc (%)") or str(c) in {"ADT", "C4.5", "LOR", "SVM"}
    ]
    if len(score_cols) < 2:
        return {}
    out: dict[int, set[int]] = {}
    col_pos = {c: list(df.columns).index(c) for c in score_cols}
    for pos, (_, row) in enumerate(df.iterrows()):
        vals = {}
        for c in score_cols:
            v = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(v):
                vals[c] = float(v)
        if not vals:
            continue
        m = max(vals.values())
        out[pos] = {col_pos[c] for c, v in vals.items() if abs(v - m) < 1e-9}
    return out


def _df_to_table(df: pd.DataFrame, col_widths=None, *, highlight: bool = True) -> Table:
    data = [list(df.columns.astype(str))]
    for _, row in df.iterrows():
        data.append([str(x) if pd.notna(x) else "" for x in row.tolist()])
    wrapped = []
    for r_i, row in enumerate(data):
        wrapped.append(
            [
                Paragraph(
                    str(c).replace("&", "&amp;").replace("<", "&lt;"),
                    ParagraphStyle(
                        "cell",
                        fontSize=9 if r_i else 10,
                        leading=12,
                        fontName="Helvetica-Bold" if r_i == 0 else "Helvetica",
                    ),
                )
                for c in row
            ]
        )
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    if highlight:
        label_col = next((c for c in ("Method", "Imputer") if c in df.columns), None)
        best_rows = _best_row_indices(df) if label_col else set()
        best_cells = {} if label_col else _best_cells_in_row(df)

        if label_col:
            for pos, val in enumerate(df[label_col].tolist()):
                table_row = pos + 1
                is_best = pos in best_rows
                is_nm = _is_nm_related(val)
                if is_best:
                    style_cmds.append(("BACKGROUND", (0, table_row), (-1, table_row), BEST_BG))
                    style_cmds.append(("FONTNAME", (0, table_row), (-1, table_row), "Helvetica-Bold"))
                    style_cmds.append(
                        ("TEXTCOLOR", (0, table_row), (-1, table_row), colors.HexColor("#742a2a"))
                    )
                elif is_nm:
                    style_cmds.append(("BACKGROUND", (0, table_row), (-1, table_row), NM_BG))
                    style_cmds.append(("FONTNAME", (0, table_row), (-1, table_row), "Helvetica-Bold"))

        nm_acc_cols = {
            i
            for i, c in enumerate(df.columns)
            if str(c).startswith("NM") and "Acc" in str(c)
        }
        for pos, winners in best_cells.items():
            table_row = pos + 1
            for col_i in winners:
                style_cmds.append(
                    ("BACKGROUND", (col_i, table_row), (col_i, table_row), BEST_BG)
                )
                style_cmds.append(
                    ("FONTNAME", (col_i, table_row), (col_i, table_row), "Helvetica-Bold")
                )
            for col_i in nm_acc_cols:
                if col_i not in winners:
                    style_cmds.append(
                        ("BACKGROUND", (col_i, table_row), (col_i, table_row), NM_BG)
                    )
                    style_cmds.append(
                        ("FONTNAME", (col_i, table_row), (col_i, table_row), "Helvetica-Bold")
                    )

    t.setStyle(TableStyle(style_cmds))
    return t


def _load(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    return None


def _section_table(story, styles, title: str, caption: str, df: pd.DataFrame | None, widths=None):
    story.append(Paragraph(title, styles["h2"]))
    if df is None or df.empty:
        story.append(Paragraph("<i>Results file not found.</i>", styles["note"]))
        return
    story.append(_df_to_table(df, col_widths=widths))
    story.append(Paragraph(caption, styles["caption"]))


def build_pdf() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=landscape(A4),
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="DengueCAD Performance Comparisons",
        author="DengueCAD",
    )
    story = []

    # Cover
    story.append(Paragraph("DengueCAD — Performance Comparison Report", styles["title"]))
    story.append(
        Paragraph(
            "NMPrediction (Algorithm 1: NMI imputation → GA wrapper + ADT feature selection → "
            "stratified k-fold ADT) compared with C4.5, logistic regression (LOR), and SVM-RBF. "
            "Based on Rao &amp; Kumar, IEEE TITB 2012.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Part I covers <b>dengue-related</b> experiments. "
            "Part II covers <b>non-dengue</b> multi-disease ARFF benchmarks and the MissForest Iris comparison.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 4))

    # ----- Abbreviations legend -----
    story.append(Paragraph("Legend — abbreviations used in this report", styles["h1"]))
    story.append(
        Paragraph(
            "<b>MICE</b> means <i>Multiple Imputation by Chained Equations</i>: an iterative method that "
            "models each incomplete column as a function of the others and cycles until convergence. "
            "In DengueCAD it is implemented with scikit-learn’s <font face='Courier'>IterativeImputer</font> "
            "(a MICE-style chained equations imputer).",
            styles["body"],
        )
    )

    legend_rows = [
        ["Abbreviation", "Full form / meaning"],
        ["ADT", "Alternating Decision Tree (Freund & Mason); used inside NMPrediction"],
        ["ARFF", "Attribute-Relation File Format (Weka / KEEL datasets)"],
        ["AUC", "Area Under the ROC Curve"],
        ["C4.5", "Quinlan decision-tree learner (entropy / information-gain style; CART stand-in in scikit-learn)"],
        ["CV", "Cross-Validation (here: stratified k-fold)"],
        ["dengue (train)", "Single 70k-row training block from dengue.csv for NMI / NMPrediction"],
        ["GA", "Genetic Algorithm (wrapper search for influential feature subsets)"],
        ["kNN", "k-Nearest Neighbours imputation"],
        ["LOR", "Logistic Regression"],
        ["MCAR", "Missing Completely At Random (synthetic missingness pattern)"],
        ["MICE", "Multiple Imputation by Chained Equations (IterativeImputer in this project)"],
        ["MissForest", "Random-forest–based missing-value imputation (Stekhoven & Bühlmann)"],
        ["MV", "Missing Value(s)"],
        ["NM", "New Methodology / NMPrediction (IEEE TITB 2012 Algorithm 1)"],
        ["NMI", "Non-parametric Missing-value Imputation (indexing / nearest-neighbour method)"],
        ["NRMSE", "Normalized Root Mean Squared Error (imputation error for continuous attributes)"],
        ["PFC", "Proportion of Falsely Classified entries (imputation error for categoricals)"],
        ["RBF", "Radial Basis Function kernel (SVM)"],
        ["ROC", "Receiver Operating Characteristic curve"],
        ["SE", "Sensitivity (true positive rate / recall)"],
        ["SP", "Specificity (true negative rate)"],
        ["SVM", "Support Vector Machine"],
        ["TITB / JBHI", "IEEE Transactions on Information Technology in Biomedicine / Journal of Biomedical Health Informatics"],
        ["UCI", "University of California, Irvine Machine Learning Repository"],
        ["Weka / KEEL", "Open-source ML workbenches; source of the multi-disease ARFF files"],
    ]
    legend_df = pd.DataFrame(legend_rows[1:], columns=legend_rows[0])
    story.append(_df_to_table(legend_df, col_widths=[3.5 * cm, 20 * cm], highlight=False))
    story.append(
        Paragraph(
            "<b>Row colours in result tables:</b> "
            "<font color='#c53030'><b>Red</b></font> = best-performing model in that comparison "
            "(including NM / NMI when it wins); "
            "<font color='#276749'><b>Green</b></font> = NM / NMI / NMPrediction row when it is "
            "<i>not</i> the best performer.",
            styles["caption"],
        )
    )
    story.append(PageBreak())

    # ========== PART I: DENGUE ==========
    story.append(Paragraph("Part I — Dengue-related performance comparisons", styles["h1"]))
    story.append(
        Paragraph(
            "Datasets: synthetic/vendored <font face='Courier'>dengue.csv</font> (70k single train set), "
            "Bangladesh <font face='Courier'>dataset.csv</font>, and "
            "<font face='Courier'>Dengue_clinical_dataset.csv</font> (Munshiganj clinical/hematology).",
            styles["body"],
        )
    )

    # dengue.csv single 70k train
    story.append(Paragraph("1. Vendored dengue.csv — NMPrediction (70k single dataset)", styles["h2"]))
    story.append(
        Paragraph(
            "Protocol: first 70,000 rows as one training dataset for NMI / NMPrediction; "
            "10,000 complete-case rows → validation; 20,000 complete-case rows → test (performance). "
            "Note: this file has only four binary symptoms and near-chance label signal, so accuracies "
            "cluster near 50%.",
            styles["note"],
        )
    )
    _section_table(
        story,
        styles,
        "Table I — Performance (Accuracy / SE / SP / AUC)",
        "Source: results/table1_performance.csv",
        _load(RESULTS / "table1_performance.csv"),
        widths=[2.2 * cm, 2.0 * cm, 2.8 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm],
    )
    _section_table(
        story,
        styles,
        "Table II — Wilcoxon matched-pairs (NM vs baselines)",
        "Source: results/table2_wilcoxon.csv",
        _load(RESULTS / "table2_wilcoxon.csv"),
        widths=[2.5 * cm, 2.5 * cm, 4.5 * cm, 2.5 * cm],
    )
    _section_table(
        story,
        styles,
        "Table III — Influential features (NM)",
        "Source: results/table3_influential_features.csv",
        _load(RESULTS / "table3_influential_features.csv"),
        widths=[2.2 * cm, 3.2 * cm, 3.5 * cm, 2.8 * cm, 6 * cm],
    )
    _section_table(
        story,
        styles,
        "Hold-out validation (10k) / test (20k) complete-case",
        "Source: results/holdout_metrics.csv",
        _load(RESULTS / "holdout_metrics.csv"),
        widths=[2.0 * cm, 2.5 * cm, 1.8 * cm, 2.5 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 4 * cm],
    )

    story.append(PageBreak())

    # Dengue clinical Munshiganj
    story.append(Paragraph("2. Dengue_clinical_dataset.csv (clinical + hematology)", styles["h2"]))
    story.append(
        Paragraph(
            "1,018 records (697 dengue-positive / 321 negative). Features include Gender, Age, "
            "Platelet Count, WBC, Location, Fever, Duration_of_Fever, Headache, Muscle_Pain, Rash, Vomiting. "
            "NM selected Platelet Count, WBC, Fever.",
            styles["body"],
        )
    )
    _section_table(
        story,
        styles,
        "Table I — Performance",
        "Source: results/nmprediction_clinical/table1_performance.csv",
        _load(RESULTS / "nmprediction_clinical" / "table1_performance.csv"),
        widths=[3.5 * cm, 2.2 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm],
    )
    _section_table(
        story,
        styles,
        "Table II — Wilcoxon",
        "Source: results/nmprediction_clinical/table2_wilcoxon.csv",
        _load(RESULTS / "nmprediction_clinical" / "table2_wilcoxon.csv"),
        widths=[3.5 * cm, 2.5 * cm, 4.5 * cm, 2.5 * cm],
    )
    _section_table(
        story,
        styles,
        "Table III — Influential features",
        "Source: results/nmprediction_clinical/table3_influential_features.csv",
        _load(RESULTS / "nmprediction_clinical" / "table3_influential_features.csv"),
        widths=[3.2 * cm, 3 * cm, 3.2 * cm, 2.8 * cm, 7 * cm],
    )

    # dataset.csv Bangladesh
    story.append(Paragraph("3. dataset.csv (Bangladesh comprehensive clinical set)", styles["h2"]))
    story.append(
        Paragraph(
            "<b>Data caveat:</b> In this file IgG equals Outcome for all 1,000 rows (perfect label leak). "
            "Body_Temperature alone also yields 100% majority accuracy. Perfect scores below reflect the "
            "dataset structure. A clinical-only run (excluding IgG/NS1) is also shown.",
            styles["note"],
        )
    )
    _section_table(
        story,
        styles,
        "Table I — All features (includes IgG)",
        "Source: results/nmprediction_dataset_csv/table1_performance.csv",
        _load(RESULTS / "nmprediction_dataset_csv" / "table1_performance.csv"),
        widths=[3.5 * cm, 2.2 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm],
    )
    _section_table(
        story,
        styles,
        "Table I — Clinical features only (no IgG / NS1)",
        "Source: results/nmprediction_dataset_csv/table1_performance_clinical.csv",
        _load(RESULTS / "nmprediction_dataset_csv" / "table1_performance_clinical.csv"),
        widths=[4.5 * cm, 2.2 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm],
    )
    _section_table(
        story,
        styles,
        "Table III — Influential features (all features run)",
        "Source: results/nmprediction_dataset_csv/table3_influential_features.csv",
        _load(RESULTS / "nmprediction_dataset_csv" / "table3_influential_features.csv"),
        widths=[3.2 * cm, 3 * cm, 3.2 * cm, 2.8 * cm, 6 * cm],
    )

    # Imputation on dataset.csv
    story.append(Paragraph("4. Missing-value imputation on dataset.csv (Joint_Pain MVs)", styles["h2"]))
    story.append(
        Paragraph(
            "Natural missingness: 491 cells in Joint_Pain. Controlled benchmark: complete cases, "
            "12% MCAR mask, compare NMI vs Mode/Mean, Mean, Median, kNN, MICE.",
            styles["body"],
        )
    )
    _section_table(
        story,
        styles,
        "Imputation accuracy comparison",
        "Source: results/imputation_dataset_csv/imputation_comparison.csv",
        _load(RESULTS / "imputation_dataset_csv" / "imputation_comparison.csv"),
        widths=[3.5 * cm, 3 * cm, 3 * cm, 2.8 * cm, 2.5 * cm],
    )

    story.append(PageBreak())

    # ========== PART II: NON-DENGUE ==========
    story.append(Paragraph("Part II — Non-dengue performance comparisons", styles["h1"]))
    story.append(
        Paragraph(
            "These experiments show that NMI / NMPrediction are <b>disease-agnostic</b>: "
            "the same pipeline runs on Weka/KEEL ARFF medical datasets and on the canonical "
            "MissForest Iris demo (UCI).",
            styles["body"],
        )
    )

    # ARFF multi-disease
    story.append(Paragraph("5. Multi-disease ARFF benchmark (11 datasets)", styles["h2"]))
    story.append(
        Paragraph(
            "Source folder: <font face='Courier'>experiments/datasets/*.arff</font> "
            "(appendicitis, bupa, contractions, heart, hepatitis, laryngeal1, lplaudiob, pima, rds, weaning, wisconsin).",
            styles["body"],
        )
    )
    arff_t1 = _load(RESULTS / "arff_disease_comparison" / "ALL_table1_performance.csv")
    if arff_t1 is not None:
        # Compact accuracy pivot for overview
        piv = arff_t1.pivot(index="Dataset", columns="Method", values="Accuracy (%)")
        for col in ("NM", "C4.5", "LOR", "SVM"):
            if col not in piv.columns:
                piv[col] = None
        piv = piv[["NM", "C4.5", "LOR", "SVM"]].reset_index()
        piv.columns = ["Dataset", "NM Acc (%)", "C4.5 Acc (%)", "LOR Acc (%)", "SVM Acc (%)"]
        _section_table(
            story,
            styles,
            "Overview — Accuracy (%) by method",
            "Pivoted from ALL_table1_performance.csv. NM best on 5/11 datasets in this run.",
            piv,
            widths=[3.5 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm],
        )

    _section_table(
        story,
        styles,
        "Table I — Full metrics (Accuracy / SE / SP / AUC)",
        "Source: results/arff_disease_comparison/ALL_table1_performance.csv",
        arff_t1,
        widths=[2.8 * cm, 2.0 * cm, 2.8 * cm, 2.2 * cm, 2.2 * cm, 1.8 * cm],
    )

    story.append(PageBreak())

    _section_table(
        story,
        styles,
        "Table II — Wilcoxon (NM vs baselines)",
        "Source: results/arff_disease_comparison/ALL_table2_wilcoxon.csv",
        _load(RESULTS / "arff_disease_comparison" / "ALL_table2_wilcoxon.csv"),
        widths=[3.0 * cm, 2.5 * cm, 5.0 * cm, 2.5 * cm],
    )
    _section_table(
        story,
        styles,
        "Table III — Influential feature subsets (NM)",
        "Source: results/arff_disease_comparison/ALL_table3_influential_features.csv",
        _load(RESULTS / "arff_disease_comparison" / "ALL_table3_influential_features.csv"),
        widths=[2.5 * cm, 2.5 * cm, 2.8 * cm, 2.5 * cm, 9 * cm],
    )

    # MissForest Iris
    story.append(Paragraph("6. MissForest canonical dataset (UCI Iris) — imputation + prediction", styles["h2"]))
    story.append(
        Paragraph(
            "Dataset: UCI Iris (Fisher 1936), the demo set used in the missForest R package "
            "(Stekhoven &amp; Bühlmann). Protocol: 20% MCAR on attributes, then impute and classify. "
            "Local copy: <font face='Courier'>data/missforest_iris/iris.csv</font>.",
            styles["body"],
        )
    )
    _section_table(
        story,
        styles,
        "Table A — Imputation error (lower NRMSE better)",
        "Source: results/missforest_comparison/table_imputation.csv",
        _load(RESULTS / "missforest_comparison" / "table_imputation.csv"),
        widths=[4 * cm, 3 * cm, 3.5 * cm, 2.8 * cm, 2.5 * cm],
    )
    _section_table(
        story,
        styles,
        "Table B — Downstream prediction accuracy (%) after imputation",
        "Source: results/missforest_comparison/table_prediction.csv",
        _load(RESULTS / "missforest_comparison" / "table_prediction.csv"),
        widths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm],
    )

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Generated by DengueCAD <font face='Courier'>scripts/build_comparison_pdf.py</font>. "
            "See the <b>Legend</b> at the start of this document for all abbreviations "
            "(including <b>MICE</b> = Multiple Imputation by Chained Equations).",
            styles["footer"],
        )
    )

    doc.build(story)
    return OUT


if __name__ == "__main__":
    path = build_pdf()
    print(f"Wrote {path}")
