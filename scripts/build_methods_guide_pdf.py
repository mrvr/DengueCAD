#!/usr/bin/env python3
"""
Reference guide to the imputation and prediction methods compared with NM.

Writes docs/DengueCAD_Methods_Guide.pdf: a plain-language explanation of each
method (NMI, MICE, kNN, MissForest, Mean/Mode; NMPrediction with its GA
wrapper and ADT, C4.5, LOR, SVM) with a flowchart, a small example, the
settings DengueCAD uses, strengths / limitations and references.

Flowcharts come from docs/methods/FLOWCHARTS.md; render them first with
    python scripts/render_architecture_diagrams.py \
        --source docs/methods/FLOWCHARTS.md --outdir docs/methods
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FLOWCHARTS = ROOT / "docs" / "methods"
OUT = ROOT / "docs" / "DengueCAD_Methods_Guide.pdf"

TEXT = colors.HexColor("#1A202C")
MUTED = colors.HexColor("#4A5568")
ACCENT = colors.HexColor("#2C5282")
GRID = colors.HexColor("#B8C4D6")
HEADER = colors.HexColor("#DDE5F0")  # light blue-grey
IDEA = colors.HexColor("#EBF4FF")  # light blue
NM = colors.HexColor("#FFF2CC")  # light yellow (NM methods)
GOOD = colors.HexColor("#E2F0D9")  # light green
CAUTION = colors.HexColor("#FCE4D6")  # light peach
ZEBRA = colors.HexColor("#F7F9FC")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

DEJAVU = Path("/usr/share/fonts/truetype/dejavu")


def _register_fonts() -> tuple[str, str, str]:
    """DejaVu Sans covers Greek letters and math symbols; fall back to Helvetica."""
    files = {
        "DejaVu": "DejaVuSans.ttf",
        "DejaVu-Bold": "DejaVuSans-Bold.ttf",
        "DejaVu-Oblique": "DejaVuSans-Oblique.ttf",
        "DejaVu-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    }
    if not all((DEJAVU / f).is_file() for f in files.values()):
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    for name, f in files.items():
        pdfmetrics.registerFont(TTFont(name, str(DEJAVU / f)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold",
        italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique",
    )
    return "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"


REGULAR, BOLD, ITALIC = _register_fonts()


def _styles() -> dict[str, ParagraphStyle]:
    def ps(name, **kw):
        base = dict(fontName=REGULAR, fontSize=10.2, leading=13.8, textColor=TEXT)
        base.update(kw)
        return ParagraphStyle(name, **base)

    return {
        "title": ps("title", fontName=BOLD, fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ps("subtitle", fontSize=13, leading=18, alignment=TA_CENTER, textColor=MUTED, spaceAfter=4),
        "h1": ps("h1", fontName=BOLD, fontSize=17, leading=22, textColor=ACCENT, spaceBefore=4, spaceAfter=8),
        "h2": ps("h2", fontName=BOLD, fontSize=13.5, leading=18, textColor=ACCENT, spaceBefore=10, spaceAfter=6),
        "h3": ps("h3", fontName=BOLD, fontSize=11.5, leading=15, textColor=TEXT, spaceBefore=8, spaceAfter=3),
        "body": ps("body", alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ps("bullet", spaceAfter=2),
        "caption": ps("caption", fontName=ITALIC, fontSize=9.5, leading=12, alignment=TA_CENTER,
                      textColor=MUTED, spaceBefore=3, spaceAfter=8),
        "box": ps("box", fontSize=11, leading=15),
        "formula": ps("formula", fontSize=11, leading=16, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
        "cell": ps("cell", fontSize=9.5, leading=12.5),
        "cell_b": ps("cell_b", fontName=BOLD, fontSize=9.5, leading=12.5),
        "ref": ps("ref", fontSize=9.5, leading=13, spaceAfter=4, leftIndent=18, firstLineIndent=-18),
    }


S = _styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str], numbered: bool = False) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(t, "bullet"), leftIndent=14) for t in items],
        bulletType="1" if numbered else "bullet",
        start="1" if numbered else "•",
        leftIndent=16,
        bulletFontName=REGULAR,
        bulletFontSize=10,
        spaceAfter=6,
    )


def box(text: str, fill=IDEA, label: str = "In one sentence") -> Table:
    t = Table([[P(f"<b>{label}:</b> {text}", "box")]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def flowchart(name: str, caption: str, max_h: float = 10 * cm) -> KeepTogether:
    path = FLOWCHARTS / f"{name}.png"
    if not path.is_file():
        raise SystemExit(f"Missing {path}; render docs/methods/FLOWCHARTS.md first.")
    w_px, h_px = PILImage.open(path).size
    w = CONTENT_W
    h = w * h_px / w_px
    if h > max_h:
        h, w = max_h, max_h * w_px / h_px
    return KeepTogether([Spacer(1, 4), Image(str(path), width=w, height=h), P(caption, "caption")])


def grid_table(rows: list[list[str]], col_widths: list[float], *, header_fill=HEADER,
               highlight_rows: dict[int, colors.Color] | None = None) -> Table:
    data = [[P(c, "cell_b" if r == 0 else "cell") for c in row] for r, row in enumerate(rows)]
    total = sum(col_widths)
    t = Table(data, colWidths=[CONTENT_W * w / total for w in col_widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    for r, fill in (highlight_rows or {}).items():
        style.append(("BACKGROUND", (0, r), (-1, r), fill))
    t.setStyle(TableStyle(style))
    return t


def pros_cons(pros: list[str], cons: list[str]) -> Table:
    def cell(items):
        return [P(f"• {i}", "cell") for i in items]

    t = Table(
        [[P("Strengths", "cell_b"), P("Watch out for", "cell_b")], [cell(pros), cell(cons)]],
        colWidths=[CONTENT_W / 2, CONTENT_W / 2],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), GOOD), ("BACKGROUND", (1, 0), (1, 0), CAUTION),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def settings_table(rows: list[tuple[str, str]]) -> Table:
    return grid_table([["Setting", "Value used in DengueCAD"], *[list(r) for r in rows]], [1.2, 2.8])


def refs_line(nums: list[int]) -> Paragraph:
    return P("<b>Learn more:</b> " + ", ".join(f"[{n}]" for n in nums), "body")


def method(story: list, *, title: str, one_liner: str, steps: list[str], chart: str,
           chart_caption: str, example: list, settings: list[tuple[str, str]],
           pros: list[str], cons: list[str], refs: list[int], nm: bool = False,
           extra: list | None = None, new_page: bool = True) -> None:
    if new_page:
        story.append(CondPageBreak(9 * cm))
    story.append(P(title, "h2"))
    story.append(box(one_liner, fill=NM if nm else IDEA))
    story.append(P("How it works", "h3"))
    story.append(bullets(steps, numbered=True))
    story.append(flowchart(chart, chart_caption))
    if extra:
        story.extend(extra)
    story.append(P("A small example", "h3"))
    story.extend(example)
    story.append(KeepTogether([P("Settings used in DengueCAD", "h3"), settings_table(settings)]))
    story.append(Spacer(1, 8))
    story.append(KeepTogether([pros_cons(pros, cons), Spacer(1, 6), refs_line(refs)]))


REFERENCES = [
    ('Rao VSH, Kumar MN. "A new intelligence-based approach for computer-aided diagnosis of dengue fever." '
     "<i>IEEE Trans. Information Technology in Biomedicine</i> 16(1):112–118, 2012.",
     "https://doi.org/10.1109/TITB.2011.2171978"),
    ('Rubin DB. "Inference and missing data." <i>Biometrika</i> 63(3):581–592, 1976.',
     "https://doi.org/10.1093/biomet/63.3.581"),
    ('van Buuren S, Groothuis-Oudshoorn K. "mice: Multivariate Imputation by Chained Equations in R." '
     "<i>Journal of Statistical Software</i> 45(3):1–67, 2011.",
     "https://doi.org/10.18637/jss.v045.i03"),
    ("scikit-learn documentation: IterativeImputer.",
     "https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html"),
    ('Troyanskaya O, et al. "Missing value estimation methods for DNA microarrays." '
     "<i>Bioinformatics</i> 17(6):520–525, 2001.",
     "https://doi.org/10.1093/bioinformatics/17.6.520"),
    ("scikit-learn documentation: KNNImputer.",
     "https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html"),
    ('Stekhoven DJ, Bühlmann P. "MissForest — non-parametric missing value imputation for mixed-type data." '
     "<i>Bioinformatics</i> 28(1):112–118, 2012.",
     "https://doi.org/10.1093/bioinformatics/btr597"),
    ('Breiman L. "Random forests." <i>Machine Learning</i> 45:5–32, 2001.',
     "https://doi.org/10.1023/A:1010933404324"),
    ("Little RJA, Rubin DB. <i>Statistical Analysis with Missing Data</i>, 3rd ed. Wiley, 2019.",
     "https://doi.org/10.1002/9781119482260"),
    ("scikit-learn documentation: SimpleImputer.",
     "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html"),
    ('Kohavi R, John GH. "Wrappers for feature subset selection." '
     "<i>Artificial Intelligence</i> 97(1–2):273–324, 1997.",
     "https://doi.org/10.1016/S0004-3702(97)00043-X"),
    ("Goldberg DE. <i>Genetic Algorithms in Search, Optimization and Machine Learning</i>. "
     "Addison-Wesley, 1989.", None),
    ('Freund Y, Mason L. "The alternating decision tree learning algorithm." '
     "<i>Proc. 16th ICML</i>, pp. 124–133, 1999.",
     "https://mlanthology.org/icml/1999/freund1999icml-alternating/"),
    ('Freund Y, Schapire RE. "A decision-theoretic generalization of on-line learning and an application '
     'to boosting." <i>J. Computer and System Sciences</i> 55(1):119–139, 1997.',
     "https://doi.org/10.1006/jcss.1997.1504"),
    ("Quinlan JR. <i>C4.5: Programs for Machine Learning</i>. Morgan Kaufmann, 1993.",
     "https://dl.acm.org/doi/abs/10.5555/152181"),
    ('Quinlan JR. "Induction of decision trees." <i>Machine Learning</i> 1:81–106, 1986.',
     "https://doi.org/10.1007/BF00116251"),
    ("scikit-learn user guide: Decision Trees.", "https://scikit-learn.org/stable/modules/tree.html"),
    ('Cox DR. "The regression analysis of binary sequences." '
     "<i>J. Royal Statistical Society B</i> 20(2):215–242, 1958.",
     "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x"),
    ("scikit-learn user guide: Logistic regression.",
     "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"),
    ('Cortes C, Vapnik V. "Support-vector networks." <i>Machine Learning</i> 20:273–297, 1995.',
     "https://doi.org/10.1007/BF00994018"),
    ('Chang C-C, Lin C-J. "LIBSVM: a library for support vector machines." '
     "<i>ACM Trans. Intelligent Systems and Technology</i> 2(3):27, 2011.",
     "https://doi.org/10.1145/1961189.1961199"),
    ("scikit-learn user guide: Support Vector Machines.", "https://scikit-learn.org/stable/modules/svm.html"),
]


def intro(story: list) -> None:
    story.append(Spacer(1, 3 * cm))
    story.append(P("DengueCAD", "title"))
    story.append(P("Methods Guide: imputation and prediction methods compared with NM", "subtitle"))
    story.append(P(f"Reference and learning guide · {date.today():%B %Y}", "subtitle"))
    story.append(Spacer(1, 1.2 * cm))
    story.append(box(
        "DengueCAD compares the New Method (NM) of Rao &amp; Kumar [1] against well-known alternatives. "
        "NM has two parts: <b>NMI</b>, which fills in missing values, and <b>NMPrediction</b>, which "
        "picks the most useful features and predicts the diagnosis. This guide explains, in plain "
        "language, every method NM is compared with, plus NM itself.",
        label="What this guide is",
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(P("How to read each method page", "h3"))
    story.append(bullets([
        "<b>In one sentence</b> — the core idea (yellow boxes mark the NM methods).",
        "<b>How it works</b> — numbered steps, followed by a flowchart of the same steps.",
        "<b>A small example</b> — a tiny worked case with made-up numbers.",
        "<b>Settings used in DengueCAD</b> — the exact configuration used in our experiments.",
        "<b>Strengths / Watch out for</b> — when the method shines and when to be careful.",
        "<b>Learn more</b> — numbered references, listed with links at the end of the guide.",
    ]))
    story.append(P("Contents", "h3"))
    story.append(grid_table([
        ["Section", "Methods"],
        ["1. Background", "Missing values, the two-step pipeline, key terms"],
        ["2. Imputation methods", "2.1 NMI · 2.2 MICE · 2.3 kNN · 2.4 MissForest · 2.5 Mean / Median / Mode"],
        ["3. Prediction methods", "3.1 NMPrediction (with GA wrapper and ADT) · 3.2 C4.5 · 3.3 Logistic "
                                  "regression · 3.4 SVM"],
        ["4. Side-by-side summary", "Quick comparison tables"],
        ["5. How the methods are scored", "Imputation accuracy, Accuracy, SE, SP, AUC"],
        ["6. References", "Papers and documentation with links"],
    ], [1.3, 3.2]))

    story.append(PageBreak())
    story.append(P("1. Background", "h1"))
    story.append(P("1.1 Why missing values matter", "h2"))
    story.append(P(
        "Clinical records are often incomplete: a lab test was not ordered, a symptom was not recorded, "
        "or a form field was left blank. Most prediction methods need a complete table, so the missing "
        "cells must be handled first. Dropping incomplete records wastes data and can bias the result; "
        "<b>imputation</b> instead fills each missing cell with a sensible estimate."
    ))
    story.append(P(
        "Statisticians describe <i>why</i> values are missing with three labels [2, 9]. "
        "<b>MCAR</b> (missing completely at random): missingness has nothing to do with the data — like "
        "a sample tube dropped by accident. <b>MAR</b> (missing at random): missingness depends on other "
        "recorded values — e.g. older patients skip a test more often. <b>MNAR</b> (missing not at "
        "random): missingness depends on the missing value itself — e.g. very sick patients miss a "
        "follow-up. DengueCAD's controlled experiments hide values MCAR, so we know the true answer and "
        "can score each method."
    ))
    story.append(P("1.2 The two-step pipeline", "h2"))
    story.append(P(
        "Every experiment follows the same two steps. First an <b>imputation</b> method completes the "
        "data; then a <b>prediction</b> method (a classifier) learns to predict the decision column — "
        "for dengue, whether the patient has the disease."
    ))
    story.append(flowchart("m00-overview", "Figure 1. Imputation methods (left) and prediction methods "
                                           "(right) compared in DengueCAD. NM's components are NMI and "
                                           "NMPrediction.", max_h=8 * cm))
    story.append(P("1.3 Key terms", "h2"))
    story.append(grid_table([
        ["Term", "Meaning"],
        ["Record / row", "One patient."],
        ["Attribute / feature", "One column describing the patient, e.g. fever, platelet count."],
        ["Decision column", "The diagnosis we want to predict (the last column), e.g. dengue yes / no."],
        ["Categorical attribute", "Takes values from a fixed set (yes / no, blood group). Averages make no "
                                  "sense, so the most frequent value (mode) is used."],
        ["Numeric attribute", "A measured number (temperature, platelets). Averages (mean) make sense."],
        ["Complete record", "A row with no missing values. NMI uses these as donors."],
        ["Donor / neighbour", "A similar complete record whose values are borrowed to fill a gap."],
        ["Cross-validation (CV)", "Split the data into k parts; train on k−1, test on the remaining part, "
                                  "rotate, and average. Gives an honest estimate of performance."],
        ["Overfitting", "A model memorises the training data and does worse on new patients."],
    ], [1.3, 3.5]))


def imputation_section(story: list) -> None:
    story.append(PageBreak())
    story.append(P("2. Imputation methods", "h1"))
    story.append(P(
        "Each method below takes a table with gaps and returns a complete table. They differ in how they "
        "decide what value is \"sensible\": a single column summary (Mean / Mode), similar patients "
        "(NMI, kNN), or a model learned from the other columns (MICE, MissForest)."
    ))

    nmi_formulas = [
        P("The four scoring cases", "h3"),
        P("For a target row R<sub>i</sub> and a donor R<sub>k</sub>, each attribute C<sub>l</sub> "
          "gets a score I<sub>Cl</sub>(R<sub>i</sub>, R<sub>k</sub>). Which formula applies depends on "
          "whether the two patients share the same diagnosis and on the attribute's type:"),
        grid_table([
            ["", "Categorical / integer attribute", "Fractional / real attribute"],
            ["<b>Case I</b><br/>same decision class",
             "I = min(γ<sub>i</sub> / γ<sub>k</sub>, γ<sub>k</sub> / γ<sub>i</sub>)<br/>"
             "γ = how often each record's value occurs among complete records of that class",
             "I = min(A<sub>il</sub> / A#, A<sub>kl</sub> / A#)<br/>A# = mean of column l"],
            ["<b>Case II</b><br/>different classes",
             "I = max(β / δ, δ / β)<br/>β, δ = how often each value occurs within its own record's class",
             "I = max(A<sub>il</sub> / Λ, A<sub>kl</sub> / Λ)<br/>Λ = min(P#, Q#), the smaller of "
             "the two class means of column l"],
        ], [0.9, 2, 2]),
        Spacer(1, 4),
        P("The scores are combined into one distance, which is standardised so that “close” means "
          "“closer than average”:"),
        P("d<sub>ik</sub> = √( Σ<sub>l</sub> I<sub>Cl</sub>(R<sub>i</sub>, R<sub>k</sub>)² )<br/>"
          "z(d<sub>ik</sub>) = (d<sub>ik</sub> − mean(d)) / sd(d)<br/>"
          "neighbours = { R<sub>k</sub> : z(d<sub>ik</sub>) ≤ 0 }", "formula"),
        P("Because the neighbour rule is “below-average distance”, NMI does not need a k to be chosen; "
          "about half of the donors usually qualify. If none do, the single closest donor is used."),
    ]
    method(
        story, new_page=False, nm=True,
        title="2.1 NMI — New Method Imputation (NM's imputer)",
        one_liner="fill each gap with the most common value (or the average) among complete records "
                  "that are closer than average to the incomplete record, where closeness also takes "
                  "the diagnosis into account.",
        steps=[
            "Separate the complete records (<b>donors</b>) from the incomplete ones (<b>targets</b>). "
            "Records whose diagnosis is missing are left alone.",
            "For a target record, compare it with every donor, one attribute at a time, using a score "
            "that depends on whether the two records share the same diagnosis (see the four cases below).",
            "Combine the attribute scores into a single distance, and convert all distances to z-scores.",
            "Every donor with a z-score ≤ 0 (closer than average) is a <b>neighbour</b>.",
            "Fill each missing categorical / integer value with the neighbours' <b>mode</b>, and each "
            "missing real value with the neighbours' <b>mean</b>. Repeat for every target record.",
        ],
        chart="m01-nmi",
        chart_caption="Figure 2. NMI: prepare donors and targets, measure closeness, fill the gaps.",
        extra=nmi_formulas,
        example=[
            P("A dengue patient has <i>fever</i> and <i>platelets</i> missing. Six complete records are "
              "donors, with distances 0.8, 1.0, 1.2, 2.5, 2.9 and 3.1 (mean 1.92). The first three are "
              "below the mean (z ≤ 0), so they are the neighbours. Their fever values are yes, yes, no → "
              "<b>mode = yes</b>. Their platelet counts are 90k, 110k, 100k → <b>mean = 100k</b>."),
        ],
        settings=[
            ("Implementation", "nmilib.non_parametric_imputation (sibling NMI repository)"),
            ("Donors", "Complete records only; rows with a missing decision are not imputed"),
            ("Neighbours", "All donors with z(d) ≤ 0 (no k to tune)"),
            ("Fill rule", "Mode for categorical / integer, mean for fractional / real"),
            ("Hold-out experiment", "All attributes ordinal-encoded, so the mode is used everywhere"),
        ],
        pros=[
            "Works on mixed categorical and numeric data.",
            "Uses the diagnosis when measuring similarity, so neighbours are clinically relevant.",
            "No parameters to tune (no k, no model).",
            "Easy to explain: values come from real, similar patients.",
        ],
        cons=[
            "Needs enough complete records; with many gaps, few donors remain.",
            "Cannot impute records whose diagnosis is missing.",
            "Comparing every target with every donor is slow on very large data.",
            "Taking the mode/mean of ~half the donors can pull values towards the class average.",
        ],
        refs=[1],
    )

    method(
        story,
        title="2.2 MICE — Multivariate Imputation by Chained Equations",
        one_liner="predict each incomplete column from all the other columns with a regression model, "
                  "and cycle through the columns several times until the filled-in values stop changing.",
        steps=[
            "Fill every gap temporarily with the column mean.",
            "Pick a column that had missing values. Fit a regression model that predicts it from all the "
            "other columns, using only the rows where it was observed.",
            "Replace that column's missing cells with the model's predictions.",
            "Move to the next incomplete column and repeat. One pass over all columns is a <b>round</b>.",
            "Run more rounds until the imputed values settle (or a maximum is reached).",
        ],
        chart="m02-mice",
        chart_caption="Figure 3. MICE: chained regressions, one column at a time, repeated in rounds.",
        example=[
            P("<i>Platelets</i> is missing for some patients. MICE fits platelets ≈ b<sub>0</sub> + "
              "b<sub>1</sub>·WBC + b<sub>2</sub>·age + … on the patients whose platelets are known, then "
              "uses the fitted equation to predict the missing platelet counts. Next it does the same "
              "for <i>WBC</i> using the newly filled platelets, and so on."),
            P("In full multiple imputation, MICE builds several completed datasets with random draws and "
              "pools the results [3]. DengueCAD uses the single, deterministic version."),
        ],
        settings=[
            ("Implementation", "scikit-learn IterativeImputer [4]"),
            ("Model per column", "Bayesian ridge regression (scikit-learn default)"),
            ("Initial fill", "Column mean"),
            ("Rounds", "Up to 20 (max_iter = 20)"),
            ("Draws", "sample_posterior = False → one deterministic imputation"),
            ("Categorical data", "Predictions are rounded to the nearest valid code"),
        ],
        pros=[
            "Uses relationships between columns, not just column averages.",
            "Very widely used and well studied in medical statistics.",
            "Flexible: any regression model can be plugged in.",
        ],
        cons=[
            "Default linear models miss non-linear patterns and interactions.",
            "Treats category codes as numbers unless a classifier is used per column.",
            "Several rounds over all columns can be slow on wide data.",
        ],
        refs=[3, 4],
    )

    method(
        story,
        title="2.3 kNN imputation — k-Nearest Neighbours",
        one_liner="find the k most similar records that have the value, and fill the gap with their "
                  "(distance-weighted) average.",
        steps=[
            "For a record with a missing value in column X, compute its distance to every other record, "
            "using only the columns both records have (nan-Euclidean distance).",
            "Keep the k = 5 closest records whose X is known.",
            "Give closer neighbours more say: weight each by 1 / distance.",
            "Fill X with the weighted average of the neighbours' values.",
        ],
        chart="m03-knn",
        chart_caption="Figure 4. kNN imputation: find the k closest records, then take a weighted average.",
        example=[
            P("A patient's <i>haemoglobin</i> is missing. Their five closest patients have 12, 13, 13, "
              "14 and 15 g/dL at distances 1, 1, 2, 2, 4. Weights are 1, 1, 0.5, 0.5, 0.25, giving "
              "(12 + 13 + 6.5 + 7 + 3.75) / 3.25 ≈ <b>13.0 g/dL</b>."),
        ],
        settings=[
            ("Implementation", "scikit-learn KNNImputer [6], after Troyanskaya et al. [5]"),
            ("Neighbours", "k = 5"),
            ("Weights", "1 / distance (weights = \"distance\")"),
            ("Distance", "nan-Euclidean (ignores columns missing in either record)"),
            ("Categorical data", "Averages are rounded to the nearest valid code"),
        ],
        pros=[
            "Simple and intuitive: borrow from similar patients.",
            "No model to fit; captures local, non-linear structure.",
            "Can use incomplete records as neighbours.",
        ],
        cons=[
            "k must be chosen; results depend on it.",
            "Distances are sensitive to feature scaling and irrelevant columns.",
            "Ignores the diagnosis; averaging categories needs rounding.",
            "Slow for large datasets (distance to every row).",
        ],
        refs=[5, 6],
    )

    method(
        story,
        title="2.4 MissForest",
        one_liner="like MICE, but each column is predicted by a random forest, which handles mixed "
                  "categorical / numeric data and non-linear relationships.",
        steps=[
            "Fill gaps temporarily: mean for numeric columns, mode for categorical columns.",
            "Order the columns from fewest to most missing values.",
            "For each column, train a <b>random forest</b> (many decision trees on random subsets of "
            "rows and features [8]) on the rows where the column is observed, and predict the missing "
            "cells — a classification forest for categories, a regression forest for numbers.",
            "After all columns, compare the new table with the previous round.",
            "Stop when the change is tiny (or a maximum number of rounds is reached).",
        ],
        chart="m04-missforest",
        chart_caption="Figure 5. MissForest: iterative imputation with a random forest per column.",
        example=[
            P("<i>Rash</i> (yes / no) is missing for some patients. MissForest trains a classification "
              "forest that predicts rash from fever, platelets, age and the other columns. If most "
              "trees vote \"yes\" for a patient, the gap is filled with yes. Numeric columns such as "
              "platelets are filled with the average prediction of the trees."),
        ],
        settings=[
            ("Implementation", "denguecad/missforest_impute.py (follows Stekhoven &amp; Bühlmann [7])"),
            ("Forest size", "40 trees per column"),
            ("Rounds", "Up to 6 (benchmark) / 5 (hold-out experiment)"),
            ("Stopping rule", "Mean squared change between rounds &lt; 10<sup>−6</sup>"),
            ("Categorical columns", "Random forest classifier on integer codes"),
        ],
        pros=[
            "Handles mixed data types and complex, non-linear relationships.",
            "No assumptions about the data distribution.",
            "Often among the most accurate imputers in published comparisons [7].",
        ],
        cons=[
            "Much slower than the other methods (many forests per round).",
            "Less transparent: hard to explain why a value was chosen.",
            "Can overfit small datasets.",
        ],
        refs=[7, 8],
    )

    method(
        story,
        title="2.5 Mean / Median / Mode imputation",
        one_liner="fill every gap in a column with one summary value of that column — the mean, the "
                  "median or the most frequent value.",
        steps=[
            "For each column, compute one summary of the observed values: the <b>mean</b> (numeric), "
            "the <b>median</b> (numeric and skewed) or the <b>mode</b> (categorical).",
            "Write that value into every missing cell of the column.",
        ],
        chart="m05-mean-mode",
        chart_caption="Figure 6. Mean / Median / Mode: one value per column.",
        example=[
            P("<i>Fever</i> is recorded as yes, yes, no, yes for four patients and missing for a fifth: "
              "the mode is <b>yes</b>. <i>Age</i> values 20, 30, 40 and 90: the mean is 45, the median "
              "is 35 — the median is less affected by the unusual 90."),
        ],
        settings=[
            ("Implementation", "scikit-learn SimpleImputer [10]"),
            ("Imputation benchmark", "Mean, median, or mode (mode for categorical + mean for numeric)"),
            ("Hold-out experiment", "Mode on the ordinal-encoded attributes"),
        ],
        pros=[
            "Extremely fast and simple; a standard baseline [9].",
            "Never fails, even with few complete records.",
        ],
        cons=[
            "Ignores every other column and the diagnosis.",
            "Shrinks variability and weakens relationships between columns.",
            "Every patient with a gap gets the same value.",
        ],
        refs=[9, 10],
    )


def prediction_section(story: list) -> None:
    story.append(PageBreak())
    story.append(P("3. Prediction methods", "h1"))
    story.append(P(
        "A prediction method (classifier) learns from patients whose diagnosis is known and then "
        "predicts the diagnosis of new patients. NMPrediction is NM's full pipeline; C4.5, logistic "
        "regression and SVM are the standard classifiers it is compared with in the paper [1]."
    ))

    ga_and_adt = [
        P("Inside NMPrediction (a): the GA wrapper", "h3"),
        P("A <b>wrapper</b> judges a feature subset by actually training the classifier on it [11]. "
          "Trying every subset is impossible (2<sup>n</sup> subsets for n features), so a <b>genetic "
          "algorithm</b> [12] searches instead: each candidate subset is a string of bits (1 = keep), "
          "the best strings survive and are combined, and random bit flips keep exploring."),
        flowchart("m07-ga-wrapper", "Figure 8. GA wrapper: evolve bit masks scored by cross-validated "
                                    "accuracy.", max_h=8 * cm),
        P("Inside NMPrediction (b): the Alternating Decision Tree (ADT)", "h3"),
        P("An ADT [13] is built by <b>boosting</b> [14]: it adds one simple rule per round, each time "
          "focusing on the patients the current model gets wrong. Every rule carries a score for YES "
          "and for NO; to classify a patient, the scores of all rules that apply are added up and the "
          "sign of the total decides. The sum also shows how confident the prediction is."),
        flowchart("m08-adt", "Figure 9. ADT: boosted rules with additive scores.", max_h=8 * cm),
    ]
    method(
        story, new_page=False, nm=True,
        title="3.1 NMPrediction — NM's prediction pipeline (Algorithm 1)",
        one_liner="impute with NMI, let a genetic algorithm choose the most influential features, then "
                  "predict with a boosted Alternating Decision Tree.",
        steps=[
            "<b>Impute</b> the training data with NMI (Section 2.1).",
            "<b>Select features</b> with a GA wrapper: the genetic algorithm searches for the subset of "
            "attributes on which the classifier is most accurate.",
            "<b>Train an ADT</b> on the selected features.",
            "<b>Evaluate</b> with stratified 10-fold cross-validation (Accuracy, AUC, sensitivity, "
            "specificity), or predict new patients using only the selected features.",
        ],
        chart="m06-nmprediction",
        chart_caption="Figure 7. NMPrediction: NMI → GA wrapper → ADT.",
        extra=ga_and_adt,
        example=[
            P("From 20 clinical attributes, the GA might keep 6 — say fever duration, platelets, WBC, "
              "rash, headache and NS1. The ADT then learns rules such as “platelets &lt; 100k → +0.8, "
              "otherwise −0.4”. A new patient's scores are summed; a positive total means dengue. The "
              "selected attributes are also reported as the most influential symptoms (Table III)."),
        ],
        settings=[
            ("Implementation", "denguecad/nm_prediction.py, feature_selection.py, adt.py"),
            ("GA operators", "Crossover Pc = 1.0, mutation Pm = 0.001, elitism 2, tournament of 3"),
            ("GA size", "Population 12 × 8 generations (hold-out experiment: 8 × 5)"),
            ("GA fitness", "3-fold CV accuracy of the ADT − 0.01 × fraction of features kept"),
            ("ADT", "40 boosting rounds of decision stumps (AdaBoost) with additive scores"),
            ("Evaluation", "Stratified 10-fold CV; hold-out validation / test sets"),
        ],
        pros=[
            "Handles missing values as part of the pipeline.",
            "Picks a small, interpretable set of influential symptoms.",
            "ADT rules are readable and give a confidence score.",
            "Boosting usually gives strong accuracy from simple rules.",
        ],
        cons=[
            "The GA is the slowest part (many cross-validated fits).",
            "Results vary slightly with the random seed of the GA.",
            "DengueCAD's ADT approximates Freund &amp; Mason's tree with boosted stumps (rules are not "
            "nested under one another).",
        ],
        refs=[1, 11, 12, 13, 14],
    )

    method(
        story,
        title="3.2 C4.5 decision tree",
        one_liner="ask the most informative yes/no question first, split the patients by the answer, "
                  "and keep asking questions until each group is (nearly) one diagnosis.",
        steps=[
            "Start with all training patients in one node.",
            "For each attribute, measure how much splitting on it reduces uncertainty about the "
            "diagnosis (<b>information gain</b>; C4.5 uses the gain <i>ratio</i> to avoid favouring "
            "attributes with many values [15, 16]).",
            "Split on the best attribute (or threshold for numbers) and create child nodes.",
            "Repeat on each child until a node is pure or too small; it becomes a leaf.",
            "To predict, follow the questions from the root down to a leaf.",
        ],
        chart="m09-c45",
        chart_caption="Figure 10. C4.5: grow the tree by repeatedly splitting on the most informative attribute.",
        example=[
            P("Root question: “platelets &lt; 100k?”. If yes → next question “fever &gt; 3 days?” → "
              "leaf <b>dengue</b>. If no → leaf <b>not dengue</b>. Entropy measures uncertainty: a node "
              "with 50 % dengue has entropy 1 bit; a pure node has 0. The question that lowers entropy "
              "the most is asked first."),
        ],
        settings=[
            ("Implementation", "scikit-learn DecisionTreeClassifier [17]"),
            ("Split criterion", "Entropy / information gain (criterion = \"entropy\")"),
            ("Note", "scikit-learn grows binary CART-style trees without C4.5's gain ratio and pruning; "
                     "this is the usual open-source stand-in for C4.5"),
        ],
        pros=[
            "Very easy to read and explain to clinicians.",
            "Handles numeric and categorical data; no scaling needed.",
            "Fast to train and to predict.",
        ],
        cons=[
            "A single tree overfits easily (unstable to small data changes).",
            "Axis-aligned splits approximate smooth boundaries poorly.",
            "Usually less accurate than boosted or kernel methods.",
        ],
        refs=[15, 16, 17],
    )

    method(
        story,
        title="3.3 Logistic regression (LOR)",
        one_liner="add up the features with learned weights and pass the total through an S-shaped "
                  "curve to get the probability of disease.",
        steps=[
            "Standardise every feature (mean 0, standard deviation 1).",
            "Compute a score z = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + … + "
            "b<sub>p</sub>x<sub>p</sub>.",
            "Turn the score into a probability with the sigmoid p = 1 / (1 + e<sup>−z</sup>).",
            "Adjust the weights b to make the true diagnoses as likely as possible (maximum likelihood), "
            "with a small penalty on large weights.",
            "Predict positive when p ≥ 0.5.",
        ],
        chart="m10-lor",
        chart_caption="Figure 11. Logistic regression: weighted sum → sigmoid → probability.",
        example=[
            P("Suppose z = −1 + 2·(low platelets) + 1·(fever). A patient with both gets z = 2, so "
              "p = 1 / (1 + e<sup>−2</sup>) ≈ <b>0.88</b> → dengue. With neither, z = −1 and p ≈ 0.27 → "
              "not dengue. Each weight shows how much a feature raises the odds (e<sup>b</sup> is the "
              "odds ratio)."),
        ],
        settings=[
            ("Implementation", "scikit-learn LogisticRegression [19] after StandardScaler"),
            ("Solver", "L-BFGS, up to 2000 iterations"),
            ("Penalty", "L2 with C = 1.0 (scikit-learn default)"),
        ],
        pros=[
            "Gives probabilities and interpretable odds ratios [18].",
            "Fast, stable and hard to overfit with regularisation.",
            "A standard reference model in medicine.",
        ],
        cons=[
            "Boundary is linear in the features; misses interactions unless added by hand.",
            "Sensitive to strongly correlated features.",
            "Needs numeric inputs (categories must be encoded).",
        ],
        refs=[18, 19],
    )

    method(
        story,
        title="3.4 Support Vector Machine (SVM) with RBF kernel",
        one_liner="find the boundary that separates the two diagnoses with the widest possible margin, "
                  "using a kernel so the boundary can curve.",
        steps=[
            "Standardise every feature.",
            "Measure similarity between patients with the RBF (Gaussian) kernel "
            "K(x, x′) = exp(−γ ‖x − x′‖²): 1 for identical patients, near 0 for very different ones.",
            "Find the boundary with the widest margin between the classes; the parameter C trades a "
            "wider margin against training mistakes [20].",
            "The training patients on or inside the margin are the <b>support vectors</b>; only they "
            "define the boundary.",
            "Predict a new patient from its weighted similarity to the support vectors.",
        ],
        chart="m11-svm",
        chart_caption="Figure 12. SVM: kernel similarity, widest-margin boundary, support vectors.",
        example=[
            P("Plot patients by platelets and WBC. A straight line may not separate dengue from "
              "non-dengue, but the RBF kernel lets the boundary bend around clusters of dengue patients. "
              "A new patient close (in kernel terms) to dengue support vectors is predicted dengue."),
        ],
        settings=[
            ("Implementation", "scikit-learn SVC [22] (LIBSVM [21]) after StandardScaler"),
            ("Kernel", "RBF, γ = \"scale\" = 1 / (number of features × variance of X)"),
            ("Regularisation", "C = 1.0"),
            ("Large data", "Training subsampled to 8,000 rows in the hold-out experiment"),
        ],
        pros=[
            "Often the most accurate of the baselines; flexible, non-linear boundaries.",
            "Works well with many features.",
            "Only the support vectors matter, which resists outliers far from the boundary.",
        ],
        cons=[
            "Training time grows quickly with the number of rows.",
            "A black box: no simple rules or feature weights.",
            "Needs scaling and tuning of C and γ; no native probabilities.",
        ],
        refs=[20, 21, 22],
    )


def summary_section(story: list) -> None:
    story.append(CondPageBreak(12 * cm))
    story.append(P("4. Side-by-side summary", "h1"))
    story.append(P("4.1 Imputation methods", "h2"))
    story.append(grid_table([
        ["Method", "Core idea", "Uses other columns?", "Uses diagnosis?", "Mixed types", "Speed"],
        ["<b>NMI</b>", "Mode / mean of below-average-distance donors", "Yes (distance)", "Yes", "Yes", "Medium"],
        ["MICE", "Regression per column, in rounds", "Yes (model)", "No", "Partly", "Medium"],
        ["kNN", "Weighted mean of k nearest rows", "Yes (distance)", "No", "Partly", "Medium"],
        ["MissForest", "Random forest per column, in rounds", "Yes (model)", "No", "Yes", "Slow"],
        ["Mean / Mode", "One value per column", "No", "No", "Yes", "Very fast"],
    ], [1, 2.4, 1.2, 1, 0.9, 0.9], highlight_rows={1: NM}))
    story.append(Spacer(1, 10))
    story.append(P("4.2 Prediction methods", "h2"))
    story.append(grid_table([
        ["Method", "Core idea", "Feature selection", "Interpretable?", "Boundary", "Speed"],
        ["<b>NMPrediction</b>", "NMI + GA wrapper + boosted ADT", "Yes (GA)", "Yes (rules + scores)",
         "Non-linear", "Slow (GA)"],
        ["C4.5", "Tree of most informative questions", "Implicit", "Yes (tree)", "Axis-aligned", "Fast"],
        ["LOR", "Weighted sum → sigmoid", "No", "Yes (odds ratios)", "Linear", "Fast"],
        ["SVM (RBF)", "Widest margin with a kernel", "No", "No", "Non-linear", "Medium–slow"],
    ], [1.2, 2.2, 1.1, 1.3, 1, 1], highlight_rows={1: NM}))
    story.append(Spacer(1, 6))
    story.append(P("Yellow rows are NM's components. Speeds are relative, for the dataset sizes used in "
                   "DengueCAD.", "caption"))

    story.append(CondPageBreak(9 * cm))
    story.append(P("5. How the methods are scored", "h1"))
    story.append(P(
        "<b>Imputation accuracy.</b> Start from complete records, hide some values on purpose (MCAR), "
        "impute, and count the share of hidden cells restored to their true value. In the hold-out "
        "experiment all attributes are ordinal-encoded (numbers binned into deciles), so a cell counts as "
        "correct only when the exact code is recovered."
    ))
    story.append(P(
        "<b>Prediction metrics.</b> With TP / TN = correctly predicted positives / negatives and "
        "FP / FN = wrong positives / negatives:"
    ))
    story.append(grid_table([
        ["Metric", "Formula", "Meaning"],
        ["Accuracy", "(TP + TN) / all", "Share of patients classified correctly"],
        ["Sensitivity (SE)", "TP / (TP + FN)", "Share of dengue patients detected"],
        ["Specificity (SP)", "TN / (TN + FP)", "Share of healthy patients correctly cleared"],
        ["AUC", "Area under the ROC curve", "Chance that a random dengue patient is ranked above a random "
                                            "non-dengue patient (0.5 = guessing, 1 = perfect)"],
    ], [1.2, 1.5, 3]))
    story.append(P(
        "Results of these comparisons are in DengueCAD_Performance_Comparisons.pdf and "
        "DengueCAD_Imputation_Prediction_Holdout.pdf in the same docs folder.", "body"))


def references_section(story: list) -> None:
    story.append(PageBreak())
    story.append(P("6. References", "h1"))
    for i, (text, url) in enumerate(REFERENCES, start=1):
        link = f' <link href="{url}" color="#2C5282"><u>{url}</u></link>' if url else ""
        story.append(P(f"[{i}] {text}{link}", "ref"))


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(REGULAR, 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.2 * cm, "DengueCAD · Methods Guide")
    canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(GRID)
    canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
    canvas.restoreState()


def build(out: Path = OUT) -> Path:
    story: list = []
    intro(story)
    imputation_section(story)
    prediction_section(story)
    summary_section(story)
    references_section(story)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title="DengueCAD Methods Guide", author="DengueCAD",
        subject="Imputation and prediction methods compared with NM",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    return out


if __name__ == "__main__":
    print(f"Wrote {build()}")
