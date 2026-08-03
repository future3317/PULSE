#!/usr/bin/env python
"""Generate the Supplementary Information LaTeX file for the screening-resolution manuscript.

Reads frozen results from ``results/phase7c/`` and writes
``CrossPiezo_ScreeningResolution_Supplementary.tex``.  No hand-entered numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase7c"
OUTPUT = PROJECT_ROOT / "CrossPiezo_ScreeningResolution_Supplementary.tex"


def _fmt(x, decimals=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{decimals}f}"


def table_s1_high_response() -> str:
    """Corrected high-response sensitivity table."""
    df = pd.read_csv(RESULTS_ROOT / "high_response_sensitivity.csv")
    df = df[(df["panel"] == "P0") & (df["metric"] == "F1_Frobenius")
            & (df["fraction"].isin([0.10, 0.25]))]
    df["method"] = pd.Categorical(df["method"], categories=[
        "dual_high", "jarvis_anchor", "mp_anchor", "pooled_exploratory"
    ])
    df = df.sort_values(["fraction", "method"])
    method_labels = {
        "dual_high": "Dual-high",
        "jarvis_anchor": "JARVIS anchor",
        "mp_anchor": "MP anchor",
        "pooled_exploratory": "Pooled (exploratory)",
    }
    rows = []
    for _, row in df.iterrows():
        method = method_labels.get(row["method"], row["method"])
        rows.append(
            f"{method} & {int(row['fraction'] * 100)}\\% & {int(row['n_selected'])} & "
            f"{_fmt(row['kendall_tau'])} & [{_fmt(row['kendall_tau_ci95_low'])}, {_fmt(row['kendall_tau_ci95_high'])}] & "
            f"{_fmt(row['nAUCC'])} & [{_fmt(row['nAUCC_ci95_low'])}, {_fmt(row['nAUCC_ci95_high'])}] \\\\\n"
        )
    return """\\begin{table}[htbp]
\\centering
\\caption{Corrected high-response sensitivity for P0 F1 at $f=0.10$ and $f=0.25$.}
\\label{tab:si_highresponse}
\\begin{tabular}{lcccccc}
\\toprule
Method \\& $f$ \\& $N$ \\& $\tau$ \\& 95\\% CI \\& nAUCC \\& 95\\% CI \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s2_controls() -> str:
    """Full property-control comparison with provenance flags."""
    controls = pd.read_csv(RESULTS_ROOT / "property_controls.csv")
    prov = pd.read_csv(RESULTS_ROOT / "control_provenance.csv")
    p0 = controls[controls["panel"] == "P0"]
    p0 = p0[p0["attribute"].isin(["volume", "band_gap", "energy_above_hull", "dielectric_total_trace"])]
    p0_prov = prov[prov["panel"] == "P0"].set_index("attribute")
    attr_labels = {
        "F1_Frobenius": "F1 Frobenius",
        "volume": "Volume",
        "band_gap": "Band gap",
        "energy_above_hull": "Energy above hull",
        "dielectric_total_trace": "Dielectric trace",
    }
    rows = []
    for _, row in p0.iterrows():
        attr = row["attribute"]
        if attr in p0_prov.index:
            copy_flag = p0_prov.loc[attr, "same_field_copy_flag"]
            flag_str = "$\\checkmark$" if copy_flag else ""
        else:
            flag_str = ""
        rows.append(
            f"{attr_labels.get(attr, attr)} \\& {int(row['common_n'])} \\& "
            f"{_fmt(row['kendall_tau'])} \\& [{_fmt(row['kendall_tau_ci95_low'])}, {_fmt(row['kendall_tau_ci95_high'])}] \\& "
            f"{_fmt(row['nAUCC'])} \\& [{_fmt(row['nAUCC_ci95_low'])}, {_fmt(row['nAUCC_ci95_high'])}] \\& "
            f"{_fmt(row['Delta_tau'])} \\& {_fmt(row['Delta_nAUCC'])} \\& {flag_str} \\\\\n"
        )
    return """\\begin{table}[htbp]
\\centering
\\caption{Property-control comparison on P0. The same-field copy flag is raised when more than 5\\% of finite matched pairs have identical JARVIS/MP values.}
\\label{tab:si_controls}
\\begin{tabular}{lcccccccc}
\\toprule
Attribute \\& $N$ \\& $\tau$ \\& 95\\% CI \\& nAUCC \\& 95\\% CI \\& $\Delta\\tau$ \\& $\Delta$nAUCC \\& Copy flag \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s3_portfolio() -> str:
    """Equal-budget portfolio results."""
    bench = pd.read_csv(RESULTS_ROOT / "portfolio_benchmark.csv")
    p0 = bench[(bench["panel"] == "P0") & (bench["metric"] == "F1_Frobenius")
               & (bench["eval_mode"] == "full_panel") & (bench["budget_factor"] == 1.0)]
    strategy_labels = {
        "jarvis_only": "JARVIS only",
        "mp_only": "MP only",
        "average_percentile": "Average percentile",
        "borda_count": "Borda",
        "maximin_percentile": "Maximin",
        "intersection_first": "Intersection first",
        "balanced_union": "Balanced union",
        "disagreement_abstention": "Disagreement abstention",
        "minimax_oracle": "Minimax oracle",
    }
    rows = []
    for _, row in p0.iterrows():
        strat = strategy_labels.get(row["strategy"], row["strategy"])
        rows.append(
            f"{strat} \\& {_fmt(row['worst_source_recall'])} \\& {_fmt(row['worst_source_ndcg'])} \\& "
            f"{_fmt(row['portfolio_coverage'])} \\& {_fmt(row['paired_diff_recall'])} \\& "
            f"[{_fmt(row['paired_diff_recall_ci95_low'])}, {_fmt(row['paired_diff_recall_ci95_high'])}] \\\\\n"
        )
    return """\\begin{table}[htbp]
\\centering
\\caption{Equal-budget ($b=1.0$) portfolio results on P0 F1 ($q^*=10\\%$). Paired differences are versus the better single-source baseline.}
\\label{tab:si_portfolio}
\\begin{tabular}{lccccc}
\\toprule
Strategy \\& Recall \\& NDCG \\& Coverage \\& $\Delta$Recall \\& 95\\% CI \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s4_budget_sensitivity() -> str:
    """Portfolio budget sensitivity."""
    bench = pd.read_csv(RESULTS_ROOT / "portfolio_benchmark.csv")
    p0 = bench[(bench["panel"] == "P0") & (bench["metric"] == "F1_Frobenius")
               & (bench["eval_mode"] == "full_panel")]
    strategy_labels = {
        "jarvis_only": "JARVIS only",
        "mp_only": "MP only",
        "average_percentile": "Average percentile",
        "maximin_percentile": "Maximin",
        "balanced_union": "Balanced union",
    }
    rows = []
    for strategy in strategy_labels:
        sub = p0[p0["strategy"] == strategy].sort_values("budget_factor")
        vals = " \\& ".join([_fmt(r) for r in sub["worst_source_recall"]])
        rows.append(f"{strategy_labels[strategy]} \\& {vals} \\\\\n")
    return """\\begin{table}[htbp]
\\centering
\\caption{Portfolio worst-source recall on P0 F1 across budget factors $b=1.0, 1.5, 2.0$.}
\\label{tab:si_budget}
\\begin{tabular}{lccc}
\\toprule
Strategy \\& $b=1.0$ \\& $b=1.5$ \\& $b=2.0$ \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def main() -> int:
    doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{cleveref}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black}

\title{\textbf{Supplementary Information}\\CrossPiezo: Elite-Tail Instability in Cross-Database Piezoelectric Screening}
\author{Anonymous Author(s)}
\date{3 August 2026}
\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Supplementary tables}

"""
    doc += table_s1_high_response() + "\n\n"
    doc += table_s2_controls() + "\n\n"
    doc += table_s3_portfolio() + "\n\n"
    doc += table_s4_budget_sensitivity() + "\n\n"
    doc += r"""
\end{document}
"""
    OUTPUT.write_text(doc, encoding="utf-8")
    print(f"[make_supplementary] Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
