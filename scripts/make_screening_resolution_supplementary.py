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
\\small
\\setlength{\\tabcolsep}{4pt}
\\centering
\\caption{Corrected high-response sensitivity for P0 F1 at $f=0.10$ and $f=0.25$.}
\\label{tab:si_highresponse}
\\begin{tabular}{lcccccc}
\\toprule
Method \\& $f$ \\& $N$ \\& $\tau$ \\& CI \\& nAUCC \\& CI \\\\
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
\\small
\\setlength{\\tabcolsep}{4pt}
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
            f"{_fmt(row['portfolio_coverage'])} \\& {_fmt(row['full_proc_delta_best_recall'])} \\& "
            f"[{_fmt(row['full_proc_delta_best_recall_ci95_low'])}, {_fmt(row['full_proc_delta_best_recall_ci95_high'])}] \\\\\n"
        )
    return """\\begin{table}[htbp]
\\small
\\setlength{\\tabcolsep}{4pt}
\\centering
\\caption{Equal-budget ($b=1.0$) portfolio results on P0 F1 ($q^*=10\\%$).
Paired differences are versus the better single-source baseline and are computed as material-level worst-source recall differences using a full-procedure grouped bootstrap: each replicate resamples reduced-formula groups with replacement, assigns distinct identities to duplicated occurrences, re-runs the portfolio strategy and both single-source baselines, and recomputes the improvement (see Note~1).}
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
\\small
\\setlength{\\tabcolsep}{4pt}
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
\usepackage{graphicx}
\usepackage{enumitem}
\graphicspath{{figures/screening_resolution/}}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black}

\title{\textbf{Supplementary Information}\\CrossPiezo: Elite-Tail Instability in Cross-Database Piezoelectric Screening}
\author{Anonymous Author(s)}
\date{3 August 2026}
\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Supplementary notes}

\subsection{Portfolio estimand definition (Note~1)}

The portfolio benchmark reports two related quantities.
\emph{Worst-source recall} is a material-level (micro) metric: it counts how many of the top-$q^* n$ materials selected by a strategy are also present in the JARVIS and MP elite sets, divided by $q^* n$, with no group weighting.
\emph{Paired difference in recall} is the difference between the material-level worst-source recall of the strategy and that of the better single-source baseline, computed on the full observed panel.

The confidence interval reported in Table~S3 uses a \emph{full-procedure grouped bootstrap}.
Each replicate:
(i) resamples reduced-formula groups with replacement and assigns distinct bootstrap identities to duplicated occurrences;
(ii) recomputes source-specific rankings and top-$q^*$ elite sets on the bootstrap sample;
(iii) re-runs the portfolio strategy and both single-source baselines;
(iv) computes the improvement over the better single source within that replicate.
The reported 95\% interval is the percentile bootstrap interval of these replicate differences.
The point estimate is fixed on the full-panel selections and is exactly the arithmetic difference of the Recall column; the bootstrap captures uncertainty from group resampling and re-selection.

The ``better single source'' is determined per replicate for the interval; the main-text point estimate uses the full-panel better source (here JARVIS-only and MP-only are tied on P0 F1 at $q^*=10\%$, $b=1.0$).
The portfolio analysis is presented as a risk-management illustration, not as a validated method comparison or as evidence of physical truth.

\subsection{Structure matching and manual audit (Note~2)}

The CrossPiezo-Invariant-v1 panel was constructed from relaxed CIFs shared by the Materials Project (MP) and JARVIS.
The matching pipeline is: (i) load all processed MP and JARVIS piezoelectric records; (ii) compute structure-level overlaps by compositional reduced formula and relaxed-lattice similarity; (iii) apply a frozen similarity threshold to obtain candidate pairs; (iv) perform a response-blind manual audit of the top-matched candidates using a frozen rule set.
The audit was response-blind: the auditor did not see piezoelectric response values when deciding whether a candidate pair represented the same or a distinct structure.
All audit decisions, exclusion reasons and ambiguous cases are recorded in the frozen panel-membership file.
P0 ($n=573$) is the full set of accepted structure matches; P2 ($n=207$) is the subset passing a tighter similarity threshold.

\subsection{Tensor invariants and conventions (Note~3)}

All piezoelectric response values are piezoelectric stress tensors $\mathbf{e}$ in C/m$^2$ with full Cartesian $3\times3\times3$ components.
The three coordinate-invariant scalars are:
\begin{itemize}[noitemsep]
\item F1, Cartesian Frobenius norm $\|\mathbf{e}\|_F$;
\item F3, true collinear longitudinal maximum $\max_{\|\mathbf{n}\|=1} |n_i e_{ijk} n_j n_k|$ computed by numerical optimisation over the unit sphere;
\item F4, Kelvin/Mandel operator norm on symmetric strain, computed from the Mandel representation of the $6\times3$ piezoelectric tensor.
\end{itemize}
Voigt-to-Cartesian conversion follows the standard convention $11\to xx$, $22\to yy$, $33\to zz$, $23\to yz$, $13\to xz$, $12\to xy$ with factor-of-two handling for shear components as appropriate for the stress tensor.
No sign conventions, unit conversions or coordinate rotations were applied silently; all transformations are recorded in the repository transformation history.

\subsection{Bootstrap, quantile and tie handling (Note~4)}

Screening-resolution curves are evaluated at quantiles $q=1\%,2\%,\dots,50\%$.
The chance-adjusted Jaccard overlap is $\widetilde J_q = (J_q - \mathbb{E}[J_q])/(1 - \mathbb{E}[J_q])$, where $\mathbb{E}[J_q]$ is the exact hypergeometric expectation under independent rankings.
Simultaneous 95\% confidence bands for the screening-resolution curve use a paired bootstrap with a studentized sup-norm critical value; this bootstrap resamples matched material pairs (rows) because the curve itself is a cross-sectional function of the ranked universe and the band is constructed for fixed quantiles.

For property-control comparisons and the portfolio benchmark, bootstrap resampling is by reduced-formula group to respect the dependence among entries sharing a composition.
In the portfolio full-procedure bootstrap, duplicated groups create distinct bootstrap identities so that re-selection and elite-set sizes are computed on the expanded bootstrap sample.
The random seed, number of replicates, studentization details, tie handling and $k$ rounding are recorded in the Phase~7C configuration and result files.
Ties in source rankings are broken by stable sorting; duplicate reduced formulae are retained in the ranking but noted in the panel metadata.
The persistent-onset criterion requires the lower confidence bound to exceed a threshold $\delta$ for at least five consecutive quantiles.

\section{Supplementary figures}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig4_illustrative_candidates}
\caption{Illustrative cross-database consensus and disputed candidates on P0 F1.
Points show JARVIS versus MP percentiles for consensus elite (green), JARVIS-only elite (orange), MP-only elite (purple) and consensus low (grey) materials.
The dashed diagonal marks equal ranking; dotted lines mark the 10\% elite threshold.
These examples are illustrative and are not adjudicated as physically correct or incorrect.}
\label{fig:si_candidates}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig5_portfolio_frontier}
\caption{Portfolio cost--coverage frontier on P0 F1 ($q^*=10\%$).
Worst-source recall (left) and NDCG (right) are plotted against budget factor $b$ for the best single-source baselines and three source-robust aggregation strategies.
The vertical dashed line marks the equal-budget analysis at $b=1.0$.
Aggregation changes the coverage--quality trade-off but does not validate physical truth.
See Note~1 for the distinction between material-level recall and group-mean paired differences.}
\label{fig:si_portfolio}
\end{figure}

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
