#!/usr/bin/env python
"""Generate the Supplementary Information LaTeX file for CrossPiezo.

Reads the frozen result layer and post-approval diagnostics under
``results/phase9/`` and writes the supplementary LaTeX source. No hand-entered
numerical results are used.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase7c"
SOURCE_ROOT = Path(os.environ.get("CROSSPIEZO_SOURCE_ROOT", r"E:\CODE\PULSE"))
UPGRADE_RESULTS_ROOT = SOURCE_ROOT / "results" / "phase9"
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
Method & $f$ & $N$ & $\\tau$ & CI & nAUCC & CI \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s2_controls() -> str:
    """Full property-control comparison with provenance flags."""
    controls = pd.read_csv(RESULTS_ROOT / "property_controls.csv")
    provenance_path = UPGRADE_RESULTS_ROOT / "control_provenance_clean.csv"
    if not provenance_path.exists():
        provenance_path = RESULTS_ROOT / "control_provenance.csv"
    prov = pd.read_csv(provenance_path)
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
            flag_column = "high_identical_value_fraction_flag"
            if flag_column not in p0_prov.columns:
                flag_column = "same_field_copy_flag"
            copy_flag = p0_prov.loc[attr, flag_column]
            flag_str = "$\\checkmark$" if copy_flag else ""
        else:
            flag_str = ""
        rows.append(
            f"{attr_labels.get(attr, attr)} & {int(row['common_n'])} & "
            f"{_fmt(row['kendall_tau'])} & [{_fmt(row['kendall_tau_ci95_low'])}, {_fmt(row['kendall_tau_ci95_high'])}] & "
            f"{_fmt(row['nAUCC'])} & [{_fmt(row['nAUCC_ci95_low'])}, {_fmt(row['nAUCC_ci95_high'])}] & "
            f"{_fmt(row['Delta_tau'])} & {_fmt(row['Delta_nAUCC'])} & {flag_str} \\\\\n"
        )
    return """\\begin{table}[htbp]
\\scriptsize
\\setlength{\\tabcolsep}{2pt}
\\centering
\\caption{Property-control comparison on P0. The high identical-value fraction flag marks a high fraction of identical cross-source values; for energy above hull this is largely associated with tied zero-hull entries. The flag does not establish copying or shared provenance. The $\\Delta\\tau$ and $\\Delta$nAUCC columns are full-panel point estimates.}
\\label{tab:si_controls}
\\begin{tabular}{lcccccccc}
\\toprule
Attribute & $N$ & $\\tau$ & 95\\% CI & nAUCC & 95\\% CI & $\\Delta\\tau$ & $\\Delta$nAUCC & Identical-value flag \\\\
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
        "maximin_percentile": "Maximin",
        "balanced_union": "Balanced union",
        "minimax_oracle": "Minimax oracle",
    }
    p0 = p0[p0["strategy"].isin(strategy_labels)]
    rows = []
    for _, row in p0.iterrows():
        strat = strategy_labels.get(row["strategy"], row["strategy"])
        rows.append(
            f"{strat} & {_fmt(row['worst_source_recall'])} & {_fmt(row['worst_source_ndcg'])} & "
            f"{_fmt(row['portfolio_coverage'])} & {_fmt(row['full_proc_delta_best_recall'])} & "
            f"[{_fmt(row['full_proc_delta_best_recall_ci95_low'])}, {_fmt(row['full_proc_delta_best_recall_ci95_high'])}] \\\\\n"
        )
    return """\\begin{table}[htbp]
\\small
\\setlength{\\tabcolsep}{4pt}
\\centering
\\caption{Equal-budget ($b=1.0$) portfolio results on P0 F1 ($q^*=10\\%$), retained as an operational illustration rather than a validated method comparison.
Paired differences are versus the better single-source baseline and are computed as material-level worst-source recall differences using a full-procedure grouped bootstrap: each replicate resamples reduced-formula groups with replacement, assigns distinct identities to duplicated occurrences, re-runs the portfolio strategy and both single-source baselines, and recomputes the improvement (see Note~1).}
\\label{tab:si_portfolio}
\\begin{tabular}{lccccc}
\\toprule
Strategy & Recall & NDCG & Coverage & $\\Delta$Recall & 95\\% CI \\\\
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
        vals = " & ".join([_fmt(r) for r in sub["worst_source_recall"]])
        rows.append(f"{strategy_labels[strategy]} & {vals} \\\\\n")
    return """\\begin{table}[htbp]
\\small
\\setlength{\\tabcolsep}{4pt}
\\centering
\\caption{Portfolio worst-source recall on P0 F1 across budget factors $b=1.0, 1.5, 2.0$.}
\\label{tab:si_budget}
\\begin{tabular}{lccc}
\\toprule
Strategy & $b=1.0$ & $b=1.5$ & $b=2.0$ \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s5_baseline_metrics() -> str:
    """Conventional top-k / listwise ranking diagnostics for P0 F1."""
    df = pd.read_csv(RESULTS_ROOT / "baseline_metrics_comparison.csv")
    df = df[(df["panel"] == "P0") & (df["metric"] == "F1_Frobenius")].sort_values("q_percentile")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{int(row['q_percentile'])} & {int(row['k'])} & "
            f"{_fmt(row['plain_jaccard'])} & {_fmt(row['chance_adjusted_jaccard'])} & "
            f"{_fmt(row['top_weighted_kendall_tau'])} & "
            f"{_fmt(row['top_weighted_spearman_rho'])} & {_fmt(row['partial_naucc_band_value'])} \\\\\n"
        )
    global_tau = _fmt(df["global_kendall_tau"].iloc[0])
    global_rho = _fmt(df["global_spearman_rho"].iloc[0])
    return """\\begin{table}[htbp]
\\small
\\setlength{\\tabcolsep}{3pt}
\\centering
\\caption{Baseline ranking diagnostics on P0 F1 (global Kendall $\\tau$ = """ + global_tau + """, global Spearman $\\rho$ = """ + global_rho + """).
Columns are, from left to right: screened quantile $q$, corresponding top-$k$ set size, plain Jaccard, chance-adjusted Jaccard $\\widetilde J_q$, top-weighted Kendall $\\tau$ on the union of the two top-$k$ sets, top-weighted Spearman $\\rho$ on the same union, and the partial nAUCC of the band containing $q$ (elite $1$--$10$, intermediate $10$--$20$, broad $20$--$50$). The depth-agnostic RBO$_{0.95}$ is shown once in Figure~\\ref{fig:si_baseline_comparison}, rather than repeated in every row.}
\\label{tab:si_baseline_metrics}
\\begin{tabular}{rcccccc}
\\toprule
$q$ (\\%) & $k$ & Plain $J$ & $\\widetilde J_q$ & $\\tau_{q}$ & $\\rho_{q}$ & Band nAUCC \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_s6_matching_audit() -> str:
    """Frozen structure-matching audit summary."""
    df = pd.read_csv(RESULTS_ROOT / "matching_audit_summary.csv")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['panel']} & {int(row['count'])} & "
            f"{int(row['expected_count'])} & "
            f"{_fmt(row['mean_rms_distance'])} & {_fmt(row['median_rms_distance'])} & "
            f"{_fmt(row['max_rms_distance'])} & {_fmt(row['mean_max_distance'])} & "
            f"{_fmt(row['median_max_distance'])} & {_fmt(row['max_max_distance'])} & "
            f"{_fmt(row['mean_lattice_distance'])} & {_fmt(row['median_lattice_distance'])} & "
            f"{_fmt(row['max_lattice_distance'])} \\\\\n"
        )
    return """\\begin{table}[htbp]
\\small
\\setlength{\\tabcolsep}{3pt}
\\centering
\\caption{Frozen structure-matching audit summary. P0 is the full relaxed-structure matched universe; P2 is the tighter subset retained by the RMS and lattice thresholds recorded in ``panel\\_counts.json''.}
\\label{tab:si_matching_audit}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccccccccc}
\\toprule
Panel & Count & Expected & RMS mean & RMS med. & RMS max & Max mean & Max med. & Max max & Lat. mean & Lat. med. & Lat. max \\\\
\\midrule
""" + "".join(rows) + """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""


def table_s7_scientific_upgrade() -> str:
    """Unified cluster-bootstrap intervals, split into readable S7a/S7b tables."""
    df = pd.read_csv(UPGRADE_RESULTS_ROOT / "cluster_bootstrap_summary.csv")
    metric_labels = {
        "F1_Frobenius": "F1 Frobenius",
        "F3_Longitudinal": "F3 Longitudinal",
        "F4_KelvinOp": "F4 Kelvin",
    }
    overall_rows = []
    band_rows = []
    for _, row in df.sort_values(["panel", "metric"]).iterrows():
        onset = "--" if pd.isna(row["persistent_onset_delta0.05"]) else f"{row['persistent_onset_delta0.05']:.0f}\\%"
        overall_rows.append(
            f"{row['panel']} & {metric_labels[row['metric']]} & {int(row['n'])} & {int(row['n_groups'])} & "
            f"{_fmt(row['tau'])} [{_fmt(row['tau_ci95_low'])}, {_fmt(row['tau_ci95_high'])}] & "
            f"{_fmt(row['nAUCC'])} [{_fmt(row['nAUCC_ci95_low'])}, {_fmt(row['nAUCC_ci95_high'])}] & {onset} "
            + r"\\"
            + "\n"
        )
        band_rows.append(
            f"{row['panel']} & {metric_labels[row['metric']]} & "
            f"{_fmt(row['partial_nAUCC_elite'])} [{_fmt(row['partial_nAUCC_elite_ci95_low'])}, {_fmt(row['partial_nAUCC_elite_ci95_high'])}] & "
            f"{_fmt(row['partial_nAUCC_intermediate'])} [{_fmt(row['partial_nAUCC_intermediate_ci95_low'])}, {_fmt(row['partial_nAUCC_intermediate_ci95_high'])}] & "
            f"{_fmt(row['partial_nAUCC_broad'])} [{_fmt(row['partial_nAUCC_broad_ci95_low'])}, {_fmt(row['partial_nAUCC_broad_ci95_high'])}] "
            + r"\\"
            + "\n"
        )
    return r"""\begin{table}[htbp]
\small
\setlength{\tabcolsep}{4pt}
\centering
\renewcommand{\thetable}{S\arabic{table}a}
\renewcommand{\theHtable}{\arabic{table}a}
\caption{Overall screening-resolution results from the unified reduced-formula cluster bootstrap. Each of 2,000 replicates resamples reduced-formula groups, recomputes the bootstrap universe size and exact hypergeometric null, and yields a simultaneous studentized confidence band for the full curve. The nAUCC intervals are percentile intervals over the same replicates. The final column is the persistent practical-onset threshold $q^{\mathrm{persist}}_{0.05}$, requiring five consecutive lower confidence bounds above 0.05.}
\label{tab:si_upgrade_overall}
\begin{tabular}{llrrccc}
\toprule
Panel & Metric & $N$ & Groups & $\tau$ [95\% CI] & nAUCC [95\% CI] & $q^{\mathrm{persist}}_{0.05}$ \\
\midrule
""" + "".join(overall_rows) + r"""\bottomrule
\end{tabular}
\end{table}
\addtocounter{table}{-1}
\renewcommand{\thetable}{S\arabic{table}b}
\renewcommand{\theHtable}{\arabic{table}b}
\begin{table}[htbp]
\small
\setlength{\tabcolsep}{4pt}
\centering
\caption{Banded partial nAUCC estimates for the elite (1--10\%), intermediate (10--20\%) and broad (20--50\%) screening bands. Entries are estimates followed by percentile-bootstrap 95\% intervals from the same reduced-formula cluster-bootstrap replicates; simultaneous coverage applies to the confidence band for the full screening-resolution curve.}
\label{tab:si_upgrade_bands}
\begin{tabular}{llccc}
\toprule
Panel & Metric & Elite [95\% CI] & Intermediate [95\% CI] & Broad [95\% CI] \\
\midrule
""" + "".join(band_rows) + r"""\bottomrule
\end{tabular}
\end{table}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theHtable}{\arabic{table}}
"""


def table_s8_matching_trim() -> str:
    """Largest-distance trimming sensitivity for the primary P0 F1 panel."""
    df = pd.read_csv(UPGRADE_RESULTS_ROOT / "matching_distance_trim_sensitivity.csv")
    df = df[(df["panel"] == "P0") & (df["metric"] == "F1_Frobenius")
            & (df["distance_metric"] == "rms_distance")]
    rows = []
    for _, row in df.iterrows():
        trim_label = f"{row['trim_fraction']:.0%}".replace("%", r"\%")
        rows.append(
            f"{trim_label} & {int(row['n_removed'])} & {int(row['n_retained'])} & "
            f"{_fmt(row['nAUCC'])} & {_fmt(row['elite_partial_nAUCC'])} & "
            f"{int(row['top10_overlap'])}/{int(row['top10_k'])} \\\\\n"
        )
    return r"""\begin{table}[htbp]
\small
\setlength{\tabcolsep}{5pt}
\centering
\caption{Largest-RMS-distance trimming sensitivity on P0 F1. The primary matched panel is retained at 0\%; the other rows remove the largest RMS distances and recompute the point-estimate resolution summaries.}
\label{tab:si_matching_trim}
\begin{tabular}{r r r c c c}
\toprule
Trim fraction & Removed count & Retained $N$ & nAUCC & Elite partial nAUCC & Top-10 overlap \\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
\end{table}
"""


def table_s9_control_ties() -> str:
    """Tie-induced sensitivity for P0 control rankings."""
    df = pd.read_csv(UPGRADE_RESULTS_ROOT / "control_tie_sensitivity.csv")
    df = df[df["panel"] == "P0"]
    labels = {
        "band_gap": "Band gap",
        "energy_above_hull": "Energy above hull",
        "dielectric_total_trace": "Dielectric trace",
    }
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{labels.get(row['attribute'], row['attribute'])} & "
            f"{int(row['n_tied_rows_jarvis'])}/{int(row['n_tied_rows_mp'])} & "
            f"[{_fmt(row['nAUCC_min'])}, {_fmt(row['nAUCC_max'])}] & "
            f"[{_fmt(row['elite_partial_nAUCC_min'])}, {_fmt(row['elite_partial_nAUCC_max'])}] & "
            f"[{int(row['top10_overlap_min'])}, {int(row['top10_overlap_max'])}] \\\\\n"
        )
    return r"""\begin{table}[htbp]
\scriptsize
\setlength{\tabcolsep}{2pt}
\centering
\caption{Tie-induced sensitivity of P0 control rankings. Exact ties were independently shuffled 1,000 times; ranges are tie-breaking sensitivity ranges, not sampling confidence intervals.}
\label{tab:si_control_ties}
\begin{tabular}{l c c c c}
\toprule
Attribute & Tied rows (J/M) & nAUCC range & Elite partial range & \shortstack{Top-10\% shared-count range\\(out of 57)} \\
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
\end{table}
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
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}

\title{\textbf{Supplementary Information}\\CrossPiezo: Limited Elite-Set Reproducibility in Cross-Database Piezoelectric Screening}
\author{Anonymous Author(s)}
\date{}
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

The frozen matcher tolerances are: fractional length tolerance $\ell_\mathrm{tol}=0.2$, site tolerance $s_\mathrm{tol}=0.3$\,\AA, angle tolerance $5^\circ$, primitive-cell reduction enabled, and lattice scaling enabled (see ``configs/matching.yaml'').
P0 ($n=573$) is the full set of accepted structure matches; P2 ($n=207$) is the subset passing tighter RMS-distance and lattice-distance thresholds ($\mathrm{RMS}_\mathrm{thr}\approx0.0078$\,\AA, lattice-distance threshold $\approx0.0036$).
Table~\ref{tab:si_matching_audit} gives the resulting distance distributions.
Figure~\ref{fig:si_distance_discrepancy} shows an association that is statistically detectable at this panel size but negligible in magnitude (Spearman $\rho=-0.09$, $p=0.0312$); this is a sensitivity diagnostic rather than an exclusion of threshold or matching-error effects.

\subsection{Tensor invariants and conventions (Note~3)}

All piezoelectric response values are piezoelectric stress tensors $\mathbf{e}$ in C/m$^2$ with full Cartesian $3\times3\times3$ components.
Using the orthonormal Mandel representation
\[
M = \begin{bmatrix}
e_{1xx} & e_{1yy} & e_{1zz} & \sqrt{2}e_{1yz} & \sqrt{2}e_{1xz} & \sqrt{2}e_{1xy}\\
e_{2xx} & e_{2yy} & e_{2zz} & \sqrt{2}e_{2yz} & \sqrt{2}e_{2xz} & \sqrt{2}e_{2xy}\\
e_{3xx} & e_{3yy} & e_{3zz} & \sqrt{2}e_{3yz} & \sqrt{2}e_{3xz} & \sqrt{2}e_{3xy}
\end{bmatrix},
\]
and $m(\mathbf n)=(n_x^2,n_y^2,n_z^2,\sqrt{2}n_yn_z,\sqrt{2}n_xn_z,\sqrt{2}n_xn_y)^\mathsf{T}$ for $\|\mathbf n\|_2=1$, we define
\[
F_1=\|M\|_F,\qquad F_4=\|M\|_2=\sigma_{\max}(M),\qquad
F_3=\max_{\|\mathbf n\|_2=1}|\mathbf n^\mathsf{T}M m(\mathbf n)|.
\]
The internal check $0\leq F_3\leq F_4\leq F_1$ follows from these definitions.
The deterministic F3 implementation uses a 10,000-point Fibonacci sphere, 20 projected-gradient starts, at most 100 iterations per start with improvement tolerance $10^{-9}$, and a bounded Nelder--Mead spherical-coordinate polish with 400 iterations and objective tolerances $10^{-9}$/$10^{-12}$; it is cross-checked against a dense brute-force oracle in the convention audit.
Voigt-to-Cartesian conversion follows the standard convention $11\to xx$, $22\to yy$, $33\to zz$, $23\to yz$, $13\to xz$, $12\to xy$ with factor-of-two handling for shear components as appropriate for the stress tensor.
No sign conventions, unit conversions or coordinate rotations were applied silently; all transformations are recorded in the repository transformation history.

\subsection{Bootstrap, quantile and tie handling (Note~4)}

For equations below, $q$ is written as a fraction; the reported grid is $q=1\%,2\%,\dots,50\%$.
For a panel of size $N$, let $k(q)=\max\{1,\lfloor qN\rfloor\}$ and let $T_A(q)$ and $T_B(q)$ be the two top-$k(q)$ sets. With $X_q=|T_A(q)\cap T_B(q)|$, $J_q=X_q/(2k(q)-X_q)$.
Under independent random rankings, $X_q\sim\operatorname{Hypergeom}(N,k,k)$ and
\[
E_0[J_q]=\sum_{x=\max(0,2k-N)}^k
\frac{x}{2k-x}\frac{\binom{k}{x}\binom{N-k}{k-x}}{\binom{N}{k}},
\]
not $J_q$ evaluated at $E[X_q]$. The chance-adjusted overlap is $\widetilde J_q = (J_q-E_0[J_q])/(1-E_0[J_q])$; it can be negative, so nAUCC is not restricted to $[0,1]$.
For the primary cluster-bootstrap analysis, curve and onset inference use one reduced-formula cluster bootstrap with a studentized sup-norm critical value; scalar nAUCC and partial nAUCC intervals are percentile intervals from the same replicates. Each replicate resamples reduced-formula groups with replacement, recomputes the bootstrap universe size and exact hypergeometric null, and then recomputes the complete screening curve.

For property-control comparisons and the portfolio benchmark, bootstrap resampling is also by reduced-formula group to respect the dependence among entries sharing a composition.
In the portfolio full-procedure bootstrap, duplicated groups create distinct bootstrap identities so that re-selection and elite-set sizes are computed on the expanded bootstrap sample.
The random seed, number of replicates, studentization details, tie handling and $k$ rounding are recorded in the configuration and result files.
Ties in source rankings are broken by stable sorting; duplicate reduced formulae are retained in the ranking but noted in the panel metadata.
We distinguish two onset concepts. The persistent above-chance onset is $q^{\mathrm{persist}}_{0}=\min\{q_j:\operatorname{LCB}(q_{j+r})>0,\ r=0,\ldots,L-1\}$, whereas the prespecified persistent practical-onset threshold is $q^{\mathrm{persist}}_{0.05}=\min\{q_j:\operatorname{LCB}(q_{j+r})>0.05,\ r=0,\ldots,L-1\}$. We use $L=5$ and report $q^{\mathrm{persist}}_{0.05}$ as the primary practical threshold; $0.05$ is a prespecified practical threshold, not the unique criterion for statistical distinguishability from chance.
The legacy row-bootstrap curve is retained as a sensitivity analysis. Tables~\ref{tab:si_upgrade_overall} and~\ref{tab:si_upgrade_bands} report the primary cluster-bootstrap results; the tie-aware cutoff audit additionally reports exact overlap bounds when a score tie straddles $k$, together with absolute and relative score gaps. No such ambiguity occurs at the P0 F1 cutoffs $q=1\%,5\%,10\%$.

\subsection{Source workflow and tensor conventions (Note~5)}

Both JARVIS and MP report the piezoelectric stress tensor $\mathbf{e}$ in C/m$^2$ using the same Voigt order ($xx,yy,zz,yz,xz,xy$) and the same engineering-shear factor-of-two convention.
JARVIS values are expressed in the source-structure Cartesian frame; MP values are expressed in the MP IEEE-oriented structure frame.
Before any response comparison, MP structures are aligned to JARVIS structures (or vice versa) with a frozen ``pymatgen'' ``StructureMatcher'' configuration and the recovered rotation is applied to the MP Cartesian tensor.
The three reported scalars (F1, F3, F4) are coordinate-frame invariants, so they are unaffected by the source-specific structure-frame choice.

Several calculation settings are not documented in the release metadata used by CrossPiezo: exchange--correlation functional, pseudopotential family, DFPT vs finite-difference ionic response, plane-wave cutoff, k-point density, relaxation thresholds, primitive vs conventional defaults, and sign-convention history.
These items are therefore reported as unknown rather than assumed identical (see \path{docs/SCIENTIFIC_CONTRACT.md}).
The repository convention audit (\path{scripts/run_convention_audit.py}) confirms that F1, F3 and F4 are rotation invariant, that Voigt/Cartesian conversion and engineering-shear handling are internally consistent, and that the F3 optimiser converges to a dense-grid brute-force oracle.
This audit rules out low-level convention errors in the CrossPiezo pipeline; it does not prove that JARVIS and MP use identical calculation settings.

\subsection{Screening resolution vs conventional ranking diagnostics (Note~6)}

Screening resolution is reported with the chance-adjusted Jaccard curve $\widetilde J_q$ and its area (nAUCC). These metrics focus on the elite tail, where cross-database disagreement is largest.
For comparison, Table~\ref{tab:si_baseline_metrics} and Figures~\ref{fig:si_synthetic_baseline}--\ref{fig:si_baseline_comparison} report conventional diagnostics:
plain Jaccard $J_q$, chance-adjusted Jaccard $\widetilde J_q$, precision@$k$ (equal here to recall@$k$ and the overlap coefficient), Rank-Biased Overlap (RBO) with $p=0.95$, top-weighted Kendall $\tau$ on the union of the two top-$k$ sets, and top-weighted Spearman $\rho$ on the same union.

These diagnostics illustrate that global rank association and elite-set agreement answer different questions.
The synthetic example in Figure~\ref{fig:si_synthetic_baseline} constructs two length-100 rankings with global Kendall $\tau \approx 0.98$ but disjoint top-$5\%$ sets; global metrics are near-perfect while top-$5\%$ precision and Jaccard are zero.
On P0 F1, global Kendall $\tau=0.245$ and Spearman $\rho=0.350$ are weak-to-moderate rather than high; elite-tail agreement is nevertheless near chance for small $q$ (see Table~\ref{tab:si_baseline_metrics}).

\section{Supplementary figures}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig4_illustrative_candidates}
\caption{Illustrative cross-database consensus and disputed candidates on P0 F1.
Points show JARVIS versus MP rank percentiles (0 = highest) for consensus elite (green), JARVIS-only elite (orange), MP-only elite (purple) and consensus low (grey) materials.
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

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig_synthetic_baseline_metrics}
\caption{Synthetic motivating example for screening resolution.
Two length-100 rankings are constructed with global Kendall $\tau \approx 0.98$ but disjoint top-$5\%$ sets, showing that high global rank correlation does not guarantee agreement on elite-screening decisions.
The reported conventional metrics (global $\tau$, Spearman $\rho$, Rank-Biased Overlap) are near-perfect while top-$5\%$ precision and Jaccard are zero.
See Note~6 for the full metric definitions.}
\label{fig:si_synthetic_baseline}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig_s1_baseline_metric_comparison}
\caption{Comparison of conventional ranking diagnostics and screening-resolution on P0 F1.
Left: depth-agnostic global Kendall $\tau$, global Spearman $\rho$, and RBO. Right: set-based top-$q$ diagnostics on P0 F1.
Global rank association is modest, and elite-tail agreement is near-chance for small $q$; global association alone does not determine elite-set reproducibility.
See Table~\ref{tab:si_baseline_metrics} for the numerical values and Note~6 for definitions.}
\label{fig:si_baseline_comparison}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{fig_s2_distance_vs_rank_discrepancy}
\caption{Relaxed-structure RMS distance versus absolute F1 percentile gap for the 573 P0 matched pairs.
The association is statistically detectable at this panel size but negligible in magnitude ($\rho=-0.09$, $p=0.0312$); this is a sensitivity diagnostic rather than an exclusion of threshold or matching-error effects.
The LOWESS curve is illustrative and is not a fitted model.}
\label{fig:si_distance_discrepancy}
\end{figure}

\section{Supplementary tables}

"""
    doc += table_s1_high_response() + "\n\n"
    doc += table_s2_controls() + "\n\n"
    doc += table_s3_portfolio() + "\n\n"
    doc += table_s4_budget_sensitivity() + "\n\n"
    doc += table_s5_baseline_metrics() + "\n\n"
    doc += table_s6_matching_audit() + "\n\n"
    doc += table_s7_scientific_upgrade() + "\n\n"
    doc += table_s8_matching_trim() + "\n\n"
    doc += table_s9_control_ties() + "\n\n"
    doc += r"""
\end{document}
"""
    OUTPUT.write_text(doc, encoding="utf-8")
    print(f"[make_supplementary] Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
