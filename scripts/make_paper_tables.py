#!/usr/bin/env python
"""Generate CrossPiezo invariant manuscript LaTeX tables.

Reads frozen results from ``results/phase6a/`` and writes ``.tex`` fragments to
``artifacts/phase6b/tables/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_ROOT = PROJECT_ROOT / "PiezoProtocol_LaTeX_Draft_v0.1" / "tables"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase6a"


def _write(name: str, content: str) -> None:
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    (TABLES_ROOT / f"{name}.tex").write_text(content, encoding="utf-8")


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def table_primary_ranking() -> None:
    df = pd.read_csv(RESULTS_ROOT / "ranking_primary.csv")
    cols = ["panel", "functional", "n_pairs", "kendall_tau",
            "kendall_tau_ci95_low", "kendall_tau_ci95_high",
            "top_10pct_observed_jaccard", "top_10pct_expected_jaccard",
            "top_10pct_chance_adjusted_jaccard", "median_normalized_rank_displacement"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Primary ranking stability for the strict structure-matched invariant panels P0 and P2.}",
        r"\label{tab:primary_ranking}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        "Panel & Metric & N & $\\tau$ & $\\tau$ CI low & $\\tau$ CI high & $J_{\\text{obs}}$ & $E[J]$ & $J_{\\text{adj}}$ & Median rank displacement \\\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['panel']} & {row['functional']} & {int(row['n_pairs'])} & "
            f"{_fmt(row['kendall_tau'])} & {_fmt(row['kendall_tau_ci95_low'])} & "
            f"{_fmt(row['kendall_tau_ci95_high'])} & {_fmt(row['top_10pct_observed_jaccard'])} & "
            f"{_fmt(row['top_10pct_expected_jaccard'])} & {_fmt(row['top_10pct_chance_adjusted_jaccard'])} & "
            f"{_fmt(row['median_normalized_rank_displacement'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write("table_primary_ranking", "\n".join(lines))


def table_robustness() -> None:
    df = pd.read_csv(RESULTS_ROOT / "robustness_primary.csv")
    df = df[df["functional"] == "F1_Frobenius"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness matrix for F1 Frobenius across subsets and transformations.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Subset & N & $\\tau$ & $\\tau$ CI low & $\\tau$ CI high & $J_{\\text{adj}}$ (top 10\\%) & Median rank displacement \\\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['subset']} & {int(row['n_pairs'])} & {_fmt(row['kendall_tau'])} & "
            f"{_fmt(row['kendall_tau_ci95_low'])} & {_fmt(row['kendall_tau_ci95_high'])} & "
            f"{_fmt(row['top_10pct_chance_adjusted_jaccard'])} & {_fmt(row['median_normalized_rank_displacement'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write("table_robustness", "\n".join(lines))


def table_top_quantile_consensus() -> None:
    df = pd.read_csv(RESULTS_ROOT / "robust_summary.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Top-quantile consensus and source-only counts across primary metrics (P0).}",
        r"\label{tab:top_quantile_consensus}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        "Metric & Quantile & Both top & JARVIS only & MP only & Neither \\\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['functional']} & {int(row['quantile']*100)}\\% & {int(row['both_top'])} & "
            f"{int(row['jarvis_only_top'])} & {int(row['mp_only_top'])} & {int(row['neither_top'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write("table_top_quantile_consensus", "\n".join(lines))


def main() -> int:
    table_primary_ranking()
    table_robustness()
    table_top_quantile_consensus()
    print(f"[make_paper_tables] Wrote tables to {TABLES_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
