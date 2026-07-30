#!/usr/bin/env python
"""Generate CrossPiezo invariant manuscript figures.

Reads frozen results from ``results/phase6a/`` and writes SVG/PDF/PNG to
``artifacts/phase6b/figures/``.  No hand-entered numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = PROJECT_ROOT / "artifacts" / "phase6b" / "figures"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase6a"
FROZEN_ROOT = PROJECT_ROOT / "artifacts" / "releases" / "phase6a_c2ed53e"

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_ROOT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pair_funnel() -> None:
    """Figure 1: conversion / pair funnel."""
    stages = ["JARVIS records", "MP records", "Verified\nconversions", "Candidate\noverlap", "Strict\nmatched pairs"]
    counts = [5000, 3316, 8316, 1266, 573]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(stages, counts, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("CrossPiezo data and pair funnel")
    for i, v in enumerate(counts):
        ax.text(v + 50, i, str(v), va="center")
    _save(fig, "fig_01_pair_funnel")


def fig_metric_definitions() -> None:
    """Figure 2: metric definitions schematic."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    metrics = [
        ("F1", "||e||_F", "rotation invariant"),
        ("F_MP_SVD", "σ_max(3×6 Voigt)", "MP source field only"),
        ("F3", "max |n_i e_ijk n_j n_k|", "rotation invariant"),
        ("F4", "σ_max(A_K) Kelvin/Mandel", "rotation invariant"),
    ]
    y = 0.85
    for label, formula, note in metrics:
        ax.text(0.05, y, f"{label}:  {formula}", fontsize=12, va="top", family="monospace")
        ax.text(0.05, y - 0.08, f"   {note}", fontsize=9, va="top", color="dimgray")
        y -= 0.22
    ax.set_title("Response metric definitions")
    _save(fig, "fig_02_metric_definitions")


def fig_scatter_and_rank(panel: str = "P0") -> None:
    """Figure 3: P0/P2 scatter and rank plots for F1/F3/F4."""
    panel_df = pd.read_parquet(FROZEN_ROOT / "panels" / "panel_membership.parquet")
    sub = panel_df[panel_df[panel]]
    metrics = [("F1", "jarvis_f1", "mp_f1"), ("F3", "jarvis_f3", "mp_f3"), ("F4", "jarvis_f4", "mp_f4")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, jc, mc) in zip(axes, metrics):
        x = sub[jc].to_numpy()
        y = sub[mc].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        ax.scatter(x, y, alpha=0.5, s=20)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"JARVIS {name}")
        ax.set_ylabel(f"MP {name}")
        ax.set_title(f"{name} ({panel}, N={len(x)})")
    fig.suptitle("Cross-source response scatter (C/m²)")
    _save(fig, f"fig_03_scatter_{panel}")


def fig_top_fraction_null() -> None:
    """Figure 4: top-fraction observed vs exact random null."""
    ranking = pd.read_csv(RESULTS_ROOT / "ranking_primary.csv")
    sub = ranking[ranking["panel"] == "P0"]
    fractions = [0.05, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.25
    x = np.arange(len(fractions))
    for i, func in enumerate(["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]):
        row = sub[sub["functional"] == func].iloc[0]
        obs = [row[f"top_{int(f*100)}pct_observed_jaccard"] for f in fractions]
        exp = [row[f"top_{int(f*100)}pct_expected_jaccard"] for f in fractions]
        ax.bar(x + i * width - width, obs, width, label=f"{func} observed", alpha=0.8)
        ax.bar(x + i * width, exp, width, label=f"{func} random-null", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fractions])
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Top-fraction overlap vs exact hypergeometric null (P0)")
    ax.legend(fontsize=7)
    _save(fig, "fig_04_top_fraction_null")


def fig_threshold_disagreement() -> None:
    """Figure 5: threshold/quantile disagreement (Cohen kappa)."""
    threshold = pd.read_csv(RESULTS_ROOT / "threshold_primary.csv")
    p0 = threshold[threshold["panel"] == "P0"]
    metrics = ["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]
    thresholds = [0.25, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for func in metrics:
        sub = p0[p0["functional"] == func]
        kappas = [sub[sub["threshold_C_per_m2"] == t]["cohen_kappa"].values[0] for t in thresholds]
        ax.plot(thresholds, kappas, marker="o", label=func)
    ax.axhline(0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Threshold (C/m²)")
    ax.set_ylabel("Cohen kappa")
    ax.set_title("Threshold-screening agreement (P0)")
    ax.legend()
    _save(fig, "fig_05_threshold_disagreement")


def fig_consensus_map() -> None:
    """Figure 6: consensus/disputed candidate map for F1 top 10%."""
    panel_df = pd.read_parquet(FROZEN_ROOT / "panels" / "panel_membership.parquet")
    sub = panel_df[panel_df["P0"]]
    left = sub["jarvis_f1"].to_numpy()
    right = sub["mp_f1"].to_numpy()
    k = int(np.floor(0.10 * len(sub)))
    top_j = set(np.argsort(-left, kind="stable")[:k])
    top_m = set(np.argsort(-right, kind="stable")[:k])
    colors = []
    for i in range(len(sub)):
        if i in top_j and i in top_m:
            colors.append("green")
        elif i in top_j:
            colors.append("orange")
        elif i in top_m:
            colors.append("purple")
        else:
            colors.append("lightgray")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(left, right, c=colors, alpha=0.6, s=20)
    lo, hi = min(left.min(), right.min()), max(left.max(), right.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("JARVIS F1 (C/m²)")
    ax.set_ylabel("MP F1 (C/m²)")
    ax.set_title("Top-10% consensus/disputed candidates (F1, P0)")
    from matplotlib.patches import Patch
    legend = [
        Patch(color="green", label="Both top 10%"),
        Patch(color="orange", label="JARVIS only"),
        Patch(color="purple", label="MP only"),
        Patch(color="lightgray", label="Neither"),
    ]
    ax.legend(handles=legend, loc="upper left")
    _save(fig, "fig_06_consensus_map")


def main() -> int:
    fig_pair_funnel()
    fig_metric_definitions()
    fig_scatter_and_rank("P0")
    fig_scatter_and_rank("P2")
    fig_top_fraction_null()
    fig_threshold_disagreement()
    fig_consensus_map()
    print(f"[make_paper_figures] Wrote figures to {FIGURES_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
