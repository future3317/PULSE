#!/usr/bin/env python
"""Generate figures for the CrossPiezo screening-resolution manuscript.

Reads frozen results from ``results/phase7c/`` and writes PDF/PNG to
``figures/screening_resolution/``.  No hand-entered numbers.

Style: Times New Roman (or serif fallback), Nature-inspired low-saturation palette,
unified font sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = PROJECT_ROOT / "figures" / "screening_resolution"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase7c"

# --- Typography --------------------------------------------------------------
FONT_NAME = "Times New Roman"
_available = {f.name for f in font_manager.fontManager.ttflist}
if FONT_NAME in _available:
    _family = FONT_NAME
else:
    _family = "serif"
    print(f"[warn] {FONT_NAME} not found; falling back to generic serif", file=sys.stderr)

plt.rcParams.update(
    {
        "font.family": _family,
        "font.serif": [FONT_NAME, "DejaVu Serif", "serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# --- Nature-inspired low-saturation palette ----------------------------------
COLORS = {
    "f1": "#4C78A8",      # muted blue
    "f3": "#F58518",      # muted orange
    "f4": "#54A24B",      # muted green
    "red": "#E45756",     # muted red
    "purple": "#B279A2",  # muted purple
    "teal": "#72B7B2",    # muted teal
    "gray": "#BAB0AC",    # light gray
    "dark_gray": "#555555",
}

METRIC_LABELS = {
    "F1_Frobenius": "F1 Frobenius",
    "F3_Longitudinal": "F3 Longitudinal",
    "F4_KelvinOp": "F4 KelvinOp",
}

METRIC_COLOR = {
    "F1_Frobenius": COLORS["f1"],
    "F3_Longitudinal": COLORS["f3"],
    "F4_KelvinOp": COLORS["f4"],
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_screening_resolution_curves() -> None:
    """P0 screening-resolution curves for F1/F3/F4 with simultaneous bands."""
    curve = pd.read_csv(RESULTS_ROOT / "concordance_curve.csv")
    p0 = curve[curve["panel"] == "P0"]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    # Shaded bands for elite / intermediate / broad
    band_colors = {
        "elite": (0.95, 0.95, 0.95),
        "intermediate": (0.90, 0.90, 0.90),
        "broad": (0.85, 0.85, 0.85),
    }
    ax.axvspan(1, 10, color=band_colors["elite"], lw=0, zorder=0)
    ax.axvspan(10, 20, color=band_colors["intermediate"], lw=0, zorder=0)
    ax.axvspan(20, 50, color=band_colors["broad"], lw=0, zorder=0)

    for metric in ["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]:
        sub = p0[p0["metric"] == metric].sort_values("q_percentile")
        color = METRIC_COLOR[metric]
        ax.plot(
            sub["q_percentile"],
            sub["chance_adjusted_jaccard"],
            color=color,
            lw=1.5,
            label=METRIC_LABELS[metric],
            zorder=3,
        )
        ax.fill_between(
            sub["q_percentile"],
            sub["adj_jaccard_ci95_low"],
            sub["adj_jaccard_ci95_high"],
            color=color,
            alpha=0.20,
            lw=0,
            zorder=2,
        )

    ax.axhline(0, color=COLORS["dark_gray"], linestyle="--", lw=0.8, zorder=1)
    ax.set_xlim(1, 50)
    ax.set_xlabel("Screened quantile $q$ (%)")
    ax.set_ylabel("Chance-adjusted Jaccard $\widetilde J_q$")
    ax.set_title("Screening-resolution curves (P0, $n=573$)")
    ax.legend(loc="lower right", frameon=False)

    # Band labels at top
    ax.text(5.5, ax.get_ylim()[1] * 0.92, "elite", ha="center", fontsize=7, color=COLORS["dark_gray"])
    ax.text(15, ax.get_ylim()[1] * 0.92, "intermediate", ha="center", fontsize=7, color=COLORS["dark_gray"])
    ax.text(35, ax.get_ylim()[1] * 0.92, "broad", ha="center", fontsize=7, color=COLORS["dark_gray"])

    _save(fig, "fig1_screening_resolution_curves")


def fig2_banded_naucc() -> None:
    """Partial nAUCC by band for P0 and P2."""
    bands = pd.read_csv(RESULTS_ROOT / "concordance_bands.csv")
    band_order = ["elite", "intermediate", "broad"]
    panels = ["P0", "P2"]
    metrics = ["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    width = 0.25
    x = np.arange(len(band_order))

    for ax, panel in zip(axes, panels):
        sub = bands[bands["panel"] == panel]
        for i, metric in enumerate(metrics):
            vals = [
                sub[(sub["metric"] == metric) & (sub["band"] == b)]["partial_nAUCC"].values[0]
                for b in band_order
            ]
            ax.bar(x + i * width, vals, width, label=METRIC_LABELS[metric], color=METRIC_COLOR[metric])
        ax.axhline(0, color=COLORS["dark_gray"], lw=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([b.capitalize() for b in band_order])
        ax.set_xlabel("Band")
        ax.set_title(f"{panel} ($n={573 if panel == 'P0' else 207}$)")

    axes[0].set_ylabel("Partial nAUCC")
    axes[1].legend(loc="upper left", frameon=False)
    _save(fig, "fig2_banded_naucc")


def fig3_property_controls() -> None:
    """Delta tau and Delta nAUCC for P0 controls vs F1."""
    controls = pd.read_csv(RESULTS_ROOT / "property_controls.csv")
    p0 = controls[(controls["panel"] == "P0") & (controls["attribute"] != "F1_Frobenius")]
    attributes = ["volume", "band_gap", "energy_above_hull", "dielectric_total_trace"]
    labels = ["Volume", "Band gap", "Energy above hull", "Dielectric trace"]
    colors = [COLORS["f1"], COLORS["f3"], COLORS["f4"], COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    for ax, metric, ylabel in zip(
        axes, ["Delta_tau", "Delta_nAUCC"], [r"$\Delta\tau$", r"$\Delta$nAUCC"]
    ):
        vals = [p0[p0["attribute"] == a][metric].values[0] for a in attributes]
        ax.barh(labels, vals, color=colors, height=0.6)
        ax.axvline(0, color=COLORS["dark_gray"], lw=0.8)
        ax.set_xlabel(ylabel)
        ax.set_title("Control -- F1 consistency advantage (P0)")
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=7)

    _save(fig, "fig3_property_controls")


def fig4_portfolio_benchmark() -> None:
    """Worst-source recall by strategy and budget factor (P0 F1)."""
    bench = pd.read_csv(RESULTS_ROOT / "portfolio_benchmark.csv")
    p0_f1 = bench[(bench["panel"] == "P0") & (bench["metric"] == "F1_Frobenius") & (bench["eval_mode"] == "full_panel")]

    strategies = ["jarvis_only", "mp_only", "average_percentile", "borda_count", "maximin_percentile", "balanced_union"]
    labels = ["JARVIS only", "MP only", "Avg percentile", "Borda", "Maximin", "Balanced union"]
    budget_factors = [1.0, 2.0]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(strategies))
    width = 0.35
    color_b1 = COLORS["f1"]
    color_b2 = COLORS["gray"]

    for i, bf in enumerate(budget_factors):
        vals = [p0_f1[(p0_f1["strategy"] == s) & (p0_f1["budget_factor"] == bf)]["worst_source_recall"].values[0] for s in strategies]
        label = f"$b={bf:.1f}$" + (" (coverage upper bound)" if bf == 2.0 else "")
        ax.bar(x + i * width, vals, width, label=label, color=color_b1 if bf == 1.0 else color_b2)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Worst-source recall")
    ax.set_title("Equal-budget portfolio benchmark (P0 F1, $q^*=10\\%$)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, 1.05)

    _save(fig, "fig4_portfolio_benchmark")


def main() -> int:
    fig1_screening_resolution_curves()
    fig2_banded_naucc()
    fig3_property_controls()
    fig4_portfolio_benchmark()
    print(f"[make_screening_resolution_figures] Wrote figures to {FIGURES_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
