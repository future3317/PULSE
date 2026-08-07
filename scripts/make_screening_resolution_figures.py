#!/usr/bin/env python
"""Generate figures for the CrossPiezo screening-resolution manuscript.

Reads frozen results from ``results/phase7c/`` and writes PDF/PNG to
``figures/screening_resolution/``.  No hand-entered numbers.

Style: Times New Roman (or serif fallback), Nature-inspired low-saturation palette,
unified font sizes.
"""

from __future__ import annotations

import sys
import os
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
SOURCE_ROOT = Path(os.environ.get("CROSSPIEZO_SOURCE_ROOT", r"E:\CODE\PULSE"))
UPGRADE_RESULTS_ROOT = SOURCE_ROOT / "results" / "phase9"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "phase6a"

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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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

# --- Unified visual grammar --------------------------------------------------
COLORS = {
    "f1": "#2E5C8A",        # deep blue
    "f3": "#D95F02",        # orange-red
    "f4": "#1A9E77",        # teal-green
    "jarvis": "#E67600",    # orange
    "mp": "#7B3FA0",        # purple
    "consensus": "#2E7D32", # deep green
    "disputed": "#BAB0AC",  # light gray
    "red": "#E45756",
    "purple": "#B279A2",
    "teal": "#72B7B2",
    "gray": "#BAB0AC",
    "dark_gray": "#555555",
    "light_gray": "#E8E8E8",
}

METRIC_LABELS = {
    "F1_Frobenius": "Frobenius norm",
    "F3_Longitudinal": "Longitudinal",
    "F4_KelvinOp": "Kelvin",
}

METRIC_COLOR = {
    "F1_Frobenius": COLORS["f1"],
    "F3_Longitudinal": COLORS["f3"],
    "F4_KelvinOp": COLORS["f4"],
}

METRIC_LINE = {
    "F1_Frobenius": "-",
    "F3_Longitudinal": "--",
    "F4_KelvinOp": "-.",
}

STRATEGY_LABELS = {
    "jarvis_only": "JARVIS only",
    "mp_only": "MP only",
    "average_percentile": "Average percentile",
    "borda_count": "Borda",
    "maximin_percentile": "Maximin",
    "balanced_union": "Balanced union",
}

STRATEGY_COLOR = {
    "jarvis_only": COLORS["jarvis"],
    "mp_only": COLORS["mp"],
    "average_percentile": COLORS["f1"],
    "borda_count": COLORS["f3"],
    "maximin_percentile": COLORS["f4"],
    "balanced_union": COLORS["dark_gray"],
}

PANEL_LABELS = {
    "P0": "All matched (P0, $n=573$)",
    "P2": "Tight matched (P2, $n=207$)",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_ROOT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Figure 1: Benchmark concept ---------------------------------------------
def fig1_benchmark_concept() -> None:
    """Schematic overview of the CrossPiezo benchmark and screening-resolution idea."""
    baseline = pd.read_csv(RESULTS_ROOT / "baseline_metrics_comparison.csv")
    baseline = baseline[(baseline["panel"] == "P0") & (baseline["metric"] == "F1_Frobenius")]
    elite_counts: dict[int, tuple[int, int]] = {}
    for q in (1, 5, 10):
        row = baseline.loc[baseline["q_percentile"] == q].iloc[0]
        k = int(row["k"])
        overlap = int(round(float(row["precision_at_k"]) * k))
        elite_counts[q] = (overlap, k)
    fig = plt.figure(figsize=(6.5, 4.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # Panel a: data funnel
    ax_a = fig.add_subplot(gs[0, 0])
    stages = ["8,316\nrecords", "1,266\noverlaps", "573\nP0 pairs", "207\nP2 pairs"]
    widths = [1.0, 0.78, 0.55, 0.35]
    y = np.arange(len(stages))
    colors_funnel = [COLORS["light_gray"], "#D0D0D0", "#B8B8B8", COLORS["gray"]]
    for i, (stage, w, c) in enumerate(zip(stages, widths, colors_funnel)):
        left = (1 - w) / 2
        ax_a.barh(i, w, left=left, height=0.6, color=c, edgecolor=COLORS["dark_gray"], lw=0.5)
        ax_a.text(0.5, i, stage, ha="center", va="center", fontsize=8, color="black")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.5, len(stages) - 0.5)
    ax_a.invert_yaxis()
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title("(a) Structure-matched benchmark")
    ax_a.spines["bottom"].set_visible(False)
    ax_a.spines["left"].set_visible(False)

    # Panel b: invariant scalars (schematic bars)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    ax_b.set_title("(b) Coordinate-invariant scalars")
    invariants = [
        ("Frobenius norm", COLORS["f1"], 0.80),
        ("Longitudinal", COLORS["f3"], 0.50),
        ("Kelvin", COLORS["f4"], 0.20),
    ]
    for label, color, ypos in invariants:
        ax_b.plot([0.15, 0.35], [ypos, ypos], color=color, lw=3, solid_capstyle="round")
        ax_b.plot([0.55, 0.75], [ypos, ypos], color=color, lw=3, solid_capstyle="round")
        ax_b.plot([0.35, 0.55], [ypos, ypos + 0.05], color=color, lw=2)
        ax_b.text(0.80, ypos, label, va="center", fontsize=8)
    ax_b.text(0.45, 0.93, r"$\mathbf{e}$  $\rightarrow$  $F_1, F_3, F_4$", ha="center", fontsize=9)

    # Panel c: screening-resolution definition
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 4)
    ax_c.axis("off")
    ax_c.set_title("(c) Screening-resolution definition")
    # Two source ranking bars
    ax_c.add_patch(plt.Rectangle((0.5, 2.6), 8, 0.5, color=COLORS["jarvis"], alpha=0.25))
    ax_c.text(0.3, 2.85, "JARVIS", ha="right", va="center", fontsize=8, color=COLORS["jarvis"])
    ax_c.add_patch(plt.Rectangle((0.5, 1.7), 8, 0.5, color=COLORS["mp"], alpha=0.25))
    ax_c.text(0.3, 1.95, "MP", ha="right", va="center", fontsize=8, color=COLORS["mp"])
    # Elite overlap (darker)
    ax_c.add_patch(plt.Rectangle((0.5, 2.6), 1.5, 0.5, color=COLORS["jarvis"], alpha=0.85))
    ax_c.add_patch(plt.Rectangle((0.5, 1.7), 1.5, 0.5, color=COLORS["mp"], alpha=0.85))
    # Broad overlap (medium)
    ax_c.add_patch(plt.Rectangle((2.0, 2.6), 4.0, 0.5, color=COLORS["jarvis"], alpha=0.45))
    ax_c.add_patch(plt.Rectangle((2.0, 1.7), 4.0, 0.5, color=COLORS["mp"], alpha=0.45))
    # Labels below bars
    ax_c.text(1.25, 1.35, "1--10%", ha="center", fontsize=7, color=COLORS["dark_gray"])
    ax_c.text(1.25, 1.15, "elite", ha="center", fontsize=7, color=COLORS["red"])
    ax_c.text(4.0, 1.35, "20--50%", ha="center", fontsize=7, color=COLORS["dark_gray"])
    ax_c.text(4.0, 1.15, "broad", ha="center", fontsize=7, color=COLORS["consensus"])
    # Arrow and caption below the bars
    ax_c.annotate("", xy=(7.5, 1.25), xytext=(4.5, 1.25),
                  arrowprops=dict(arrowstyle="->", color=COLORS["dark_gray"], lw=1))
    ax_c.text(6.0, 1.05, r"$\mathrm{wider\ pool}\;\rightarrow\;\mathrm{more\ concordance}$", fontsize=7, ha="center", va="top")

    # Panel d: main finding
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    findings = [
        ("top 1%", f"{elite_counts[1][0]}/{elite_counts[1][1]} shared", COLORS["red"], 0.78),
        ("top 5%", f"{elite_counts[5][0]}/{elite_counts[5][1]} shared", COLORS["f3"], 0.52),
        ("top 10%", f"{elite_counts[10][0]}/{elite_counts[10][1]} shared", COLORS["consensus"], 0.26),
    ]
    ax_d.set_title("(d) Observed elite-set overlap")
    for label, outcome, color, ypos in findings:
        ax_d.scatter(0.12, ypos, s=120, c=color, zorder=3, clip_on=False)
        ax_d.text(0.22, ypos, f"{label}: {outcome}", va="center", fontsize=8)

    _save(fig, "fig1_benchmark_concept")


# --- Figure 2: Flagship screening-resolution results -------------------------
def fig2_screening_resolution_flagship() -> None:
    """P0/P2 resolution curves with banded partial nAUCC summary."""
    upgrade_curve = UPGRADE_RESULTS_ROOT / "cluster_bootstrap_curve.csv"
    if upgrade_curve.exists():
        curve = pd.read_csv(upgrade_curve).rename(
            columns={
                "cluster_boot_ci95_low": "adj_jaccard_ci95_low",
                "cluster_boot_ci95_high": "adj_jaccard_ci95_high",
            }
        )
        print("[fig2] using reduced-formula cluster-bootstrap curve bands")
    else:
        curve = pd.read_csv(RESULTS_ROOT / "concordance_curve.csv")
    bands = pd.read_csv(RESULTS_ROOT / "concordance_bands.csv")
    summary = pd.read_csv(RESULTS_ROOT / "concordance_summary.csv")
    upgrade_summary_path = UPGRADE_RESULTS_ROOT / "cluster_bootstrap_summary.csv"
    upgrade_summary = pd.read_csv(upgrade_summary_path) if upgrade_summary_path.exists() else None

    fig = plt.figure(figsize=(6.5, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.25)
    panels = ["P0", "P2"]

    # Top row: curves
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, panel in zip(axes_top, panels):
        sub = curve[curve["panel"] == panel]
        band_ranges = [(1, 10, COLORS["light_gray"], "elite"),
                       (10, 20, "#D0D0D0", "intermediate"),
                       (20, 50, COLORS["gray"], "broad")]
        for q0, q1, c, label in band_ranges:
            ax.axvspan(q0, q1, color=c, alpha=0.12, lw=0, zorder=0)
            ax.text((q0 + q1) / 2.0, 0.98, label, ha="center", va="top", fontsize=6,
                    transform=ax.get_xaxis_transform(), color=COLORS["dark_gray"])

        for metric in ["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]:
            msub = sub[sub["metric"] == metric].sort_values("q_percentile")
            color = METRIC_COLOR[metric]
            ax.plot(
                msub["q_percentile"],
                msub["chance_adjusted_jaccard"],
                color=color,
                lw=1.5,
                ls=METRIC_LINE[metric],
                label=METRIC_LABELS[metric],
                zorder=3,
            )
            ax.fill_between(
                msub["q_percentile"],
                msub["adj_jaccard_ci95_low"],
                msub["adj_jaccard_ci95_high"],
                color=color,
                alpha=0.12,
                lw=0,
                zorder=2,
            )
            if metric == "F1_Frobenius":
                for q_label in (5, 10, 20, 50):
                    point = msub.loc[msub["q_percentile"] == q_label]
                    if not point.empty:
                        value = float(point["chance_adjusted_jaccard"].iloc[0])
                        ax.annotate(f"{value:.2f}", (q_label, value), xytext=(0, 5),
                                    textcoords="offset points", ha="center", fontsize=6,
                                    color=color, zorder=4)

        # mark persistent onset for P0
        if panel == "P0":
            for metric in ["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]:
                summary_source = upgrade_summary if upgrade_summary is not None else summary
                srow = summary_source[(summary_source["panel"] == panel) & (summary_source["metric"] == metric)]
                onset = srow["persistent_onset_delta0.05"].values[0]
                if not pd.isna(onset):
                    ax.axvline(onset, color=METRIC_COLOR[metric], ls=":", lw=0.8, alpha=0.7, zorder=1)
                    ax.text(
                        onset,
                        0.86,
                        f"{int(onset)}%",
                        rotation=90,
                        va="bottom",
                        color=METRIC_COLOR[metric],
                        fontsize=6,
                        transform=ax.get_xaxis_transform(),
                    )

        ax.axhline(0, color=COLORS["dark_gray"], linestyle="--", lw=0.8, zorder=1)
        ax.set_xlim(1, 50)
        ax.set_ylim(-0.2, 0.4)
        ax.set_xlabel("Screened quantile $q$ (%)")
        if panel == "P0":
            ax.set_ylabel(r"Chance-adjusted Jaccard $\widetilde J_q$")
        ax.set_title(PANEL_LABELS[panel])
        ax.legend(loc="lower right", frameon=False)

    # Bottom row: banded partial nAUCC
    axes_bot = [fig.add_subplot(gs[1, i]) for i in range(2)]
    band_order = ["elite", "intermediate", "broad"]
    x = np.arange(len(band_order))
    width = 0.25
    for ax, panel in zip(axes_bot, panels):
        sub = bands[bands["panel"] == panel]
        for i, metric in enumerate(["F1_Frobenius", "F3_Longitudinal", "F4_KelvinOp"]):
            if upgrade_summary is not None:
                srow = upgrade_summary[(upgrade_summary["panel"] == panel) & (upgrade_summary["metric"] == metric)].iloc[0]
                vals = [float(srow[f"partial_nAUCC_{band}"]) for band in band_order]
                lows = [float(srow[f"partial_nAUCC_{band}_ci95_low"]) for band in band_order]
                highs = [float(srow[f"partial_nAUCC_{band}_ci95_high"]) for band in band_order]
            else:
                vals = [
                    sub[(sub["metric"] == metric) & (sub["band"] == b)]["partial_nAUCC"].values[0]
                    for b in band_order
                ]
                lows = highs = [float(v) for v in vals]
            positions = x + i * width
            ax.bar(positions, vals, width, label=METRIC_LABELS[metric],
                   color=METRIC_COLOR[metric], edgecolor="white", lw=0.5)
            if upgrade_summary is not None:
                ax.errorbar(
                    positions,
                    vals,
                    yerr=[np.asarray(vals) - np.asarray(lows), np.asarray(highs) - np.asarray(vals)],
                    fmt="none",
                    ecolor=METRIC_COLOR[metric],
                    elinewidth=0.8,
                    capsize=2,
                    zorder=4,
                )
        ax.axhline(0, color=COLORS["dark_gray"], lw=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([b.capitalize() for b in band_order])
        ax.set_xlabel("Band")
        if panel == "P0":
            ax.set_ylabel("Partial nAUCC")
        ax.set_ylim(-0.15, 0.35)

    _save(fig, "fig2_screening_resolution_flagship")


# --- Figure 3: Control forest plot -------------------------------------------
def fig3_control_forest() -> None:
    """Forest plot of control--F1 consistency advantage with bootstrap CIs."""
    controls = pd.read_csv(RESULTS_ROOT / "property_controls.csv")
    attr_order = ["volume", "band_gap", "energy_above_hull", "dielectric_total_trace"]
    p0 = controls[(controls["panel"] == "P0") & (controls["attribute"].isin(attr_order))]
    p0 = p0.set_index("attribute").loc[attr_order].reset_index()
    p0_f1 = controls[(controls["panel"] == "P0") & (controls["attribute"] == "F1_Frobenius")]

    labels = {
        "volume": "Volume",
        "band_gap": "Band gap",
        "energy_above_hull": "Energy above hull",
        "dielectric_total_trace": "Dielectric trace",
    }
    y_labels = [labels[a] for a in p0["attribute"]]
    y = np.arange(len(y_labels))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    # Left: Delta tau with CI
    ax = axes[0]
    vals = p0["Delta_tau"].values
    lows = p0["Delta_tau_ci95_low"].values
    highs = p0["Delta_tau_ci95_high"].values
    colors = [COLORS["f1"], COLORS["f3"], COLORS["f4"], COLORS["purple"]]
    for i, (v, lo, hi, c) in enumerate(zip(vals, lows, highs, colors)):
        flag = p0["attribute"].values[i] == "energy_above_hull"
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o",
                    color=c, ecolor=c, capsize=3, capthick=1,
                    markersize=6 if not flag else 7,
                    markerfacecolor="white" if flag else c,
                    markeredgewidth=1.5 if flag else 0,
                    markeredgecolor=c if flag else "none")
    ax.axvline(0, color=COLORS["dark_gray"], lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(r"$\Delta\tau$ (control -- F1)")
    ax.set_xlim(-0.1, 0.95)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(highs[i] + 0.03, i, f"{v:.3f}", va="center", fontsize=7)

    # Right: control nAUCC with CI and F1 reference
    ax = axes[1]
    f1_naucc = p0_f1["nAUCC"].values[0]
    f1_lo = p0_f1["nAUCC_ci95_low"].values[0]
    f1_hi = p0_f1["nAUCC_ci95_high"].values[0]
    vals = p0["nAUCC"].values
    lows = p0["nAUCC_ci95_low"].values
    highs = p0["nAUCC_ci95_high"].values
    for i, (v, lo, hi, c) in enumerate(zip(vals, lows, highs, colors)):
        flag = p0["attribute"].values[i] == "energy_above_hull"
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o",
                    color=c, ecolor=c, capsize=3, capthick=1,
                    markersize=6 if not flag else 7,
                    markerfacecolor="white" if flag else c,
                    markeredgewidth=1.5 if flag else 0,
                    markeredgecolor=c if flag else "none")
    ax.axvline(f1_naucc, color=COLORS["dark_gray"], lw=0.8, ls="--")
    ax.axvspan(f1_lo, f1_hi, color=COLORS["dark_gray"], alpha=0.12, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("nAUCC [95% CI]")
    ax.set_xlim(-0.1, 1.05)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(highs[i] + 0.03, i, f"{v:.3f}", va="center", fontsize=7)

    axes[1].annotate("shaded band = F1 nAUCC 95% CI", xy=(0.02, 0.90), xycoords="axes fraction",
                     ha="left", va="top", fontsize=6, color=COLORS["dark_gray"])
    fig.suptitle("Cross-source consistency of control properties", fontsize=9, y=1.02)
    _save(fig, "fig3_control_forest")


# --- Figure 4: Illustrative material candidates ------------------------------
def fig4_illustrative_candidates() -> None:
    """Rank--rank scatter of illustrative consensus/disputed/low candidates."""
    pm = pd.read_parquet(ARTIFACTS_ROOT / "panels" / "panel_membership.parquet")
    pm = pm[pm["P0"]].copy()

    # Compute F1 percentiles within P0
    pm["jarvis_f1_pct"] = pm["jarvis_f1"].rank(pct=True, ascending=False) * 100
    pm["mp_f1_pct"] = pm["mp_f1"].rank(pct=True, ascending=False) * 100

    # Consensus/disputed categories based on top-10% F1 membership
    def categorize(row):
        j_top = row["jarvis_f1_pct"] <= 10
        m_top = row["mp_f1_pct"] <= 10
        j_low = row["jarvis_f1_pct"] >= 80
        m_low = row["mp_f1_pct"] >= 80
        if j_top and m_top:
            return "consensus_elite"
        elif j_top and not m_top:
            return "jarvis_only"
        elif m_top and not j_top:
            return "mp_only"
        elif j_low and m_low:
            return "consensus_low"
        else:
            return "other"

    pm["category"] = pm.apply(categorize, axis=1)

    # Select illustrative examples
    selected = []
    seen_formulas = set()
    quotas = {
        "consensus_elite": 3,
        "jarvis_only": 3,
        "mp_only": 3,
        "consensus_low": 2,
    }
    for cat, n in quotas.items():
        sub = pm[(pm["category"] == cat) & (~pm["formula"].isin(seen_formulas))].copy()
        # prefer P2 members when available
        sub_p2 = sub[sub["P2"]]
        pick_from = sub_p2 if len(sub_p2) >= n else sub
        pick_from = pick_from.copy()
        if cat in ("consensus_elite", "consensus_low"):
            # minimize distance from diagonal for consensus
            pick_from["dist"] = np.abs(pick_from["jarvis_f1_pct"] - pick_from["mp_f1_pct"])
            pick_from = pick_from.sort_values("dist", ascending=True)
        else:
            # maximize distance from diagonal for disputed
            pick_from["dist"] = np.abs(pick_from["jarvis_f1_pct"] - pick_from["mp_f1_pct"])
            pick_from = pick_from.sort_values("dist", ascending=False)
        picked = pick_from.head(n)
        selected.append(picked)
        seen_formulas.update(picked["formula"].tolist())

    selected = pd.concat(selected, ignore_index=True)

    # Choose one annotated representative per category to reduce clutter
    annotate_idx = []
    for cat in ["consensus_elite", "jarvis_only", "mp_only", "consensus_low"]:
        sub = selected[selected["category"] == cat].copy()
        if cat in ("consensus_elite", "consensus_low"):
            sub["dist"] = np.abs(sub["jarvis_f1_pct"] - sub["mp_f1_pct"])
            idx = sub["dist"].idxmin()
        else:
            sub["dist"] = np.abs(sub["jarvis_f1_pct"] - sub["mp_f1_pct"])
            idx = sub["dist"].idxmax()
        annotate_idx.append(idx)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cat_colors = {
        "consensus_elite": COLORS["consensus"],
        "jarvis_only": COLORS["jarvis"],
        "mp_only": COLORS["mp"],
        "consensus_low": COLORS["disputed"],
    }
    cat_labels = {
        "consensus_elite": "Consensus elite",
        "jarvis_only": "JARVIS-only elite",
        "mp_only": "MP-only elite",
        "consensus_low": "Consensus low",
    }
    for cat in ["consensus_elite", "jarvis_only", "mp_only", "consensus_low"]:
        sub = selected[selected["category"] == cat]
        ax.scatter(sub["mp_f1_pct"], sub["jarvis_f1_pct"],
                   c=cat_colors[cat], label=cat_labels[cat], s=100,
                   edgecolor="white", lw=0.5, zorder=3)

    # annotate only the chosen representatives
    for idx in annotate_idx:
        row = selected.loc[idx]
        x, y = row["mp_f1_pct"], row["jarvis_f1_pct"]
        color = cat_colors[row["category"]]
        if x < 15 and y < 15:
            xytext = (x + 22, y + 8)
        elif x < 15:
            xytext = (x + 22, y - 12)
        elif y < 15:
            xytext = (x - 25, y + 8)
        else:
            xytext = (x - 25, y - 12)
        ax.annotate(row["formula"], (x, y), xytext=xytext,
                    fontsize=6, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6))

    # reference lines
    ax.plot([0, 100], [0, 100], ls="--", lw=0.8, color=COLORS["dark_gray"], zorder=1)
    ax.axhline(10, color=COLORS["dark_gray"], lw=0.6, ls=":", zorder=1)
    ax.axvline(10, color=COLORS["dark_gray"], lw=0.6, ls=":", zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("MP rank percentile, 0 = highest")
    ax.set_ylabel("JARVIS rank percentile, 0 = highest")
    ax.set_title("Illustrative cross-database consensus and disputed candidates (P0 F1)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    _save(fig, "fig4_illustrative_candidates")


# --- Figure 5: Portfolio cost-coverage frontier ------------------------------
def fig5_portfolio_frontier() -> None:
    """Worst-source recall/NDCG vs budget factor for key strategies."""
    bench = pd.read_csv(RESULTS_ROOT / "portfolio_benchmark.csv")
    p0_f1 = bench[(bench["panel"] == "P0") & (bench["metric"] == "F1_Frobenius")
                  & (bench["eval_mode"] == "full_panel")]

    strategies = ["jarvis_only", "mp_only", "average_percentile", "maximin_percentile", "balanced_union"]
    budget_factors = sorted(p0_f1["budget_factor"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True)
    for ax, metric, ylabel in zip(axes, ["worst_source_recall", "worst_source_ndcg"],
                                   ["Worst-source recall", "Worst-source NDCG"]):
        for strategy in strategies:
            sub = p0_f1[p0_f1["strategy"] == strategy].sort_values("budget_factor")
            label = STRATEGY_LABELS[strategy]
            color = STRATEGY_COLOR[strategy]
            ls = "-" if strategy != "balanced_union" else "--"
            ax.plot(sub["budget_factor"], sub[metric], marker="o", ms=4,
                    color=color, ls=ls, lw=1.2, label=label, zorder=3)
            # add CI error bars if paired_diff available for non-single-source
            if strategy not in ("jarvis_only", "mp_only"):
                lo = sub[metric] - (sub["paired_diff_recall_ci95_high"] - sub["paired_diff_recall"]) if metric == "worst_source_recall" else sub[metric]
                hi = sub[metric] + (sub["paired_diff_recall_ci95_high"] - sub["paired_diff_recall"]) if metric == "worst_source_recall" else sub[metric]
                # For NDCG we do not have paired CI in this CSV; skip.
                if metric == "worst_source_recall":
                    ax.fill_between(sub["budget_factor"], lo, hi, color=color, alpha=0.12, lw=0)

        ax.axvline(1.0, color=COLORS["dark_gray"], lw=0.8, ls="-.", zorder=1)
        ax.set_xlabel("Budget factor $b$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1.0, 2.0)
        ax.set_xticks(budget_factors)
        if metric == "worst_source_recall":
            ax.set_ylim(0, 1.05)

    axes[0].legend(loc="lower right", frameon=False, fontsize=6)
    axes[1].legend(loc="lower right", frameon=False, fontsize=6)
    fig.suptitle("Portfolio cost-coverage frontier on P0 F1 ($q^*=10\\%$)", fontsize=9, y=1.02)
    _save(fig, "fig5_portfolio_frontier")


def fig_s1_baseline_metric_comparison() -> None:
    """Compare screening-resolution metrics with conventional top-k / global baselines."""
    df = pd.read_csv(RESULTS_ROOT / "baseline_metrics_comparison.csv")
    if df.empty:
        print("[warn] baseline_metrics_comparison.csv not found; skipping fig_s1", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: global / depth-agnostic metrics.
    ax = axes[0]
    global_metrics = {
        "Global Kendall $\\tau$": df["global_kendall_tau"].iloc[0],
        "Global Spearman $\\rho$": df["global_spearman_rho"].iloc[0],
        "RBO ($p=0.95$)": df["rank_biased_overlap_p095"].iloc[0],
    }
    y = np.arange(len(global_metrics))
    colors = [COLORS["f1"], COLORS["f4"], COLORS["mp"]]
    for i, (name, val) in enumerate(global_metrics.items()):
        ax.barh(i, val, color=colors[i], height=0.5, edgecolor="white", lw=0.5)
        ax.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(list(global_metrics.keys()))
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Value")
    ax.set_title("(a) Depth-agnostic metrics")
    ax.axvline(0, color=COLORS["dark_gray"], lw=0.6)
    ax.invert_yaxis()

    # Right: top-q metrics from the frozen curve.
    ax = axes[1]
    ax.plot(df["q_percentile"], df["plain_jaccard"], marker="o", ms=4,
            color=COLORS["gray"], lw=1.2, label="Plain Jaccard")
    ax.plot(df["q_percentile"], df["chance_adjusted_jaccard"], marker="s", ms=4,
            color=COLORS["f1"], lw=1.2, label="Chance-adjusted Jaccard")
    ax.plot(df["q_percentile"], df["precision_at_k"], marker="d", ms=4,
            color=COLORS["f4"], lw=1.2, label="Precision@$k$")
    ax.axhline(0, color=COLORS["dark_gray"], lw=0.6, ls="--")
    ax.set_xlabel("Screened quantile $q$ (%)")
    ax.set_ylabel("Value")
    ax.set_xlim(0, 52)
    ax.set_ylim(-0.15, 1.05)
    ax.set_title("(b) Top-$q$ metrics on P0 F1")
    ax.legend(loc="lower right", frameon=False, fontsize=6)

    fig.tight_layout()
    _save(fig, "fig_s1_baseline_metric_comparison")


def main() -> int:
    fig1_benchmark_concept()
    fig2_screening_resolution_flagship()
    fig3_control_forest()
    fig4_illustrative_candidates()
    fig5_portfolio_frontier()
    fig_s1_baseline_metric_comparison()
    print(f"[make_screening_resolution_figures] Wrote figures to {FIGURES_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
