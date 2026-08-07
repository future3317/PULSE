#!/usr/bin/env python
"""Plot RMS structure distance versus cross-source F1 rank discrepancy for P0.

Reads ``artifacts/phase6a/panels/panel_membership.parquet``, computes
percentiles of JARVIS and MP F1 within P0, and plots each pair's RMS relaxed
structure distance against the absolute percentile gap.

Saves ``figures/screening_resolution/fig_s2_distance_vs_rank_discrepancy.{pdf,png}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = (
    PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_membership.parquet"
)
FIGURE_DIR = PROJECT_ROOT / "figures" / "screening_resolution"
OUTPUT_PDF = FIGURE_DIR / "fig_s2_distance_vs_rank_discrepancy.pdf"
OUTPUT_PNG = FIGURE_DIR / "fig_s2_distance_vs_rank_discrepancy.png"


def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Simple LOWESS via local linear regression on sorted x."""
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    n = len(xs)
    window = max(3, int(frac * n))
    y_smooth = np.empty(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, lo + window)
        weights = np.exp(-0.5 * ((xs[lo:hi] - xs[i]) / (xs.std() + 1e-12)) ** 2)
        weights = weights / weights.sum()
        y_smooth[i] = np.sum(weights * ys[lo:hi])
    return xs, y_smooth


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pm = pd.read_parquet(PANEL_PATH)
    p0 = pm[pm["P0"]].copy()

    p0["jarvis_f1_pct"] = p0["jarvis_f1"].rank(pct=True, ascending=False)
    p0["mp_f1_pct"] = p0["mp_f1"].rank(pct=True, ascending=False)
    p0["abs_pct_gap"] = (p0["jarvis_f1_pct"] - p0["mp_f1_pct"]).abs()

    x = p0["rms_distance"].to_numpy(dtype=np.float64)
    y = p0["abs_pct_gap"].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    rho, pvalue = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(
        x,
        y * 100.0,
        alpha=0.35,
        s=15,
        color="#1f4e79",
        edgecolors="none",
        label="P0 matched pairs",
    )

    if len(x) > 10:
        xs, ys = _lowess(x, y * 100.0, frac=0.3)
        ax.plot(xs, ys, color="#c44e52", linewidth=2.0, label="LOWESS trend")

    ax.set_xlabel("RMS relaxed structure distance (Å)", fontsize=10)
    ax.set_ylabel("|JARVIS F1 percentile − MP F1 percentile| (%)", fontsize=10)
    ax.set_title(
        "Structure-match distance versus cross-source F1 rank discrepancy", fontsize=11
    )

    textstr = (
        f"Spearman rho = {rho:.3f}\n"
        f"p-value = {pvalue:.2e}\n"
        f"n = {len(x)}"
    )
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3, edgecolor="gray"),
    )

    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUTPUT_PDF, dpi=300)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"[make_matching_distance_discrepancy_figure] Wrote {OUTPUT_PDF} and {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
