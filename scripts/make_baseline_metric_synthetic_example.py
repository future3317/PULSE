#!/usr/bin/env python
"""Synthetic example: high global Kendall tau but disjoint elite tail decisions.

Constructs two length-100 rankings whose global Kendall correlation is high
(>0.9) but whose top-5% selected sets are disjoint.  This is used to argue that
screening resolution captures decision-relevant tail disagreement that global
correlation metrics miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from crosspiezo.analysis.ranking import (
    precision_at_k,
    top_weighted_kendall_tau,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase7c"
FIGURES_ROOT = PROJECT_ROOT / "figures" / "screening_resolution"


def main() -> int:
    rng = np.random.default_rng(2026)
    n = 100
    elite = 5  # top 5%

    left = np.arange(n, 0, -1, dtype=float) + rng.normal(scale=0.5, size=n)
    right = left.copy() + rng.normal(scale=0.5, size=n)

    # Swap the elite block with the next elite block so the top-5% sets differ
    # but the bulk ordering stays almost intact.
    block_a = right[:elite].copy()
    block_b = right[elite : 2 * elite].copy()
    right[:elite] = block_b - 2.0  # ensure they are no longer top elite
    right[elite : 2 * elite] = block_a + 2.0  # promote the second block

    tau_global, _ = stats.kendalltau(left, right)
    prec = precision_at_k(left, right, elite)
    tail_tau = top_weighted_kendall_tau(left, right, elite)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "left_score": left,
        "right_score": right,
        "left_rank": stats.rankdata(-left, method="average"),
        "right_rank": stats.rankdata(-right, method="average"),
        "in_left_top5pct": stats.rankdata(-left, method="average") <= elite,
        "in_right_top5pct": stats.rankdata(-right, method="average") <= elite,
    })
    csv_path = RESULTS_ROOT / "baseline_synthetic_example.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    base = ax.scatter(df["left_rank"], df["right_rank"], c="#BAB0AC", s=30, alpha=0.7, label="Other")
    left_elite = df[df["in_left_top5pct"]]
    right_elite = df[df["in_right_top5pct"]]
    ax.scatter(left_elite["left_rank"], left_elite["right_rank"], c="#E67600", s=60, marker="s", label="Left elite only")
    ax.scatter(right_elite["left_rank"], right_elite["right_rank"], c="#7B3FA0", s=60, marker="D", label="Right elite only")
    ax.plot([1, n], [1, n], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Left ranking")
    ax.set_ylabel("Right ranking")
    ax.set_title(
        f"Global Kendall $\\tau$ = {tau_global:.3f}, precision@5% = {prec:.2f}, "
        f"tail $\\tau$ = {tail_tau:.3f}"
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_ROOT / f"fig_synthetic_baseline_metrics.{ext}", dpi=300)
    plt.close(fig)

    print(f"[synthetic_example] wrote {csv_path}")
    print(f"[synthetic_example] global tau={tau_global:.3f}, precision@5%={prec:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
