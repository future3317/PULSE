#!/usr/bin/env python
"""Compute conventional top-k / listwise ranking baselines for P0 F1.

Reads the frozen panel membership and concordance curve, computes metrics such
as precision@k, recall@k, plain Jaccard, overlap coefficient, RBO, and
top-weighted Kendall/Spearman, and writes a comparison table.  No hand-entered
numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from crosspiezo.analysis.ranking import (
    _spearman_rho_no_pvalue,
    overlap_coefficient_at_k,
    plain_jaccard_at_k,
    precision_at_k,
    rank_biased_overlap,
    recall_at_k,
    top_weighted_kendall_tau,
    top_weighted_spearman_rho,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_membership.parquet"
CURVE_PATH = PROJECT_ROOT / "results" / "phase7c" / "concordance_curve.csv"
BANDS_PATH = PROJECT_ROOT / "results" / "phase7c" / "concordance_bands.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase7c" / "baseline_metrics_comparison.csv"

PANEL = "P0"
METRIC = "F1_Frobenius"
Q_LEVELS = [1.0, 5.0, 10.0, 20.0, 50.0]


def _partial_naucc(curve: pd.DataFrame, q_min: float, q_max: float) -> float:
    sub = curve[(curve["q_percentile"] >= q_min) & (curve["q_percentile"] <= q_max)].copy()
    sub = sub.sort_values("q_percentile")
    x = sub["q_percentile"].to_numpy(dtype=float)
    y = sub["chance_adjusted_jaccard"].to_numpy(dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) / (q_max - q_min))


def main() -> int:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"panel membership not found: {PANEL_PATH}")

    panels = pd.read_parquet(PANEL_PATH)
    panel = panels[panels[PANEL].fillna(False)].copy()
    left = panel["jarvis_f1"].to_numpy(dtype=float)
    right = panel["mp_f1"].to_numpy(dtype=float)
    n = len(left)

    curve = pd.read_csv(CURVE_PATH)
    curve = curve[(curve["panel"] == PANEL) & (curve["metric"] == METRIC)].copy()
    curve = curve.sort_values("q_percentile")
    curve_map = curve.set_index("q_percentile")

    bands = pd.read_csv(BANDS_PATH)
    bands = bands[(bands["panel"] == PANEL) & (bands["metric"] == METRIC)].copy()
    band_map = bands.set_index("band")

    global_tau, _ = stats.kendalltau(left, right)
    global_rho = _spearman_rho_no_pvalue(left, right)

    rows = []
    for q in Q_LEVELS:
        k = int(curve_map.loc[q, "k"])
        observed_jaccard = float(curve_map.loc[q, "observed_jaccard"])
        adj_jaccard = float(curve_map.loc[q, "chance_adjusted_jaccard"])
        adj_low = float(curve_map.loc[q, "adj_jaccard_ci95_low"])
        adj_high = float(curve_map.loc[q, "adj_jaccard_ci95_high"])

        rows.append({
            "panel": PANEL,
            "metric": METRIC,
            "n_pairs": n,
            "q_percentile": q,
            "k": k,
            "global_kendall_tau": global_tau,
            "global_spearman_rho": global_rho,
            "observed_jaccard": observed_jaccard,
            "chance_adjusted_jaccard": adj_jaccard,
            "chance_adjusted_jaccard_ci95_low": adj_low,
            "chance_adjusted_jaccard_ci95_high": adj_high,
            "plain_jaccard": plain_jaccard_at_k(left, right, k),
            "overlap_coefficient": overlap_coefficient_at_k(left, right, k),
            "precision_at_k": precision_at_k(left, right, k),
            "recall_at_k": recall_at_k(left, right, k),
            "rank_biased_overlap_p095": rank_biased_overlap(left, right, p=0.95),
            "top_weighted_kendall_tau": top_weighted_kendall_tau(left, right, k),
            "top_weighted_spearman_rho": top_weighted_spearman_rho(left, right, k),
            "partial_naucc_1_to_q": _partial_naucc(curve, 1.0, q),
            "partial_naucc_band": (
                "elite" if q <= 10 else ("intermediate" if q <= 20 else "broad")
            ),
            "partial_naucc_band_value": float(band_map.loc[
                ("elite" if q <= 10 else ("intermediate" if q <= 20 else "broad"))
            ]["partial_nAUCC"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"[compute_baseline_metrics] wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
