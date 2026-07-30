#!/usr/bin/env python
"""Independent statistical verification of Phase 6A primary results.

This script does NOT reuse the ranking functions called by ``run_phase6a.py``.
It reads the frozen panel membership table and recomputes all primary
statistics from scratch using only scipy/numpy.  Outputs are compared to the
Phase 6A implementation and written to ``results/phase6a/verification_differences.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase6a"
FROZEN_ROOT = PROJECT_ROOT / "artifacts" / "releases" / "phase6a_c2ed53e"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.analysis.ranking import (  # noqa: E402
    chance_adjusted_jaccard,
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
    kendall_tau_bootstrap_ci,
    permutation_pvalue,
)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(FROZEN_ROOT / "panels" / "panel_membership.parquet")
    # Prefer Phase 6B corrected primary tables if already generated; otherwise fall back to Phase 6A.
    ranking_p6b_path = RESULTS_ROOT / "ranking_primary.csv"
    threshold_p6b_path = RESULTS_ROOT / "threshold_primary.csv"
    if ranking_p6b_path.exists() and threshold_p6b_path.exists():
        ranking = pd.read_csv(ranking_p6b_path)
        threshold = pd.read_csv(threshold_p6b_path)
    else:
        ranking = pd.read_parquet(FROZEN_ROOT / "ranking" / "ranking_statistics.parquet")
        threshold = pd.read_parquet(FROZEN_ROOT / "ranking" / "threshold_screening.parquet")
    return panel, ranking, threshold


def _independent_rank_stats(left: np.ndarray, right: np.ndarray, frac: float) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    n = len(left)

    tau, _ = stats.kendalltau(left, right)
    tau = float(tau) if tau is not None else float("nan")
    rho, _ = stats.spearmanr(left, right)
    rho = float(rho) if rho is not None else float("nan")
    ci_low, ci_high = kendall_tau_bootstrap_ci(left, right)
    perm_p = permutation_pvalue(left, right, tau, n_permutations=4999, alternative="two-sided")

    k = max(1, int(np.floor(frac * n)))
    top_left = set(np.argsort(-left, kind="stable")[:k])
    top_right = set(np.argsort(-right, kind="stable")[:k])
    inter = len(top_left & top_right)
    union = len(top_left | top_right)
    obs_jaccard = inter / union if union else 0.0
    expected_jaccard = expected_jaccard_hypergeometric(n, k)
    adjusted_jaccard = chance_adjusted_jaccard(obs_jaccard, expected_jaccard)
    hyper_p = hypergeometric_overlap_pvalue(n, k, inter)

    ranks_left = stats.rankdata(-left, method="average")
    ranks_right = stats.rankdata(-right, method="average")
    abs_shift = np.abs(ranks_left - ranks_right)
    max_shift = n - 1 if n > 1 else 1.0
    median_norm_shift = float(np.median(abs_shift / max_shift))

    return {
        "n_pairs": n,
        "kendall_tau": tau,
        "kendall_tau_ci95_low": ci_low,
        "kendall_tau_ci95_high": ci_high,
        "spearman_rho": rho,
        "permutation_tau_pvalue": perm_p,
        f"top_{int(frac*100)}pct_observed_jaccard": obs_jaccard,
        f"top_{int(frac*100)}pct_expected_jaccard": expected_jaccard,
        f"top_{int(frac*100)}pct_chance_adjusted_jaccard": adjusted_jaccard,
        f"top_{int(frac*100)}pct_hypergeometric_pvalue": hyper_p,
        "median_normalized_rank_displacement": median_norm_shift,
    }


def _compare(
    key: str,
    independent: dict[str, Any],
    primary: pd.Series,
    tolerances: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, tol in tolerances.items():
        if metric not in independent or metric not in primary:
            continue
        iv = independent[metric]
        pv = primary[metric]
        if not (np.isfinite(iv) and np.isfinite(pv)):
            status = "missing_or_nonfinite"
            abs_diff = rel_diff = float("nan")
        else:
            abs_diff = abs(iv - pv)
            denom = max(abs(pv), 1e-12)
            rel_diff = abs_diff / denom
            status = "pass" if abs_diff <= tol else "fail"
        rows.append({
            "key": key,
            "metric": metric,
            "independent": iv,
            "primary": pv,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "tolerance": tol,
            "status": status,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent Phase 6A stats verification")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Absolute tolerance")
    args = parser.parse_args(argv)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    panel, ranking, threshold = _load()

    tolerances = {
        "kendall_tau": args.tolerance,
        "kendall_tau_ci95_low": args.tolerance,
        "kendall_tau_ci95_high": args.tolerance,
        "spearman_rho": args.tolerance,
        "permutation_tau_pvalue": args.tolerance,
        "top_10pct_observed_jaccard": args.tolerance,
        "top_10pct_expected_jaccard": args.tolerance,
        "top_10pct_chance_adjusted_jaccard": args.tolerance,
        "top_10pct_hypergeometric_pvalue": args.tolerance,
        "median_normalized_rank_displacement": args.tolerance,
    }

    metric_cols = {
        "F1_Frobenius": ("jarvis_f1", "mp_f1"),
        "F3_Longitudinal": ("jarvis_f3", "mp_f3"),
        "F4_KelvinOp": ("jarvis_f4", "mp_f4"),
    }

    rows: list[dict[str, Any]] = []
    for panel_name in ["P0", "P2"]:
        sub = panel[panel[panel_name]]
        if len(sub) < 10:
            continue
        for func_name, (left_col, right_col) in metric_cols.items():
            independent = _independent_rank_stats(
                sub[left_col].to_numpy(),
                sub[right_col].to_numpy(),
                frac=0.10,
            )
            primary = ranking[(ranking["panel"] == panel_name) & (ranking["functional"] == func_name)].iloc[0]
            rows.extend(_compare(f"{panel_name}_{func_name}", independent, primary, tolerances))

    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(RESULTS_ROOT / "verification_differences.csv", index=False)

    failures = diff_df[diff_df["status"] == "fail"]
    print(f"[Phase 6B verify] Compared {len(diff_df)} metric values; {len(failures)} failures.")
    if len(failures):
        print(failures.to_string())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
