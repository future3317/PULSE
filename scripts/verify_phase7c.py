#!/usr/bin/env python
"""Independent verification of Phase 7C Work Packages A and F.

This script re-implements the core calculations using separate code paths and
compares the resulting numbers against the CSVs produced by ``run_phase7c.py``.
It does not import ``crosspiezo.analysis.phase7c_stats``.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULT_ROOT = PROJECT_ROOT / "results" / "phase7c"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.analysis.ranking import (  # noqa: E402
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
)


def _load_config() -> dict[str, Any]:
    with open(CONFIG_ROOT / "phase7c.yaml") as f:
        return yaml.safe_load(f)


def _simultaneous_band(
    boot_curves: np.ndarray, observed: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    se = np.std(boot_curves, axis=0, ddof=1)
    se = np.where(se > 1e-12, se, 1e-12)
    studentized = np.max(np.abs(boot_curves - observed[None, :]) / se[None, :], axis=1)
    crit = float(np.quantile(studentized, 1.0 - alpha))
    low = np.clip(observed - crit * se, -1.0, 1.0)
    high = np.clip(observed + crit * se, -1.0, 1.0)
    return low, high


def independent_screening_curve(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    rng = np.random.default_rng(seed)
    qs = np.asarray(q_percentiles, dtype=np.float64)
    ks = np.array([max(1, int(math.floor(q / 100.0 * n))) for q in qs], dtype=np.int64)

    # Observed curve: use rank positions for an O(n + len(qs)) computation.
    pos_left = np.empty(n, dtype=np.int64)
    pos_left[np.argsort(-left, kind="stable")] = np.arange(n, dtype=np.int64)
    pos_right = np.empty(n, dtype=np.int64)
    pos_right[np.argsort(-right, kind="stable")] = np.arange(n, dtype=np.int64)
    inter_obs = np.array([int(((pos_left < k) & (pos_right < k)).sum()) for k in ks])
    union_obs = 2 * ks - inter_obs
    obs_jacc = np.where(union_obs > 0, inter_obs / union_obs, 0.0)
    exp_obs = np.array([expected_jaccard_hypergeometric(n, k) for k in ks])
    observed_curve = np.array(
        [
            (oj - ej) / (1.0 - ej) if np.isfinite(ej) and ej < 1.0 else float("nan")
            for oj, ej in zip(obs_jacc, exp_obs, strict=False)
        ],
        dtype=np.float64,
    )

    records: list[dict[str, Any]] = []
    for i, q in enumerate(qs):
        records.append(
            {
                "q_percentile": q,
                "observed_overlap": int(inter_obs[i]),
                "chance_adjusted_jaccard": float(observed_curve[i]),
                "hypergeometric_pvalue": hypergeometric_overlap_pvalue(n, ks[i], int(inter_obs[i])),
            }
        )

    # Bootstrap curves: use descending positions, matching the main pipeline.
    positions = np.arange(n, dtype=np.int64)
    boot_array = np.empty((n_boot, len(qs)), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        pos_left = np.empty(n, dtype=np.int64)
        pos_left[np.argsort(-left[idx], kind="stable")] = positions
        pos_right = np.empty(n, dtype=np.int64)
        pos_right[np.argsort(-right[idx], kind="stable")] = positions
        inter = np.array([int(((pos_left < k) & (pos_right < k)).sum()) for k in ks])
        union = 2 * ks - inter
        obs_jacc = np.where(union > 0, inter / union, 0.0)
        boot_array[b] = (obs_jacc - exp_obs) / (1.0 - exp_obs)

    valid_boot = np.isfinite(boot_array).all(axis=1)
    if valid_boot.sum() >= max(1, n_boot // 2):
        boot_array = boot_array[valid_boot]
        low, high = _simultaneous_band(boot_array, observed_curve)
    else:
        low = high = np.full_like(observed_curve, np.nan)

    for i, rec in enumerate(records):
        rec["adj_jaccard_ci95_low"] = float(low[i])
        rec["adj_jaccard_ci95_high"] = float(high[i])
    return pd.DataFrame(records)


def independent_naucc(curve: pd.DataFrame) -> float:
    x = curve["q_percentile"].to_numpy(dtype=np.float64)
    y = curve["chance_adjusted_jaccard"].to_numpy(dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def independent_partial_naucc(curve: pd.DataFrame, q_min: float, q_max: float) -> float:
    x = curve["q_percentile"].to_numpy(dtype=np.float64)
    y = curve["chance_adjusted_jaccard"].to_numpy(dtype=np.float64)
    mask = (x >= q_min) & (x <= q_max)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    denom = q_max - q_min
    return float(np.trapezoid(y, x) / denom) if denom > 0 else float("nan")


def independent_persistent_onset(
    curve: pd.DataFrame, delta: float, min_consecutive: int = 5
) -> float | None:
    qs = curve["q_percentile"].to_numpy(dtype=np.float64)
    low = curve["adj_jaccard_ci95_low"].to_numpy(dtype=np.float64)
    above = low > delta
    for i in range(len(qs) - min_consecutive + 1):
        if above[i : i + min_consecutive].all():
            return float(qs[i])
    return None


def _source_percentiles(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    ranks = stats.rankdata(scores, method="average")
    return (ranks - 1.0) / (n - 1.0)


def independent_portfolio_select(
    strategy: str,
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
) -> list[int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0:
        return []
    target_k = max(1, int(math.floor(q_star * n)))
    budget = min(n, int(round(budget_factor * target_k)))
    pl = _source_percentiles(left)
    pr = _source_percentiles(right)
    avg = 0.5 * (pl + pr)

    if strategy == "jarvis_only":
        selected = set(np.argsort(-left, kind="stable")[:budget].tolist())
    elif strategy == "mp_only":
        selected = set(np.argsort(-right, kind="stable")[:budget].tolist())
    elif strategy == "average_percentile":
        selected = set(np.argsort(-avg, kind="stable")[:budget].tolist())
    elif strategy == "borda_count":
        rl = stats.rankdata(-left, method="average")
        rr = stats.rankdata(-right, method="average")
        selected = set(np.argsort(rl + rr, kind="stable")[:budget].tolist())
    elif strategy == "maximin_percentile":
        selected = set(np.argsort(-np.minimum(pl, pr), kind="stable")[:budget].tolist())
    else:
        selected = set()

    selected_list = sorted(selected, key=lambda i: (-avg[i], i))
    return selected_list[:budget]


def independent_portfolio_metrics(
    selected: list[int], left: np.ndarray, right: np.ndarray, q_star: float
) -> dict[str, float]:
    n = len(left)
    k_q = max(1, int(math.floor(q_star * n)))
    top_j = set(np.argsort(-left, kind="stable")[:k_q].tolist())
    top_m = set(np.argsort(-right, kind="stable")[:k_q].tolist())
    sel = set(selected)
    recall_j = len(sel & top_j) / k_q
    recall_m = len(sel & top_m) / k_q
    worst_recall = min(recall_j, recall_m)

    pl = _source_percentiles(left)
    pr = _source_percentiles(right)
    avg = 0.5 * (pl + pr)
    order = sorted(selected, key=lambda i: (-avg[i], i))
    m = len(order)

    def ndcg(rel: np.ndarray, universe: np.ndarray) -> float:
        rel = rel[:m]
        positions = np.arange(2, m + 2, dtype=np.float64)
        dcg = np.sum((2.0**rel - 1.0) / np.log2(positions))
        ideal_idx = np.argsort(-universe, kind="stable")[:m]
        ideal = universe[ideal_idx]
        idcg = np.sum((2.0**ideal - 1.0) / np.log2(positions))
        return float(dcg / idcg) if idcg > 0 else 0.0

    ndcg_j = ndcg(pl[order], pl)
    ndcg_m = ndcg(pr[order], pr)
    worst_ndcg = min(ndcg_j, ndcg_m)
    return {
        "worst_source_recall": float(worst_recall),
        "worst_source_ndcg": float(worst_ndcg),
        "portfolio_size": len(selected),
        "portfolio_coverage": len(selected) / n,
    }


def verify_wp_a(cfg: dict[str, Any], panel_df: pd.DataFrame) -> dict[str, Any]:
    print("[Verify] WP A: independent screening-resolution curves and bands...")
    sr = cfg["screening_resolution"]
    q = np.arange(sr["q_percentiles"]["start"], sr["q_percentiles"]["stop"] + 1, sr["q_percentiles"]["step"])
    seed = sr["random_seed"]
    n_boot = sr["bootstrap_replicates"]
    deltas = sr["delta_thresholds"]
    min_consecutive = sr["persistent_onset_min_consecutive"]
    bands = sr["bands"]
    metrics = cfg["metrics"]["primary"]

    main_summary = pd.read_csv(RESULT_ROOT / "concordance_summary.csv")
    main_bands = pd.read_csv(RESULT_ROOT / "concordance_bands.csv")
    flagged: list[dict[str, Any]] = []
    max_diff = 0.0

    panels = [cfg["panels"]["primary"], cfg["panels"]["sensitivity"]]
    for base in panels:
        sub = panel_df[panel_df[base]]
        for metric in metrics:
            left = sub[metric["jarvis_col"]].to_numpy(dtype=float)
            right = sub[metric["mp_col"]].to_numpy(dtype=float)
            curve = independent_screening_curve(left, right, q, n_boot=n_boot, seed=seed)
            naucc_val = independent_naucc(curve)

            main_row = main_summary[(main_summary["panel"] == base) & (main_summary["metric"] == metric["name"])]
            if main_row.empty:
                flagged.append({"panel": base, "metric": metric["name"], "field": "nAUCC", "reason": "missing main row"})
                continue
            main_naucc = float(main_row.iloc[0]["nAUCC"])
            diff = abs(naucc_val - main_naucc)
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                flagged.append(
                    {
                        "panel": base,
                        "metric": metric["name"],
                        "field": "nAUCC",
                        "independent": naucc_val,
                        "main": main_naucc,
                        "diff": diff,
                    }
                )

            for delta in deltas:
                onset = independent_persistent_onset(curve, delta, min_consecutive)
                main_onset = main_row.iloc[0].get(f"persistent_onset_delta{delta:.2f}")
                if (onset is None) != (pd.isna(main_onset) if main_onset is not None else True):
                    flagged.append(
                        {
                            "panel": base,
                            "metric": metric["name"],
                            "field": f"persistent_onset_delta{delta:.2f}",
                            "independent": onset,
                            "main": main_onset,
                            "reason": "None mismatch",
                        }
                    )
                elif onset is not None and abs(onset - float(main_onset)) > 1e-6:
                    diff_onset = abs(onset - float(main_onset))
                    max_diff = max(max_diff, diff_onset)
                    flagged.append(
                        {
                            "panel": base,
                            "metric": metric["name"],
                            "field": f"persistent_onset_delta{delta:.2f}",
                            "independent": onset,
                            "main": float(main_onset),
                            "diff": diff_onset,
                        }
                    )

            for band in bands:
                val = independent_partial_naucc(curve, band["q_min"], band["q_max"])
                main_band_row = main_bands[
                    (main_bands["panel"] == base)
                    & (main_bands["metric"] == metric["name"])
                    & (main_bands["band"] == band["name"])
                ]
                if main_band_row.empty:
                    flagged.append(
                        {
                            "panel": base,
                            "metric": metric["name"],
                            "band": band["name"],
                            "field": "partial_nAUCC",
                            "reason": "missing main band row",
                        }
                    )
                    continue
                main_val = float(main_band_row.iloc[0]["partial_nAUCC"])
                d = abs(val - main_val)
                max_diff = max(max_diff, d)
                if d > 1e-6:
                    flagged.append(
                        {
                            "panel": base,
                            "metric": metric["name"],
                            "band": band["name"],
                            "field": "partial_nAUCC",
                            "independent": val,
                            "main": main_val,
                            "diff": d,
                        }
                    )

    return {"max_diff": max_diff, "flagged": flagged}


def verify_wp_f(cfg: dict[str, Any], panel_df: pd.DataFrame) -> dict[str, Any]:
    print("[Verify] WP F: independent equal-budget portfolio metrics...")
    rp = cfg["robust_portfolio"]
    q_star = rp["q_star"]
    primary_budget = rp["primary_budget_factor"]
    strategies = ["jarvis_only", "mp_only", "average_percentile", "borda_count", "maximin_percentile"]
    metrics = cfg["metrics"]["primary"]

    main_benchmark = pd.read_csv(RESULT_ROOT / "portfolio_benchmark.csv")
    flagged: list[dict[str, Any]] = []
    max_diff = 0.0

    panels = [cfg["panels"]["primary"], cfg["panels"]["sensitivity"]]
    for base in panels:
        sub = panel_df[panel_df[base]].copy()
        for metric in metrics:
            left = sub[metric["jarvis_col"]].to_numpy(dtype=float)
            right = sub[metric["mp_col"]].to_numpy(dtype=float)
            valid = np.isfinite(left) & np.isfinite(right)
            left, right = left[valid], right[valid]
            if len(left) < 20:
                continue
            for strategy in strategies:
                selected = independent_portfolio_select(
                    strategy, left, right, q_star, primary_budget
                )
                metrics_dict = independent_portfolio_metrics(selected, left, right, q_star)
                main_row = main_benchmark[
                    (main_benchmark["panel"] == base)
                    & (main_benchmark["metric"] == metric["name"])
                    & (main_benchmark["strategy"] == strategy)
                    & (main_benchmark["budget_factor"] == primary_budget)
                    & (main_benchmark["eval_mode"] == "full_panel")
                ]
                if main_row.empty:
                    flagged.append(
                        {
                            "panel": base,
                            "metric": metric["name"],
                            "strategy": strategy,
                            "reason": "missing main row",
                        }
                    )
                    continue
                for field in ("worst_source_recall", "worst_source_ndcg", "portfolio_coverage"):
                    ind = metrics_dict[field]
                    main_val = float(main_row.iloc[0][field])
                    diff = abs(ind - main_val)
                    max_diff = max(max_diff, diff)
                    if diff > 1e-6:
                        flagged.append(
                            {
                                "panel": base,
                                "metric": metric["name"],
                                "strategy": strategy,
                                "field": field,
                                "independent": ind,
                                "main": main_val,
                                "diff": diff,
                            }
                        )
    return {"max_diff": max_diff, "flagged": flagged}


def main() -> int:
    cfg = _load_config()
    panel_path = PROJECT_ROOT / cfg["panels"]["membership_path"]
    try:
        panel_df = pd.read_parquet(panel_path)
    except OSError as exc:
        if "Repetition level histogram size mismatch" in str(exc):
            raise RuntimeError(
                "The frozen panel Parquet requires pyarrow>=23.0; the current "
                "PyArrow runtime cannot decode its data pages. "
                f"Use a compatible environment to read {panel_path}."
            ) from exc
        raise

    result_a = verify_wp_a(cfg, panel_df)
    result_f = verify_wp_f(cfg, panel_df)

    total_max_diff = max(result_a["max_diff"], result_f["max_diff"])
    total_flags = result_a["flagged"] + result_f["flagged"]
    status = "reconciled" if not total_flags else "flagged"

    summary = {
        "status": status,
        "max_absolute_difference": total_max_diff,
        "wp_a": result_a,
        "wp_f": result_f,
        "n_flags": len(total_flags),
        "flagged_items": total_flags,
    }

    path = RESULT_ROOT / "verification_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[Verify] Reconciliation status: {status}")
    print(f"[Verify] Max absolute difference: {total_max_diff:.6e}")
    print(f"[Verify] Flags: {len(total_flags)}")
    print(f"[Verify] Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0 if status == "reconciled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
