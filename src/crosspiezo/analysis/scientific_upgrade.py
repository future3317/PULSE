"""Statistical primitives for the post-Version-A scientific upgrade.

The functions in this module deliberately do not mutate the frozen Phase 6A
panel or any source data.  They add three safeguards that were missing from
the original screening-resolution run:

* reduced-formula cluster bootstrap resampling with the null expectation
  recomputed for each bootstrap universe;
* exact top-k cutoff diagnostics, including score ties and overlap bounds;
* raw-value summaries that complement rank-only diagnostics.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from crosspiezo.analysis.phase7b_stats import naucc, persistent_onset
from crosspiezo.analysis.ranking import (
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
)


PARTIAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("elite", 1.0, 10.0),
    ("intermediate", 10.0, 20.0),
    ("broad", 20.0, 50.0),
)


def _finite_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(0, min(int(k), len(scores)))
    return np.argsort(-scores, kind="stable")[:k]


@lru_cache(maxsize=50_000)
def _expected_jaccard_cached(n: int, k: int) -> float:
    """Cache the exact null expectation across repeated bootstrap universes."""
    return expected_jaccard_hypergeometric(int(n), int(k))


def _score_cutoff(scores: np.ndarray, k: int) -> dict[str, Any]:
    """Return a deterministic cutoff and its exact tie diagnostics."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    values = scores[finite]
    n = len(values)
    if n == 0 or k <= 0:
        return {
            "n_valid": n,
            "k": int(k),
            "cutoff_score": float("nan"),
            "next_score": float("nan"),
            "absolute_gap": float("nan"),
            "relative_gap": float("nan"),
            "tie_block_size": 0,
            "mandatory_count": 0,
            "optional_count": 0,
            "tie_ambiguous": False,
        }
    kk = min(int(k), n)
    ordered = np.sort(values)[::-1]
    cutoff = float(ordered[kk - 1])
    next_score = float(ordered[kk]) if kk < n else float("nan")
    gap = cutoff - next_score if kk < n else float("nan")
    rel_gap = gap / max(abs(cutoff), np.finfo(float).eps) if np.isfinite(gap) else float("nan")
    tie_block = int(np.sum(values == cutoff))
    mandatory = int(np.sum(values > cutoff))
    optional = tie_block
    needed_from_tie = kk - mandatory
    return {
        "n_valid": n,
        "k": kk,
        "cutoff_score": cutoff,
        "next_score": next_score,
        "absolute_gap": float(gap),
        "relative_gap": float(rel_gap),
        "tie_block_size": tie_block,
        "mandatory_count": mandatory,
        "optional_count": optional,
        "optional_needed": int(needed_from_tie),
        "tie_ambiguous": bool(optional > needed_from_tie),
    }


def _top_family(scores: np.ndarray, k: int) -> tuple[set[int], set[int], int, bool]:
    """Return mandatory/optional top-k membership for a score vector."""
    values = np.asarray(scores, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if not len(valid_idx):
        return set(), set(), 0, False
    kk = min(max(int(k), 1), len(valid_idx))
    ordered = np.sort(values[valid_idx])[::-1]
    cutoff = ordered[kk - 1]
    mandatory = set(valid_idx[values[valid_idx] > cutoff].tolist())
    optional = set(valid_idx[values[valid_idx] == cutoff].tolist())
    needed = kk - len(mandatory)
    return mandatory, optional, needed, len(optional) > needed


def _enumerated_top_sets(
    mandatory: set[int], optional: set[int], needed: int, max_sets: int = 200_000
) -> list[set[int]] | None:
    if needed < 0 or needed > len(optional):
        return None
    n_sets = math.comb(len(optional), needed)
    if n_sets > max_sets:
        return None
    return [mandatory | set(choice) for choice in itertools.combinations(sorted(optional), needed)]


def overlap_bounds_with_ties(
    left: np.ndarray, right: np.ndarray, k: int
) -> dict[str, Any]:
    """Compute observed and tie-induced bounds for top-k overlap.

    If a cutoff tie is too large to enumerate, the returned bounds are
    conservative and ``bounds_exact`` is false.  The frozen data have small
    top-k tie blocks, so the exact path is normally used.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    kk = min(max(int(k), 1), len(left)) if len(left) else 0
    if kk == 0:
        return {"k": 0, "observed_overlap": 0, "min_overlap": 0, "max_overlap": 0, "bounds_exact": True}
    top_l = set(_top_indices(left, kk).tolist())
    top_r = set(_top_indices(right, kk).tolist())
    lm, lo, ln, la = _top_family(left, kk)
    rm, ro, rn, ra = _top_family(right, kk)
    left_sets = _enumerated_top_sets(lm, lo, ln)
    right_sets = _enumerated_top_sets(rm, ro, rn)
    if left_sets is not None and right_sets is not None:
        intersections = [len(a & b) for a in left_sets for b in right_sets]
        min_overlap, max_overlap = min(intersections), max(intersections)
        exact = True
    else:
        # A safe fallback for an unusually large cutoff tie.
        min_overlap, max_overlap = 0, kk
        exact = False
    return {
        "k": kk,
        "observed_overlap": int(len(top_l & top_r)),
        "min_overlap": int(min_overlap),
        "max_overlap": int(max_overlap),
        "left_tie_ambiguous": bool(la),
        "right_tie_ambiguous": bool(ra),
        "bounds_exact": exact,
    }


def cutoff_diagnostics(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return per-cutoff raw gaps, tie flags, and overlap bounds."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    rows: list[dict[str, Any]] = []
    for q in np.asarray(q_percentiles, dtype=float):
        k = max(1, int(math.floor(q / 100.0 * n)))
        dl = _score_cutoff(left, k)
        dr = _score_cutoff(right, k)
        bounds = overlap_bounds_with_ties(left, right, k)
        exp = _expected_jaccard_cached(n, k)
        denom = 1.0 - exp
        j_min = bounds["min_overlap"] / (2 * k - bounds["min_overlap"])
        j_max = bounds["max_overlap"] / (2 * k - bounds["max_overlap"])
        rows.append(
            {
                "q_percentile": float(q),
                "n": n,
                "k": k,
                "observed_overlap": bounds["observed_overlap"],
                "min_overlap": bounds["min_overlap"],
                "max_overlap": bounds["max_overlap"],
                "expected_jaccard": exp,
                "observed_jaccard": bounds["observed_overlap"] / (2 * k - bounds["observed_overlap"]),
                "min_jaccard": j_min,
                "max_jaccard": j_max,
                "observed_adjusted_jaccard": (bounds["observed_overlap"] / (2 * k - bounds["observed_overlap"]) - exp) / denom,
                "min_adjusted_jaccard": (j_min - exp) / denom,
                "max_adjusted_jaccard": (j_max - exp) / denom,
                "bounds_exact": bool(bounds["bounds_exact"]),
                "tie_ambiguous": bool(dl["tie_ambiguous"] or dr["tie_ambiguous"]),
                "left_cutoff_score": dl["cutoff_score"],
                "left_next_score": dl["next_score"],
                "left_absolute_gap": dl["absolute_gap"],
                "left_relative_gap": dl["relative_gap"],
                "left_tie_block_size": dl["tie_block_size"],
                "left_optional_needed": dl["optional_needed"],
                "right_cutoff_score": dr["cutoff_score"],
                "right_next_score": dr["next_score"],
                "right_absolute_gap": dr["absolute_gap"],
                "right_relative_gap": dr["relative_gap"],
                "right_tie_block_size": dr["tie_block_size"],
                "right_optional_needed": dr["optional_needed"],
                "hypergeometric_pvalue": hypergeometric_overlap_pvalue(
                    n, k, bounds["observed_overlap"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _curve_point(left: np.ndarray, right: np.ndarray, q_percentiles: np.ndarray) -> pd.DataFrame:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    # Rank once per universe; the bootstrap calls this function thousands of
    # times and q-specific re-sorting needlessly dominates runtime.
    order_l = np.argsort(-left, kind="stable")
    order_r = np.argsort(-right, kind="stable")
    pos_l = np.empty(n, dtype=np.int64)
    pos_r = np.empty(n, dtype=np.int64)
    pos_l[order_l] = np.arange(n, dtype=np.int64)
    pos_r[order_r] = np.arange(n, dtype=np.int64)
    ks = np.maximum(1, np.floor(np.asarray(q_percentiles, dtype=float) / 100.0 * n).astype(int))
    left_positions = pos_l[:, None] < ks[None, :]
    right_positions = pos_r[:, None] < ks[None, :]
    intersections = np.sum(left_positions & right_positions, axis=0).astype(int)
    rows: list[dict[str, Any]] = []
    for q, k, inter in zip(q_percentiles, ks, intersections):
        union = 2 * int(k) - int(inter)
        observed = int(inter) / union if union else 0.0
        expected = _expected_jaccard_cached(n, int(k))
        adjusted = (observed - expected) / (1.0 - expected)
        rows.append(
            {
                "q_percentile": float(q),
                "k": int(k),
                "n": n,
                "observed_overlap": int(inter),
                "observed_jaccard": observed,
                "expected_jaccard": expected,
                "chance_adjusted_jaccard": adjusted,
            }
        )
    return pd.DataFrame(rows)


def _simultaneous_band(
    bootstrap_curves: np.ndarray, observed: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(bootstrap_curves) < 2:
        nan = np.full_like(observed, np.nan)
        return nan, nan
    se = np.std(bootstrap_curves, axis=0, ddof=1)
    se = np.where(se > 1e-12, se, 1e-12)
    sup = np.max(np.abs(bootstrap_curves - observed[None, :]) / se[None, :], axis=1)
    crit = float(np.quantile(sup, 1.0 - alpha))
    return np.clip(observed - crit * se, -1.0, 1.0), np.clip(observed + crit * se, -1.0, 1.0)


def _area_average(
    q_percentiles: np.ndarray,
    values: np.ndarray,
    q_min: float,
    q_max: float,
) -> float:
    """Return the trapezoidal area average on a closed quantile interval.

    Endpoint values are used when a diagnostic grid extends beyond the
    requested interval; disjoint intervals still return NaN.
    """
    q = np.asarray(q_percentiles, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(q) & np.isfinite(y)
    q, y = q[valid], y[valid]
    if len(q) < 2 or q_max <= q_min:
        return float("nan")
    order = np.argsort(q, kind="stable")
    q, y = q[order], y[order]
    if q_max < q[0] or q_min > q[-1]:
        return float("nan")
    interior = (q > q_min) & (q < q_max)
    q_eval = np.concatenate(([q_min], q[interior], [q_max]))
    y_eval = np.interp(q_eval, q, y)
    return float(np.trapezoid(y_eval, q_eval) / (q_max - q_min))


def _bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([idx_by_group[g] for g in sampled])


def cluster_bootstrap_screening(
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    min_consecutive: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Unified reduced-formula cluster bootstrap for tau, nAUCC and the curve."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    valid = _finite_mask(left, right) & pd.notna(groups)
    left, right, groups = left[valid], right[valid], groups[valid]
    qs = np.asarray(q_percentiles, dtype=float)
    point_curve = _curve_point(left, right, qs)
    observed_curve = point_curve["chance_adjusted_jaccard"].to_numpy(float)
    point_naucc = naucc(point_curve)
    tau_point, _ = stats.kendalltau(left, right)
    tau_point = float(tau_point) if tau_point is not None else float("nan")

    rng = np.random.default_rng(seed)
    boot_curves: list[np.ndarray] = []
    boot_naucc: list[float] = []
    boot_tau: list[float] = []
    for _ in range(int(n_boot)):
        idx = _bootstrap_indices(groups, rng)
        curve = _curve_point(left[idx], right[idx], qs)
        values = curve["chance_adjusted_jaccard"].to_numpy(float)
        if np.isfinite(values).all():
            boot_curves.append(values)
            boot_naucc.append(naucc(curve))
        tau, _ = stats.kendalltau(left[idx], right[idx])
        if tau is not None and np.isfinite(tau):
            boot_tau.append(float(tau))

    curve_array = np.asarray(boot_curves, dtype=float)
    low, high = _simultaneous_band(curve_array, observed_curve, alpha)
    point_curve["cluster_boot_ci95_low"] = low
    point_curve["cluster_boot_ci95_high"] = high
    onset = {}
    for delta in (0.0, 0.05, 0.10):
        curve_for_onset = point_curve.rename(
            columns={"cluster_boot_ci95_low": "adj_jaccard_ci95_low"}
        )
        onset[f"persistent_onset_delta{delta:.2f}"] = persistent_onset(
            curve_for_onset, delta=delta, min_consecutive=min_consecutive
        )
    summary = {
        "n": int(len(left)),
        "n_groups": int(pd.Series(groups).nunique()),
        "tau": tau_point,
        "tau_ci95_low": float(np.percentile(boot_tau, 2.5)) if boot_tau else float("nan"),
        "tau_ci95_high": float(np.percentile(boot_tau, 97.5)) if boot_tau else float("nan"),
        "nAUCC": float(point_naucc),
        "nAUCC_ci95_low": float(np.percentile(boot_naucc, 2.5)) if boot_naucc else float("nan"),
        "nAUCC_ci95_high": float(np.percentile(boot_naucc, 97.5)) if boot_naucc else float("nan"),
        "n_boot_requested": int(n_boot),
        "n_boot_curve": int(len(boot_curves)),
        "n_boot_tau": int(len(boot_tau)),
        "bootstrap_seed": int(seed),
        **onset,
    }
    for name, q_min, q_max in PARTIAL_BANDS:
        point_partial = _area_average(qs, observed_curve, q_min, q_max)
        boot_partial = [
            _area_average(qs, values, q_min, q_max)
            for values in curve_array
            if np.isfinite(values).all()
        ]
        summary[f"partial_nAUCC_{name}"] = point_partial
        summary[f"partial_nAUCC_{name}_ci95_low"] = (
            float(np.percentile(boot_partial, 2.5)) if boot_partial else float("nan")
        )
        summary[f"partial_nAUCC_{name}_ci95_high"] = (
            float(np.percentile(boot_partial, 97.5)) if boot_partial else float("nan")
        )
        summary[f"partial_nAUCC_{name}_n_boot"] = int(len(boot_partial))
    return point_curve, summary


__all__ = [
    "PARTIAL_BANDS",
    "cluster_bootstrap_screening",
    "cutoff_diagnostics",
    "overlap_bounds_with_ties",
]
