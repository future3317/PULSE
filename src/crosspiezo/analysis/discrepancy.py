"""Cross-protocol tensor discrepancy metrics for CrossPiezo Phase 3."""

from __future__ import annotations

import numpy as np
from scipy import stats


def frobenius_norm(tensor: np.ndarray) -> float:
    """Frobenius norm of a Cartesian tensor."""
    return float(np.linalg.norm(tensor))


def normalized_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """Normalized Frobenius discrepancy used in the manuscript."""
    diff = left - right
    denom = 0.5 * (frobenius_norm(left) + frobenius_norm(right)) + epsilon
    return frobenius_norm(diff) / denom


def absolute_discrepancy(left: np.ndarray, right: np.ndarray) -> float:
    return frobenius_norm(left - right)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine of the flattened tensors."""
    left_flat = left.ravel()
    right_flat = right.ravel()
    denom = np.linalg.norm(left_flat) * np.linalg.norm(right_flat) + 1e-12
    return float(np.dot(left_flat, right_flat) / denom)


def amplitude_ratio(left: np.ndarray, right: np.ndarray) -> float:
    """Ratio of Frobenius norms (right / left)."""
    denom = frobenius_norm(left) + 1e-12
    return frobenius_norm(right) / denom


def sign_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of components with opposite sign, ignoring near-zero entries."""
    mask = (np.abs(left) > 1e-6) | (np.abs(right) > 1e-6)
    if not np.any(mask):
        return 0.0
    return float(np.mean((left[mask] * right[mask]) < 0))


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "median",
    n_replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    point = float(np.median(values)) if statistic == "median" else float(np.mean(values))
    reps = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        sample = rng.choice(values, size=n, replace=True)
        reps[i] = float(np.median(sample)) if statistic == "median" else float(np.mean(sample))
    alpha = 1.0 - confidence
    lower = float(np.percentile(reps, 100 * alpha / 2))
    upper = float(np.percentile(reps, 100 * (1.0 - alpha / 2)))
    return point, lower, upper


def discrepancy_summary(values: np.ndarray, name: str = "metric") -> dict[str, float]:
    """Non-parametric summary with bootstrap CI."""
    vals = np.asarray(values, dtype=np.float64)
    point, lo, hi = bootstrap_ci(vals, statistic="median")
    return {
        f"{name}_n": int(len(vals)),
        f"{name}_median": point,
        f"{name}_median_ci95_low": lo,
        f"{name}_median_ci95_high": hi,
        f"{name}_mean": float(np.mean(vals)),
        f"{name}_std": float(np.std(vals)),
        f"{name}_iqr_low": float(np.percentile(vals, 25)),
        f"{name}_iqr_high": float(np.percentile(vals, 75)),
        f"{name}_p05": float(np.percentile(vals, 5)),
        f"{name}_p95": float(np.percentile(vals, 95)),
    }


def rank_stability(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute top-k Jaccard and global rank correlations."""
    if k_values is None:
        k_values = [20, 50, 100]
    left_scores = np.asarray(left_scores)
    right_scores = np.asarray(right_scores)
    n = len(left_scores)
    results: dict[str, float] = {}
    for k in k_values:
        kk = min(k, n)
        top_left = set(np.argsort(-left_scores)[:kk])
        top_right = set(np.argsort(-right_scores)[:kk])
        inter = len(top_left & top_right)
        union = len(top_left | top_right)
        results[f"top_{k}_jaccard"] = inter / union if union else 0.0
    if n > 1:
        results["kendall_tau"], results["kendall_pvalue"] = stats.kendalltau(
            left_scores, right_scores
        )
        results["spearman_rho"], results["spearman_pvalue"] = stats.spearmanr(
            left_scores, right_scores
        )
    return results
