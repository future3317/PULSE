"""Ranking stability metrics with preregistered tensor functionals.

All functionals operate on the same paired universe and use the same
contribution type and units, so that rank instability is not an artifact of
different candidate sets or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RankingResult:
    """Ranking stability for a single scalar functional."""

    functional_name: str
    n_pairs: int
    top_20_jaccard: float
    top_50_jaccard: float
    top_100_jaccard: float
    kendall_tau: float
    kendall_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau_ci_low: float
    kendall_tau_ci_high: float
    mean_absolute_rank_shift: float
    median_absolute_rank_shift: float


def frobenius_norm_score(tensor: np.ndarray) -> float:
    """Symmetry-adapted Frobenius norm of a Cartesian tensor."""
    return float(np.linalg.norm(np.asarray(tensor, dtype=np.float64)))


def max_longitudinal_response(tensor: np.ndarray, n_samples: int = 20000, seed: int = 42) -> float:
    """Maximum directional piezoelectric response, max_{||n||=1} |n_i e_ijk n_j n_k|.

    This is a rotation-invariant scalar functional.  It is evaluated by uniform
    sphere sampling followed by a small local polish using the largest sample as
    a starting point.
    """
    t = np.asarray(tensor, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_samples, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    vals = np.abs(np.einsum("ni,ijk,nj,nk->n", dirs, t, dirs, dirs))
    best_idx = int(np.argmax(vals))
    best_dir = dirs[best_idx]
    best_val = float(vals[best_idx])

    # Lightweight local refinement: tiny gradient-free coordinate ascent.
    step = 0.05
    for _ in range(50):
        grad = np.zeros(3)
        for eps_axis in range(3):
            perturb = best_dir.copy()
            perturb[eps_axis] += 1e-5
            perturb /= np.linalg.norm(perturb)
            v = abs(float(np.einsum("i,ijk,j,k->", perturb, t, perturb, perturb)))
            grad[eps_axis] = (v - best_val) / 1e-5
        trial = best_dir + step * grad
        trial /= np.linalg.norm(trial)
        v = abs(float(np.einsum("i,ijk,j,k->", trial, t, trial, trial)))
        if v > best_val:
            best_dir = trial
            best_val = v
        else:
            step *= 0.5
    return best_val


def max_shear_response(tensor: np.ndarray) -> float:
    """Shear response functional is not uniquely defined for rank-3 tensors.

    The old implementation used arbitrary coordinate-axis components and was not
    rotation invariant.  It is withdrawn pending a physically motivated definition.
    """
    raise NotImplementedError(
        "max_shear_response has been removed because no rotation-invariant "
        "physical definition was preregistered."
    )


def derived_d_score(
    piezo: np.ndarray,
    elastic: np.ndarray | None,
) -> float:
    """||d||_F where d = e C^{-1}, if an elastic tensor is supplied."""
    if elastic is None:
        return float("nan")
    e = np.asarray(piezo, dtype=np.float64)
    c = np.asarray(elastic, dtype=np.float64)
    # Convert Voigt 6x6 to matrix if needed; assume 6x6 in C11,C12,... order.
    if c.shape == (6, 6):
        c_mat = c
    else:
        return float("nan")
    try:
        c_inv = np.linalg.inv(c_mat)
        # e is 3x3x3; contract last two indices with C^{-1} in Voigt form.
        from crosspiezo.conventions.voigt import cartesian_to_voigt, voigt_to_cartesian

        e_voigt = cartesian_to_voigt(e, engineering_shear=True)
        d_voigt = e_voigt @ c_inv.T
        d = voigt_to_cartesian(d_voigt, engineering_shear=True)
        return float(np.linalg.norm(d))
    except Exception:  # noqa: BLE001
        return float("nan")


def rank_stability_functional(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    functional_name: str,
    k_values: list[int] | None = None,
) -> RankingResult:
    """Compute rank-stability metrics for a single functional on paired records.

    Scores must already be aligned: left_scores[i] and right_scores[i] refer to
    the same matched material from the two sources.
    """
    if k_values is None:
        k_values = [20, 50, 100]
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    n = len(left)

    jaccards: dict[str, float] = {}
    for k in k_values:
        kk = min(k, n)
        top_left = set(np.argsort(-left, kind="stable")[:kk])
        top_right = set(np.argsort(-right, kind="stable")[:kk])
        inter = len(top_left & top_right)
        union = len(top_left | top_right)
        jaccards[f"top_{k}_jaccard"] = inter / union if union else 0.0

    if n > 1:
        tau, tau_p = stats.kendalltau(left, right)
        rho, rho_p = stats.spearmanr(left, right)
        tau = float(tau) if tau is not None else float("nan")
        tau_p = float(tau_p) if tau_p is not None else float("nan")
        rho = float(rho) if rho is not None else float("nan")
        rho_p = float(rho_p) if rho_p is not None else float("nan")
    else:
        tau = tau_p = rho = rho_p = float("nan")

    # Bootstrap CI for Kendall tau.
    tau_lo, tau_hi = _kendall_tau_bootstrap(left, right)

    # Rank shifts.
    ranks_left = stats.rankdata(-left, method="average")
    ranks_right = stats.rankdata(-right, method="average")
    abs_shift = np.abs(ranks_left - ranks_right)
    mean_shift = float(np.mean(abs_shift)) if n else float("nan")
    median_shift = float(np.median(abs_shift)) if n else float("nan")

    return RankingResult(
        functional_name=functional_name,
        n_pairs=n,
        top_20_jaccard=jaccards.get("top_20_jaccard", float("nan")),
        top_50_jaccard=jaccards.get("top_50_jaccard", float("nan")),
        top_100_jaccard=jaccards.get("top_100_jaccard", float("nan")),
        kendall_tau=tau,
        kendall_pvalue=tau_p,
        spearman_rho=rho,
        spearman_pvalue=rho_p,
        kendall_tau_ci_low=tau_lo,
        kendall_tau_ci_high=tau_hi,
        mean_absolute_rank_shift=mean_shift,
        median_absolute_rank_shift=median_shift,
    )


def _kendall_tau_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    n_replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Kendall tau."""
    n = len(left)
    if n < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        tau, _ = stats.kendalltau(left[idx], right[idx])
        reps[i] = float(tau) if tau is not None else float("nan")
    valid = np.isfinite(reps)
    if np.sum(valid) < n_replicates // 2:
        return float("nan"), float("nan")
    reps = reps[valid]
    alpha = 1.0 - confidence
    lo = float(np.percentile(reps, 100 * alpha / 2))
    hi = float(np.percentile(reps, 100 * (1.0 - alpha / 2)))
    return lo, hi


def ranking_summary_table(results: list[RankingResult]) -> list[dict[str, float | str | int]]:
    """Convert a list of RankingResult objects to plain records."""
    rows: list[dict[str, float | str | int]] = []
    for r in results:
        rows.append({
            "functional": r.functional_name,
            "n_pairs": r.n_pairs,
            "top_20_jaccard": r.top_20_jaccard,
            "top_50_jaccard": r.top_50_jaccard,
            "top_100_jaccard": r.top_100_jaccard,
            "kendall_tau": r.kendall_tau,
            "kendall_tau_ci95_low": r.kendall_tau_ci_low,
            "kendall_tau_ci95_high": r.kendall_tau_ci_high,
            "spearman_rho": r.spearman_rho,
            "mean_abs_rank_shift": r.mean_absolute_rank_shift,
            "median_abs_rank_shift": r.median_absolute_rank_shift,
        })
    return rows
