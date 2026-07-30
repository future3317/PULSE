"""Ranking stability metrics with preregistered tensor functionals.

All functionals operate on the same paired universe and use the same
contribution type and units, so that rank instability is not an artifact of
different candidate sets or metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


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
    top_10_jaccard: float = float("nan")
    top_30_jaccard: float = float("nan")


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


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic quasi-uniform sampling of the unit sphere."""
    golden = np.pi * (3.0 - np.sqrt(5.0))
    ys = np.linspace(1.0, -1.0, n)
    radius = np.sqrt(np.maximum(0.0, 1.0 - ys * ys))
    theta = golden * np.arange(n)
    return np.column_stack((radius * np.cos(theta), ys, radius * np.sin(theta)))


def max_longitudinal_modulus(
    tensor: np.ndarray,
    *,
    grid_N: int = 10000,
    n_starts: int = 20,
    max_iter: int = 100,
    tol: float = 1e-9,
    cross_check: bool = True,
    cross_check_tol: float = 1e-4,
) -> float:
    """Deterministic maximum longitudinal piezoelectric modulus.

    Computes ``max_{||n||=1} | n_i e_ijk n_j n_k |`` with:

    1. a deterministic dense Fibonacci sphere grid;
    2. multi-start projected gradient ascent on the unit sphere;
    3. an optional SLSQP constrained-polish cross-check of the best direction;
    4. internal symmetrisation of the last two tensor indices.

    Parameters
    ----------
    grid_N:
        Number of deterministic sphere samples.
    n_starts:
        Number of grid points used as optimisation starting points.
    max_iter:
        Maximum projected-gradient iterations per start.
    tol:
        Convergence tolerance for the projected-gradient improvement.
    cross_check:
        If True, refine the best projected-gradient direction with SLSQP
        and warn if the two methods disagree by more than ``cross_check_tol``.
    cross_check_tol:
        Relative tolerance for agreement between the projected-gradient and
        SLSQP solutions.
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.shape != (3, 3, 3):
        raise ValueError(f"Expected Cartesian tensor shape (3, 3, 3), got {t.shape}")
    # Enforce the minor symmetry required by the Voigt/Cartesian convention.
    t = 0.5 * (t + t.transpose(0, 2, 1))

    dirs = _fibonacci_sphere(grid_N)
    vals = np.abs(np.einsum("ni,ijk,nj,nk->n", dirs, t, dirs, dirs))
    top_idx = np.argsort(-vals, kind="stable")[:n_starts]

    def _f_and_grad(n: np.ndarray) -> tuple[float, np.ndarray]:
        f = float(np.einsum("i,ijk,j,k->", n, t, n, n))
        # Gradient of f with respect to n, respecting minor symmetry in j,k.
        g1 = np.einsum("ljk,j,k->l", t, n, n)
        g2 = np.einsum("i,k,ikl->l", n, n, t)
        return f, g1 + 2.0 * g2

    best_val = 0.0
    best_dir: np.ndarray | None = None

    for start in dirs[top_idx]:
        n = start.copy()
        alpha = 0.3
        prev_f2 = float("inf")
        for _ in range(max_iter):
            f, grad_f = _f_and_grad(n)
            # Gradient of f^2.
            grad = 2.0 * f * grad_f
            grad_tan = grad - np.dot(grad, n) * n
            gt_norm = np.linalg.norm(grad_tan)
            if gt_norm < 1e-12:
                break

            cur_f2 = f * f
            step = alpha
            improved = False
            for _ in range(12):
                n_try = n + step * grad_tan
                n_try /= np.linalg.norm(n_try)
                f_try, _ = _f_and_grad(n_try)
                if f_try * f_try > cur_f2 + 1e-14:
                    n = n_try
                    f = f_try
                    improved = True
                    break
                step *= 0.5

            abs_f = abs(f)
            if abs_f > best_val:
                best_val = abs_f
                best_dir = n.copy()

            if not improved:
                alpha *= 0.7
                if alpha < 1e-4:
                    break
                continue

            if abs(prev_f2 - f * f) < tol and f * f <= prev_f2:
                break
            prev_f2 = f * f

    if cross_check and best_dir is not None:
        def neg_f2(x: np.ndarray) -> float:
            f = float(np.einsum("i,ijk,j,k->", x, t, x, x))
            return -(f * f)

        def neg_f2_jac(x: np.ndarray) -> np.ndarray:
            f, g = _f_and_grad(x)
            return -2.0 * f * g

        constraint = {
            "type": "eq",
            "fun": lambda x: float(x @ x) - 1.0,
            "jac": lambda x: 2.0 * x,
        }
        result = optimize.minimize(
            neg_f2,
            best_dir,
            jac=neg_f2_jac,
            method="SLSQP",
            constraints=constraint,
            options={"ftol": 1e-12, "maxiter": 200, "disp": False},
        )
        if result.success:
            f_ref = float(np.einsum("i,ijk,j,k->", result.x, t, result.x, result.x))
            ref_val = abs(f_ref)
            if ref_val > best_val:
                best_val = ref_val
            denom = max(best_val, 1.0)
            if abs(ref_val - best_val) / denom > cross_check_tol:
                warnings.warn(
                    f"max_longitudinal_modulus cross-check mismatch: "
                    f"projected-gradient={best_val:.6e}, SLSQP={ref_val:.6e}",
                    stacklevel=2,
                )

    return float(best_val)


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
        top_10_jaccard=jaccards.get("top_10_jaccard", float("nan")),
        top_30_jaccard=jaccards.get("top_30_jaccard", float("nan")),
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
            "top_10_jaccard": r.top_10_jaccard,
            "top_20_jaccard": r.top_20_jaccard,
            "top_30_jaccard": r.top_30_jaccard,
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
