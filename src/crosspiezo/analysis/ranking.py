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
    """F1: full Cartesian Frobenius norm ||e||_F.

    This is a coordinate-invariant scalar for any rank-3 Cartesian tensor.
    """
    return float(np.linalg.norm(np.asarray(tensor, dtype=np.float64)))


def mp_reported_svd_scalar(voigt: np.ndarray) -> float:
    """F2: MP-reported plain Voigt SVD scalar.

    MP publishes ``e_ij_max`` as the largest singular value of the 3x6 Voigt
    matrix.  This value is source-field-native to MP and is only guaranteed to
    be a property of the MP-reported Voigt matrix, not necessarily a physical
    coordinate invariant of the full Cartesian tensor.  Do not call it the
    "maximum longitudinal modulus" unless equivalence to the directional
    maximum is proven.
    """
    a = np.asarray(voigt, dtype=np.float64)
    if a.shape != (3, 6):
        raise ValueError(f"Expected Voigt shape (3, 6), got {a.shape}")
    # Use the symmetric eigenvalue of A @ A.T instead of SVD to avoid the
    # sporadic BLAS aborts seen with np.linalg.svd on some Windows builds.
    return float(np.sqrt(np.linalg.eigvalsh(a @ a.T).max()))


def kelvin_operator_norm(tensor: np.ndarray) -> float:
    """F4: Kelvin/Mandel operator norm of the piezoelectric tensor.

    Treats the tensor as the linear map
    ``E in Sym(3) -> D_i = e_{ijk} E_{jk}``.  In Kelvin/Mandel basis the
    matrix representation is

        A_K = [e_{i11}, e_{i22}, e_{i33},
               sqrt(2) e_{i23}, sqrt(2) e_{i13}, sqrt(2) e_{i12}]

    and the operator norm is the largest singular value of A_K.  This is a
    coordinate-invariant induced matrix norm.
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.shape != (3, 3, 3):
        raise ValueError(f"Expected Cartesian tensor shape (3, 3, 3), got {t.shape}")
    # Enforce minor symmetry in the strain indices.
    t = 0.5 * (t + t.transpose(0, 2, 1))
    a_k = np.zeros((3, 6), dtype=np.float64)
    a_k[:, 0] = t[:, 0, 0]
    a_k[:, 1] = t[:, 1, 1]
    a_k[:, 2] = t[:, 2, 2]
    a_k[:, 3] = np.sqrt(2.0) * t[:, 1, 2]
    a_k[:, 4] = np.sqrt(2.0) * t[:, 0, 2]
    a_k[:, 5] = np.sqrt(2.0) * t[:, 0, 1]
    # Largest singular value via the 3x3 symmetric eigenproblem; avoids the
    # BLAS aborts observed with np.linalg.svd on some Windows/Anaconda builds.
    return float(np.sqrt(np.linalg.eigvalsh(a_k @ a_k.T).max()))


def max_longitudinal_response(tensor: np.ndarray, n_samples: int = 20000, seed: int = 42) -> float:
    """F3 (stochastic estimate): true directional longitudinal maximum.

    Computes ``max_{||n||=1} |n_i e_ijk n_j n_k|``.

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
    """F3 (deterministic): true maximum longitudinal piezoelectric modulus.

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
        # Nelder-Mead spherical-coordinate polish avoids the Fortran aborts
        # observed on some Windows/SciPy builds with SLSQP and L-BFGS-B.
        # theta in (0, pi), phi in [0, 2*pi), radius fixed to 1.
        theta0 = float(np.arccos(np.clip(best_dir[2], -1.0, 1.0)))
        phi0 = float(np.arctan2(best_dir[1], best_dir[0])) % (2.0 * np.pi)

        def _n_from_spherical(params: np.ndarray) -> np.ndarray:
            theta, phi = params
            return np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                dtype=np.float64,
            )

        def neg_f2_spherical(params: np.ndarray) -> float:
            theta, phi = params
            # Soft barrier keeping theta inside (0, pi).
            if not (1e-8 < theta < np.pi - 1e-8):
                return 1e6
            n = _n_from_spherical(params)
            f = float(np.einsum("i,ijk,j,k->", n, t, n, n))
            return -(f * f)

        result = optimize.minimize(
            neg_f2_spherical,
            np.array([theta0, phi0], dtype=np.float64),
            method="Nelder-Mead",
            bounds=[(1e-8, np.pi - 1e-8), (0.0, 2.0 * np.pi)],
            options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 400, "disp": False},
        )
        if result.success:
            n_ref = _n_from_spherical(result.x)
            f_ref = float(np.einsum("i,ijk,j,k->", n_ref, t, n_ref, n_ref))
            ref_val = abs(f_ref)
            if ref_val > best_val:
                denom = max(best_val, 1.0)
                if (ref_val - best_val) / denom > cross_check_tol:
                    warnings.warn(
                        f"max_longitudinal_modulus cross-check found a better optimum: "
                        f"projected-gradient={best_val:.6e}, "
                        f"spherical-polish={ref_val:.6e}",
                        stacklevel=2,
                    )
                best_val = ref_val

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


# -----------------------------------------------------------------------------
# Top-k / listwise baseline ranking diagnostics
# -----------------------------------------------------------------------------


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top ``k`` items by descending score (stable tie-break)."""
    k = int(min(k, len(scores)))
    return np.argsort(-scores, kind="stable")[:k]


def _top_k_union(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    """Indices that appear in the top-``k`` set of either side."""
    top_l = _top_k_indices(left, k)
    top_r = _top_k_indices(right, k)
    return np.unique(np.concatenate([top_l, top_r]))


def precision_at_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Precision of the top-``k`` overlap for paired, equal-length rankings."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or k <= 0:
        return float("nan")
    kk = min(k, n)
    top_l = set(_top_k_indices(left, kk))
    top_r = set(_top_k_indices(right, kk))
    return len(top_l & top_r) / kk


def recall_at_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Recall of the top-``k`` overlap for paired, equal-length rankings.

    Because the two rankings share the same candidate universe, this equals
    ``precision_at_k``; it is exposed separately to match conventional
    information-retrieval terminology.
    """
    return precision_at_k(left, right, k)


def overlap_coefficient_at_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Overlap coefficient of the top-``k`` sets: |intersection| / min(|A|,|B|)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or k <= 0:
        return float("nan")
    kk = min(k, n)
    top_l = set(_top_k_indices(left, kk))
    top_r = set(_top_k_indices(right, kk))
    denom = min(len(top_l), len(top_r))
    return len(top_l & top_r) / denom if denom > 0 else 0.0


def plain_jaccard_at_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Plain Jaccard index of the top-``k`` sets (no chance adjustment)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or k <= 0:
        return float("nan")
    kk = min(k, n)
    top_l = set(_top_k_indices(left, kk))
    top_r = set(_top_k_indices(right, kk))
    union = len(top_l | top_r)
    return len(top_l & top_r) / union if union > 0 else 0.0


def rank_biased_overlap(left: np.ndarray, right: np.ndarray, p: float = 0.95) -> float:
    """Rank-Biased Overlap (RBO) between two paired rankings.

    RBO is a top-weighted similarity that does not require a fixed depth and is
    monotonic in the depth of the prefix considered.  The parameter ``p`` in
    (0,1) controls the tail weight: values close to 1 put more weight on deep
    ranks.  This implementation uses the standard extrapolated form

        RBO = (1-p) * sum_{d=1}^{n} p^{d-1} * A_d + p^n * A_n,

    where ``A_d`` is the fraction of common items in the top-``d`` prefixes.

    References
    ----------
    Webber, W., Moffat, A., & Zobel, J. (2010). A similarity measure for
    indefinite rankings. ACM Transactions on Information Systems, 28(4), 1-38.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or not 0.0 < p < 1.0:
        return float("nan")

    order_l = _top_k_indices(left, n)
    order_r = _top_k_indices(right, n)
    set_l: set[int] = set()
    set_r: set[int] = set()
    agreement = 0.0
    weight = 1.0 - p
    for d in range(1, n + 1):
        set_l.add(int(order_l[d - 1]))
        set_r.add(int(order_r[d - 1]))
        inter = len(set_l & set_r)
        union = len(set_l | set_r)
        a_d = inter / union if union > 0 else 0.0
        agreement += weight * (p ** (d - 1)) * a_d
    # Extrapolated tail term: p^n * A_n.
    agreement += (p ** n) * a_d
    return float(agreement)


def top_weighted_kendall_tau(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Kendall tau computed on the union of the two top-``k`` sets."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or k <= 0:
        return float("nan")
    kk = min(k, n)
    subset = _top_k_union(left, right, kk)
    if len(subset) < 2:
        return float("nan")
    a, b = left[subset], right[subset]
    if np.allclose(a, b):
        return 1.0
    tau, _ = stats.kendalltau(a, b)
    return float(tau) if tau is not None else float("nan")


def _spearman_rho_no_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho computed without scipy's p-value path.

    Some SciPy/OpenBLAS combinations raise a floating-point exception when the
    input is constant or perfectly rank-correlated.  We compute the Pearson
    correlation of average ranks directly and handle those edge cases.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = stats.rankdata(a, method="average")
    rb = stats.rankdata(b, method="average")
    if np.allclose(ra, rb):
        return 1.0
    sa = float(np.std(ra, ddof=0))
    sb = float(np.std(rb, ddof=0))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    cov = float(np.mean((ra - ra.mean()) * (rb - rb.mean())))
    return cov / (sa * sb)


def top_weighted_spearman_rho(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Spearman rho computed on the union of the two top-``k`` sets."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0 or k <= 0:
        return float("nan")
    kk = min(k, n)
    subset = _top_k_union(left, right, kk)
    if len(subset) < 2:
        return float("nan")
    return _spearman_rho_no_pvalue(left[subset], right[subset])


# -----------------------------------------------------------------------------
# Phase 6B corrected null statistics
# -----------------------------------------------------------------------------


def expected_jaccard_hypergeometric(n: int, k: int) -> float:
    """Exact expectation of top-k Jaccard under independent random ranking.

    Two independent top-k sets are drawn from n items without replacement.
    The intersection size X ~ Hypergeometric(N=n, K=k, n=k).  The Jaccard is
    J = X / (2k - X).  This function returns E[J] exactly by summation.
    """
    if n <= 0 or k <= 0 or k > n:
        return float("nan")
    dist = stats.hypergeom(n, k, k)
    xs = np.arange(max(0, 2 * k - n), min(k, n) + 1, dtype=np.float64)
    pmf = dist.pmf(xs.astype(int))
    jaccards = xs / (2.0 * k - xs)
    return float(np.sum(pmf * jaccards))


def hypergeometric_overlap_pvalue(n: int, k: int, x: int) -> float:
    """P(intersection >= x) for two independent top-k sets under hypergeom null."""
    if n <= 0 or k <= 0 or x > k:
        return float("nan")
    return float(stats.hypergeom.sf(x - 1, n, k, k))


def chance_adjusted_jaccard(observed: float, expected: float) -> float:
    """Adjusted Jaccard = (observed - expected) / (1 - expected)."""
    if not np.isfinite(expected) or expected >= 1.0:
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def permutation_pvalue(
    left: np.ndarray,
    right: np.ndarray,
    observed_stat: float,
    n_permutations: int = 4999,
    seed: int = 42,
    alternative: str = "two-sided",
) -> float:
    """Finite-sample corrected permutation p-value: (b + 1) / (B + 1).

    Tests the null hypothesis of no association between paired scores.
    This is a test of exchangeability, not a test of "sufficient agreement".
    """
    rng = np.random.default_rng(seed)
    n = len(left)
    if n < 5:
        return float("nan")
    b = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        tau, _ = stats.kendalltau(left, right[perm])
        if tau is None:
            continue
        if alternative == "two-sided":
            if abs(float(tau)) >= abs(observed_stat):
                b += 1
        elif alternative == "greater":
            if float(tau) >= observed_stat:
                b += 1
        else:
            if float(tau) <= observed_stat:
                b += 1
    return (b + 1) / (n_permutations + 1)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary vectors of equal length."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = (a == b).mean()
    p_yes = (a.sum() + b.sum()) / (2.0 * n)
    p_no = 1.0 - p_yes
    p_e = p_yes * p_yes + p_no * p_no
    if p_e >= 1.0 - 1e-12:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def matthews_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Matthews correlation coefficient for two binary vectors."""
    tp = int((a & b).sum())
    tn = int((~a & ~b).sum())
    fp = int((~a & b).sum())
    fn = int((a & ~b).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def kendall_tau_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    n_replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for Kendall tau-b."""
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
