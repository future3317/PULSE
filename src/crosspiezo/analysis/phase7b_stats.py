"""Phase 7B statistical primitives for screening-resolution and robust portfolios.

All functions are deterministic given a random seed and operate on paired
source-aligned score vectors.  They are intentionally decoupled from I/O so
that the main script, an independent verification script, and the test suite
all share the same numerics.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from crosspiezo.analysis.ranking import (
    chance_adjusted_jaccard,
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
)


def _finite_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right)


def _source_percentiles(scores: np.ndarray) -> np.ndarray:
    """Return source-wise percentiles in [0, 1]; highest score -> 1.0."""
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    ranks = stats.rankdata(scores, method="average")
    return (ranks - 1.0) / (n - 1.0)


def anion_group(formula: str | None) -> str:
    """Classify a formula by its anion(s) using pymatgen element parsing.

    Multi-anion materials receive a ``_``-joined label ordered by the
    electronegative elements present.  This avoids the Phase 7A substring bug
    where ``Se`` was classified as sulfide and ``Ho``/``Co`` as oxide.
    """
    if not formula:
        return "unknown"
    try:
        from pymatgen.core.composition import Composition
    except Exception:  # pragma: no cover
        return "unknown"
    try:
        comp = Composition(formula)
    except Exception:
        return "unknown"
    # Electronegative anions: Pauling X above a threshold that keeps Ge out.
    anion_els = [el for el in comp.elements if el.X > 2.05]
    mapping = {
        "O": "oxide",
        "S": "sulfide",
        "Se": "selenide",
        "Te": "telluride",
        "F": "halide",
        "Cl": "halide",
        "Br": "halide",
        "I": "halide",
        "N": "nitride",
    }
    labels = []
    for el in sorted(anion_els, key=lambda e: (str(e))):
        cat = mapping.get(str(el))
        if cat and cat not in labels:
            labels.append(cat)
    if not labels:
        return "other"
    if len(labels) == 1:
        return labels[0]
    return "_".join(labels)


def reduced_formula(formula: str | None) -> str | None:
    """Return the pymatgen reduced formula, or None if unparsable."""
    if not formula:
        return None
    try:
        from pymatgen.core.composition import Composition

        return Composition(formula).reduced_formula
    except Exception:
        return None


def _top_k_indices(scores: np.ndarray, k: int) -> set[int]:
    k = max(0, min(k, len(scores)))
    return set(np.argsort(-scores, kind="stable")[:k].tolist())


def _adjusted_jaccard_at_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    n = len(left)
    kk = max(1, min(k, n))
    tl = _top_k_indices(left, kk)
    tr = _top_k_indices(right, kk)
    inter = len(tl & tr)
    union = len(tl | tr)
    obs = inter / union if union else 0.0
    exp = expected_jaccard_hypergeometric(n, kk)
    return chance_adjusted_jaccard(obs, exp)


def _curve_from_scores(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Observed chance-adjusted Jaccard curve (no bootstrap)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    qs = np.asarray(q_percentiles, dtype=np.float64)
    curve = np.full(len(qs), np.nan, dtype=np.float64)
    for i, q in enumerate(qs):
        k = max(1, int(math.floor(q / 100.0 * n)))
        curve[i] = _adjusted_jaccard_at_k(left, right, k)
    return curve


def _simultaneous_band(
    bootstrap_curves: np.ndarray,
    observed: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Studentized sup-norm simultaneous confidence band.

    Parameters
    ----------
    bootstrap_curves:
        Array of shape (B, Q) of bootstrap curve realizations.
    observed:
        Array of shape (Q,) of observed curve values.
    alpha:
        Significance level (default 0.05 for a 95% band).

    Returns
    -------
    low, high arrays of shape (Q,).  Values are clamped to [-1, 1].
    """
    n_boot_curves, n_q = bootstrap_curves.shape
    if len(observed) != n_q:
        raise ValueError("bootstrap_curves and observed must have same Q dimension")
    if n_boot_curves == 0:
        nan_arr = np.full_like(observed, np.nan)
        return nan_arr, nan_arr
    se = np.std(bootstrap_curves, axis=0, ddof=1)
    se = np.where(se > 1e-12, se, 1e-12)
    studentized = np.max(np.abs(bootstrap_curves - observed[None, :]) / se[None, :], axis=1)
    crit = float(np.quantile(studentized, 1.0 - alpha))
    low = observed - crit * se
    high = observed + crit * se
    return np.clip(low, -1.0, 1.0), np.clip(high, -1.0, 1.0)


def screening_resolution_curve(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute observed and null top-q overlap statistics for paired scores.

    Returns a DataFrame with one row per ``q_percentile`` containing observed
    overlap, chance-adjusted Jaccard, a simultaneous 95% confidence band,
    exact hypergeometric p-values, and rank-displacement summaries over the
    union of the two top-q sets.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    qs = np.asarray(q_percentiles, dtype=np.float64)
    rng = np.random.default_rng(seed)

    ranks_left = stats.rankdata(-left, method="average")
    ranks_right = stats.rankdata(-right, method="average")
    abs_displacement = np.abs(ranks_left - ranks_right)

    # Precompute descending positions and per-q constants.
    ks = np.array([max(1, int(math.floor(q / 100.0 * n))) for q in qs], dtype=np.int64)
    exp_curve = np.array([expected_jaccard_hypergeometric(n, k) for k in ks], dtype=np.float64)
    positions = np.arange(n, dtype=np.int64)
    pos_left_obs = np.empty(n, dtype=np.int64)
    pos_left_obs[np.argsort(-left, kind="stable")] = positions
    pos_right_obs = np.empty(n, dtype=np.int64)
    pos_right_obs[np.argsort(-right, kind="stable")] = positions

    records: list[dict[str, Any]] = []
    for i, q in enumerate(qs):
        k = ks[i]
        mask_left = pos_left_obs < k
        mask_right = pos_right_obs < k
        inter = int((mask_left & mask_right).sum())
        union_mask = mask_left | mask_right
        union_size = int(union_mask.sum())
        obs_jaccard = inter / union_size if union_size else 0.0
        exp_jaccard = exp_curve[i]
        adj_jaccard = (obs_jaccard - exp_jaccard) / (1.0 - exp_jaccard)
        hyper_p = hypergeometric_overlap_pvalue(n, k, inter)
        enrichment = (
            obs_jaccard / exp_jaccard if exp_jaccard and exp_jaccard > 0 else float("nan")
        )

        mean_disp = float(abs_displacement[union_mask].mean()) if union_size else float("nan")
        median_disp = (
            float(np.median(abs_displacement[union_mask])) if union_size else float("nan")
        )

        records.append(
            {
                "q_percentile": q,
                "k": k,
                "n": n,
                "observed_overlap": inter,
                "observed_jaccard": obs_jaccard,
                "expected_jaccard": exp_jaccard,
                "chance_adjusted_jaccard": adj_jaccard,
                "hypergeometric_pvalue": hyper_p,
                "overlap_enrichment": enrichment,
                "mean_rank_displacement_in_union": mean_disp,
                "median_rank_displacement_in_union": median_disp,
            }
        )

    observed_curve = np.array([r["chance_adjusted_jaccard"] for r in records], dtype=np.float64)

    # Fast paired bootstrap: compute descending positions once per replicate.
    boot_array = np.empty((n_boot, len(qs)), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        pos_left = np.empty(n, dtype=np.int64)
        pos_left[np.argsort(-left[idx], kind="stable")] = positions
        pos_right = np.empty(n, dtype=np.int64)
        pos_right[np.argsort(-right[idx], kind="stable")] = positions
        inter = np.array([int(((pos_left < k) & (pos_right < k)).sum()) for k in ks])
        union = 2 * ks - inter
        obs_jaccard = np.where(union > 0, inter / union, 0.0)
        boot_array[b] = (obs_jaccard - exp_curve) / (1.0 - exp_curve)
    valid_boot = np.isfinite(boot_array).all(axis=1)
    if valid_boot.sum() >= max(1, n_boot // 2):
        boot_array = boot_array[valid_boot]
        low, high = _simultaneous_band(boot_array, observed_curve, alpha=alpha)
    else:
        low = high = np.full_like(observed_curve, np.nan)

    for i, rec in enumerate(records):
        rec["adj_jaccard_ci95_low"] = float(low[i])
        rec["adj_jaccard_ci95_high"] = float(high[i])

    return pd.DataFrame(records)


def naucc(q_curve: pd.DataFrame) -> float:
    """Normalized area under the chance-adjusted concordance curve.

    The trapezoidal AUC is divided by ``q_max - q_min`` so that nAUCC is a
    weighted average of adjusted Jaccard over the reported quantile range.
    """
    x = q_curve["q_percentile"].to_numpy(dtype=np.float64)
    y = q_curve["chance_adjusted_jaccard"].to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    denom = x[-1] - x[0]
    if denom <= 0:
        return float("nan")
    return float(np.trapezoid(y, x) / denom)


def persistent_onset(
    q_curve: pd.DataFrame,
    delta: float,
    min_consecutive: int = 5,
) -> float | None:
    """Smallest q whose simultaneous lower band exceeds delta persistently.

    Requires ``min_consecutive`` consecutive q-values with
    ``adj_jaccard_ci95_low > delta``.  Returns ``None`` if no such run exists.
    """
    qs = q_curve["q_percentile"].to_numpy(dtype=np.float64)
    low = q_curve["adj_jaccard_ci95_low"].to_numpy(dtype=np.float64)
    above = low > delta
    for i in range(len(qs) - min_consecutive + 1):
        if above[i : i + min_consecutive].all():
            return float(qs[i])
    return None


def q_consensus_pointwise(
    q_curve: pd.DataFrame,
    delta: float = 0.0,
) -> float | None:
    """Pointwise first crossing (legacy definition, for contrast only)."""
    sub = q_curve[q_curve["adj_jaccard_ci95_low"] > delta]
    if sub.empty:
        return None
    return float(sub["q_percentile"].min())


# ---------------------------------------------------------------------------
# Grouped paired bootstrap for tau and nAUCC
# ---------------------------------------------------------------------------


def _group_indices(groups: np.ndarray) -> dict[Any, np.ndarray]:
    idx_by_group: dict[Any, np.ndarray] = {}
    for g in np.unique(groups):
        idx_by_group[g] = np.where(groups == g)[0]
    return idx_by_group


def grouped_paired_bootstrap_tau(
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and grouped paired-bootstrap 95% CI for Kendall tau.

    Groups are resampled with replacement; all observations within a sampled
    group are kept, preserving within-group dependence.

    Returns
    -------
    (tau_point, ci_low, ci_high)
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    valid = _finite_mask(left, right)
    left, right, groups = left[valid], right[valid], groups[valid]
    n = len(left)
    if n < 5:
        return float("nan"), float("nan"), float("nan")

    tau_point, _ = stats.kendalltau(left, right)
    tau_point = float(tau_point) if tau_point is not None else float("nan")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return tau_point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx_by_group = _group_indices(groups)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
        if len(idx) < 5:
            reps[b] = float("nan")
            continue
        tau, _ = stats.kendalltau(left[idx], right[idx])
        reps[b] = float(tau) if tau is not None else float("nan")

    reps = reps[np.isfinite(reps)]
    if len(reps) < n_boot // 2:
        return tau_point, float("nan"), float("nan")
    ci_low, ci_high = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    return tau_point, ci_low, ci_high


def _naucc_from_scores(
    left: np.ndarray,
    right: np.ndarray,
    qs: np.ndarray,
    ks: np.ndarray,
    exp_curve: np.ndarray,
) -> float:
    """Compute nAUCC directly without building a DataFrame."""
    n = len(left)
    positions = np.arange(n, dtype=np.int64)
    pos_left = np.empty(n, dtype=np.int64)
    pos_left[np.argsort(-left, kind="stable")] = positions
    pos_right = np.empty(n, dtype=np.int64)
    pos_right[np.argsort(-right, kind="stable")] = positions
    inter = np.array([int(((pos_left < k) & (pos_right < k)).sum()) for k in ks])
    union = 2 * ks - inter
    obs_jaccard = np.where(union > 0, inter / union, 0.0)
    adj = (obs_jaccard - exp_curve) / (1.0 - exp_curve)
    order = np.argsort(qs, kind="stable")
    x, y = qs[order], adj[order]
    if len(x) < 2:
        return float("nan")
    denom = x[-1] - x[0]
    return float(np.trapezoid(y, x) / denom) if denom > 0 else float("nan")


def grouped_paired_bootstrap_naucc(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and grouped paired-bootstrap 95% CI for nAUCC."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    valid = _finite_mask(left, right)
    left, right, groups = left[valid], right[valid], groups[valid]

    qs = np.asarray(q_percentiles, dtype=np.float64)
    ks = np.array([max(1, int(math.floor(q / 100.0 * len(left)))) for q in qs], dtype=np.int64)
    exp_curve = np.array([expected_jaccard_hypergeometric(len(left), k) for k in ks], dtype=np.float64)
    point = _naucc_from_scores(left, right, qs, ks, exp_curve)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(left) < 5:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx_by_group = _group_indices(groups)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
        if len(idx) < 5:
            reps[b] = float("nan")
            continue
        reps[b] = _naucc_from_scores(left[idx], right[idx], qs, ks, exp_curve)
    reps = reps[np.isfinite(reps)]
    if len(reps) < n_boot // 2:
        return point, float("nan"), float("nan")
    ci_low, ci_high = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    return point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Permutation test for difference in tau between a control and piezo metric
# ---------------------------------------------------------------------------


def permutation_test_delta_tau(
    left_ctrl: np.ndarray,
    right_ctrl: np.ndarray,
    left_piezo: np.ndarray,
    right_piezo: np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    n_perm: int = 4999,
    seed: int = 43,
) -> float:
    """Group-label permutation p-value for H0: tau_ctrl == tau_piezo.

    The test assumes the two measurements share the same group labels (e.g.
    the same matched pairs).  Under the null, the ``control`` / ``piezo``
    assignment for each group is exchangeable.
    """
    left_ctrl = np.asarray(left_ctrl, dtype=np.float64)
    right_ctrl = np.asarray(right_ctrl, dtype=np.float64)
    left_piezo = np.asarray(left_piezo, dtype=np.float64)
    right_piezo = np.asarray(right_piezo, dtype=np.float64)
    groups = np.asarray(groups)

    valid_c = _finite_mask(left_ctrl, right_ctrl)
    valid_p = _finite_mask(left_piezo, right_piezo)
    left_ctrl, right_ctrl, groups_c = left_ctrl[valid_c], right_ctrl[valid_c], groups[valid_c]
    left_piezo, right_piezo, groups_p = left_piezo[valid_p], right_piezo[valid_p], groups[valid_p]

    tau_c, _ = stats.kendalltau(left_ctrl, right_ctrl)
    tau_p, _ = stats.kendalltau(left_piezo, right_piezo)
    tau_c = float(tau_c) if tau_c is not None else float("nan")
    tau_p = float(tau_p) if tau_p is not None else float("nan")
    if not (np.isfinite(tau_c) and np.isfinite(tau_p)):
        return float("nan")
    observed = tau_p - tau_c

    common = np.intersect1d(np.unique(groups_c), np.unique(groups_p))
    if len(common) == 0:
        return float("nan")

    idx_c = _group_indices(groups_c)
    idx_p = _group_indices(groups_p)
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(n_perm):
        assign_piezo = rng.random(len(common)) > 0.5
        c_idx = np.concatenate([idx_c[g] for g in common[~assign_piezo]])
        p_idx = np.concatenate([idx_p[g] for g in common[assign_piezo]])
        if len(c_idx) < 5 or len(p_idx) < 5:
            continue
        t_c, _ = stats.kendalltau(left_ctrl[c_idx], right_ctrl[c_idx])
        t_p, _ = stats.kendalltau(left_piezo[p_idx], right_piezo[p_idx])
        t_c = float(t_c) if t_c is not None else float("nan")
        t_p = float(t_p) if t_p is not None else float("nan")
        if not (np.isfinite(t_c) and np.isfinite(t_p)):
            continue
        if abs(t_p - t_c) >= abs(observed):
            b += 1
    return (b + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Robust portfolio selection and evaluation
# ---------------------------------------------------------------------------


def portfolio_select(
    strategy: str,
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
    lambda_param: float = 0.0,
) -> list[int]:
    """Deterministically select a robust cross-source portfolio.

    Parameters
    ----------
    strategy:
        One of ``jarvis_only``, ``mp_only``, ``average_percentile``,
        ``borda_count``, ``maximin_percentile``, ``intersection_first``,
        ``balanced_union``, ``disagreement_abstention``, ``minimax_oracle``.
    left, right:
        Paired source scores.
    q_star:
        Target elite fraction (e.g. 0.10 for the top 10%).
    budget_factor:
        Multiplier on the target elite size.
    lambda_param:
        Disagreement penalty for ``disagreement_abstention``.

    Returns
    -------
    Sorted list of selected indices (length == budget, deterministic).
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    if n == 0:
        return []

    target_k = max(1, int(math.floor(q_star * n)))
    budget = min(n, int(round(budget_factor * target_k)))

    pl = _source_percentiles(left)
    pr = _source_percentiles(right)
    avg_pct = 0.5 * (pl + pr)

    def _sort_indices(scores: np.ndarray, descending: bool = True) -> list[int]:
        order = np.argsort(-scores if descending else scores, kind="stable")
        return order.tolist()

    selected: set[int] = set()

    if strategy == "jarvis_only":
        selected = set(_sort_indices(left)[:budget])
    elif strategy == "mp_only":
        selected = set(_sort_indices(right)[:budget])
    elif strategy == "average_percentile":
        selected = set(_sort_indices(avg_pct)[:budget])
    elif strategy == "borda_count":
        # Lower sum of ranks is better; both rankings treat higher scores as better.
        rl = stats.rankdata(-left, method="average")
        rr = stats.rankdata(-right, method="average")
        borda = rl + rr
        selected = set(_sort_indices(borda, descending=False)[:budget])
    elif strategy == "maximin_percentile":
        selected = set(_sort_indices(np.minimum(pl, pr))[:budget])
    elif strategy == "intersection_first":
        top_j = _top_k_indices(left, target_k)
        top_m = _top_k_indices(right, target_k)
        inter = top_j & top_m
        order = _sort_indices(avg_pct)
        selected = set([i for i in order if i in inter][:budget])
        if len(selected) < budget:
            for i in order:
                if i not in selected:
                    selected.add(i)
                if len(selected) >= budget:
                    break
    elif strategy == "balanced_union":
        top_j = _top_k_indices(left, target_k)
        top_m = _top_k_indices(right, target_k)
        uni = top_j | top_m
        order = _sort_indices(avg_pct)
        selected = set([i for i in order if i in uni][:budget])
        if len(selected) < budget:
            for i in order:
                if i not in selected:
                    selected.add(i)
                if len(selected) >= budget:
                    break
    elif strategy == "disagreement_abstention":
        disagreement = np.abs(pl - pr)
        score = avg_pct - lambda_param * disagreement
        selected = set(_sort_indices(score)[:budget])
    elif strategy == "minimax_oracle":
        selected = _minimax_oracle(left, right, q_star, budget)
    else:
        raise ValueError(f"Unknown portfolio strategy: {strategy}")

    # Deterministic final ordering: average percentile desc, then min percentile desc,
    # then index ascending as an unambiguous tie-breaker.
    selected_list = sorted(
        selected,
        key=lambda i: (-avg_pct[i], -min(pl[i], pr[i]), i),
    )
    return selected_list[:budget]


def _minimax_oracle(left: np.ndarray, right: np.ndarray, q_star: float, budget: int) -> set[int]:
    """Exact max--min selector for two observed elite sets.

    The objective depends only on four membership categories: intersection,
    JARVIS-only, MP-only and neither. Selecting the intersection first and
    then balancing the two one-sided categories is therefore exact for the
    worst-source recall objective; average percentile is used only to choose
    among selections with the same objective value.
    """
    n = len(left)
    if budget <= 0 or n == 0:
        return set()
    target_k = max(1, int(math.floor(q_star * n)))
    top_j_idx = np.argsort(-left, kind="stable")[:target_k]
    top_m_idx = np.argsort(-right, kind="stable")[:target_k]
    top_j = np.zeros(n, dtype=bool)
    top_m = np.zeros(n, dtype=bool)
    top_j[top_j_idx] = True
    top_m[top_m_idx] = True

    pl = _source_percentiles(left)
    pr = _source_percentiles(right)
    avg_pct = 0.5 * (pl + pr)

    categories = {
        "intersection": np.where(top_j & top_m)[0].tolist(),
        "jarvis_only": np.where(top_j & ~top_m)[0].tolist(),
        "mp_only": np.where(~top_j & top_m)[0].tolist(),
        "neither": np.where(~top_j & ~top_m)[0].tolist(),
    }
    for values in categories.values():
        values.sort(key=lambda i: (-avg_pct[i], i))

    intersection = categories["intersection"][:budget]
    selected = list(intersection)
    if len(selected) == budget:
        return set(selected)

    remaining = budget - len(selected)
    n_j = len(categories["jarvis_only"])
    n_m = len(categories["mp_only"])
    best: tuple[int, int, int] | None = None
    best_take_j = 0
    best_take_m = 0
    for take_j in range(min(n_j, remaining) + 1):
        take_m = min(n_m, remaining - take_j)
        objective = min(len(intersection) + take_j, len(intersection) + take_m)
        candidate = (objective, take_j + take_m, -abs(take_j - take_m))
        if best is None or candidate > best:
            best = candidate
            best_take_j = take_j
            best_take_m = take_m

    assert best is not None
    selected.extend(categories["jarvis_only"][:best_take_j])
    selected.extend(categories["mp_only"][:best_take_m])

    fill = budget - len(selected)
    selected.extend(categories["neither"][:fill])
    return set(selected)


def portfolio_metrics(selected: Sequence[int], left: np.ndarray, right: np.ndarray, q_star: float) -> dict[str, Any]:
    """Evaluate a deterministic portfolio on the two-source task.

    Metrics are worst-source summaries.  NDCG ideal is computed from the full
    universe, not only the selected set.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    selected = [int(i) for i in selected]
    n = len(left)
    if n == 0 or not selected:
        return {
            "worst_source_recall": float("nan"),
            "worst_source_ndcg": float("nan"),
            "worst_source_normalized_utility": float("nan"),
            "minimax_recall_regret": float("nan"),
            "minimax_ndcg_regret": float("nan"),
            "portfolio_size": 0,
            "portfolio_coverage": 0.0,
        }

    k_q = max(1, int(math.floor(q_star * n)))
    top_j = _top_k_indices(left, k_q)
    top_m = _top_k_indices(right, k_q)

    sel_set = set(selected)
    recall_j = len(sel_set & top_j) / k_q
    recall_m = len(sel_set & top_m) / k_q
    worst_recall = min(recall_j, recall_m)

    pl = _source_percentiles(left)
    pr = _source_percentiles(right)
    avg_pct = 0.5 * (pl + pr)

    def _ndcg_for_source(selected_relevances: np.ndarray, source_relevances: np.ndarray, m: int) -> float:
        selected_relevances = np.asarray(selected_relevances, dtype=np.float64)[:m]
        if m == 0:
            return 0.0
        positions = np.arange(2, m + 2, dtype=np.float64)
        dcg = np.sum((2.0**selected_relevances - 1.0) / np.log2(positions))
        # Ideal DCG uses the full universe's top-m relevances, not just the selected set.
        ideal_idx = np.argsort(-source_relevances, kind="stable")[:m]
        ideal = source_relevances[ideal_idx]
        idcg = np.sum((2.0**ideal - 1.0) / np.log2(positions))
        return float(dcg / idcg) if idcg > 0 else 0.0

    order = sorted(selected, key=lambda i: (-avg_pct[i], i))
    ndcg_j = _ndcg_for_source(pl[order], pl, len(order))
    ndcg_m = _ndcg_for_source(pr[order], pr, len(order))
    worst_ndcg = min(ndcg_j, ndcg_m)

    return {
        "worst_source_recall": float(worst_recall),
        "worst_source_ndcg": float(worst_ndcg),
        "worst_source_normalized_utility": float(worst_recall),
        "minimax_recall_regret": float(1.0 - worst_recall),
        "minimax_ndcg_regret": float(1.0 - worst_ndcg),
        "portfolio_size": len(selected),
        "portfolio_coverage": len(selected) / n,
    }


def portfolio_bootstrap_ci(
    strategy: str,
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 47,
    lambda_param: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Grouped paired-bootstrap 95% CI for each portfolio metric.

    Returns a dict mapping metric name to ``{"ci95_low": ..., "ci95_high": ...}``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    valid = _finite_mask(left, right)
    left, right, groups = left[valid], right[valid], groups[valid]

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {}

    rng = np.random.default_rng(seed)
    idx_by_group = _group_indices(groups)
    metric_keys = [
        "worst_source_recall",
        "worst_source_ndcg",
        "worst_source_normalized_utility",
        "minimax_recall_regret",
        "minimax_ndcg_regret",
        "portfolio_coverage",
    ]
    reps: dict[str, list[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
        if len(idx) < 5:
            continue
        sel = portfolio_select(
            strategy,
            left[idx],
            right[idx],
            q_star,
            budget_factor,
            lambda_param=lambda_param,
        )
        metrics = portfolio_metrics(sel, left[idx], right[idx], q_star)
        for k in metric_keys:
            v = metrics.get(k)
            if v is not None and np.isfinite(v):
                reps[k].append(float(v))

    result: dict[str, dict[str, float]] = {}
    for k, vals in reps.items():
        if len(vals) < n_boot // 2:
            result[k] = {"ci95_low": float("nan"), "ci95_high": float("nan")}
        else:
            arr = np.asarray(vals, dtype=np.float64)
            result[k] = {
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
            }
    return result


def tune_disagreement_abstention(
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
    lambda_grid: Sequence[float],
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 200,
    seed: int = 48,
) -> float:
    """Choose the disagreement-abstention lambda by grouped bootstrap on dev data.

    For each candidate lambda we compute the median worst-source recall over
    grouped bootstrap replicates and return the lambda with the highest median.
    Ties favour smaller lambda for parsimony.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    valid = _finite_mask(left, right)
    left, right, groups = left[valid], right[valid], groups[valid]
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(lambda_grid) == 0:
        return float(lambda_grid[0]) if lambda_grid else 0.0

    rng = np.random.default_rng(seed)
    idx_by_group = _group_indices(groups)
    best_lambda = float(lambda_grid[0])
    best_median = -1.0
    for lam in lambda_grid:
        recalls: list[float] = []
        for _ in range(n_boot):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
            if len(idx) < 5:
                continue
            sel = portfolio_select(
                "disagreement_abstention",
                left[idx],
                right[idx],
                q_star,
                budget_factor,
                lambda_param=lam,
            )
            rec = portfolio_metrics(sel, left[idx], right[idx], q_star)["worst_source_recall"]
            if np.isfinite(rec):
                recalls.append(float(rec))
        if not recalls:
            continue
        med = float(np.median(recalls))
        if med > best_median + 1e-12:
            best_median = med
            best_lambda = float(lam)
    return best_lambda


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Return BH-adjusted p-values (q-values)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    # Benjamini-Hochberg linear step-up.
    q = np.minimum.accumulate(sorted_p * n / np.arange(1, n + 1))
    # Enforce monotonicity and bound by 1.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = q
    return out.tolist()
