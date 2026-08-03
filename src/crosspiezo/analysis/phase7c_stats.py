"""Phase 7C statistical primitives for reviewer-proofing.

This module extends the frozen Phase 7B primitives with:

* banded partial nAUCC (elite / intermediate / broad),
* corrected high-response sensitivity masks and conditional permutation,
* grouped paired-bootstrap CI for paired differences,
* cross-evaluation of robust portfolios with paired single-source baselines,
* control provenance records.

All functions are deterministic given a seed and operate on paired
source-aligned score vectors.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from crosspiezo.analysis.phase7b_stats import (
    _finite_mask,
    benjamini_hochberg,
    grouped_paired_bootstrap_naucc,
    grouped_paired_bootstrap_tau,
    naucc,
    portfolio_metrics,
    portfolio_select,
    screening_resolution_curve,
    tune_disagreement_abstention,
)
from crosspiezo.analysis.ranking import expected_jaccard_hypergeometric


def partial_naucc(q_curve: pd.DataFrame, q_min: float, q_max: float) -> float:
    """Normalised area under the chance-adjusted concordance curve over a band.

    The band is inclusive of ``q_min`` and ``q_max``.  The integral is
    normalised by ``q_max - q_min`` so that partial nAUCC is a weighted
    average of adjusted Jaccard over the band.
    """
    x = q_curve["q_percentile"].to_numpy(dtype=np.float64)
    y = q_curve["chance_adjusted_jaccard"].to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    mask = (x >= q_min) & (x <= q_max)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    denom = q_max - q_min
    if denom <= 0:
        return float("nan")
    return float(np.trapezoid(y, x) / denom)


def _fast_naucc_for_selected(
    left: np.ndarray,
    right: np.ndarray,
    q_percentiles: Sequence[float] | np.ndarray,
) -> float:
    """nAUCC for a selected subset without constructing a DataFrame."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left, right = left[valid], right[valid]
    n = len(left)
    qs = np.asarray(q_percentiles, dtype=np.float64)
    if n < 2 or len(qs) < 2:
        return float("nan")
    ks = np.array([max(1, int(math.floor(q / 100.0 * n))) for q in qs], dtype=np.int64)
    exp_curve = np.array([expected_jaccard_hypergeometric(n, k) for k in ks], dtype=np.float64)
    positions = np.arange(n, dtype=np.int64)
    pos_left = np.empty(n, dtype=np.int64)
    pos_left[np.argsort(-left, kind="stable")] = positions
    pos_right = np.empty(n, dtype=np.int64)
    pos_right[np.argsort(-right, kind="stable")] = positions
    inter = np.array([int(((pos_left < k) & (pos_right < k)).sum()) for k in ks])
    union = 2 * ks - inter
    obs = np.where(union > 0, inter / union, 0.0)
    adj = (obs - exp_curve) / (1.0 - exp_curve)
    order = np.argsort(qs, kind="stable")
    x, y = qs[order], adj[order]
    denom = x[-1] - x[0]
    return float(np.trapezoid(y, x) / denom) if denom > 0 else float("nan")


def dual_high_response_mask(
    left: np.ndarray,
    right: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Mask for the dual-high subset: both sources exceed the min-quantile threshold.

    The threshold is the ``1 - fraction`` quantile of ``min(F_J, F_M)``.
    Items where either value is missing are excluded.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    min_score = np.minimum(left, right)
    threshold = float(np.quantile(min_score[valid], 1.0 - fraction))
    mask = valid & (left > threshold) & (right > threshold)
    return mask


def anchor_high_response_mask(
    anchor: np.ndarray,
    other: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Mask selecting the top ``fraction`` by ``anchor`` and evaluating on ``other``.

    ``other`` is only used to drop pairs with missing values.
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    valid = _finite_mask(anchor, other)
    n = int(valid.sum())
    if n == 0:
        return valid
    k = max(1, int(math.floor(fraction * n)))
    order = np.argsort(-anchor[valid], kind="stable")
    top_local = order[:k]
    mask = np.zeros(len(anchor), dtype=bool)
    valid_idx = np.where(valid)[0]
    mask[valid_idx[top_local]] = True
    return mask


_DEFAULT_Q_PERCENTILES = np.arange(1, 51, dtype=float)


def conditional_permutation_high_response(
    left: np.ndarray,
    right: np.ndarray,
    fraction: float,
    q_percentiles: Sequence[float] | np.ndarray | None = None,
    n_perm: int = 4999,
    seed: int = 50,
    min_n: int = 10,
) -> dict[str, Any]:
    """Conditional permutation p-value for pooled high-response selection.

    The observed pooled subset is defined by ``(left + right) / 2`` exceeding
    its ``1 - fraction`` quantile.  Under each permutation the JARVIS/MP
    labels are randomly swapped per pair, the pooled selection is recomputed,
    and the same nAUCC statistic is calculated on the selected subset.

    This procedure is kept only as an exploratory sensitivity check; the
    small-n elite-tail diagnostics rely on ``dual_high`` and ``anchor_*``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = _finite_mask(left, right)
    left_v, right_v = left[valid], right[valid]
    n = len(left_v)
    if n < min_n:
        return {
            "n_selected": 0,
            "observed_naucc": float("nan"),
            "permuted_mean_naucc": float("nan"),
            "permuted_std_naucc": float("nan"),
            "p_value": float("nan"),
            "n_perm": n_perm,
        }

    if q_percentiles is None:
        q_percentiles = _DEFAULT_Q_PERCENTILES
    qs = np.asarray(q_percentiles, dtype=np.float64)

    def _naucc_of_selected(lv: np.ndarray, rv: np.ndarray) -> float:
        pooled = 0.5 * (lv + rv)
        threshold = float(np.quantile(pooled, 1.0 - fraction))
        sel = np.where(pooled >= threshold)[0]
        if len(sel) < min_n:
            return float("nan")
        return _fast_naucc_for_selected(lv[sel], rv[sel], qs)

    observed = _naucc_of_selected(left_v, right_v)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        swap = rng.random(n) > 0.5
        lv_perm = np.where(swap, right_v, left_v)
        rv_perm = np.where(swap, left_v, right_v)
        reps[b] = _naucc_of_selected(lv_perm, rv_perm)

    finite_reps = reps[np.isfinite(reps)]
    if len(finite_reps) == 0:
        p_value = float("nan")
    else:
        p_value = (1.0 + float(np.sum(finite_reps >= observed))) / (len(finite_reps) + 1.0)

    pooled = 0.5 * (left_v + right_v)
    threshold = float(np.quantile(pooled, 1.0 - fraction))
    n_selected = int((pooled >= threshold).sum())

    return {
        "n_selected": n_selected,
        "observed_naucc": float(observed),
        "permuted_mean_naucc": float(np.mean(finite_reps)) if len(finite_reps) else float("nan"),
        "permuted_std_naucc": float(np.std(finite_reps, ddof=1)) if len(finite_reps) else float("nan"),
        "p_value": float(p_value),
        "n_perm": n_perm,
    }


def paired_bootstrap_ci_difference(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 51,
) -> tuple[float, float, float]:
    """Grouped paired-bootstrap 95% CI for ``mean(a - b)``.

    Differences are computed within groups, then groups are resampled with
    replacement to preserve within-group dependence.
    """
    metric_a = np.asarray(metric_a, dtype=np.float64)
    metric_b = np.asarray(metric_b, dtype=np.float64)
    groups = np.asarray(groups)
    valid = np.isfinite(metric_a) & np.isfinite(metric_b)
    metric_a, metric_b, groups = metric_a[valid], metric_b[valid], groups[valid]

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan"), float("nan")

    idx_by_group: dict[Any, np.ndarray] = {}
    for g in unique_groups:
        idx_by_group[g] = np.where(groups == g)[0]

    group_diffs = np.array(
        [np.mean(metric_a[idx_by_group[g]] - metric_b[idx_by_group[g]]) for g in unique_groups],
        dtype=np.float64,
    )
    point = float(np.mean(group_diffs))

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        reps[b] = float(np.mean(group_diffs[sampled]))

    ci_low = float(np.percentile(reps, 2.5))
    ci_high = float(np.percentile(reps, 97.5))
    return point, ci_low, ci_high


def material_level_paired_diff_ci(
    selected: Sequence[int],
    baseline_selected: Sequence[int],
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 52,
) -> tuple[float, float, float]:
    """Grouped paired-bootstrap CI for the material-level worst-source recall difference.

    The point estimate is ``worst_source_recall(strategy) - worst_source_recall(baseline)``
    on the full panel.  For each bootstrap replicate groups are resampled with
    replacement, the existing global selections are projected onto the bootstrap
    sample, and the material-level worst-source recall difference is recomputed.
    """
    selected = [int(i) for i in selected]
    baseline_selected = [int(i) for i in baseline_selected]
    selected_set = set(selected)
    baseline_set = set(baseline_selected)

    def _diff_on_sample(idx: np.ndarray) -> float:
        l, r = left[idx], right[idx]
        local_selected = [i for i, global_idx in enumerate(idx) if global_idx in selected_set]
        local_baseline = [i for i, global_idx in enumerate(idx) if global_idx in baseline_set]
        rec_a = portfolio_metrics(local_selected, l, r, q_star)["worst_source_recall"]
        rec_b = portfolio_metrics(local_baseline, l, r, q_star)["worst_source_recall"]
        return float(rec_a - rec_b)

    point = _diff_on_sample(np.arange(len(left)))

    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan"), float("nan"), float("nan")

    idx_by_group: dict[Any, np.ndarray] = {g: np.where(groups == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
        reps.append(_diff_on_sample(boot_idx))

    reps_arr = np.asarray(reps)
    return point, float(np.percentile(reps_arr, 2.5)), float(np.percentile(reps_arr, 97.5))


def full_procedure_portfolio_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
    strategies: Sequence[str],
    groups: Sequence[Hashable] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 53,
) -> pd.DataFrame:
    """Full-procedure grouped bootstrap CI for portfolio strategies.

    Each bootstrap replicate:

    1. resamples reduced-formula groups with replacement;
    2. creates distinct bootstrap identities for duplicated occurrences;
    3. recomputes source-specific rankings and top-:math:`q^*` elite sets;
    4. re-runs every portfolio strategy and both single-source baselines;
    5. reports recall differences versus JARVIS-only, MP-only, and the
       better single source *within that replicate*.

    The point estimate is computed on the full observed panel.  The
    returned intervals are unconditional percentile bootstrap CIs.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)

    valid = _finite_mask(left, right)
    left, right, groups = left[valid], right[valid], groups[valid]
    n = len(left)
    if n < 2:
        return _empty_portfolio_ci(strategies)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return _empty_portfolio_ci(strategies)

    idx_by_group: dict[Any, np.ndarray] = {g: np.where(groups == g)[0] for g in unique_groups}

    def _evaluate(_l: np.ndarray, _r: np.ndarray) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for strategy in list(strategies) + ["jarvis_only", "mp_only"]:
            sel = portfolio_select(strategy, _l, _r, q_star, budget_factor)
            out[strategy] = portfolio_metrics(sel, _l, _r, q_star)
        return out

    # Full-panel point estimates.
    full = _evaluate(left, right)
    full_better = (
        "jarvis_only"
        if full["jarvis_only"]["worst_source_recall"] >= full["mp_only"]["worst_source_recall"]
        else "mp_only"
    )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    reps: dict[str, dict[str, list[float]]] = {
        s: {"delta_j": [], "delta_m": [], "delta_best": []} for s in strategies
    }

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
        if len(boot_idx) < 5:
            continue
        boot_left, boot_right = left[boot_idx], right[boot_idx]
        boot = _evaluate(boot_left, boot_right)
        jarvis_rec = boot["jarvis_only"]["worst_source_recall"]
        mp_rec = boot["mp_only"]["worst_source_recall"]
        better_rec = max(jarvis_rec, mp_rec)
        for strategy in strategies:
            rec = boot[strategy]["worst_source_recall"]
            reps[strategy]["delta_j"].append(float(rec - jarvis_rec))
            reps[strategy]["delta_m"].append(float(rec - mp_rec))
            reps[strategy]["delta_best"].append(float(rec - better_rec))

    for strategy in strategies:
        point = float(full[strategy]["worst_source_recall"])
        delta_j_point = float(full[strategy]["worst_source_recall"] - full["jarvis_only"]["worst_source_recall"])
        delta_m_point = float(full[strategy]["worst_source_recall"] - full["mp_only"]["worst_source_recall"])
        delta_best_point = float(full[strategy]["worst_source_recall"] - full[full_better]["worst_source_recall"])

        for delta_name, point_val, values in (
            ("delta_j", delta_j_point, reps[strategy]["delta_j"]),
            ("delta_m", delta_m_point, reps[strategy]["delta_m"]),
            ("delta_best", delta_best_point, reps[strategy]["delta_best"]),
        ):
            if values:
                arr = np.asarray(values)
                rows.append(
                    {
                        "strategy": strategy,
                        "delta_type": delta_name,
                        "point": point_val,
                        "ci95_low": float(np.percentile(arr, 2.5)),
                        "ci95_high": float(np.percentile(arr, 97.5)),
                        "n_boot": n_boot,
                        "seed": seed,
                    }
                )
            else:
                rows.append(
                    {
                        "strategy": strategy,
                        "delta_type": delta_name,
                        "point": point_val,
                        "ci95_low": float("nan"),
                        "ci95_high": float("nan"),
                        "n_boot": n_boot,
                        "seed": seed,
                    }
                )

    return pd.DataFrame(rows)


def _empty_portfolio_ci(strategies: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for delta_name in ("delta_j", "delta_m", "delta_best"):
            rows.append(
                {
                    "strategy": strategy,
                    "delta_type": delta_name,
                    "point": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n_boot": 0,
                    "seed": 0,
                }
            )
    return pd.DataFrame(rows)


def _assign_folds(pair_ids: Sequence[str], n_folds: int, seed: int = 0) -> np.ndarray:
    """Deterministic fold assignment from hashed pair identifiers."""
    folds = np.empty(len(pair_ids), dtype=np.int64)
    for i, pid in enumerate(pair_ids):
        digest = hashlib.md5(str(pid).encode("utf-8")).hexdigest()
        folds[i] = int(digest, 16) % n_folds
    return folds


def portfolio_cross_evaluation(
    left: np.ndarray,
    right: np.ndarray,
    q_star: float,
    budget_factor: float,
    strategies: Sequence[str],
    groups: Sequence[Hashable] | np.ndarray,
    pair_ids: Sequence[str],
    n_folds: int = 5,
    lambda_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    n_inner_boot: int = 200,
    cross_eval_seed: int = 49,
    tune_seed: int = 48,
) -> pd.DataFrame:
    """Grouped cross-evaluation of robust portfolio strategies.

    For each outer fold the portfolio is selected on the test fold (the
    selection rule depends only on the source scores in that fold).  The
    disagreement-abstention penalty is tuned on the training folds via grouped
    bootstrap.  For every robust strategy the paired difference versus the
    better of the two single-source baselines is reported within the same fold.

    Returns a DataFrame with one row per (fold, strategy).
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    groups = np.asarray(groups)
    pair_ids = list(pair_ids)
    valid = _finite_mask(left, right)
    left, right, groups, pair_ids = left[valid], right[valid], groups[valid], [
        p for i, p in enumerate(pair_ids) if valid[i]
    ]

    folds = _assign_folds(pair_ids, n_folds, seed=cross_eval_seed)
    rows: list[dict[str, Any]] = []

    for fold in range(n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        if test_mask.sum() < 5:
            continue

        l_test, r_test = left[test_mask], right[test_mask]
        l_train, r_train = left[train_mask], right[train_mask]
        g_train = groups[train_mask]

        # Single-source baselines on the test fold.
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for strategy in ("jarvis_only", "mp_only"):
            sel = portfolio_select(strategy, l_test, r_test, q_star, budget_factor)
            baseline_metrics[strategy] = portfolio_metrics(sel, l_test, r_test, q_star)

        better_recall = max(
            baseline_metrics["jarvis_only"]["worst_source_recall"],
            baseline_metrics["mp_only"]["worst_source_recall"],
        )
        better_ndcg = max(
            baseline_metrics["jarvis_only"]["worst_source_ndcg"],
            baseline_metrics["mp_only"]["worst_source_ndcg"],
        )

        for strategy in strategies:
            lam = 0.0
            if strategy == "disagreement_abstention":
                if len(np.unique(g_train)) >= 2:
                    lam = tune_disagreement_abstention(
                        l_train,
                        r_train,
                        q_star,
                        budget_factor,
                        lambda_grid,
                        g_train,
                        n_boot=n_inner_boot,
                        seed=tune_seed + fold,
                    )
                else:
                    lam = float(lambda_grid[0])

            sel = portfolio_select(
                strategy, l_test, r_test, q_star, budget_factor, lambda_param=lam
            )
            metrics = portfolio_metrics(sel, l_test, r_test, q_star)

            rows.append(
                {
                    "fold": int(fold),
                    "n_test": int(test_mask.sum()),
                    "strategy": strategy,
                    "disagreement_lambda": float(lam) if strategy == "disagreement_abstention" else float("nan"),
                    "worst_source_recall": metrics["worst_source_recall"],
                    "worst_source_ndcg": metrics["worst_source_ndcg"],
                    "portfolio_size": metrics["portfolio_size"],
                    "portfolio_coverage": metrics["portfolio_coverage"],
                    "minimax_recall_regret": metrics["minimax_recall_regret"],
                    "paired_diff_recall": metrics["worst_source_recall"] - better_recall,
                    "paired_diff_ndcg": metrics["worst_source_ndcg"] - better_ndcg,
                }
            )

    return pd.DataFrame(rows)


def control_provenance_record(
    attr_name: str,
    jarvis_col: str | None,
    mp_col: str | None,
    jarvis_source_db: str,
    mp_source_db: str,
    unit: str,
    calculation_type_version: str,
    jarvis_arr: np.ndarray,
    mp_arr: np.ndarray,
) -> dict[str, Any]:
    """Return a provenance audit record for a single control attribute.

    The ``same_field_copy_flag`` is raised when more than 5% of the finite
    matched pairs have identical JARVIS and MP values, which would be
    consistent with direct field copying rather than independent calculation.
    """
    jarvis_arr = np.asarray(jarvis_arr, dtype=np.float64)
    mp_arr = np.asarray(mp_arr, dtype=np.float64)
    n = len(jarvis_arr)
    valid = _finite_mask(jarvis_arr, mp_arr)
    common_n = int(valid.sum())
    jarvis_missing = int((~np.isfinite(jarvis_arr)).sum())
    mp_missing = int((~np.isfinite(mp_arr)).sum())

    identical = 0
    if common_n > 0:
        identical = int(np.sum(jarvis_arr[valid] == mp_arr[valid]))
    same_field_copy_flag = bool(common_n > 0 and identical / common_n > 0.05)

    return {
        "attribute": attr_name,
        "jarvis_source_field": jarvis_col,
        "mp_source_field": mp_col,
        "jarvis_source_database": jarvis_source_db,
        "mp_source_database": mp_source_db,
        "unit": unit,
        "calculation_type_version": calculation_type_version,
        "n_pairs": n,
        "common_n": common_n,
        "jarvis_missing": jarvis_missing,
        "mp_missing": mp_missing,
        "identical_values": identical,
        "fraction_identical": float(identical / common_n) if common_n else float("nan"),
        "same_field_copy_flag": same_field_copy_flag,
    }


__all__ = [
    "partial_naucc",
    "dual_high_response_mask",
    "anchor_high_response_mask",
    "conditional_permutation_high_response",
    "paired_bootstrap_ci_difference",
    "full_procedure_portfolio_bootstrap_ci",
    "portfolio_cross_evaluation",
    "control_provenance_record",
    "benjamini_hochberg",
    "grouped_paired_bootstrap_naucc",
    "grouped_paired_bootstrap_tau",
    "naucc",
    "portfolio_metrics",
    "portfolio_select",
    "screening_resolution_curve",
]
