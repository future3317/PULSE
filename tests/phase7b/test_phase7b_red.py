"""Red-team tests for Phase 7B methodological corrections.

These tests encode the Phase 7A bugs that Phase 7B must avoid:
normalized AUC, persistent onset, monotonic invariance, Borda direction,
NDCG ideal, minimax regret, anion parsing, determinism, grouped bootstrap,
and disagreement-abstention tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from crosspiezo.analysis.phase7b_stats import (
    _source_percentiles,
    anion_group,
    benjamini_hochberg,
    grouped_paired_bootstrap_tau,
    naucc,
    persistent_onset,
    portfolio_metrics,
    portfolio_select,
    tune_disagreement_abstention,
)


def test_naucc_normalized_between_minus_one_and_one():
    """nAUCC is a weighted average of adjusted Jaccard, bounded by [-1, 1]."""
    qs = np.arange(0, 51, dtype=np.float64)
    df = pd.DataFrame(
        {
            "q_percentile": qs,
            "chance_adjusted_jaccard": qs / 50.0,
        }
    )
    value = naucc(df)
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx(0.5, abs=1e-9)


def test_persistent_onset_requires_five_consecutive_crossings():
    """A single LCB crossing is not enough; need min_consecutive=5."""
    qs = np.arange(1, 21, dtype=np.float64)
    # Only three consecutive low-confidence bounds above 0.
    low = np.array([-0.2] * 7 + [0.1, 0.2, 0.3] + [-0.1] * 10)
    df = pd.DataFrame({"q_percentile": qs, "adj_jaccard_ci95_low": low})
    assert persistent_onset(df, delta=0.0, min_consecutive=5) is None

    # Five consecutive values above 0 starting at q=10.
    low2 = np.array([-0.2] * 9 + [0.1, 0.2, 0.3, 0.4, 0.5] + [-0.1] * 6)
    df2 = pd.DataFrame({"q_percentile": qs, "adj_jaccard_ci95_low": low2})
    assert persistent_onset(df2, delta=0.0, min_consecutive=5) == 10.0


def test_quantile_normalized_kendall_tau_equals_raw_tau():
    """Strict monotonic calibration cannot change rank correlations."""
    rng = np.random.default_rng(7)
    left = rng.exponential(scale=1.0, size=80)
    right = left + rng.normal(scale=0.3, size=80)
    tau_raw, _ = stats.kendalltau(left, right)
    tau_q, _ = stats.kendalltau(_source_percentiles(left), _source_percentiles(right))
    assert tau_q == pytest.approx(tau_raw, abs=1e-9)


def test_borda_prefers_lower_sum_of_ranks():
    """Lower Borda score (sum of ranks) must select the consensus top items."""
    left = np.array([10.0, 5.0, 1.0])
    right = np.array([9.0, 4.0, 2.0])
    selected = portfolio_select("borda_count", left, right, q_star=0.5, budget_factor=1.5)
    # n=3, q_star=0.5 -> target_k=1, budget=2.
    assert selected[:2] == [0, 1]


def test_borda_wrong_direction_reproduces_7a_bug():
    """The Phase 7A bug (negative sum sorted descending) picks the worst items."""
    left = np.array([10.0, 5.0, 1.0])
    right = np.array([9.0, 4.0, 2.0])
    # Old (buggy) implementation.
    rl = stats.rankdata(-left, method="average")
    rr = stats.rankdata(-right, method="average")
    buggy_score = -(rl + rr)
    buggy_selected = np.argsort(buggy_score, kind="stable")[:2].tolist()
    assert buggy_selected == [2, 1]


def test_ndcg_ideal_uses_full_universe():
    """NDCG must be < 1 when the selected set is not the universe's best."""
    left = np.array([4.0, 3.0, 2.0, 1.0])
    right = np.array([4.0, 3.0, 2.0, 1.0])
    selected = [0, 3]  # index 0 is best, index 3 is worst
    metrics = portfolio_metrics(selected, left, right, q_star=0.5)
    assert metrics["worst_source_ndcg"] < 1.0


def test_minimax_recall_regret_equals_one_minus_worst_recall():
    """With budget >= target, a perfect portfolio has zero recall regret."""
    n = 20
    scores = np.arange(n, 0, -1, dtype=np.float64)
    selected = portfolio_select(
        "average_percentile", scores, scores, q_star=0.2, budget_factor=1.0
    )
    metrics = portfolio_metrics(selected, scores, scores, q_star=0.2)
    assert metrics["worst_source_recall"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["minimax_recall_regret"] == pytest.approx(0.0, abs=1e-9)


def test_anion_group_no_substring_confusion():
    """Composition-based parsing must distinguish Se/S and Ho/Co/O."""
    assert anion_group("LiAlSe2") == "selenide"
    assert anion_group("CoS2") == "sulfide"
    assert anion_group("HoCoO3") == "oxide"
    assert anion_group("NaClO3") == "halide_oxide"


def test_portfolio_select_deterministic_and_exact_budget():
    """Selection is deterministic and returns exactly the computed budget."""
    rng = np.random.default_rng(5)
    left = rng.normal(size=100)
    right = left + rng.normal(scale=0.5, size=100)
    sel1 = portfolio_select("maximin_percentile", left, right, q_star=0.1, budget_factor=1.5)
    sel2 = portfolio_select("maximin_percentile", left, right, q_star=0.1, budget_factor=1.5)
    assert sel1 == sel2
    target_k = max(1, int(np.floor(0.1 * 100)))
    budget = min(100, int(round(1.5 * target_k)))
    assert len(sel1) == budget
    assert len(sel1) == len(set(sel1))


def test_grouped_bootstrap_ci_wider_than_naive():
    """Grouped resampling must produce a wider tau CI than naive row resampling."""
    rng = np.random.default_rng(11)
    n_groups = 20
    group_means = rng.normal(size=n_groups)
    groups = np.repeat(np.arange(n_groups), 5)
    left = group_means[groups] + rng.normal(scale=0.5, size=len(groups))
    right = group_means[groups] + rng.normal(scale=0.5, size=len(groups))

    _, g_lo, g_hi = grouped_paired_bootstrap_tau(
        left, right, groups, n_boot=1000, seed=22
    )
    g_width = g_hi - g_lo

    # Naive percentile bootstrap (resample rows).
    reps = []
    n = len(left)
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        tau, _ = stats.kendalltau(left[idx], right[idx])
        reps.append(tau)
    reps = np.asarray(reps)
    n_lo, n_hi = np.percentile(reps[np.isfinite(reps)], [2.5, 97.5])
    n_width = n_hi - n_lo

    assert np.isfinite(g_width)
    assert g_width > n_width


def test_disagreement_abstention_tuning_sanity():
    """Tuning the disagreement penalty on a dev split should not hurt holdout."""
    rng = np.random.default_rng(13)
    n = 120
    left = rng.exponential(scale=1.0, size=n)
    right = left + rng.normal(scale=0.4, size=n)
    # Add a few high-disagreement outliers.
    outlier = np.array([0, 1, 2])
    left[outlier] *= 5.0
    groups = np.repeat(np.arange(n // 4), 4)

    dev_mask = np.zeros(n, dtype=bool)
    dev_mask[::2] = True
    hold_mask = ~dev_mask

    lam = tune_disagreement_abstention(
        left[dev_mask],
        right[dev_mask],
        q_star=0.1,
        budget_factor=1.0,
        lambda_grid=[0.0, 0.5, 1.0, 2.0, 4.0],
        groups=groups[dev_mask],
        n_boot=100,
        seed=99,
    )
    sel_tuned = portfolio_select(
        "disagreement_abstention",
        left[hold_mask],
        right[hold_mask],
        q_star=0.1,
        budget_factor=1.0,
        lambda_param=lam,
    )
    sel_zero = portfolio_select(
        "disagreement_abstention",
        left[hold_mask],
        right[hold_mask],
        q_star=0.1,
        budget_factor=1.0,
        lambda_param=0.0,
    )
    rec_tuned = portfolio_metrics(sel_tuned, left[hold_mask], right[hold_mask], q_star=0.1)[
        "worst_source_recall"
    ]
    rec_zero = portfolio_metrics(sel_zero, left[hold_mask], right[hold_mask], q_star=0.1)[
        "worst_source_recall"
    ]
    assert rec_tuned >= rec_zero - 1e-6


def test_benjamini_hochberg_bounds_and_monotonicity():
    pvals = [0.01, 0.04, 0.10, 0.50, 0.001]
    adj = benjamini_hochberg(pvals)
    assert len(adj) == len(pvals)
    assert all(0.0 <= q <= 1.0 for q in adj)
    # Sorted adjusted p-values should be non-decreasing.
    assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))
