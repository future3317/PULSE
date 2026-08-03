"""Unit tests for top-k / listwise baseline ranking diagnostics.

These metrics are used to argue that screening resolution captures tail-specific
disagreement not visible to global rank-correlation measures.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from crosspiezo.analysis.ranking import (
    overlap_coefficient_at_k,
    plain_jaccard_at_k,
    precision_at_k,
    rank_biased_overlap,
    recall_at_k,
    top_weighted_kendall_tau,
    top_weighted_spearman_rho,
)


def test_perfect_agreement_gives_perfect_top_k_metrics():
    """Identical rankings should yield perfect top-k overlap."""
    scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    for k in [1, 3, 5, 10]:
        assert precision_at_k(scores, scores, k) == pytest.approx(1.0)
        assert recall_at_k(scores, scores, k) == pytest.approx(1.0)
        assert plain_jaccard_at_k(scores, scores, k) == pytest.approx(1.0)
        assert overlap_coefficient_at_k(scores, scores, k) == pytest.approx(1.0)
    # Top-weighted tau/rho need at least two items in the union.
    for k in [3, 5, 10]:
        assert top_weighted_kendall_tau(scores, scores, k) == pytest.approx(1.0)
        assert top_weighted_spearman_rho(scores, scores, k) == pytest.approx(1.0)


def test_disjoint_top_k_sets_have_zero_overlap():
    """Two rankings whose top-k sets are disjoint score zero on overlap metrics."""
    left = np.array([10.0, 9.0, 8.0, 0.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 10.0, 9.0, 8.0])
    k = 3
    assert precision_at_k(left, right, k) == pytest.approx(0.0)
    assert plain_jaccard_at_k(left, right, k) == pytest.approx(0.0)
    assert overlap_coefficient_at_k(left, right, k) == pytest.approx(0.0)


def test_partial_overlap_gives_expected_jaccard_and_overlap():
    """When top-k sets share two of three items, Jaccard = 0.5 and overlap = 2/3."""
    left = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    right = np.array([10.0, 9.0, 0.0, 8.0, 0.0, 0.0])
    k = 3
    assert overlap_coefficient_at_k(left, right, k) == pytest.approx(2.0 / 3.0)
    assert precision_at_k(left, right, k) == pytest.approx(2.0 / 3.0)
    assert plain_jaccard_at_k(left, right, k) == pytest.approx(0.5)


def test_top_weighted_tau_on_perfect_tail_agreement():
    """If the top-k unions agree on relative ordering, tail tau/rho are 1."""
    left = np.array([10.0, 9.0, 8.0, 1.0, 0.5, 0.1])
    right = np.array([10.5, 9.5, 8.5, 1.2, 0.4, 0.2])
    assert top_weighted_kendall_tau(left, right, k=3) == pytest.approx(1.0)
    assert top_weighted_spearman_rho(left, right, k=3) == pytest.approx(1.0)


def test_rbo_perfect_agreement_is_one():
    """RBO of identical rankings is 1 for any valid p."""
    scores = np.arange(20, 0, -1, dtype=float)
    for p in [0.5, 0.8, 0.95, 0.99]:
        assert rank_biased_overlap(scores, scores, p=p) == pytest.approx(1.0, abs=1e-9)


def test_rbo_monotonic_in_p_for_noisy_rankings():
    """As p increases, RBO puts more weight on deeper ranks and should not decrease."""
    rng = np.random.default_rng(7)
    left = rng.exponential(scale=1.0, size=50)
    right = left + rng.normal(scale=0.3, size=50)
    rbo_low = rank_biased_overlap(left, right, p=0.5)
    rbo_high = rank_biased_overlap(left, right, p=0.99)
    assert 0.0 <= rbo_low <= 1.0
    assert 0.0 <= rbo_high <= 1.0
    assert rbo_high >= rbo_low - 1e-9


def test_rbo_reversed_ranking_decreases_with_p():
    """For reversed rankings, lower p gives more top-weighted penalty."""
    left = np.arange(10, 0, -1, dtype=float)
    right = np.arange(1, 11, dtype=float)
    rbo_top = rank_biased_overlap(left, right, p=0.5)
    rbo_tail = rank_biased_overlap(left, right, p=0.95)
    assert rbo_top < rbo_tail
    assert rbo_top < 0.2


def test_metrics_handle_nan_pairs():
    """Finite-only masks are applied before computing metrics."""
    left = np.array([10.0, 9.0, np.nan, 7.0, 6.0])
    right = np.array([10.0, 8.0, 8.5, np.nan, 6.0])
    # After dropping nan at indices 2 and 3, the valid universe is indices 0,1,4.
    # Top-3 of both sides is the whole universe; overlap is complete.
    assert precision_at_k(left, right, k=3) == pytest.approx(1.0)
    assert plain_jaccard_at_k(left, right, k=3) == pytest.approx(1.0)


def test_global_tau_can_be_high_while_tail_metrics_are_low():
    """The motivating example: high global correlation but disjoint elite decisions."""
    n = 100
    left = np.arange(n, 0, -1, dtype=float)
    # Swap the top-5 block with the next-5 block in right. Global tau stays high,
    # but the top-5 selected sets are disjoint.
    right = left.copy()
    right[:10] = left[:10].copy()
    right[:5], right[5:10] = right[5:10].copy(), right[:5].copy()
    tau_global, _ = stats.kendalltau(left, right)
    assert tau_global > 0.9
    assert precision_at_k(left, right, k=5) == pytest.approx(0.0)
    assert top_weighted_kendall_tau(left, right, k=5) < 0.5


def test_overlap_coefficient_bounds():
    """Overlap coefficient is bounded in [0,1]."""
    rng = np.random.default_rng(11)
    left = rng.exponential(scale=1.0, size=40)
    right = rng.exponential(scale=1.0, size=40)
    for k in [1, 5, 10, 40]:
        val = overlap_coefficient_at_k(left, right, k)
        assert 0.0 <= val <= 1.0 or np.isnan(val)


def test_top_weighted_metrics_return_nan_for_insufficient_subset():
    """If the top-k union has fewer than 2 items, tau/rho are undefined."""
    left = np.array([1.0, 0.0])
    right = np.array([0.9, 0.0])
    assert np.isnan(top_weighted_kendall_tau(left, right, k=1))
    assert np.isnan(top_weighted_spearman_rho(left, right, k=1))
