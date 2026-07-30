"""Tests for Phase 6B corrected null statistics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from crosspiezo.analysis.ranking import (
    chance_adjusted_jaccard,
    cohen_kappa,
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
    matthews_correlation,
    permutation_pvalue,
)


def test_expected_jaccard_hypergeometric_against_monte_carlo():
    """Exact E[J] must agree with a Monte Carlo estimate."""
    n, k = 200, 20
    rng = np.random.default_rng(123)
    jaccards = []
    for _ in range(10000):
        top_a = set(rng.choice(n, size=k, replace=False))
        top_b = set(rng.choice(n, size=k, replace=False))
        inter = len(top_a & top_b)
        union = len(top_a | top_b)
        jaccards.append(inter / union if union else 0.0)
    mc_mean = float(np.mean(jaccards))
    exact = expected_jaccard_hypergeometric(n, k)
    assert exact == pytest.approx(mc_mean, abs=1e-3)


def test_expected_jaccard_hypergeometric_matches_simple_approx_for_large_n():
    """For large n, E[J] ~ k/(2n-k)."""
    n, k = 10000, 100
    exact = expected_jaccard_hypergeometric(n, k)
    approx = k / (2.0 * n - k)
    assert exact == pytest.approx(approx, rel=1e-2)


def test_chance_adjusted_jaccard_zero_when_observed_equals_expected():
    assert chance_adjusted_jaccard(0.05, 0.05) == pytest.approx(0.0, abs=1e-9)


def test_chance_adjusted_jaccard_one_when_observed_is_one():
    assert chance_adjusted_jaccard(1.0, 0.05) == pytest.approx(1.0, abs=1e-9)


def test_hypergeometric_overlap_pvalue_bounds():
    n, k, x = 100, 10, 0
    p = hypergeometric_overlap_pvalue(n, k, x)
    assert 0.0 <= p <= 1.0


def test_permutation_pvalue_uniform_null():
    """For uncorrelated data the two-sided permutation p-value should not be tiny."""
    rng = np.random.default_rng(42)
    left = rng.normal(size=50)
    right = rng.normal(size=50)
    tau, _ = stats.kendalltau(left, right)
    p = permutation_pvalue(left, right, float(tau), n_permutations=999, seed=42)
    assert p > 0.05


def test_permutation_pvalue_small_for_perfect_correlation():
    left = np.arange(20, dtype=np.float64)
    right = left.copy()
    tau, _ = stats.kendalltau(left, right)
    p = permutation_pvalue(left, right, float(tau), n_permutations=199, seed=42)
    assert p <= 0.01


def test_cohen_kappa_perfect_agreement():
    a = np.array([True, False, True, True])
    assert cohen_kappa(a, a) == pytest.approx(1.0, abs=1e-9)


def test_cohen_kappa_random_guess():
    """For independent balanced labels kappa ~ 0."""
    rng = np.random.default_rng(42)
    a = rng.random(200) > 0.5
    b = rng.random(200) > 0.5
    k = cohen_kappa(a, b)
    assert -0.15 < k < 0.15


def test_matthews_correlation_perfect_agreement():
    a = np.array([True, False, True, True])
    assert matthews_correlation(a, a) == pytest.approx(1.0, abs=1e-9)
