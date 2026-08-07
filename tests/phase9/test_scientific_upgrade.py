import numpy as np

from crosspiezo.analysis.scientific_upgrade import (
    cluster_bootstrap_screening,
    cutoff_diagnostics,
    overlap_bounds_with_ties,
)


def test_overlap_bounds_expose_cutoff_tie_interval():
    result = overlap_bounds_with_ties(
        np.array([4.0, 3.0, 3.0, 1.0]),
        np.array([4.0, 3.0, 2.0, 1.0]),
        k=2,
    )
    assert result["observed_overlap"] == 2
    assert result["min_overlap"] == 1
    assert result["max_overlap"] == 2
    assert result["left_tie_ambiguous"] is True
    assert result["bounds_exact"] is True


def test_cutoff_gap_is_zero_for_an_exact_cutoff_tie():
    result = cutoff_diagnostics(
        np.array([4.0, 3.0, 3.0, 1.0]),
        np.array([4.0, 3.0, 2.0, 1.0]),
        np.array([50.0]),
    )
    row = result.iloc[0]
    assert bool(row["tie_ambiguous"])
    assert row["left_absolute_gap"] == 0.0


def test_cluster_bootstrap_returns_reduced_formula_group_count():
    curve, summary = cluster_bootstrap_screening(
        np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.2]),
        np.array([4.0, 2.5, 2.0, 1.5, 0.4, 0.1]),
        np.array(["A", "A", "B", "C", "D", "E"], dtype=object),
        np.array([10.0, 50.0]),
        n_boot=25,
        seed=7,
    )
    assert len(curve) == 2
    assert summary["n"] == 6
    assert summary["n_groups"] == 5
    assert summary["n_boot_curve"] == 25
    assert np.isfinite(summary["partial_nAUCC_elite"])
    assert summary["partial_nAUCC_elite_n_boot"] == 25
    assert summary["partial_nAUCC_elite_ci95_low"] <= summary["partial_nAUCC_elite_ci95_high"]
    assert np.isfinite(summary["nAUCC"])
