"""Red-team tests for Phase 7C reviewer-proofing corrections.

These tests encode the statistical and narrative risks identified in the
Phase 7C review:

* corrected high-response masks avoid pooled collider bias,
* balanced_union at budget factor 2.0 is a coverage upper bound,
* portfolio comparison uses paired-difference confidence intervals,
* control provenance flags suspiciously identical cross-source values,
* screening-resolution bands are integrated separately,
* the v0.6 manuscript avoids forbidden phrases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crosspiezo.analysis.phase7b_stats import naucc, portfolio_metrics, portfolio_select, screening_resolution_curve
from crosspiezo.analysis.phase7c_stats import (
    anchor_high_response_mask,
    conditional_permutation_high_response,
    control_provenance_record,
    dual_high_response_mask,
    full_procedure_portfolio_bootstrap_ci,
    paired_bootstrap_ci_difference,
    partial_naucc,
    portfolio_cross_evaluation,
    portfolio_select,
)


def _make_band_curve(slope: float = 1.0) -> pd.DataFrame:
    """Linear adjusted-Jaccard curve for partial-nAUCC unit tests."""
    qs = np.arange(1, 51, dtype=np.float64)
    return pd.DataFrame(
        {
            "q_percentile": qs,
            "chance_adjusted_jaccard": qs * slope / 50.0,
        }
    )


def test_partial_naucc_over_elite_band():
    """partial_naucc integrates only the requested quantile band."""
    curve = _make_band_curve(slope=1.0)
    # y = q / 50; average over q in [1, 10] is 5.5 / 50.
    value = partial_naucc(curve, q_min=1.0, q_max=10.0)
    assert value == pytest.approx(5.5 / 50.0, abs=1e-9)


def test_partial_naucc_constant_curve():
    """A constant curve has the same partial nAUCC in every band."""
    qs = np.arange(1, 51, dtype=np.float64)
    curve = pd.DataFrame({"q_percentile": qs, "chance_adjusted_jaccard": np.ones(50)})
    assert partial_naucc(curve, 1.0, 10.0) == pytest.approx(1.0, abs=1e-9)
    assert partial_naucc(curve, 20.0, 50.0) == pytest.approx(1.0, abs=1e-9)


def test_minimax_oracle_balances_one_sided_elites_at_fixed_budget():
    left = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    right = [6.0, 5.0, 1.0, 0.0, 4.0, 3.0]

    selected = portfolio_select("minimax_oracle", left, right, q_star=0.5, budget_factor=1.0)
    metrics = portfolio_metrics(selected, left, right, q_star=0.5)

    assert len(selected) == 3
    assert metrics["worst_source_recall"] == pytest.approx(2 / 3)


def test_minimax_oracle_is_not_below_balanced_union():
    left = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    right = [6.0, 5.0, 1.0, 0.0, 4.0, 3.0]

    oracle = portfolio_metrics(
        portfolio_select("minimax_oracle", left, right, q_star=0.5, budget_factor=1.0),
        left,
        right,
        q_star=0.5,
    )
    heuristic = portfolio_metrics(
        portfolio_select("balanced_union", left, right, q_star=0.5, budget_factor=1.0),
        left,
        right,
        q_star=0.5,
    )

    assert oracle["worst_source_recall"] >= heuristic["worst_source_recall"]


def test_dual_high_avoids_extreme_high_low_outliers():
    """Dual-high selection is less distorted by high-low cross-source outliers.

    Pooled selection can include pairs where one source is elite and the other
    is not, inflating apparent disagreement.  Dual-high excludes those pairs.
    """
    rng = np.random.default_rng(7)
    # Strongly correlated bulk plus a few high-disagreement outliers.
    base = rng.exponential(scale=1.0, size=100)
    left = base + rng.normal(scale=0.05, size=100)
    right = base + rng.normal(scale=0.05, size=100)
    outliers = np.array([0, 1, 2])
    left[outliers] *= 5.0

    pooled = 0.5 * (left + right)
    pooled_mask = pooled >= np.quantile(pooled, 0.90)
    dual_mask = dual_high_response_mask(left, right, fraction=0.10)

    pooled_curve = screening_resolution_curve(
        left[pooled_mask], right[pooled_mask], np.arange(1, 51, dtype=float), n_boot=0
    )
    dual_curve = screening_resolution_curve(
        left[dual_mask], right[dual_mask], np.arange(1, 51, dtype=float), n_boot=0
    )
    # Dual-high must be at least as concordant as pooled here.
    assert naucc(dual_curve) >= naucc(pooled_curve) - 1e-9


def test_anchor_mask_selects_by_anchor_only():
    """The anchor mask selects exactly the top fraction by the anchor source."""
    rng = np.random.default_rng(9)
    left = rng.exponential(scale=1.0, size=80)
    right = rng.exponential(scale=1.0, size=80)
    mask = anchor_high_response_mask(left, right, fraction=0.10)
    assert mask.sum() == 8
    # All selected left values exceed any non-selected left value.
    assert left[mask].min() >= left[~mask].max()


def test_conditional_permutation_returns_exploratory_stats():
    """Conditional permutation for pooled selection reports observed nAUCC and p."""
    rng = np.random.default_rng(11)
    left = rng.exponential(scale=1.0, size=60)
    right = left + rng.normal(scale=0.3, size=60)
    result = conditional_permutation_high_response(
        left, right, fraction=0.20, n_perm=199, seed=11
    )
    assert result["n_selected"] == 12
    assert 0.0 <= result["p_value"] <= 1.0
    assert np.isfinite(result["observed_naucc"])


def test_balanced_union_budget_two_has_full_recall():
    """balanced_union at budget factor 2.0 covers both source elite sets."""
    rng = np.random.default_rng(3)
    n = 100
    left = rng.exponential(scale=1.0, size=n)
    right = rng.exponential(scale=1.0, size=n)
    selected = portfolio_select("balanced_union", left, right, q_star=0.10, budget_factor=2.0)
    metrics = portfolio_metrics(selected, left, right, q_star=0.10)
    assert metrics["worst_source_recall"] == pytest.approx(1.0, abs=1e-9)


def test_paired_bootstrap_ci_difference_detects_gain():
    """paired_bootstrap_ci_difference returns a positive CI when a > b in every group."""
    rng = np.random.default_rng(5)
    n_groups = 15
    groups = np.repeat(np.arange(n_groups), 6)
    a = groups.astype(float) + rng.normal(scale=0.1, size=len(groups))
    b = groups.astype(float) - 1.0 + rng.normal(scale=0.1, size=len(groups))
    point, lo, hi = paired_bootstrap_ci_difference(a, b, groups, n_boot=1000, seed=22)
    assert point > 0.5
    assert lo > 0.0
    assert hi > point


def test_portfolio_cross_evaluation_has_paired_differences():
    """Cross-evaluation returns paired-difference columns versus single-source baselines."""
    rng = np.random.default_rng(13)
    n = 120
    left = rng.exponential(scale=1.0, size=n)
    right = left + rng.normal(scale=0.4, size=n)
    groups = np.repeat(np.arange(n // 4), 4)
    pair_ids = [f"p{i}" for i in range(n)]
    df = portfolio_cross_evaluation(
        left,
        right,
        q_star=0.10,
        budget_factor=1.0,
        strategies=["average_percentile", "maximin_percentile"],
        groups=groups,
        pair_ids=pair_ids,
        n_folds=5,
        n_inner_boot=50,
    )
    assert {"fold", "strategy", "paired_diff_recall", "paired_diff_ndcg"}.issubset(df.columns)
    assert not df["paired_diff_recall"].isna().any()


def test_control_provenance_flags_identical_values():
    """>5% identical finite cross-source values raise the copy flag."""
    jarvis = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rec = control_provenance_record(
        attr_name="dummy",
        jarvis_col="jv_col",
        mp_col="mp_col",
        jarvis_source_db="JARVIS",
        mp_source_db="MP",
        unit="u",
        calculation_type_version="DFT/PBE",
        jarvis_arr=jarvis,
        mp_arr=mp,
    )
    assert rec["same_field_copy_flag"] is True
    assert rec["fraction_identical"] == pytest.approx(1.0, abs=1e-9)


def test_control_provenance_no_flag_below_threshold():
    """Exactly one identical pair out of 100 stays below the 5% threshold."""
    jarvis = np.arange(100, dtype=float)
    mp = jarvis.copy()
    mp[1:] += 0.1
    rec = control_provenance_record(
        attr_name="dummy",
        jarvis_col="jv_col",
        mp_col="mp_col",
        jarvis_source_db="JARVIS",
        mp_source_db="MP",
        unit="u",
        calculation_type_version="DFT/PBE",
        jarvis_arr=jarvis,
        mp_arr=mp,
    )
    assert rec["same_field_copy_flag"] is False


def test_current_manuscript_avoids_forbidden_phrases():
    """The consolidated manuscript must contain required sections and avoid forbidden claims."""
    tex_path = Path(__file__).resolve().parents[2] / "CrossPiezo_ScreeningResolution_Manuscript.tex"
    assert tex_path.exists(), "consolidated manuscript not found"
    text = tex_path.read_text(encoding="utf-8")

    required_sections = [
        r"\section{Introduction}",
        r"\section{Discussion}",
        r"\section{Methods}",
        r"\section{Data availability}",
        r"\bibliography",
    ]
    for section in required_sections:
        assert section in text, f"missing required section: {section}"

    forbidden = [
        "large fraction stems from conventions",
        "high-response negative nAUCC confirms",
        "balanced_union=1.0",
        "true tensor",
        "ground truth consensus",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in text.lower(), f"forbidden phrase present: {phrase}"


def _make_portfolio_data(rng: np.random.Generator, n: int = 80):
    """Correlated left/right scores with a few groups."""
    base = rng.exponential(scale=1.0, size=n)
    left = base + rng.normal(scale=0.1, size=n)
    right = base + rng.normal(scale=0.1, size=n)
    groups = np.repeat(np.arange(n // 4), 4)[:n]
    return left, right, groups


def test_full_procedure_point_equals_arithmetic_difference():
    """The reported point equals the arithmetic recall difference on the full panel."""
    rng = np.random.default_rng(21)
    left, right, groups = _make_portfolio_data(rng)
    df = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["average_percentile", "balanced_union"],
        groups=groups, n_boot=500, seed=31,
    )
    # Manual full-panel check for balanced_union.
    sel = portfolio_select("balanced_union", left, right, q_star=0.10, budget_factor=1.0)
    rec = portfolio_metrics(sel, left, right, q_star=0.10)["worst_source_recall"]
    jarvis_rec = portfolio_metrics(
        portfolio_select("jarvis_only", left, right, q_star=0.10, budget_factor=1.0),
        left, right, q_star=0.10,
    )["worst_source_recall"]
    expected_delta_j = rec - jarvis_rec
    row = df[(df["strategy"] == "balanced_union") & (df["delta_type"] == "delta_j")]
    assert len(row) == 1
    assert float(row["point"].iloc[0]) == pytest.approx(expected_delta_j, abs=1e-9)


def test_full_procedure_same_strategy_has_zero_ci():
    """When strategy == JARVIS-only, delta_j point and CI are exactly zero."""
    rng = np.random.default_rng(23)
    left, right, groups = _make_portfolio_data(rng)
    df = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["jarvis_only"],
        groups=groups, n_boot=500, seed=33,
    )
    row = df[(df["strategy"] == "jarvis_only") & (df["delta_type"] == "delta_j")]
    assert float(row["point"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(row["ci95_low"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(row["ci95_high"].iloc[0]) == pytest.approx(0.0, abs=1e-9)


def test_full_procedure_seed_reproducible():
    """Two runs with the same seed produce identical CIs."""
    rng = np.random.default_rng(25)
    left, right, groups = _make_portfolio_data(rng)
    df1 = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["balanced_union"], groups=groups, n_boot=300, seed=41,
    )
    df2 = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["balanced_union"], groups=groups, n_boot=300, seed=41,
    )
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_full_procedure_duplicate_groups_expand_sample():
    """Duplicated groups create distinct bootstrap identities and do not collapse."""
    rng = np.random.default_rng(27)
    left, right, groups = _make_portfolio_data(rng, n=80)
    df = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["balanced_union"], groups=groups, n_boot=200, seed=43,
    )
    assert len(df) == 3  # delta_j, delta_m, delta_best
    assert all(np.isfinite(df["point"]))
    assert all(np.isfinite(df["ci95_low"]))
    assert all(np.isfinite(df["ci95_high"]))


def test_full_procedure_delta_best_non_negative():
    """Delta versus the better single source is never negative for a sensible strategy."""
    rng = np.random.default_rng(29)
    left, right, groups = _make_portfolio_data(rng)
    df = full_procedure_portfolio_bootstrap_ci(
        left, right, q_star=0.10, budget_factor=1.0,
        strategies=["balanced_union"], groups=groups, n_boot=500, seed=47,
    )
    row = df[(df["strategy"] == "balanced_union") & (df["delta_type"] == "delta_best")]
    assert float(row["point"].iloc[0]) >= -1e-9
