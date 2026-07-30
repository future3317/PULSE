"""Red tests for C-15: PMR must use consistent per-sample statistics and paired
ratio bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosspiezo.phase5b.pmr import compute_pmr_table


def _synth_metrics() -> pd.DataFrame:
    rng = np.random.default_rng(701)
    rows = []
    for model in ["ridge", "mlp"]:
        for src in ["jarvis", "mp"]:
            for seed in range(3):
                rows.append({
                    "model_name": model,
                    "train_source": src,
                    "eval_source": src,
                    "split_type": f"in_source_{src}",
                    "seed": seed,
                    "absolute_frobenius_mae": rng.uniform(0.3, 0.7),
                    "normalized_frobenius_mae": rng.uniform(0.2, 0.5),
                })
    return pd.DataFrame(rows)


def test_pmr_is_ratio_of_consistent_statistics():
    """PMR_mean_absolute must equal mean(disc) / mean(err), not mix scopes."""
    metrics = _synth_metrics()
    discs = np.array([1.0, 2.0, 3.0, 4.0])
    pmr = compute_pmr_table(metrics, {"all": discs})
    assert not pmr.empty
    row = pmr.iloc[0]
    expected = row["mean_paired_discrepancy"] / row["mean_in_source_mae"]
    assert np.isclose(row["PMR_mean_absolute"], expected, atol=1e-6)


def test_pmr_includes_per_sample_fields():
    """The output must expose per-sample discrepancies and model errors."""
    metrics = _synth_metrics()
    discs = np.array([1.0, 2.0, 3.0])
    pmr = compute_pmr_table(metrics, {"all": discs})
    assert "per_sample_discrepancy" in pmr.columns
    assert "per_sample_model_error" in pmr.columns


def test_pmr_bootstrap_resamples_full_ratio():
    """Changing the pairing of discrepancy and model error must change the CI."""
    rng = np.random.default_rng(702)
    errors = rng.uniform(0.3, 0.7, size=20)
    discs_shuffled = rng.permutation(rng.uniform(0.5, 1.5, size=20))
    discs_paired = errors + rng.normal(0.0, 0.05, size=20)

    metrics1 = pd.DataFrame({
        "model_name": ["mlp"] * 20,
        "train_source": ["jarvis"] * 20,
        "eval_source": ["jarvis"] * 20,
        "split_type": ["in_source_jarvis"] * 20,
        "seed": list(range(20)),
        "absolute_frobenius_mae": errors,
        "normalized_frobenius_mae": errors,
    })
    pmr1 = compute_pmr_table(metrics1, {"all": discs_shuffled})
    pmr2 = compute_pmr_table(metrics1, {"all": discs_paired})
    # The CI should differ when the joint distribution differs.
    assert not np.isclose(
        float(pmr1["PMR_mean_absolute_ci95_low"].iloc[0]),
        float(pmr2["PMR_mean_absolute_ci95_low"].iloc[0]),
        atol=1e-3,
    )
