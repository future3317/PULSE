"""Valid protocol-to-model ratio (PMR) computation for Phase 5B."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crosspiezo.analysis.discrepancy import bootstrap_ci


def _median_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    point, lo, hi = bootstrap_ci(values, statistic="median", seed=seed)
    return point, lo, hi


def _mean_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    point, lo, hi = bootstrap_ci(values, statistic="mean", seed=seed)
    return point, lo, hi


def _ratio_bootstrap(
    discrepancies: np.ndarray,
    errors: np.ndarray,
    n_replicates: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a ratio of paired statistics.

    The same bootstrap index is applied to both ``discrepancies`` and ``errors``
    so that the joint pairing is preserved in each replicate.
    """
    discs = np.asarray(discrepancies, dtype=np.float64)
    errs = np.asarray(errors, dtype=np.float64)
    n = min(len(discs), len(errs))
    if n < 2:
        return float("nan"), float("nan")
    discs = discs[:n]
    errs = errs[:n]
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_replicates, dtype=np.float64)
    for i in range(n_replicates):
        idx = rng.choice(n, size=n, replace=True)
        d = discs[idx]
        e = errs[idx]
        ratios[i] = float(np.mean(d) / (np.mean(e) + 1e-12))
    alpha = 1.0 - confidence
    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1.0 - alpha / 2)))
    return lo, hi


def compute_valid_models(
    metrics: pd.DataFrame,
    baseline_names: list[str] | None = None,
) -> pd.DataFrame:
    """Flag models that beat zero and composition-mean baselines in-source."""
    if baseline_names is None:
        baseline_names = ["zero", "composition_mean"]

    # Aggregate over seeds.
    agg = metrics.groupby(["model_name", "train_source", "eval_source", "split_type"]).agg(
        absolute_frobenius_mae_mean=("absolute_frobenius_mae", "mean"),
        absolute_frobenius_mae_std=("absolute_frobenius_mae", "std"),
        normalized_frobenius_mae_mean=("normalized_frobenius_mae", "mean"),
        normalized_frobenius_mae_std=("normalized_frobenius_mae", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    # Baseline thresholds per eval_source and split_type.
    baseline_df = agg[agg["model_name"].isin(baseline_names)].copy()
    baseline_thresholds = baseline_df.groupby(["eval_source", "split_type"])["absolute_frobenius_mae_mean"].min().to_dict()

    def _valid(row: pd.Series) -> bool:
        key = (row["eval_source"], row["split_type"])
        in_source = row["split_type"] in ("in_source_jarvis", "in_source_mp")
        return in_source and row["absolute_frobenius_mae_mean"] < baseline_thresholds.get(key, float("inf"))

    agg["valid"] = agg.apply(_valid, axis=1)
    return agg


def compute_pmr_table(
    metrics: pd.DataFrame,
    paired_discrepancies: dict[str, np.ndarray],
    valid_models: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute mean/mean and median/median PMR for each valid model and scope."""
    if valid_models is None:
        valid_models = compute_valid_models(metrics)

    in_source = valid_models[valid_models["split_type"].isin(["in_source_jarvis", "in_source_mp"]) & valid_models["valid"]]

    rows: list[dict[str, Any]] = []
    for (model_name, train_src), _model_grp in in_source.groupby(["model_name", "train_source"]):
        # In-source error: mean and median across eval sources/seeds.
        abs_mae_values = metrics[
            (metrics["model_name"] == model_name)
            & (metrics["train_source"] == train_src)
            & (metrics["split_type"].isin(["in_source_jarvis", "in_source_mp"]))
        ]["absolute_frobenius_mae"].dropna().values
        norm_mae_values = metrics[
            (metrics["model_name"] == model_name)
            & (metrics["train_source"] == train_src)
            & (metrics["split_type"].isin(["in_source_jarvis", "in_source_mp"]))
        ]["normalized_frobenius_mae"].dropna().values
        if len(abs_mae_values) == 0:
            continue
        mean_abs_err = float(np.mean(abs_mae_values))
        median_abs_err = float(np.median(abs_mae_values))
        mean_norm_err = float(np.mean(norm_mae_values))
        median_norm_err = float(np.median(norm_mae_values))

        for scope_name, discs in paired_discrepancies.items():
            discs = np.asarray(discs, dtype=np.float64)
            if len(discs) == 0:
                continue
            mean_disc = float(np.mean(discs))
            median_disc = float(np.median(discs))
            _, mean_lo, mean_hi = _mean_ci(discs)
            _, median_lo, median_hi = _median_ci(discs)

            mean_pmr = mean_disc / (mean_abs_err + 1e-12)
            median_pmr = median_disc / (median_abs_err + 1e-12)
            mean_pmr_lo, mean_pmr_hi = _ratio_bootstrap(discs, abs_mae_values, seed=42)
            median_pmr_lo, median_pmr_hi = _ratio_bootstrap(discs, abs_mae_values, seed=43)

            rows.append({
                "model_name": model_name,
                "train_source": train_src,
                "scope": scope_name,
                "n_pairs": len(discs),
                "mean_paired_discrepancy": mean_disc,
                "mean_paired_ci95_low": mean_lo,
                "mean_paired_ci95_high": mean_hi,
                "median_paired_discrepancy": median_disc,
                "median_paired_ci95_low": median_lo,
                "median_paired_ci95_high": median_hi,
                "mean_in_source_mae": mean_abs_err,
                "median_in_source_mae": median_abs_err,
                "PMR_mean_absolute": mean_pmr,
                "PMR_median_absolute": median_pmr,
                "PMR_mean_normalized": mean_disc / (mean_norm_err + 1e-12),
                "PMR_median_normalized": median_disc / (median_norm_err + 1e-12),
                "PMR_mean_absolute_ci95_low": mean_pmr_lo,
                "PMR_mean_absolute_ci95_high": mean_pmr_hi,
                "PMR_median_absolute_ci95_low": median_pmr_lo,
                "PMR_median_absolute_ci95_high": median_pmr_hi,
                "per_sample_discrepancy": np.asarray(discs, dtype=np.float64),
                "per_sample_model_error": np.asarray(abs_mae_values, dtype=np.float64),
            })

    return pd.DataFrame(rows)


def compute_spg(
    paired_discrepancies: dict[str, np.ndarray],
    in_source_errors: list[float],
) -> dict[str, float]:
    """Smallest protocol gap relative to the best valid in-source error."""
    best = min(in_source_errors) if in_source_errors else float("nan")
    return {
        scope: float(np.median(discs) / (best + 1e-12))
        for scope, discs in paired_discrepancies.items()
    }
