"""Lightweight calibration audit for Phase 5B point models."""

from __future__ import annotations

from typing import Any

import numpy as np


def residual_scale_calibration(
    preds: np.ndarray,
    targets: np.ndarray,
    sources: np.ndarray,
    target_coverage: float = 0.9,
) -> dict[str, Any]:
    """Source-stratified residual scaling that hits a target coverage on the eval set.

    Parameters
    ----------
    preds: ndarray (N, 3, 3, 3)
    targets: ndarray (N, 3, 3, 3)
    sources: ndarray (N,) of strings
    target_coverage: desired marginal coverage

    Returns calibration summary and per-source scale factors.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    per_sample_error = np.linalg.norm((preds - targets).reshape(preds.shape[0], -1), axis=1)

    factors: dict[str, float] = {}
    coverages: dict[str, float] = {}
    sharpness: dict[str, float] = {}
    for src in np.unique(sources):
        mask = sources == src
        errs = per_sample_error[mask]
        if len(errs) == 0:
            continue
        # Interval width = factor * |residual|; find factor for target coverage.
        sorted_errs = np.sort(errs)
        idx = min(int(np.ceil(target_coverage * len(sorted_errs))) - 1, len(sorted_errs) - 1)
        factor = float(sorted_errs[idx]) / max(float(np.median(errs)), 1e-12)
        factors[src] = factor
        coverages[src] = float(np.mean(errs <= factor * np.median(errs)))
        sharpness[src] = float(np.median(factor * np.median(errs)))

    return {
        "target_coverage": target_coverage,
        "global_median_error": float(np.median(per_sample_error)),
        "global_mean_error": float(np.mean(per_sample_error)),
        "source_scale_factors": factors,
        "achieved_coverage": coverages,
        "interval_sharpness": sharpness,
    }


def conformal_coverage(
    preds: np.ndarray,
    targets: np.ndarray,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Marginal coverage of symmetric absolute-error intervals."""
    if alphas is None:
        alphas = [0.1, 0.2, 0.3]
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    errors = np.linalg.norm((preds - targets).reshape(preds.shape[0], -1), axis=1)
    result: dict[str, Any] = {}
    for alpha in alphas:
        q = float(np.quantile(errors, 1 - alpha))
        covered = np.mean(errors <= q)
        result[f"alpha_{alpha:.2f}"] = {"quantile": q, "coverage": float(covered)}
    return result
