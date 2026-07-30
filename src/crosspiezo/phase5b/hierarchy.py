"""Discrepancy hierarchy for Phase 5B."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crosspiezo.analysis.discrepancy import absolute_discrepancy, discrepancy_summary, normalized_discrepancy
from crosspiezo.analysis.o3_transport import (
    domain_aware_discrepancy,
    exact_transported_discrepancy,
    point_group_equivalent_discrepancy,
    proper_orbit_discrepancy,
    symmetry_projected_discrepancy,
)
from crosspiezo.conventions.symmetry import point_group_rotations, project_piezo_tensor
from crosspiezo.phase5b.panels import _space_group_symbol


VARIANTS = [
    "source_native_invariant",
    "exact_transported",
    "proper_orbit",
    "domain_aware",
    "point_group_equivalent",
    "symmetry_projected",
]


def compute_hierarchy(enriched: pd.DataFrame) -> pd.DataFrame:
    """Compute discrepancy variants for every pair."""
    records: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        jid = row["jarvis_id"]
        mid = row["mp_id"]
        left = row["jarvis_tensor"]
        right = row["mp_tensor_raw"]
        rot = row["rotation"]
        sg = _space_group_symbol(row["space_group"])

        # Source-native invariant: compare Frobenius norms (frame independent).
        jnorm = float(np.linalg.norm(left))
        mnorm = float(np.linalg.norm(row["mp_tensor_aligned"]))
        records.append({
            "jarvis_id": jid,
            "mp_id": mid,
            "formula": row["formula"],
            "sublayer": row.get("sublayer", "unknown"),
            "variant": "source_native_invariant",
            "absolute": abs(jnorm - mnorm),
            "normalized": 2 * abs(jnorm - mnorm) / (jnorm + mnorm + 1e-12),
            "sign_flip_fraction": 0.0,
            "cosine_similarity": 1.0,
            "amplitude_ratio": mnorm / (jnorm + 1e-12),
            "rotation_available": 1.0 if rot is not None else 0.0,
            "polar_domain_flip": 0.0,
            "norm_retention": 1.0,
        })

        exact = exact_transported_discrepancy(left, right, rot)
        domain = domain_aware_discrepancy(left, right, rot)
        proper = proper_orbit_discrepancy(left, right, sg) if sg else _nan_variant()
        pg_equiv = point_group_equivalent_discrepancy(left, right, sg) if sg else _nan_variant()
        sym_proj = symmetry_projected_discrepancy(left, right, sg, rot) if sg else _nan_variant()

        for name, vals, retention in [
            ("exact_transported", exact, 1.0),
            ("proper_orbit", proper, 1.0),
            ("domain_aware", domain, 1.0),
            ("point_group_equivalent", pg_equiv, 1.0),
            ("symmetry_projected", sym_proj, _norm_retention(left, right, sg, rot)),
        ]:
            records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "formula": row["formula"],
                "sublayer": row.get("sublayer", "unknown"),
                "variant": name,
                "absolute": vals["absolute"],
                "normalized": vals["normalized"],
                "sign_flip_fraction": vals.get("sign_flip_fraction", float("nan")),
                "cosine_similarity": vals.get("cosine_similarity", float("nan")),
                "amplitude_ratio": vals.get("amplitude_ratio", float("nan")),
                "rotation_available": vals.get("rotation_available", 0.0),
                "polar_domain_flip": vals.get("polar_domain_flip", 0.0),
                "norm_retention": retention,
            })

    return pd.DataFrame(records)


def _nan_variant() -> dict[str, float]:
    return {
        "absolute": float("nan"),
        "normalized": float("nan"),
        "sign_flip_fraction": float("nan"),
        "cosine_similarity": float("nan"),
        "amplitude_ratio": float("nan"),
        "rotation_available": 0.0,
    }


def _norm_retention(left: np.ndarray, right: np.ndarray, sg: str | None, rot: np.ndarray | None) -> float:
    if sg is None:
        return float("nan")
    try:
        rots = point_group_rotations(sg)
    except Exception:  # noqa: BLE001
        return float("nan")
    right_t = right
    if rot is not None:
        right_t = np.einsum("il,jm,kn,lmn->ijk", rot, rot, rot, right)
    left_proj = project_piezo_tensor(left, rots)
    right_proj = project_piezo_tensor(right_t, rots)
    num = float(np.linalg.norm(left_proj) + np.linalg.norm(right_proj))
    denom = float(np.linalg.norm(left) + np.linalg.norm(right_t)) + 1e-12
    return num / denom


def hierarchy_summary(hierarchy_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize discrepancy variants with non-parametric statistics."""
    rows: list[dict[str, Any]] = []
    for variant, grp in hierarchy_df.groupby("variant"):
        abs_vals = grp["absolute"].dropna().values
        norm_vals = grp["normalized"].dropna().values
        if len(abs_vals) == 0:
            continue
        abs_summary = discrepancy_summary(abs_vals, name="absolute")
        norm_summary = discrepancy_summary(norm_vals, name="normalized")
        rows.append({
            "variant": variant,
            "n": len(abs_vals),
            "median_absolute": abs_summary["absolute_median"],
            "median_absolute_ci95_low": abs_summary["absolute_median_ci95_low"],
            "median_absolute_ci95_high": abs_summary["absolute_median_ci95_high"],
            "p95_absolute": float(np.percentile(abs_vals, 95)),
            "median_normalized": norm_summary["normalized_median"],
            "median_normalized_ci95_low": norm_summary["normalized_median_ci95_low"],
            "median_normalized_ci95_high": norm_summary["normalized_median_ci95_high"],
            "p95_normalized": float(np.percentile(norm_vals, 95)),
            "mean_sign_flip_fraction": float(np.nanmean(grp["sign_flip_fraction"].values)),
            "mean_polar_domain_flip": float(np.nanmean(grp["polar_domain_flip"].values)),
            "mean_norm_retention": float(np.nanmean(grp["norm_retention"].values)),
        })
    return pd.DataFrame(rows)


def hierarchy_by_sublayer(hierarchy_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, sublayer), grp in hierarchy_df.groupby(["variant", "sublayer"]):
        abs_vals = grp["absolute"].dropna().values
        norm_vals = grp["normalized"].dropna().values
        if len(abs_vals) == 0:
            continue
        rows.append({
            "variant": variant,
            "sublayer": sublayer,
            "n": len(abs_vals),
            "median_absolute": float(np.median(abs_vals)),
            "median_normalized": float(np.median(norm_vals)),
        })
    return pd.DataFrame(rows)
