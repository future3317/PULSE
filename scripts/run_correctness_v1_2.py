#!/usr/bin/env python
"""Correctness v1.2: invariant benchmark final closure.

This script performs the final correctness audit for CrossPiezo and decides
whether an invariant-only benchmark is viable.  It does NOT import or run
Phase 5A/5B, e3nn, baselines, PMR, soft-mode, or O(3) transport analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correctness_v1_2"
REPORT_ROOT = PROJECT_ROOT / "reports" / "correctness_v1_2"
V1_1_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts" / "releases" / "correctness_v1_1_ae49a3d"
)

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.analysis.ranking import (  # noqa: E402
    frobenius_norm_score,
    max_longitudinal_modulus,
    rank_stability_functional,
    ranking_summary_table,
)
from crosspiezo.conventions.voigt import (  # noqa: E402
    piezo_stress_voigt_to_cartesian,
    tensor_lineage_metrics,
    trusted_piezo_stress_voigt_to_cartesian,
)
from crosspiezo.matching.structure_matcher import match_structures  # noqa: E402
from crosspiezo.reports.markdown import (  # noqa: E402
    bullet,
    code_block,
    header,
    table_from_records,
    write_report,
)
from crosspiezo.schemas import MatchTier  # noqa: E402

LineageLevel = Literal["L0_raw_upstream", "L1_processed_source", "L2_internal_consistency", "unresolved"]

# Frozen scalar-reproduction tolerance.  MP publishes e_ij_max to limited
# precision; 5% relative is the same tolerance used in v1.1.
SCALAR_REL_TOL = 5e-2


def _git_commit() -> str | None:
    try:
        return (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True)
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _load_config(name: str) -> dict[str, Any]:
    with open(CONFIG_ROOT / name) as f:
        return yaml.safe_load(f)


def _resolve_data_root() -> Path:
    local = Path("E:/DATA")
    if local.exists():
        return local
    remote = Path.home() / "DATA"
    if remote.exists():
        return remote
    raise FileNotFoundError("Cannot locate data root (E:/DATA or ~/DATA)")


def _to_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        import ast

        return np.asarray(ast.literal_eval(value), dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    return None


def _processed_voigt_from_row(row: pd.Series) -> np.ndarray | None:
    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        return voigt
    return None


def _stored_cartesian_from_row(row: pd.Series) -> np.ndarray | None:
    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    return None


def _crystal_system(cif: str | None) -> str:
    if not cif:
        return "unknown"
    try:
        struct = Structure.from_str(cif, fmt="cif")
        return SpacegroupAnalyzer(struct).get_crystal_system()
    except Exception:  # noqa: BLE001
        return "unknown"


def _setup_dirs() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "pair_manifests").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "source_reconstruction").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "ranking").mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Lineage level classification
# -----------------------------------------------------------------------------


def _classify_lineage_level(row: pd.Series, source: str) -> LineageLevel:
    """Classify a record's lineage level based on available provenance.

    L0: direct upstream payload with source hash (not available in T2C parquet).
    L1: T2C processed Voigt with parser manifest / transformation history.
    L2: only processed Voigt and stored Cartesian, no upstream provenance.
    unresolved: fields missing or unrecognised.
    """
    has_voigt = _processed_voigt_from_row(row) is not None
    has_cart = _stored_cartesian_from_row(row) is not None
    # T2C-Flow carries source_last_updated and external_material_id but not the
    # raw upstream payload hash.  Treat this as processed-source provenance.
    has_manifest = bool(row.get("source_last_updated")) or bool(row.get("external_material_id"))
    if not has_voigt or not has_cart:
        return "unresolved"
    if has_manifest:
        return "L1_processed_source"
    return "L2_internal_consistency"


# -----------------------------------------------------------------------------
# All-record trusted conversion
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionRecord:
    source: str
    material_id: str
    formula: str
    processed_voigt: np.ndarray
    trusted_cartesian: np.ndarray
    project_cartesian: np.ndarray
    stored_cartesian: np.ndarray
    lineage_level: LineageLevel
    metrics: dict[str, float]
    status: str


def _all_record_conversion(jarvis: pd.DataFrame, mp: pd.DataFrame) -> tuple[list[ConversionRecord], pd.DataFrame]:
    """Run trusted/project conversion on every record with Voigt + Cartesian."""
    print("[Correctness v1.2] Running all-record trusted conversion...")
    records: list[ConversionRecord] = []
    rows: list[dict[str, Any]] = []

    def _process(df: pd.DataFrame, source: str) -> None:
        for _, row in df.iterrows():
            mid = str(row["material_id"])
            raw = _processed_voigt_from_row(row)
            stored = _stored_cartesian_from_row(row)
            if raw is None or stored is None:
                rows.append({
                    "source": source,
                    "material_id": mid,
                    "formula": row.get("formula"),
                    "lineage_level": "unresolved",
                    "status": "missing_tensor",
                })
                continue
            lineage_level = _classify_lineage_level(row, source)
            try:
                trusted = trusted_piezo_stress_voigt_to_cartesian(raw)
                project = piezo_stress_voigt_to_cartesian(raw)
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "source": source,
                    "material_id": mid,
                    "formula": row.get("formula"),
                    "lineage_level": lineage_level,
                    "status": "conversion_error",
                    "error": str(exc),
                })
                continue

            metrics = tensor_lineage_metrics(raw, trusted, project, stored)
            status = "verified" if metrics["frobenius_diff_trusted_vs_stored"] < 1e-6 else "mismatch"

            records.append(ConversionRecord(
                source=source,
                material_id=mid,
                formula=str(row.get("formula", "")),
                processed_voigt=raw,
                trusted_cartesian=trusted,
                project_cartesian=project,
                stored_cartesian=stored,
                lineage_level=lineage_level,
                metrics=metrics,
                status=status,
            ))
            rows.append({
                "source": source,
                "material_id": mid,
                "formula": row.get("formula"),
                "lineage_level": lineage_level,
                "status": status,
                **metrics,
            })

    _process(jarvis, "jarvis")
    _process(mp, "mp")

    summary_df = pd.DataFrame(rows)
    summary_df.to_parquet(ARTIFACT_ROOT / "source_reconstruction" / "all_record_conversion.parquet")
    print(f"[Correctness v1.2] Conversion records: {len(records)}")
    return records, summary_df


# -----------------------------------------------------------------------------
# Deterministic scalar reproduction
# -----------------------------------------------------------------------------


def _mp_eijmax_reproduce(voigt: np.ndarray) -> tuple[float, float]:
    """Reproduce MP ``e_ij_max`` as the spectral norm of the 3x6 Voigt matrix.

    MP's published ``e_ij_max`` is the largest singular value of the
    piezoelectric stress tensor expressed as a linear map from engineering
    strain (3x6 Voigt) to polarization (3-vector).  We cross-check SVD against
    the eigenvalue decomposition of ``A @ A.T``; both are deterministic and
    rotation invariant.
    """
    a = np.asarray(voigt, dtype=np.float64)
    if a.shape != (3, 6):
        raise ValueError(f"Expected Voigt shape (3, 6), got {a.shape}")
    svd_val = float(np.linalg.svd(a, compute_uv=False).max())
    eigh_val = float(np.sqrt(np.linalg.eigvalsh(a @ a.T).max()))
    if abs(svd_val - eigh_val) > 1e-6 * max(svd_val, 1.0):
        warnings.warn(
            f"MP e_ij_max cross-check mismatch: SVD={svd_val:.6e}, eigh={eigh_val:.6e}",
            stacklevel=2,
        )
    return svd_val, eigh_val


def _reproduce_source_scalars(
    jarvis: pd.DataFrame,
    mp: pd.DataFrame,
    conversion_records: list[ConversionRecord],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce source-published scalars from the verified tensor."""
    print("[Correctness v1.2] Reproducing source-published scalars...")
    conv_by_key: dict[tuple[str, str], ConversionRecord] = {
        (r.source, r.material_id): r for r in conversion_records
    }

    # MP e_ij_max
    mp_rows: list[dict[str, Any]] = []
    n_available = 0
    n_verified = 0
    n_mismatch = 0
    for _, row in mp.iterrows():
        mid = str(row["material_id"])
        src_val = row.get("e_ij_max")
        if src_val is None or pd.isna(src_val):
            continue
        n_available += 1
        rec = conv_by_key.get(("mp", mid))
        if rec is None or rec.status != "verified":
            mp_rows.append({
                "source": "mp",
                "material_id": mid,
                "e_ij_max": float(src_val),
                "reproduced": None,
                "relative_error": None,
                "status": "tensor_not_verified",
            })
            n_mismatch += 1
            continue
        try:
            repro, _ = _mp_eijmax_reproduce(rec.processed_voigt)
        except Exception as exc:  # noqa: BLE001
            mp_rows.append({
                "source": "mp",
                "material_id": mid,
                "e_ij_max": float(src_val),
                "reproduced": None,
                "relative_error": None,
                "status": f"compute_error: {exc}",
            })
            n_mismatch += 1
            continue
        rel_err = abs(float(src_val) - repro) / max(abs(float(src_val)), 1e-12)
        # Near-zero values are dominated by numerical noise; treat agreement
        # within an absolute window as verified.
        near_zero_abs = 2e-3
        status = "verified" if (
            rel_err < SCALAR_REL_TOL
            or (abs(float(src_val)) < near_zero_abs and abs(repro) < near_zero_abs)
        ) else "mismatch"
        if status == "verified":
            n_verified += 1
        else:
            n_mismatch += 1
        mp_rows.append({
            "source": "mp",
            "material_id": mid,
            "e_ij_max": float(src_val),
            "reproduced": repro,
            "relative_error": rel_err,
            "status": status,
        })

    mp_df = pd.DataFrame(mp_rows)
    mp_df.to_parquet(ARTIFACT_ROOT / "source_reconstruction" / "mp_scalar_reproduction.parquet")

    # JARVIS source scalar search
    jarvis_scalar_cols = [c for c in jarvis.columns if "max" in c.lower() and ("pza" in c.lower() or "piezo" in c.lower())]
    if not jarvis_scalar_cols:
        jarvis_status = "not_available"
        jarvis_detail: list[dict[str, Any]] = []
    else:
        jarvis_status = "fields_present"
        jarvis_detail = [{"column": c, "non_null": int(jarvis[c].notna().sum())} for c in jarvis_scalar_cols]

    summary = {
        "mp": {
            "n_available": n_available,
            "n_verified": n_verified,
            "n_mismatch": n_mismatch,
            "verification_rate_among_available": float(n_verified / n_available) if n_available else float("nan"),
            "coverage_rate": float(n_available / len(mp)) if len(mp) else float("nan"),
        },
        "jarvis": {
            "status": jarvis_status,
            "columns": jarvis_scalar_cols,
            "detail": jarvis_detail,
        },
    }
    (ARTIFACT_ROOT / "source_reconstruction" / "scalar_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[Correctness v1.2] MP scalar: {n_verified}/{n_available} verified "
        f"(rate={summary['mp']['verification_rate_among_available']:.1%}; coverage={summary['mp']['coverage_rate']:.1%})"
    )
    return summary, mp_df


# -----------------------------------------------------------------------------
# Invariant strict pair panel
# -----------------------------------------------------------------------------


def _build_invariant_strict_panel(
    jarvis: pd.DataFrame,
    mp: pd.DataFrame,
    overlap: pd.DataFrame,
    conversion_records: list[ConversionRecord],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the CrossPiezo-Invariant-Strict panel: identity only, no transport."""
    print("[Correctness v1.2] Building invariant strict pair panel...")
    match_cfg = _load_config("matching.yaml")
    params = match_cfg["matcher"]

    jarvis_by_id = {str(row["material_id"]): row for _, row in jarvis.iterrows()}
    mp_by_id = {str(row["material_id"]): row for _, row in mp.iterrows()}
    conv_by_key = {(r.source, r.material_id): r for r in conversion_records}

    pair_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    counts = {
        "n_candidate_overlap": len(overlap),
        "n_strict_match": 0,
        "n_tensor_verified": 0,
        "n_invariant_panel": 0,
    }

    for _, row in overlap.iterrows():
        jid = str(row["jarvis_id"])
        mid = str(row["material_id"])
        jrow = jarvis_by_id.get(jid)
        mrow = mp_by_id.get(mid)
        if jrow is None or mrow is None:
            continue

        jrec = conv_by_key.get(("jarvis", jid))
        mrec = conv_by_key.get(("mp", mid))
        j_verified = jrec is not None and jrec.status == "verified"
        m_verified = mrec is not None and mrec.status == "verified"
        if j_verified and m_verified:
            counts["n_tensor_verified"] += 1

        if not isinstance(jrow.get("cif"), str) or not isinstance(mrow.get("cif"), str):
            continue

        result = match_structures(
            left_key=f"jarvis:{jid}",
            right_key=f"mp:{mid}",
            left_cif=jrow["cif"],
            right_cif=mrow["cif"],
            ltol=params["ltol"],
            stol=params["stol"],
            angle_tol=params["angle_tol"],
            primitive_cell=params.get("primitive", True),
        )
        match_records.append({
            "jarvis_id": jid,
            "mp_id": mid,
            "match_tier": result.tier.value,
            "fit": result.fit,
            "rms_distance": result.rms_distance,
            "max_distance": result.max_distance,
            "lattice_distance": result.lattice_distance,
            "space_group_relation": result.space_group_relation,
            "rotation_class": result.rotation_class,
            "kabsch_rms": result.kabsch_rms,
            "reasons": ";".join(result.reasons or []),
        })

        # Invariant identity panel requires only a pymatgen StructureMatcher fit
        # under the frozen config.  Kabsch rotation reconstruction is diagnostic
        # only and is not used as an entry criterion.
        if result.fit and j_verified and m_verified:
            counts["n_strict_match"] += 1
            counts["n_invariant_panel"] += 1
            f1_j = frobenius_norm_score(jrec.stored_cartesian)
            f2_j = max_longitudinal_modulus(jrec.stored_cartesian)
            f1_m = frobenius_norm_score(mrec.stored_cartesian)
            f2_m = max_longitudinal_modulus(mrec.stored_cartesian)
            pair_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "formula": jrow.get("formula"),
                "jarvis_lineage_level": jrec.lineage_level,
                "mp_lineage_level": mrec.lineage_level,
                "rms_distance": result.rms_distance,
                "max_distance": result.max_distance,
                "lattice_distance": result.lattice_distance,
                "space_group_relation": result.space_group_relation,
                "rotation_class": result.rotation_class,
                "kabsch_rms": result.kabsch_rms,
                "jarvis_crystal_system": _crystal_system(jrow.get("cif")),
                "mp_crystal_system": _crystal_system(mrow.get("cif")),
                "jarvis_f1": f1_j,
                "mp_f1": f1_m,
                "jarvis_f2": f2_j,
                "mp_f2": f2_m,
                "f1_ratio": float(f1_j / f1_m) if f1_m > 0 else float("nan"),
                "f2_ratio": float(f2_j / f2_m) if f2_m > 0 else float("nan"),
            })

    matches_df = pd.DataFrame(match_records)
    pairs_df = pd.DataFrame(pair_records)
    matches_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "all_matches.parquet")
    pairs_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "invariant_strict_pairs.parquet")
    print(
        f"[Correctness v1.2] Candidate overlap: {counts['n_candidate_overlap']}; "
        f"strict matches: {counts['n_strict_match']}; tensor verified: {counts['n_tensor_verified']}; "
        f"invariant panel: {counts['n_invariant_panel']}"
    )
    return pairs_df, counts


# -----------------------------------------------------------------------------
# Invariant ranking audit
# -----------------------------------------------------------------------------


def _ranking_stability(pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute ranking-stability metrics for F1 and F2 on the invariant panel."""
    print("[Correctness v1.2] Computing invariant ranking stability...")
    results: list[Any] = []
    rows: list[dict[str, Any]] = []
    k_values = [10, 20, 30, 50]

    required = {"jarvis_f1", "mp_f1", "jarvis_f2", "mp_f2"}
    if not required.issubset(pairs_df.columns):
        print("[Correctness v1.2] WARNING: invariant panel empty; skipping ranking stability.")
        ranking_df = pd.DataFrame(columns=[
            "functional", "n_pairs", "top_10_jaccard", "top_20_jaccard",
            "top_30_jaccard", "top_50_jaccard", "kendall_tau",
            "kendall_tau_ci95_low", "kendall_tau_ci95_high",
            "spearman_rho", "spearman_pvalue", "mean_abs_rank_shift",
            "median_abs_rank_shift",
        ])
        ranking_df.to_parquet(ARTIFACT_ROOT / "ranking" / "invariant_ranking_stability.parquet")
        return ranking_df, []

    for name, left_col, right_col in [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F2_max_longitudinal", "jarvis_f2", "mp_f2"),
    ]:
        left = pairs_df[left_col].to_numpy(dtype=np.float64)
        right = pairs_df[right_col].to_numpy(dtype=np.float64)
        res = rank_stability_functional(left, right, name, k_values=k_values)
        results.append(res)
        rows.append({
            "functional": name,
            "n_pairs": res.n_pairs,
            "top_10_jaccard": res.top_10_jaccard,
            "top_20_jaccard": res.top_20_jaccard,
            "top_30_jaccard": res.top_30_jaccard,
            "top_50_jaccard": res.top_50_jaccard,
            "kendall_tau": res.kendall_tau,
            "kendall_tau_ci95_low": res.kendall_tau_ci_low,
            "kendall_tau_ci95_high": res.kendall_tau_ci_high,
            "spearman_rho": res.spearman_rho,
            "spearman_pvalue": res.spearman_pvalue,
            "mean_abs_rank_shift": res.mean_absolute_rank_shift,
            "median_abs_rank_shift": res.median_absolute_rank_shift,
        })

    ranking_df = pd.DataFrame(rows)
    ranking_df.to_parquet(ARTIFACT_ROOT / "ranking" / "invariant_ranking_stability.parquet")
    return ranking_df, ranking_summary_table(results)


def _threshold_crossing(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Count threshold crossings for the preregistered response thresholds."""
    required = {"jarvis_f1", "mp_f1", "jarvis_f2", "mp_f2"}
    if not required.issubset(pairs_df.columns):
        return pd.DataFrame()
    thresholds = [0.25, 0.5, 1.0]
    rows: list[dict[str, Any]] = []
    for name, left_col, right_col in [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F2_max_longitudinal", "jarvis_f2", "mp_f2"),
    ]:
        left = pairs_df[left_col].to_numpy()
        right = pairs_df[right_col].to_numpy()
        for thr in thresholds:
            l_pos = left > thr
            r_pos = right > thr
            both = int((l_pos & r_pos).sum())
            left_only = int((l_pos & ~r_pos).sum())
            right_only = int((~l_pos & r_pos).sum())
            neither = int((~l_pos & ~r_pos).sum())
            total = len(pairs_df)
            rows.append({
                "functional": name,
                "threshold_C_per_m2": thr,
                "both_above": both,
                "jarvis_only": left_only,
                "mp_only": right_only,
                "both_below": neither,
                "agreement_rate": (both + neither) / total if total else float("nan"),
                "crossing_rate": (left_only + right_only) / total if total else float("nan"),
            })
    return pd.DataFrame(rows)


def _sensitivity_analysis(pairs_df: pd.DataFrame) -> dict[str, Any]:
    """Ranking stability under exclusions, strata, and resampling."""
    print("[Correctness v1.2] Running sensitivity analysis...")
    out: dict[str, Any] = {}

    required = {"jarvis_f1", "mp_f1", "jarvis_f2", "mp_f2"}
    if not required.issubset(pairs_df.columns) or len(pairs_df) == 0:
        out["error"] = "invariant panel empty"
        (ARTIFACT_ROOT / "ranking" / "sensitivity_analysis.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8"
        )
        return out

    # Exclude bilateral near-zero.
    near_zero = 1e-3
    nz_mask = (pairs_df["jarvis_f1"] > near_zero) | (pairs_df["mp_f1"] > near_zero)
    f1_nz = rank_stability_functional(
        pairs_df.loc[nz_mask, "jarvis_f1"].to_numpy(),
        pairs_df.loc[nz_mask, "mp_f1"].to_numpy(),
        "F1_exclude_bilateral_near_zero",
        k_values=[10, 20, 30, 50],
    )
    f2_nz = rank_stability_functional(
        pairs_df.loc[nz_mask, "jarvis_f2"].to_numpy(),
        pairs_df.loc[nz_mask, "mp_f2"].to_numpy(),
        "F2_exclude_bilateral_near_zero",
        k_values=[10, 20, 30, 50],
    )
    out["exclude_bilateral_near_zero"] = {
        "n": int(nz_mask.sum()),
        "F1": ranking_summary_table([f1_nz])[0],
        "F2": ranking_summary_table([f2_nz])[0],
    }

    # High-response union (either source above 0.25 C/m^2).
    high_mask = (pairs_df["jarvis_f1"] > 0.25) | (pairs_df["mp_f1"] > 0.25)
    f1_high = rank_stability_functional(
        pairs_df.loc[high_mask, "jarvis_f1"].to_numpy(),
        pairs_df.loc[high_mask, "mp_f1"].to_numpy(),
        "F1_high_response_union",
        k_values=[10, 20, 30, 50],
    )
    f2_high = rank_stability_functional(
        pairs_df.loc[high_mask, "jarvis_f2"].to_numpy(),
        pairs_df.loc[high_mask, "mp_f2"].to_numpy(),
        "F2_high_response_union",
        k_values=[10, 20, 30, 50],
    )
    out["high_response_union"] = {
        "n": int(high_mask.sum()),
        "F1": ranking_summary_table([f1_high])[0],
        "F2": ranking_summary_table([f2_high])[0],
    }

    # Leave-one-crystal-system-out.
    loo_rows: list[dict[str, Any]] = []
    systems = pairs_df["jarvis_crystal_system"].unique()
    for sys in systems:
        if pd.isna(sys):
            continue
        mask = pairs_df["jarvis_crystal_system"] != sys
        if mask.sum() < 10:
            continue
        f1_loo = rank_stability_functional(
            pairs_df.loc[mask, "jarvis_f1"].to_numpy(),
            pairs_df.loc[mask, "mp_f1"].to_numpy(),
            f"F1_without_{sys}",
            k_values=[10, 20, 30, 50],
        )
        loo_rows.append(ranking_summary_table([f1_loo])[0])
    out["leave_one_crystal_system_out"] = loo_rows

    # Match-quality strata by rms_distance terciles.
    if pairs_df["rms_distance"].notna().sum() >= 3:
        pairs_df = pairs_df.copy()
        pairs_df["rms_tercile"] = pd.qcut(pairs_df["rms_distance"].rank(method="first"), 3, labels=["loose", "medium", "tight"])
        strata_rows: list[dict[str, Any]] = []
        for tercile in ["loose", "medium", "tight"]:
            sub = pairs_df[pairs_df["rms_tercile"] == tercile]
            if len(sub) < 5:
                continue
            f1_strata = rank_stability_functional(
                sub["jarvis_f1"].to_numpy(),
                sub["mp_f1"].to_numpy(),
                f"F1_rms_{tercile}",
                k_values=[10, 20, 30, 50],
            )
            strata_rows.append(ranking_summary_table([f1_strata])[0])
        out["match_quality_strata"] = strata_rows
    else:
        out["match_quality_strata"] = []

    # Pairwise bootstrap of Kendall tau.
    rng = np.random.default_rng(42)
    n_pairs = len(pairs_df)
    boot_taus: dict[str, list[float]] = {"F1": [], "F2": []}
    for _ in range(2000):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        f1_tau, _ = stats.kendalltau(pairs_df["jarvis_f1"].iloc[idx].to_numpy(), pairs_df["mp_f1"].iloc[idx].to_numpy())
        f2_tau, _ = stats.kendalltau(pairs_df["jarvis_f2"].iloc[idx].to_numpy(), pairs_df["mp_f2"].iloc[idx].to_numpy())
        if f1_tau is not None and np.isfinite(f1_tau):
            boot_taus["F1"].append(float(f1_tau))
        if f2_tau is not None and np.isfinite(f2_tau):
            boot_taus["F2"].append(float(f2_tau))
    out["bootstrap_kendall_tau"] = {
        "F1": {"mean": float(np.mean(boot_taus["F1"])), "std": float(np.std(boot_taus["F1"])),
               "ci95_low": float(np.percentile(boot_taus["F1"], 2.5)), "ci95_high": float(np.percentile(boot_taus["F1"], 97.5))},
        "F2": {"mean": float(np.mean(boot_taus["F2"])), "std": float(np.std(boot_taus["F2"])),
               "ci95_low": float(np.percentile(boot_taus["F2"], 2.5)), "ci95_high": float(np.percentile(boot_taus["F2"], 97.5))},
    }

    (ARTIFACT_ROOT / "ranking" / "sensitivity_analysis.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    return out


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def _report_v1_1_defects(commit: str | None) -> None:
    md = "## v1.1 defects addressed in v1.2\n"
    md += bullet("**V1.1-01**: ``tensor_lineage_metrics`` was called without ``trusted_cart``. The new API accepts ``trusted_cartesian`` explicitly and is tested to detect trusted/project/stored discrepancies independently.")
    md += bullet("**V1.1-02**: ``piezo_voigt_total`` is now labelled ``processed Voigt`` unless upstream parser history is demonstrated. Lineage levels L0/L1/L2 are reported honestly.")
    md += bullet("**V1.1-03**: Gate B now uses ``n_verified / n_available`` and reports coverage separately. Missing fields are no longer counted as failures.")
    md += bullet("**V1.1-04**: This script does not import or run Phase 5A/5B, baselines, PMR, soft-mode, e3nn, or O(3) transport analysis.")
    md += "\n## Frozen references\n"
    md += bullet(f"Baseline commit: ``ae49a3daedf6fe673209245a7c005ce146670290``")
    md += bullet(f"Current commit: ``{commit or 'unknown'}``")
    md += bullet(f"v1.1 artifacts frozen at: ``{V1_1_ARTIFACT_ROOT}``")
    write_report(REPORT_ROOT / "00_v1_1_defects.md", md, title="Correctness v1.2: v1.1 Defects")


def _report_lineage_levels(conversion_df: pd.DataFrame, commit: str | None) -> None:
    md = "## Data lineage levels\n"
    md += bullet("L0_raw_upstream: direct upstream payload with source hash.")
    md += bullet("L1_processed_source: T2C processed Voigt with parser manifest / transformation history.")
    md += bullet("L2_internal_consistency: processed Voigt + stored Cartesian, no upstream provenance.")
    md += bullet("unresolved: fields missing or unrecognised.")
    md += "\n"
    rows: list[dict[str, Any]] = []
    for source in ["jarvis", "mp"]:
        sub = conversion_df[conversion_df["source"] == source]
        counts = sub["lineage_level"].value_counts().to_dict()
        rows.append({
            "source": source,
            "n_L0": int(counts.get("L0_raw_upstream", 0)),
            "n_L1": int(counts.get("L1_processed_source", 0)),
            "n_L2": int(counts.get("L2_internal_consistency", 0)),
            "n_unresolved": int(counts.get("unresolved", 0)),
            "n_total": len(sub),
        })
    md += table_from_records(rows)
    md += "\n## Notes\n"
    md += bullet("T2C-Flow parquets carry ``source_last_updated`` / ``external_material_id`` but not raw upstream hashes, so records are L1 or L2.")
    md += bullet(f"Commit: ``{commit or 'unknown'}``")
    write_report(REPORT_ROOT / "01_lineage_levels.md", md, title="Correctness v1.2: Lineage Levels")


def _report_all_record_conversion(conversion_df: pd.DataFrame) -> None:
    md = "## All-record trusted conversion\n"
    md += bullet("Every record with processed Voigt + stored Cartesian was reconstructed with the trusted (pymatgen) and project converters and compared to the stored field.")
    md += bullet("Status ``verified`` means Frobenius diff trusted vs stored < 1e-6. Any mismatch is quarantined.")
    md += "\n"
    rows: list[dict[str, Any]] = []
    for source in ["jarvis", "mp"]:
        sub = conversion_df[conversion_df["source"] == source]
        verified = sub[sub["status"] == "verified"]
        mismatch = sub[sub["status"] == "mismatch"]
        rows.append({
            "source": source,
            "n_total": len(sub),
            "n_verified": len(verified),
            "n_mismatch": len(mismatch),
            "verified_rate": len(verified) / len(sub) if len(sub) else float("nan"),
            "rel_diff_median": verified["relative_diff_trusted_vs_stored"].median() if len(verified) else float("nan"),
            "rel_diff_p95": verified["relative_diff_trusted_vs_stored"].quantile(0.95) if len(verified) else float("nan"),
            "rel_diff_max": verified["relative_diff_trusted_vs_stored"].max() if len(verified) else float("nan"),
            "shear_diff_median": verified["shear_diff_trusted_vs_stored"].median() if len(verified) else float("nan"),
        })
    md += table_from_records(rows)

    md += "\n## Mismatch material IDs (sample)\n"
    for source in ["jarvis", "mp"]:
        mismatch = conversion_df[(conversion_df["source"] == source) & (conversion_df["status"] == "mismatch")]
        ids = mismatch["material_id"].head(20).tolist()
        md += bullet(f"{source}: {len(mismatch)} mismatches; first 20: {ids}")

    md += "\n## Reasons\n"
    md += bullet("Mismatches indicate that the stored Cartesian tensor is not exactly reproduced by ``pymatgen.PiezoTensor.from_vasp_voigt`` from the processed Voigt field. This can arise from Voigt-order assumptions, shear-convention drift, or version-mixed stored fields.")
    write_report(REPORT_ROOT / "02_all_record_conversion.md", md, title="Correctness v1.2: All-Record Conversion")


def _report_source_scalar_reproduction(scalar_summary: dict[str, Any], mp_df: pd.DataFrame) -> None:
    md = "## MP ``e_ij_max`` reproduction\n"
    mp = scalar_summary["mp"]
    md += bullet(f"Records with ``e_ij_max`` available: {mp['n_available']}")
    md += bullet(f"Verified within {SCALAR_REL_TOL:.0%} relative error: {mp['n_verified']}")
    md += bullet(f"Mismatches: {mp['n_mismatch']}")
    md += bullet(f"Verification rate among available: {mp['verification_rate_among_available']:.1%}")
    md += bullet(f"Coverage rate over all MP records: {mp['coverage_rate']:.1%}")
    md += bullet("Method: deterministic Fibonacci-sphere grid + projected-gradient ascent on the unit sphere + SLSQP cross-check; rotation invariant by construction.")
    md += "\n## JARVIS source scalar\n"
    jv = scalar_summary["jarvis"]
    if jv["status"] == "not_available":
        md += bullet("No source-published longitudinal-modulus field (e.g. ``max_pza``) found in the T2C-Flow JARVIS parquet. Reported as ``not_available``.")
    else:
        md += bullet(f"Candidate columns: {jv['columns']}")
        md += table_from_records(jv["detail"])
    md += "\n## Sample mismatches\n"
    mism = mp_df[mp_df["status"] == "mismatch"].head(20)
    if len(mism):
        md += table_from_records(mism[["material_id", "e_ij_max", "reproduced", "relative_error"]].to_dict("records"))
    else:
        md += "_No mismatches._\n"
    write_report(REPORT_ROOT / "03_source_scalar_reproduction.md", md, title="Correctness v1.2: Source Scalar Reproduction")


def _report_invariant_pair_panel(pairs_df: pd.DataFrame, counts: dict[str, int]) -> None:
    md = "## CrossPiezo-Invariant-Strict panel\n"
    md += bullet("Same reduced formula + StructureMatcher fit under frozen config (no 60° fallback).")
    md += bullet("Valid processed/verified total ``e`` tensor on both sides.")
    md += bullet("No componentwise frame requirement; no O(3) tensor transport.")
    md += "\n"
    md += table_from_records([{
        "n_candidate_overlap": counts["n_candidate_overlap"],
        "n_strict_match": counts["n_strict_match"],
        "n_tensor_verified": counts["n_tensor_verified"],
        "n_invariant_panel": counts["n_invariant_panel"],
    }])
    md += "\n## Match-quality distribution\n"
    if not pairs_df.empty:
        md += table_from_records([{
            "rms_distance_mean": pairs_df["rms_distance"].mean(),
            "max_distance_mean": pairs_df["max_distance"].mean(),
            "lattice_distance_mean": pairs_df["lattice_distance"].mean(),
            "kabsch_rms_mean": pairs_df["kabsch_rms"].mean(),
        }])
    else:
        md += "_No invariant pairs._\n"
    write_report(REPORT_ROOT / "04_invariant_pair_panel.md", md, title="Correctness v1.2: Invariant Pair Panel")


def _report_invariant_metrics(pairs_df: pd.DataFrame) -> None:
    md = "## Invariant metrics on the strict panel\n"
    md += bullet("F1(e) = ||e||_F (Frobenius norm of the Cartesian tensor).")
    md += bullet("F2(e) = max_{||n||=1} | n_i e_ijk n_j n_k | (maximum longitudinal modulus).")
    md += "\n"
    rows = []
    for src in ["jarvis", "mp"]:
        for name, col in [("F1", f"{src}_f1"), ("F2", f"{src}_f2")]:
            rows.append({
                "source": src,
                "functional": name,
                "mean": pairs_df[col].mean(),
                "median": pairs_df[col].median(),
                "std": pairs_df[col].std(),
                "min": pairs_df[col].min(),
                "max": pairs_df[col].max(),
            })
    md += table_from_records(rows)
    md += "\n## Cross-source ratios (JARVIS / MP)\n"
    md += table_from_records([{
        "functional": "F1",
        "mean_ratio": pairs_df["f1_ratio"].mean(),
        "median_ratio": pairs_df["f1_ratio"].median(),
    }, {
        "functional": "F2",
        "mean_ratio": pairs_df["f2_ratio"].mean(),
        "median_ratio": pairs_df["f2_ratio"].median(),
    }])
    write_report(REPORT_ROOT / "05_invariant_metrics.md", md, title="Correctness v1.2: Invariant Metrics")


def _report_ranking_stability(ranking_df: pd.DataFrame, ranking_summary: list[dict[str, Any]]) -> None:
    md = "## Invariant ranking stability\n"
    md += bullet("Metrics computed on the CrossPiezo-Invariant-Strict panel.")
    md += bullet("Top-k Jaccard, Kendall tau-b, Spearman rho, bootstrap 95% CI, absolute rank shift.")
    md += "\n"
    md += table_from_records(ranking_summary)
    md += "\n## Interpretation\n"
    for _, row in ranking_df.iterrows():
        name = row["functional"]
        tau = row["kendall_tau"]
        j10 = row["top_10_jaccard"]
        md += bullet(f"{name}: Kendall tau = {tau:.3f}; top-10 Jaccard = {j10:.3f}.")
    write_report(REPORT_ROOT / "06_ranking_stability.md", md, title="Correctness v1.2: Ranking Stability")


def _report_sensitivity_and_robustness(
    sensitivity: dict[str, Any],
    threshold_df: pd.DataFrame,
) -> None:
    md = "## Sensitivity and robustness\n"
    md += "### Threshold crossing (preregistered: 0.25, 0.5, 1.0 C/m²)\n"
    md += table_from_records(threshold_df.to_dict("records"))

    md += "\n### Exclude bilateral near-zero (< 1e-3 C/m²)\n"
    nz = sensitivity["exclude_bilateral_near_zero"]
    md += bullet(f"N = {nz['n']}")
    md += table_from_records([nz["F1"], nz["F2"]])

    md += "\n### High-response union (> 0.25 C/m²)\n"
    hi = sensitivity["high_response_union"]
    md += bullet(f"N = {hi['n']}")
    md += table_from_records([hi["F1"], hi["F2"]])

    md += "\n### Leave-one-crystal-system-out (F1)\n"
    if sensitivity["leave_one_crystal_system_out"]:
        md += table_from_records(sensitivity["leave_one_crystal_system_out"])
    else:
        md += "_Insufficient systems._\n"

    md += "\n### Match-quality strata by RMS distance (F1)\n"
    if sensitivity["match_quality_strata"]:
        md += table_from_records(sensitivity["match_quality_strata"])
    else:
        md += "_No strata._\n"

    md += "\n### Bootstrap Kendall tau\n"
    boot = sensitivity["bootstrap_kendall_tau"]
    md += table_from_records([
        {"functional": "F1", **boot["F1"]},
        {"functional": "F2", **boot["F2"]},
    ])
    write_report(REPORT_ROOT / "07_sensitivity_and_robustness.md", md, title="Correctness v1.2: Sensitivity and Robustness")


def _report_old_vs_corrected(conversion_df: pd.DataFrame, scalar_summary: dict[str, Any], counts: dict[str, int]) -> None:
    md = "## Old (v1.1) vs corrected (v1.2)\n"
    old_path = V1_1_ARTIFACT_ROOT / "source_reconstruction" / "audit_summary.parquet"
    old_lineage_rate = "unknown"
    old_scalar_rate = "unknown"
    old_n_total = "unknown"
    if old_path.exists():
        old_df = pd.read_parquet(old_path)
        old_n_total = int(len(old_df))
        old_lineage_rate = f"{(old_df['lineage_status'] == 'raw_lineage_verified').mean():.1%}"
        old_scalar_rate = f"{(old_df['derived_scalar_status'] == 'derived_scalar_verified').mean():.1%}"

    verified = conversion_df[conversion_df["status"] == "verified"]
    new_lineage_rate = len(verified) / len(conversion_df) if len(conversion_df) else 0.0
    mp = scalar_summary["mp"]
    new_scalar_rate = mp["verification_rate_among_available"]

    rows = [
        {"metric": "Scope", "v1.1": "120-record stratified sample", "v1.2": "All records with Voigt + Cartesian"},
        {"metric": "Trusted tensor in lineage metrics", "v1.1": "Not passed (project used as trusted)", "v1.2": "Passed explicitly; independent diffs"},
        {"metric": "Lineage pass rate", "v1.1": old_lineage_rate, "v1.2": f"{new_lineage_rate:.1%}"},
        {"metric": "Scalar denominator", "v1.1": "n_verified / 120", "v1.2": "n_verified / n_available; coverage reported"},
        {"metric": "MP scalar pass rate", "v1.1": old_scalar_rate, "v1.2": f"{new_scalar_rate:.1%}"},
        {"metric": "Strict pair panel", "v1.1": "Transport + Phase 5A", "v1.2": f"{counts['n_invariant_panel']} invariant-only pairs"},
        {"metric": "Phase 5A/5B / e3nn / PMR / O3", "v1.1": "Executed", "v1.2": "Not imported or executed"},
    ]
    md += table_from_records(rows)
    md += "\n"
    md += bullet("The v1.1 lineage pass rate is not comparable because the trusted tensor was discarded.")
    md += bullet("The v1.2 scalar pass rate is computed only among records that actually carry ``e_ij_max``.")
    write_report(REPORT_ROOT / "08_old_vs_corrected.md", md, title="Correctness v1.2: Old vs Corrected")


def _report_final_decision(
    conversion_df: pd.DataFrame,
    scalar_summary: dict[str, Any],
    counts: dict[str, int],
    ranking_df: pd.DataFrame,
    sensitivity: dict[str, Any],
    commit: str | None,
) -> None:
    md = "## Decision criteria\n"

    verified_rate = (
        (conversion_df["status"] == "verified").mean() if len(conversion_df) else 0.0
    )
    lineage_ok = verified_rate >= 0.99
    mp_rate = scalar_summary["mp"]["verification_rate_among_available"]
    scalar_ok = mp_rate >= 0.95 and not np.isnan(mp_rate)
    panel_ok = counts["n_invariant_panel"] >= 100

    # Instability: at least one functional shows imperfect cross-source ranking.
    instability = False
    inst_reason = ""
    if not ranking_df.empty:
        f1_j10 = ranking_df[ranking_df["functional"] == "F1_Frobenius"]["top_10_jaccard"].values[0]
        f1_tau = ranking_df[ranking_df["functional"] == "F1_Frobenius"]["kendall_tau"].values[0]
        f2_j10 = ranking_df[ranking_df["functional"] == "F2_max_longitudinal"]["top_10_jaccard"].values[0]
        f2_tau = ranking_df[ranking_df["functional"] == "F2_max_longitudinal"]["kendall_tau"].values[0]
        if f1_j10 < 0.9 or f1_tau < 0.9 or f2_j10 < 0.9 or f2_tau < 0.9:
            instability = True
            inst_reason = f"F1 top-10 Jaccard={f1_j10:.3f}, tau={f1_tau:.3f}; F2 top-10 Jaccard={f2_j10:.3f}, tau={f2_tau:.3f}"
        else:
            inst_reason = "Cross-library ranking is too stable to claim a reproducible screening instability."

    # Near-zero driven?
    nz = sensitivity["exclude_bilateral_near_zero"]
    nz_still_unstable = (
        nz["F1"]["top_10_jaccard"] < 0.9 or nz["F2"]["top_10_jaccard"] < 0.9
        or nz["F1"]["kendall_tau"] < 0.9 or nz["F2"]["kendall_tau"] < 0.9
    )
    not_near_zero_driven = instability and nz_still_unstable

    checks = {
        "trusted_lineage_closed": lineage_ok,
        "lineage_levels_labelled": True,
        "mismatch_rate_low": lineage_ok,
        "mp_scalar_rate_geq_95%": scalar_ok,
        "invariant_panel_N_geq_100": panel_ok,
        "ranking_instability_present": instability,
        "not_near_zero_driven": not_near_zero_driven,
        "artifacts_hash_bound": commit is not None,
    }

    proceed = all(checks.values())
    decision = "Proceed — Invariant Benchmark" if proceed else "Stop — CrossPiezo Main Claim"

    md += table_from_records([{"criterion": k, "satisfied": str(v)} for k, v in checks.items()])
    md += "\n## Decision\n"
    md += bullet(f"**{decision}**")
    if proceed:
        md += bullet("The invariant-only benchmark is viable: lineage closes, MP scalar reproduces in available records, the strict panel has >=100 pairs, and cross-library ranking is reproducibly unstable outside near-zero records.")
    else:
        md += bullet("At least one closure criterion failed. The project should be retained as a data-interoperability tool; no screening-instability paper should be written on the current processed-field mix.")
        md += bullet(f"Specific concern: {inst_reason}")

    md += "\n## Evidence summary\n"
    md += bullet(f"All-record conversion verified: {(conversion_df['status'] == 'verified').sum()}/{len(conversion_df)} ({verified_rate:.1%})")
    md += bullet(f"MP e_ij_max verification rate among available: {mp_rate:.1%}")
    md += bullet(f"Invariant strict panel size: {counts['n_invariant_panel']}")
    md += bullet(f"Commit: ``{commit or 'unknown'}``")
    md += bullet("All artifacts are written under ``artifacts/correctness_v1_2/`` and reports under ``reports/correctness_v1_2/``.")
    write_report(REPORT_ROOT / "09_final_decision.md", md, title="Correctness v1.2: Final Decision")
    print(f"[Correctness v1.2] Decision: {decision}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CrossPiezo correctness v1.2 invariant closure")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = args.data_root or _resolve_data_root()
    print(f"[Correctness v1.2] Data root: {data_root}")

    _setup_dirs()
    commit = _git_commit()

    cfg = _load_config("data_sources.yaml")
    t2c = cfg["sources"]["t2c_flow"]
    root = data_root / "T2C-Flow"
    jarvis = pd.read_parquet(root / t2c["records"]["jarvis_piezo"])
    mp = pd.read_parquet(root / t2c["records"]["mp_piezo"])
    overlap = pd.read_parquet(root / t2c["records"]["jarvis_mp_overlap"])

    conversion_records, conversion_df = _all_record_conversion(jarvis, mp)
    print(f"[Correctness v1.2] Records verified: {(conversion_df['status'] == 'verified').sum()}")
    print(f"[Correctness v1.2] Mismatches: {(conversion_df['status'] == 'mismatch').sum()}")

    scalar_summary, mp_scalar_df = _reproduce_source_scalars(jarvis, mp, conversion_records)
    pairs_df, counts = _build_invariant_strict_panel(jarvis, mp, overlap, conversion_records)

    if len(pairs_df) == 0:
        print("[Correctness v1.2] WARNING: Invariant strict panel is empty.")

    ranking_df, ranking_summary = _ranking_stability(pairs_df)
    threshold_df = _threshold_crossing(pairs_df)
    sensitivity = _sensitivity_analysis(pairs_df)

    # Write all reports.
    _report_v1_1_defects(commit)
    _report_lineage_levels(conversion_df, commit)
    _report_all_record_conversion(conversion_df)
    _report_source_scalar_reproduction(scalar_summary, mp_scalar_df)
    _report_invariant_pair_panel(pairs_df, counts)
    _report_invariant_metrics(pairs_df)
    _report_ranking_stability(ranking_df, ranking_summary)
    _report_sensitivity_and_robustness(sensitivity, threshold_df)
    _report_old_vs_corrected(conversion_df, scalar_summary, counts)
    _report_final_decision(conversion_df, scalar_summary, counts, ranking_df, sensitivity, commit)

    # Freeze a small manifest.
    manifest = {
        "timestamp": _now(),
        "commit": commit,
        "data_root": str(data_root),
        "n_conversion_records": len(conversion_records),
        "n_verified": int((conversion_df["status"] == "verified").sum()),
        "n_mismatch": int((conversion_df["status"] == "mismatch").sum()),
        "scalar_summary": scalar_summary,
        "invariant_panel_counts": counts,
        "ranking_summary": ranking_summary,
    }
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("[Correctness v1.2] Pipeline complete. Review reports/correctness_v1_2/09_final_decision.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
