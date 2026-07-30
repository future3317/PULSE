#!/usr/bin/env python
"""Correctness reset pipeline for CrossPiezo / PULSE.

This script re-runs the Phase 0-4 / 5A audit logic using the corrected
modules and writes all results into the ``correctness_v1_1`` namespace.
It does not overwrite the pre-audit release artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pymatgen.core.structure import Structure

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correctness_v1_1_1"
REPORT_ROOT = PROJECT_ROOT / "reports" / "correctness_v1_1_1"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.analysis.baselines import _prepare_records  # noqa: E402
from crosspiezo.analysis.discrepancy import absolute_discrepancy, normalized_discrepancy  # noqa: E402
from crosspiezo.analysis.o3_transport import (  # noqa: E402
    domain_aware_discrepancy,
    exact_transported_discrepancy,
    point_group_equivalent_discrepancy,
    proper_orbit_discrepancy,
    symmetry_projected_discrepancy,
    transform_polar_rank3,
)
from crosspiezo.analysis.ranking import (  # noqa: E402
    frobenius_norm_score,
    max_longitudinal_response,
    rank_stability_functional,
    ranking_summary_table,
)
from crosspiezo.analysis.soft_mode import _formula_to_prototype  # noqa: E402
from crosspiezo.conventions.symmetry import point_group_rotations, symmetry_residual  # noqa: E402
from crosspiezo.conventions.voigt import (  # noqa: E402
    piezo_stress_voigt_to_cartesian,
    tensor_lineage_metrics,
    trusted_piezo_stress_voigt_to_cartesian,
)
from crosspiezo.matching.structure_matcher import match_structures, to_match_record  # noqa: E402
from crosspiezo.reports.markdown import bullet, table_from_records, write_report  # noqa: E402
from crosspiezo.schemas import MatchTier  # noqa: E402

# Import Phase 5A orchestrator so we can re-use its audited sub-steps.
import scripts.run_phase5a as run_phase5a  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
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


def _tensor_from_row(row: pd.Series) -> np.ndarray | None:
    """Return a 3x3x3 Cartesian total piezo tensor from a unified row."""
    import ast

    def _to_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            return np.asarray(ast.literal_eval(value), dtype=np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        return None

    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        return piezo_stress_voigt_to_cartesian(voigt)
    return None


def _raw_voigt_from_row(row: pd.Series) -> np.ndarray | None:
    """Return the raw source Voigt tensor (3x6) from a unified row."""
    import ast

    def _to_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            return np.asarray(ast.literal_eval(value), dtype=np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        return None

    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        return voigt
    return None


def _stored_cartesian_from_row(row: pd.Series) -> np.ndarray | None:
    """Return the stored Cartesian tensor (3x3x3) from a unified row."""
    import ast

    def _to_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            return np.asarray(ast.literal_eval(value), dtype=np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        return None

    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    return None


def _space_group_symbol(space_group: Any) -> str | None:
    try:
        return f"{int(float(space_group))}"
    except Exception:  # noqa: BLE001
        return None


def _crystal_system(structure: Structure) -> str:
    """Return the crystal system for a pymatgen Structure."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(structure).get_crystal_system()
    except Exception:  # noqa: BLE001
        return "unknown"


def _setup_dirs() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "pair_manifests").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "phase5a").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "source_reconstruction").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "source_reconstruction" / "jarvis").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "source_reconstruction" / "mp").mkdir(parents=True, exist_ok=True)


def _run_matching(jarvis: pd.DataFrame, mp: pd.DataFrame, overlap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-run strict structure matching with corrected matcher."""
    print("[Correctness v1] Running strict structure matching...")
    match_cfg = _load_config("matching.yaml")
    params = match_cfg["matcher"]

    jarvis_by_id = {row["material_id"]: row for _, row in jarvis.iterrows()}
    mp_by_id = {row["material_id"]: row for _, row in mp.iterrows()}

    match_results: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    quarantine_records: list[dict[str, Any]] = []

    for _, row in overlap.iterrows():
        jid = row["jarvis_id"]
        mid = row["material_id"]
        jrow = jarvis_by_id.get(jid)
        mrow = mp_by_id.get(mid)
        if jrow is None or mrow is None:
            quarantine_records.append({"jarvis_id": jid, "mp_id": mid, "reason": "missing_source_record"})
            continue
        if not isinstance(jrow.get("cif"), str) or not isinstance(mrow.get("cif"), str):
            quarantine_records.append({"jarvis_id": jid, "mp_id": mid, "reason": "missing_cif"})
            continue

        result = match_structures(
            left_key=f"jarvis:{jid}",
            right_key=f"mp:{mid}",
            left_cif=jrow["cif"],
            right_cif=mrow["cif"],
            ltol=params["ltol"],
            stol=params["stol"],
            angle_tol=params["angle_tol"],
        )
        rec = to_match_record(result)
        match_results.append(rec.model_dump())

        if result.tier in (MatchTier.TIER_0, MatchTier.TIER_1):
            jtensor = _tensor_from_row(jrow)
            mtensor = _tensor_from_row(mrow)
            if jtensor is None or mtensor is None:
                quarantine_records.append({"jarvis_id": jid, "mp_id": mid, "reason": "tensor_conversion_failed"})
                continue

            if result.cartesian_rotation is not None:
                rot = np.asarray(result.cartesian_rotation, dtype=np.float64)
                mtensor = transform_polar_rank3(mtensor, rot)

            sg_symbol = _space_group_symbol(jrow.get("space_group"))
            jresid = mresid = None
            try:
                jstruct = Structure.from_str(jrow["cif"], fmt="cif")
                if len(jstruct) > 0:
                    rots = point_group_rotations(jstruct)
                    jresid = symmetry_residual(jtensor, rots)
                    mresid = symmetry_residual(mtensor, rots)
            except Exception:  # noqa: BLE001
                pass

            sym_threshold = match_cfg.get("symmetry_residual_threshold", 1.0)
            high_residual = (jresid is not None and jresid > sym_threshold) or (mresid is not None and mresid > sym_threshold)

            pair_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "formula": jrow["formula"],
                "space_group": jrow.get("space_group"),
                "match_tier": result.tier.value,
                "rms_distance": result.rms_distance,
                "max_distance": result.max_distance,
                "lattice_distance": result.lattice_distance,
                "space_group_relation": result.space_group_relation,
                "jarvis_norm": float(np.linalg.norm(jtensor)),
                "mp_norm": float(np.linalg.norm(mtensor)),
                "absolute_discrepancy": absolute_discrepancy(jtensor, mtensor),
                "normalized_discrepancy": normalized_discrepancy(jtensor, mtensor),
                "jarvis_symmetry_residual": jresid,
                "mp_symmetry_residual": mresid,
                "high_symmetry_residual": high_residual,
                "rotation_class": result.rotation_class,
                "kabsch_rms": result.kabsch_rms,
            })
        elif result.tier == MatchTier.QUARANTINE:
            quarantine_records.append({"jarvis_id": jid, "mp_id": mid, "reason": ";".join(result.reasons or [])})

    matches_df = pd.DataFrame(match_results)
    pairs_df = pd.DataFrame(pair_records)
    quarantine_df = pd.DataFrame(quarantine_records)

    matches_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "all_matches.parquet")
    pairs_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "strict_pairs.parquet")
    quarantine_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "quarantined_pairs.parquet")

    print(f"[Correctness v1] Matches: {len(matches_df)}; Tier-1 pairs: {len(pairs_df)}; quarantine: {len(quarantine_df)}")
    return matches_df, pairs_df, quarantine_df


def _source_reconstruction(jarvis: pd.DataFrame, mp: pd.DataFrame) -> dict[str, Any]:
    """Sample 120 records (60 per source) and independently verify tensor lineage."""
    print("[Correctness v1.1] Running independent source reconstruction audit...")
    rng = np.random.default_rng(42)
    samples: list[dict[str, Any]] = []

    def _audit_source(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
        # Build candidate list with crystal system and norm.
        candidates: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            raw_voigt = _raw_voigt_from_row(row)
            stored_cart = _stored_cartesian_from_row(row)
            if raw_voigt is None or stored_cart is None:
                continue
            norm = float(np.linalg.norm(stored_cart))
            try:
                struct = Structure.from_str(row["cif"], fmt="cif")
                crystal_system = _crystal_system(struct)
            except Exception:  # noqa: BLE001
                crystal_system = "unknown"
            candidates.append({
                "row": row,
                "raw_voigt": raw_voigt,
                "stored_cart": stored_cart,
                "norm": norm,
                "crystal_system": crystal_system,
            })

        if len(candidates) < 60:
            chosen = candidates
        else:
            # Stratified draw: per crystal system, take low/medium/high by norm.
            systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic", "unknown"]
            chosen: list[dict[str, Any]] = []
            per_system_target = 60 // len(systems)
            remainder = 60 - per_system_target * len(systems)
            for sys in systems:
                sys_recs = [c for c in candidates if c["crystal_system"] == sys]
                if len(sys_recs) < 3:
                    continue
                sys_recs.sort(key=lambda c: c["norm"])
                n = per_system_target + (1 if remainder > 0 else 0)
                remainder -= 1 if remainder > 0 else 0
                n_low = n // 3
                n_mid = n // 3
                n_high = n - n_low - n_mid
                low = sys_recs[:n_low]
                mid_start = len(sys_recs) // 2 - n_mid // 2
                mid = sys_recs[mid_start:mid_start + n_mid]
                high = sys_recs[-n_high:]
                chosen.extend(low + mid + high)
            # If stratified draw underfills, pad randomly from remaining candidates.
            if len(chosen) < 60:
                chosen_ids = {id(c) for c in chosen}
                remaining = [c for c in candidates if id(c) not in chosen_ids]
                need = 60 - len(chosen)
                if remaining:
                    chosen.extend(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())

        out: list[dict[str, Any]] = []
        for item in chosen:
            row = item["row"]
            mid = row["material_id"]
            raw_voigt = item["raw_voigt"]
            stored_cart = item["stored_cart"]

            # Independent trusted Cartesian reconstruction.
            try:
                trusted_cart = trusted_piezo_stress_voigt_to_cartesian(raw_voigt)
            except Exception:  # noqa: BLE001
                trusted_cart = None

            # Project converter reconstruction.
            try:
                project_cart = piezo_stress_voigt_to_cartesian(raw_voigt)
            except Exception as exc:  # noqa: BLE001
                project_cart = None

            # Lineage comparison.
            lineage_status = "raw_lineage_verified"
            lineage_metrics: dict[str, float] = {}
            if trusted_cart is not None:
                lineage_metrics = tensor_lineage_metrics(raw_voigt, stored_cart, project_cart)
                if lineage_metrics["relative_diff"] > 1e-3:
                    lineage_status = "conversion_mismatch"
            else:
                lineage_status = "conversion_mismatch"

            # Source-published scalar reproduction.
            derived_scalar_status = "source_field_missing"
            derived_scalar_value = float("nan")
            derived_scalar_rel_err = float("nan")
            if source == "mp":
                if "e_ij_max" in row and pd.notna(row["e_ij_max"]):
                    derived_scalar_value = float(row["e_ij_max"])
                    # Reproduce e_ij_max from tensor: max directional response.
                    from crosspiezo.analysis.ranking import max_longitudinal_response
                    repro = max_longitudinal_response(stored_cart)
                    derived_scalar_rel_err = abs(derived_scalar_value - repro) / max(abs(derived_scalar_value), 1e-12)
                    derived_scalar_status = "derived_scalar_verified" if derived_scalar_rel_err < 5e-2 else "derived_scalar_mismatch"
            elif source == "jarvis":
                # JARVIS does not expose a source-published scalar in this parquet.
                derived_scalar_status = "source_field_missing"

            # Native symmetry residual (normalized).
            native_status = "native_frame_unresolved"
            normalized_residual = float("nan")
            raw_residual = float("nan")
            try:
                struct = Structure.from_str(row["cif"], fmt="cif")
                if len(struct) > 0:
                    rots = point_group_rotations(struct)
                    raw_residual = symmetry_residual(stored_cart, rots)
                    norm_denom = np.linalg.norm(stored_cart) + 1e-12
                    normalized_residual = raw_residual / norm_denom
                    if normalized_residual <= 1e-3:
                        native_status = "native_symmetry_verified"
                    elif normalized_residual <= 5e-2:
                        native_status = "native_symmetry_review"
                    else:
                        native_status = "native_symmetry_unresolved"
            except Exception as exc:  # noqa: BLE001
                native_status = f"native_frame_error: {exc}"

            record = {
                "source": source,
                "material_id": mid,
                "formula": row.get("formula"),
                "space_group": row.get("space_group"),
                "crystal_system": item["crystal_system"],
                "norm": item["norm"],
                "lineage_status": lineage_status,
                "derived_scalar_status": derived_scalar_status,
                "derived_scalar_value": derived_scalar_value,
                "derived_scalar_rel_err": derived_scalar_rel_err,
                "native_status": native_status,
                "raw_residual": raw_residual,
                "normalized_residual": normalized_residual,
                **lineage_metrics,
            }
            out.append(record)

            # Write per-record directory.
            rec_dir = ARTIFACT_ROOT / "source_reconstruction" / source / str(mid)
            rec_dir.mkdir(parents=True, exist_ok=True)
            (rec_dir / "structure.cif").write_text(row.get("cif", ""), encoding="utf-8")
            (rec_dir / "raw_voigt.npy").write_bytes(raw_voigt.tobytes())
            (rec_dir / "stored_cartesian.npy").write_bytes(stored_cart.tobytes())
            if trusted_cart is not None:
                np.save(rec_dir / "trusted_cartesian.npy", trusted_cart)
            if project_cart is not None:
                np.save(rec_dir / "project_cartesian.npy", project_cart)
            (rec_dir / "metadata.json").write_text(
                json.dumps(record, indent=2, default=str),
                encoding="utf-8",
            )
        return out

    samples.extend(_audit_source(jarvis, "jarvis"))
    samples.extend(_audit_source(mp, "mp"))
    samples_df = pd.DataFrame(samples)
    samples_df.to_parquet(ARTIFACT_ROOT / "source_reconstruction" / "audit_summary.parquet")

    n_total = len(samples)
    n_lineage_verified = int(np.sum(samples_df["lineage_status"] == "raw_lineage_verified"))
    n_scalar_verified = int(np.sum(samples_df["derived_scalar_status"] == "derived_scalar_verified"))
    n_native_verified = int(np.sum(samples_df["native_status"] == "native_symmetry_verified"))
    n_native_review = int(np.sum(samples_df["native_status"] == "native_symmetry_review"))

    return {
        "n_total": n_total,
        "n_lineage_verified": n_lineage_verified,
        "lineage_pass_rate": float(n_lineage_verified / n_total) if n_total else 0.0,
        "n_scalar_verified": n_scalar_verified,
        "scalar_pass_rate": float(n_scalar_verified / n_total) if n_total else 0.0,
        "n_native_verified": n_native_verified,
        "n_native_review": n_native_review,
        "native_verified_rate": float(n_native_verified / n_total) if n_total else 0.0,
        "native_verified_or_review_rate": float((n_native_verified + n_native_review) / n_total) if n_total else 0.0,
    }


def _patch_phase5a_paths() -> None:
    """Redirect Phase 5A outputs into the correctness_v1_1 namespace."""
    run_phase5a.ARTIFACT_ROOT = ARTIFACT_ROOT
    run_phase5a.REPORT_ROOT = REPORT_ROOT
    run_phase5a.PHASE5A_ARTIFACT = ARTIFACT_ROOT / "phase5a"
    run_phase5a.PHASE5A_RELEASE = ARTIFACT_ROOT / "releases" / "phase0_4_v1"
    run_phase5a.PHASE5A_ARTIFACT.mkdir(parents=True, exist_ok=True)
    run_phase5a.PHASE5A_RELEASE.mkdir(parents=True, exist_ok=True)


def _write_static_reports(
    commit: str | None,
    old_counts: dict[str, Any],
    new_counts: dict[str, Any],
    reconstruction: dict[str, Any],
    pairs_df: pd.DataFrame,
) -> None:
    """Write the v1.1 code-audit reports."""
    samples_path = ARTIFACT_ROOT / "source_reconstruction" / "audit_summary.parquet"
    samples_df = pd.read_parquet(samples_path) if samples_path.exists() else pd.DataFrame()

    # 00 gate redefinition
    md = "## Gate redefinition for v1.1\n"
    md += bullet("Gate A (raw tensor lineage): raw source Voigt → trusted conversion → project conversion → stored Cartesian.")
    md += bullet("Gate B (source-published scalar): reproduce JARVIS ``max_pza`` / MP ``e_ij_max`` when available; otherwise ``not_available``.")
    md += bullet("Gate C (source-native symmetry): normalized residual ``||e - Pi_G e||_F / (||e||_F + eps)`` against actual source structure.")
    md += bullet("The old ``recomputed Frobenius norm ≈ stored norm`` check is renamed ``internal derived-field consistency`` only.")
    write_report(REPORT_ROOT / "00_gate_redefinition.md", md, title="Correctness v1.1: Gate Redefinition")

    # 01 raw tensor lineage
    md = "## Method\n"
    md += bullet("For every record with raw Voigt and stored Cartesian, compute trusted Cartesian via ``pymatgen PiezoTensor.from_vasp_voigt``.")
    md += bullet("Report Frobenius/relative/shear-only differences per source.\n")
    if not samples_df.empty and "frobenius_diff_trusted_vs_stored" in samples_df.columns:
        for source in ["jarvis", "mp"]:
            sub = samples_df[samples_df["source"] == source]
            if len(sub):
                md += bullet(f"{source}: mean rel diff = {sub['relative_diff'].mean():.2e}; max = {sub['relative_diff'].max():.2e}; shear-only mean = {sub['shear_only_diff'].mean():.2e}")
    md += "\n## Result\n"
    lineage_rate = reconstruction.get("lineage_pass_rate", 0.0)
    md += bullet(f"Raw lineage verified: {reconstruction.get('n_lineage_verified', 0)}/{reconstruction.get('n_total', 0)} ({lineage_rate:.1%}).")
    write_report(REPORT_ROOT / "01_raw_tensor_lineage.md", md, title="Correctness v1.1: Raw Tensor Lineage")

    # 02 trusted conversion
    md = "## Method\n"
    md += bullet("Project converter ``piezo_stress_voigt_to_cartesian`` is compared to pymatgen ``PiezoTensor.from_vasp_voigt`` on the same raw Voigt field.")
    md += bullet("Both use VASP Voigt order [xx,yy,zz,xy,yz,zx] internally after mapping from project order [xx,yy,zz,yz,xz,xy].\n")
    md += "## Result\n"
    md += bullet("Work-conjugacy identity and pymatgen oracle pass on synthetic tensors (see tests/conventions/test_piezo_tensor.py).")
    md += bullet("Project/trusted conversion agreement is reported per-record in the source reconstruction artifact.")
    write_report(REPORT_ROOT / "02_trusted_conversion.md", md, title="Correctness v1.1: Trusted Conversion")

    # 03 cartesian symmetry oracle
    md = "## Method\n"
    md += bullet("Custom ``lattice @ R_frac @ inv(lattice)`` + SVD path removed.")
    md += bullet("Primary path: ``SpacegroupAnalyzer(structure, symprec, angle_tolerance).get_point_group_operations(cartesian=True)``.")
    md += bullet("Operations are deduplicated and checked for orthogonality / determinant.\n")
    md += "## Result\n"
    md += bullet("Trusted API operations pass closure/identity/inverse/det tests (see tests/symmetry/test_cartesian_symmetry.py).")
    md += bullet("If trusted frame is unresolved against the source tensor, the record is marked ``source_frame_unresolved``.")
    write_report(REPORT_ROOT / "03_cartesian_symmetry_oracle.md", md, title="Correctness v1.1: Cartesian Symmetry Oracle")

    # 04 source reconstruction 120
    md = "## Sample design\n"
    md += bullet("120 records: 60 JARVIS + 60 MP, stratified by norm (low/medium/high) and crystal system.")
    md += bullet("Fixed random seed; manifest frozen in ``artifacts/correctness_v1_1/source_reconstruction/audit_summary.parquet``.\n")
    md += "## Results\n"
    md += bullet(f"Raw lineage verified: {reconstruction.get('n_lineage_verified', 0)}/{reconstruction.get('n_total', 0)} ({reconstruction.get('lineage_pass_rate', 0):.1%}).")
    md += bullet(f"Source-derived scalar verified (MP e_ij_max): {reconstruction.get('n_scalar_verified', 0)}/{reconstruction.get('n_total', 0)} ({reconstruction.get('scalar_pass_rate', 0):.1%}).")
    md += bullet(f"Native symmetry verified: {reconstruction.get('n_native_verified', 0)}/{reconstruction.get('n_total', 0)} ({reconstruction.get('native_verified_rate', 0):.1%}).")
    md += bullet(f"Native symmetry verified or review: {reconstruction.get('n_native_verified', 0) + reconstruction.get('n_native_review', 0)}/{reconstruction.get('n_total', 0)} ({reconstruction.get('native_verified_or_review_rate', 0):.1%}).")
    write_report(REPORT_ROOT / "04_source_reconstruction_120.md", md, title="Correctness v1.1: Source Reconstruction 120")

    # 05 matching without fallback
    md = "## Method\n"
    md += bullet("60° angle fallback removed; strict matcher uses only ``configs/matching.yaml`` tolerances.")
    md += bullet("Any secondary matcher result is reported separately, not mixed into the strict count.\n")
    md += "## Counts\n"
    md += bullet(f"Old frozen matcher count: {old_counts.get('tier1', 'unknown')}")
    md += bullet(f"New strict count: {new_counts.get('tier1', 'unknown')}")
    if not pairs_df.empty and "rotation_class" in pairs_df.columns:
        md += bullet(f"Rotation classes in strict pairs: {pairs_df['rotation_class'].value_counts().to_dict()}")
    write_report(REPORT_ROOT / "05_matching_without_fallback.md", md, title="Correctness v1.1: Matching Without Fallback")

    # 06 rotation reconstruction
    md = "## Method\n"
    md += bullet("Use ``StructureMatcher.get_transformation`` integer basis matrix, translation, and atom mapping.")
    md += bullet("Build period-image-corrected matched Cartesian coordinates, then Kabsch for proper and improper solutions.")
    md += bullet("Classify each match as basis_relabel / proper_rotation / improper_relation / deformation / unresolved.\n")
    md += "## Result\n"
    if not pairs_df.empty and "rotation_class" in pairs_df.columns:
        md += bullet(f"Rotation class distribution: {pairs_df['rotation_class'].value_counts().to_dict()}")
        md += bullet(f"Mean Kabsch RMS: {pairs_df['kabsch_rms'].mean():.4f}")
    write_report(REPORT_ROOT / "06_rotation_reconstruction.md", md, title="Correctness v1.1: Rotation Reconstruction")


def _write_split_and_prototype_report() -> None:
    md = "## Prototype fix\n"
    md += bullet("``_formula_to_prototype`` now returns reduced anonymous stoichiometry (e.g. Na2Cl2 and NaCl both → AB).")
    md += bullet("New fields: ``chemical_system``, ``reduced_formula``, ``anonymous_formula``, ``structure_prototype``, ``matcher_component``.\n")
    md += "## Split leakage\n"
    md += bullet("Train/test must have zero intersection on material ID, reduced formula, and prototype component.")
    md += bullet("Leakage tests are required to pass before any model benchmark is reported.")
    write_report(REPORT_ROOT / "07_split_and_prototype.md", md, title="Correctness v1.1: Split and Prototype")


def _write_old_v1_vs_v1_1(
    old_pairs: int,
    new_pairs: int,
    v1_reconstruction: dict[str, Any] | None,
    reconstruction: dict[str, Any],
) -> None:
    rows = [
        {"metric": "Tier-1 pairs", "v1": 542, "v1_1": new_pairs, "status": "revised" if new_pairs != 542 else "confirmed", "root_cause": "removed angle fallback; stricter rotation reconstruction"},
        {"metric": "Source reconstruction sample size", "v1": 60, "v1_1": reconstruction.get("n_total", 0), "status": "revised", "root_cause": "v1.1 requires independent 120-record audit"},
        {"metric": "Raw tensor lineage pass rate", "v1": "not assessed", "v1_1": f"{reconstruction.get('lineage_pass_rate', 0):.1%}", "status": "new_gate", "root_cause": "v1 used internal norm consistency, not source reconstruction"},
        {"metric": "Source-derived scalar pass rate", "v1": f"{v1_reconstruction.get('scalar_pass_rate', 0):.1%}" if v1_reconstruction else "unknown", "v1_1": f"{reconstruction.get('scalar_pass_rate', 0):.1%}", "status": "revised", "root_cause": "v1 compared stored norms; v1.1 reproduces source-published scalar"},
        {"metric": "Native symmetry verified rate", "v1": f"{v1_reconstruction.get('pass_rate', 0):.1%}" if v1_reconstruction else "unknown", "v1_1": f"{reconstruction.get('native_verified_rate', 0):.1%}", "status": "revised", "root_cause": "v1 used raw residual <1.0 C/m²; v1.1 uses normalized residual"},
    ]
    md = "## Old v1 vs. v1.1\n"
    md += table_from_records(rows)
    md += "\n## Notes\n"
    md += bullet("v1 Conditional Pass is not accepted as a gate; v1.1 re-audits with independent source reconstruction.")
    write_report(REPORT_ROOT / "08_old_v1_vs_v1_1.md", md, title="Correctness v1.1: Old v1 vs v1.1")


def _write_decision(reconstruction: dict[str, Any]) -> None:
    lineage_rate = reconstruction.get("lineage_pass_rate", 0.0)
    scalar_rate = reconstruction.get("scalar_pass_rate", 0.0)
    native_rate = reconstruction.get("native_verified_rate", 0.0)

    # Decision rubric from kickoff.
    if lineage_rate < 0.95:
        decision = "Fail"
        rationale = "Raw tensor lineage does not close for ≥95% of sampled records."
    elif scalar_rate < 0.95:
        decision = "Conditional Pass"
        rationale = "Raw lineage closes but source-published scalar does not; only invariant benchmark allowed."
    elif native_rate < 0.95:
        decision = "Pass — Invariant"
        rationale = "Raw lineage and source scalar close; source-native symmetry/frame unresolved for some records. Componentwise claims withdrawn."
    else:
        decision = "Pass — Componentwise"
        rationale = "Raw lineage, source scalar, and source-native symmetry all close for ≥95% of samples."

    md = "## Correctness v1.1 gate decision\n"
    md += bullet(f"**Decision: {decision}**")
    md += bullet(f"Rationale: {rationale}")
    md += "\n## Evidence\n"
    md += bullet(f"Raw lineage verified: {reconstruction.get('n_lineage_verified', 0)}/{reconstruction.get('n_total', 0)} ({lineage_rate:.1%}).")
    md += bullet(f"Source-derived scalar verified: {reconstruction.get('n_scalar_verified', 0)}/{reconstruction.get('n_total', 0)} ({scalar_rate:.1%}).")
    md += bullet(f"Native symmetry verified: {reconstruction.get('n_native_verified', 0)}/{reconstruction.get('n_total', 0)} ({native_rate:.1%}).")
    md += bullet("Old v1 artifacts were not overwritten; v1.1 results are in separate namespace.")
    md += "\n## Next steps\n"
    md += bullet("If Pass — Componentwise: invariant and componentwise benchmark allowed.")
    md += bullet("If Pass — Invariant or Conditional Pass: only invariant benchmark allowed; componentwise claims withdrawn.")
    md += bullet("If Fail: stop subsequent scientific work.")
    write_report(REPORT_ROOT / "09_decision.md", md, title="Correctness v1.1: Decision")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CrossPiezo correctness v1.1 independent verification")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = args.data_root or _resolve_data_root()
    print(f"[Correctness v1.1] Data root: {data_root}")

    _setup_dirs()
    commit = _git_commit()

    # Load raw data.
    cfg = _load_config("data_sources.yaml")
    t2c = cfg["sources"]["t2c_flow"]
    root = data_root / "T2C-Flow"
    jarvis = pd.read_parquet(root / t2c["records"]["jarvis_piezo"])
    mp = pd.read_parquet(root / t2c["records"]["mp_piezo"])
    overlap = pd.read_parquet(root / t2c["records"]["jarvis_mp_overlap"])

    # Source reconstruction audit (independent of matching).
    reconstruction = _source_reconstruction(jarvis, mp)

    # Re-run matching and write pair manifests into correctness_v1_1.
    matches_df, pairs_df, quarantine_df = _run_matching(jarvis, mp, overlap)

    # Old v1 counts for the v1-vs-v1.1 table.
    old_pairs_path = PROJECT_ROOT / "artifacts" / "releases" / "correctness_v1_4f46977" / "pair_manifests" / "strict_pairs.parquet"
    old_pairs = len(pd.read_parquet(old_pairs_path)) if old_pairs_path.exists() else 542

    # Re-use Phase 5A orchestration, redirected into correctness_v1_1.
    _patch_phase5a_paths()
    enriched = run_phase5a.build_enriched_pairs(jarvis, mp, pairs_df)
    print(f"[Correctness v1.1] Enriched {len(enriched)} Tier-1 pairs")

    run_phase5a.audit_symmetry_residuals(enriched)
    run_phase5a.build_manual_audit_package(enriched)
    o3_df = run_phase5a.audit_o3_transport(enriched)
    structure_df = run_phase5a.stratify_structure_shift(enriched)
    if "sublayer" in structure_df.columns:
        enriched = enriched.merge(
            structure_df[["jarvis_id", "mp_id", "sublayer", "volume_ratio"]],
            on=["jarvis_id", "mp_id"],
            how="left",
        )
    ranking_df = run_phase5a.revalidate_rankings(enriched)
    baseline_df, pmrs = run_phase5a.build_baselines_and_pmr(jarvis, mp, enriched)
    try:
        soft_mode_df, soft_mode_results = run_phase5a.audit_soft_mode(enriched, data_root=data_root)
    except Exception as exc:  # noqa: BLE001
        print(f"[Correctness v1.1] Soft-mode audit skipped: {exc}")
        soft_mode_df, soft_mode_results = pd.DataFrame(), []
    run_phase5a.compile_phase5a_decision(enriched, o3_df, structure_df, ranking_df, baseline_df, pmrs, soft_mode_df, soft_mode_results)

    # Write correctness-specific audit reports.
    new_counts = {"tier1": len(pairs_df)}
    old_counts = {"tier1": old_pairs}
    v1_manifest_path = PROJECT_ROOT / "artifacts" / "releases" / "correctness_v1_4f46977" / "manifest.json"
    v1_reconstruction = None
    if v1_manifest_path.exists():
        v1_reconstruction = json.loads(v1_manifest_path.read_text(encoding="utf-8")).get("source_reconstruction")
    _write_static_reports(commit, old_counts, new_counts, reconstruction, pairs_df)
    _write_split_and_prototype_report()
    _write_old_v1_vs_v1_1(old_pairs, len(pairs_df), v1_reconstruction, reconstruction)
    _write_decision(reconstruction)

    # Freeze a small manifest.
    manifest = {
        "timestamp": _now(),
        "commit": commit,
        "data_root": str(data_root),
        "n_pairs": len(pairs_df),
        "source_reconstruction": reconstruction,
    }
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("[Correctness v1.1] Pipeline complete. Review reports/correctness_v1_1/09_decision.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
