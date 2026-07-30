#!/usr/bin/env python
"""CrossPiezo / PULSE Phase 5B benchmark consolidation orchestration.

This script is read-only with respect to E:/DATA.  It writes outputs under
./artifacts/phase5b and ./reports.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crosspiezo.phase5b.hierarchy import compute_hierarchy, hierarchy_by_sublayer, hierarchy_summary
from crosspiezo.phase5b.lambda_audit import audit_piezojet_lambda
from crosspiezo.phase5b.panels import (
    assign_sublayers,
    build_core_extended_panels,
    build_enriched_pairs,
    compute_source_native_residuals,
)
from crosspiezo.reports.markdown import bullet, table_from_records, write_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = PROJECT_ROOT / "reports"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
PHASE5B_ARTIFACT = ARTIFACT_ROOT / "phase5b"
PHASE5A_RELEASE = ARTIFACT_ROOT / "releases" / "phase5a_v1"


def _load_config(name: str) -> dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _freeze_phase5a() -> dict[str, Any]:
    print("[Phase 5B.0] Freezing Phase 5A release...")
    release = ARTIFACT_ROOT / "releases" / "phase5a_v1"
    release.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"timestamp": _now(), "git_commit": _git_commit(), "files": {}}
    for cfg_file in (PROJECT_ROOT / "configs").glob("*.yaml"):
        dest = release / "configs" / cfg_file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_file, dest)
        manifest["files"][f"configs/{cfg_file.name}"] = _sha256_file(dest)
    for subdir in ["phase5a", "pair_manifests", "inventories", "feasibility"]:
        src = ARTIFACT_ROOT / subdir
        if not src.exists():
            continue
        dest = release / subdir
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
                manifest["files"][f"{subdir}/{f.name}"] = _sha256_file(dest / f.name)
    for report_file in REPORT_ROOT.glob("*.md"):
        name = report_file.name
        if name.startswith(("06_", "07_", "08_", "09_", "10_", "11_", "12_", "13_", "14_")):
            dest = release / "reports" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_file, dest)
            manifest["files"][f"reports/{name}"] = _sha256_file(dest)
    commit_path = release / "commit.txt"
    commit_path.write_text(_git_commit() or "unknown", encoding="utf-8")
    manifest_path = release / "freeze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Phase 5B.0] Froze {len(manifest['files'])} files to {release}")
    return manifest


def _data_root(cfg: dict[str, Any]) -> Path:
    env_root = cfg.get("data_root")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    # Remote fallback.
    remote = Path("/home/workspace/lrh/DATA")
    if remote.exists():
        return remote
    raise FileNotFoundError("Cannot locate data root; set data_sources.yaml data_root or mount /home/workspace/lrh/DATA")


def phase_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[Phase 5B.2] Building Core/Extended panels...")
    cfg = _load_config("data_sources.yaml")
    data_root = _data_root(cfg)
    enriched = build_enriched_pairs(data_root)
    residuals = compute_source_native_residuals(enriched)
    extended, core, exclusions = build_core_extended_panels(enriched, residuals, native_threshold=0.25)

    PHASE5B_ARTIFACT.mkdir(parents=True, exist_ok=True)
    drop_cols = ["jarvis_tensor", "mp_tensor_raw", "mp_tensor_aligned", "jarvis_cif", "mp_cif", "rotation", "atom_permutation"]
    extended_save = extended.drop(columns=[c for c in drop_cols if c in extended.columns])
    core_save = core.drop(columns=[c for c in drop_cols if c in core.columns])
    exclusions_save = exclusions.drop(columns=[c for c in drop_cols if c in exclusions.columns])
    extended_save.to_parquet(PHASE5B_ARTIFACT / "extended_pairs.parquet")
    core_save.to_parquet(PHASE5B_ARTIFACT / "core_pairs.parquet")
    exclusions_save.to_parquet(PHASE5B_ARTIFACT / "core_exclusions.parquet")
    residuals.to_parquet(PHASE5B_ARTIFACT / "source_native_residuals.parquet")

    md = ""
    md += "## Panel definitions\n"
    md += bullet("CrossPiezo-Extended: all 538 Tier-1 matched pairs with valid total piezo tensors.")
    md += bullet("CrossPiezo-Core: T1a near-identical structures whose source-native symmetry residuals are either low (<=0.25) or explained by a transportable frame mismatch.")
    md += "\n## Counts\n"
    md += table_from_records([{
        "panel": "Extended",
        "n_pairs": len(extended),
        "fraction_of_tier1": f"{len(extended)/538:.3f}",
    }, {
        "panel": "Core",
        "n_pairs": len(core),
        "fraction_of_tier1": f"{len(core)/538:.3f}",
    }])
    md += "\n## Core exclusion reasons (top)\n"
    reason_counts = exclusions["reason"].value_counts().head(10).reset_index()
    reason_counts.columns = ["reason", "count"]
    md += table_from_records(reason_counts.to_dict("records"))
    md += "\n## Selection bias note\n"
    md += bullet("Core is a strict subset of T1a; materials with unresolved source-native frame ambiguity are excluded from componentwise comparisons.")
    md += bullet("Ranking and invariant-norm analyses are reported on both Core and Extended.")
    write_report(REPORT_ROOT / "16_core_extended_panels.md", md, title="Phase 5B.1: Core/Extended Benchmark Panels")
    print(f"[Phase 5B.2] Core={len(core)} Extended={len(extended)} Exclusions={len(exclusions)}")
    return enriched, extended, core, residuals


def phase_residual_audit(residuals: pd.DataFrame) -> None:
    print("[Phase 5B.3] Auditing source-native frame residuals...")
    summary_rows: list[dict[str, Any]] = []
    for source in ["jarvis", "mp"]:
        sub = residuals[residuals["source"] == source]
        for metric in ["native", "transport", "common"]:
            vals = sub[f"{metric}_residual_raw"].dropna().values
            norm_vals = sub[f"{metric}_residual_normalized"].dropna().values
            if len(vals) == 0:
                continue
            summary_rows.append({
                "source": source,
                "frame": metric,
                "n": len(vals),
                "median_raw": float(np.median(vals)),
                "p95_raw": float(np.percentile(vals, 95)),
                "median_normalized": float(np.median(norm_vals)),
                "p95_normalized": float(np.percentile(norm_vals, 95)),
                "high_raw_count": int(np.sum(vals > 1.0)),
            })

    md = ""
    md += "## Residual definitions\n"
    md += bullet("native_residual: symmetry residual of the source tensor against the source CIF-setting point group.")
    md += bullet("transport_residual: symmetry residual of the tensor after transport to the common matched frame, against the transported source point group.")
    md += bullet("common_residual: symmetry residual of the transported tensor against the common-structure point group.")
    md += "\n## Source-native residual summary\n"
    md += table_from_records(summary_rows)
    md += "\n## Interpretation\n"
    md += bullet("If native_residual is high but transport_residual is low, the tensor is reported in a source-native frame that differs from the common CIF frame; the disagreement is a convention issue.")
    md += bullet("If both native and common residuals remain high, the tensor is inconsistent with its own symmetry and should be quarantined from componentwise comparisons.")
    write_report(REPORT_ROOT / "17_source_native_frame_audit.md", md, title="Phase 5B.2: Source-Native Frame Residual Audit")
    print("[Phase 5B.3] Wrote reports/17_source_native_frame_audit.md")


def phase_hierarchy(enriched: pd.DataFrame, core: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 5B.4] Computing discrepancy hierarchy...")
    # Restrict to rows present in panels; recompute sublayers from enriched.
    hierarchy_all = compute_hierarchy(enriched)
    core_keys = set(zip(core["jarvis_id"], core["mp_id"], strict=False))
    ext_keys = set(zip(extended["jarvis_id"], extended["mp_id"], strict=False))
    hierarchy_all["in_core"] = hierarchy_all.apply(lambda r: (r["jarvis_id"], r["mp_id"]) in core_keys, axis=1)
    hierarchy_all["in_extended"] = hierarchy_all.apply(lambda r: (r["jarvis_id"], r["mp_id"]) in ext_keys, axis=1)
    hierarchy_all.to_parquet(PHASE5B_ARTIFACT / "discrepancy_hierarchy.parquet")

    summary = hierarchy_summary(hierarchy_all)
    summary.to_parquet(PHASE5B_ARTIFACT / "discrepancy_hierarchy_summary.parquet")
    by_sub = hierarchy_by_sublayer(hierarchy_all)
    by_sub.to_parquet(PHASE5B_ARTIFACT / "discrepancy_hierarchy_by_sublayer.parquet")

    md = ""
    md += "## Discrepancy variants\n"
    md += bullet("source_native_invariant: Frobenius-norm difference; frame-independent.")
    md += bullet("exact_transported: MP tensor rotated into the JARVIS Cartesian frame.")
    md += bullet("proper_orbit: minimum over proper rotations in the common point group.")
    md += bullet("domain_aware: allows inversion-related polar-domain flip.")
    md += bullet("point_group_equivalent: minimum over the full common point group.")
    md += bullet("symmetry_projected: after Reynolds projection onto the common point group.")
    md += "\n## Whole-panel summary (Extended)\n"
    md += table_from_records(summary.to_dict("records"))
    md += "\n## By structure sublayer\n"
    md += table_from_records(by_sub.to_dict("records"))
    md += "\n## Notes\n"
    md += bullet("Normalized discrepancy after symmetry projection must be interpreted with norm-retention ratio; projection discards symmetry-forbidden components.")
    md += bullet("Component/cosine/directional metrics are only reported on Core.")
    write_report(REPORT_ROOT / "18_discrepancy_hierarchy.md", md, title="Phase 5B.3: Discrepancy Hierarchy")
    print("[Phase 5B.4] Wrote reports/18_discrepancy_hierarchy.md")
    return hierarchy_all


def phase_lambda_audit(core: pd.DataFrame, extended: pd.DataFrame) -> dict[str, Any]:
    print("[Phase 5B.9] Auditing full atom-resolved Λ recovery...")
    cfg = _load_config("data_sources.yaml")
    pj = cfg["sources"]["piezojet"]
    root = _data_root(cfg)
    factor_root = root / Path(pj["strict_factors"]["root"]).relative_to("E:/DATA")

    core_ids = set(core["jarvis_id"])
    ext_ids = set(extended["jarvis_id"])
    core_audit = audit_piezojet_lambda(factor_root, jarvis_ids=core_ids)
    ext_audit = audit_piezojet_lambda(factor_root, jarvis_ids=ext_ids)

    md = ""
    md += "## Audit scope\n"
    md += bullet(f"Core JARVIS IDs inspected: {core_audit.get('inspected', 0)}")
    md += bullet(f"Extended JARVIS IDs inspected: {ext_audit.get('inspected', 0)}")
    md += bullet(f"Total PiezoJet strict-factor files: {core_audit.get('n_total_files', 'N/A')}")
    md += "\n## Internal-strain tensor shapes (Extended)\n"
    shape_counts = ext_audit.get("internal_strain_shape_counts", {})
    md += table_from_records([{"shape": str(k), "count": v} for k, v in shape_counts.items()])
    md += f"\n- Full atom-resolved Λ candidate count (Extended): {ext_audit.get('full_lambda_candidate_count', 'N/A')}\n"
    md += "\n## Conclusion\n"
    if ext_audit.get("full_lambda_candidate_count", 0) == 0:
        md += bullet("No record contains a full 3N×6 or (n_atoms,3,3,3) atom-resolved internal-strain matrix. The available field is a reduced (3,3,3) representation.")
        md += bullet("Soft-mode mechanism claims must remain withdrawn until a third-protocol adjudication set or a recovered full Λ validates them.")
    else:
        md += bullet(f"{ext_audit['full_lambda_candidate_count']} records contain a candidate full Λ shape. A mode-resolved exploratory analysis is feasible but not executed here.")
    write_report(REPORT_ROOT / "23_full_lambda_recovery_audit.md", md, title="Phase 5B.8: Full Λ Recovery Audit")
    print(f"[Phase 5B.9] Λ audit: {ext_audit.get('full_lambda_candidate_count', 0)} candidates")
    return {"core": core_audit, "extended": ext_audit}


def phase_adjudication() -> None:
    print("[Phase 5B.10] Writing adjudication options...")
    plan = {
        "plan_name": "third_protocol_adjudication",
        "status": "pre_registered_not_executed",
        "candidate_selection": {
            "strata": [
                {"name": "core_high_disagreement", "n": 12, "criteria": "top normalized discrepancy in Core"},
                {"name": "core_low_disagreement", "n": 12, "criteria": "bottom normalized discrepancy in Core"},
                {"name": "high_response", "n": 12, "criteria": "top Frobenius norm in either source"},
                {"name": "low_response", "n": 12, "criteria": "bottom non-zero Frobenius norm"},
            ],
            "diversity": {
                "crystal_systems": ["triclinic", "monoclinic", "tetragonal", "trigonal", "hexagonal", "cubic"],
                "max_per_crystal_system": 8,
                "max_per_prototype": 3,
            },
        },
        "protocol": {
            "code": "ABINIT or Quantum ESPRESSO",
            "functional": "PBEsol or PBE",
            "pseudopotential": "ONCV or PAW (must be documented)",
            "kpoint_density": ">= 1000 k-points per reciprocal atom",
            "ecut": ">= 60 Ha (ABINIT) or equivalent plane-wave cutoff",
            "tensor_convention": "piezoelectric stress e in C/m^2, full Cartesian 3x3x3, engineering Voigt order xx,yy,zz,yz,xz,xy",
            "fail_policy": "If a candidate fails to converge or relax, replace with the next-ranked candidate in the same stratum; document all failures.",
        },
        "budget": {
            "n_materials": 48,
            "estimated_core_hours_per_material": "24-96",
            "max_wall_time_weeks": 4,
        },
        "outputs": [
            "raw_total_piezo_tensor_per_protocol",
            "relaxed_structure_and_CIF",
            "convergence_log",
            "transformation_history_json",
        ],
    }
    plan_path = PROJECT_ROOT / "configs" / "third_protocol_plan.yaml"
    with open(plan_path, "w") as f:
        yaml.dump(plan, f, sort_keys=False)

    md = ""
    md += "## Path A: database version shift\n"
    md += bullet("Survey historical MP piezo snapshots for exact old tensor, old structure, current tensor, version provenance, and matched overlap.")
    md += bullet("If available, compute version-shift discrepancy on the same strict pairs and add a version-held-out leaderboard split.")
    md += "\n## Path B: third-protocol DFPT\n"
    md += bullet(f"Pre-registered plan written to `{plan_path}`.")
    md += bullet(f"Candidate count: {plan['candidate_selection']['strata'][0]['n'] * len(plan['candidate_selection']['strata'])} before diversity filtering.")
    md += bullet(f"Target final set: {plan['budget']['n_materials']} materials.")
    md += bullet("Protocol: ABINIT or Quantum ESPRESSO, PBEsol/PBE, documented pseudopotentials, frozen tensor convention.")
    md += "\n## Recommendation\n"
    md += bullet("Proceed with Path B only after the model-validated benchmark decision is finalized; do not run new DFT during Phase 5B.")
    write_report(REPORT_ROOT / "24_adjudication_options.md", md, title="Phase 5B.9: Adjudication Options")
    print("[Phase 5B.10] Wrote reports/24_adjudication_options.md and configs/third_protocol_plan.yaml")


def main() -> int:
    _freeze_phase5a()
    enriched, extended, core, residuals = phase_panels()
    enriched = assign_sublayers(enriched)
    phase_residual_audit(residuals)
    phase_hierarchy(enriched, core, extended)
    phase_lambda_audit(core, extended)
    phase_adjudication()
    print("\n[Phase 5B data phases complete]")
    print("Next: run scripts/train_e3nn.py on the remote GPU host to produce model metrics,")
    print("      then resume this orchestrator for PMR/leaderboard/calibration/decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
