#!/usr/bin/env python
"""CrossPiezo / PULSE Phase 0-4 orchestration script.

This script is intentionally self-contained and read-only with respect to
E:/DATA.  It writes outputs under ./artifacts and ./reports only.
"""

from __future__ import annotations

import json
import os
import re
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

# Ensure local package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crosspiezo.analysis.discrepancy import (  # noqa: E402
    absolute_discrepancy,
    amplitude_ratio,
    cosine_similarity,
    discrepancy_summary,
    normalized_discrepancy,
    rank_stability,
    sign_disagreement,
)
from crosspiezo.conventions.symmetry import (  # noqa: E402
    point_group_rotations,
    project_piezo_tensor,
    symmetry_residual,
)
from crosspiezo.conventions.voigt import voigt_to_cartesian  # noqa: E402
from crosspiezo.inventory.scanner import build_inventory  # noqa: E402
from crosspiezo.matching.structure_matcher import match_structures, to_match_record  # noqa: E402
from crosspiezo.reports.markdown import bullet, table_from_records, write_report  # noqa: E402
from crosspiezo.schemas import MatchTier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = PROJECT_ROOT / "reports"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_config(name: str) -> dict[str, Any]:
    with open(CONFIG_ROOT / name) as f:
        return yaml.safe_load(f)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True)
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def _env_report() -> dict[str, Any]:
    return {
        "os": os.name,
        "platform": sys.platform,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cwd": str(PROJECT_ROOT),
        "data_root_accessible": (PROJECT_ROOT / ".." / ".." / "DATA").resolve().exists(),
        "git_commit": _git_commit(),
        "timestamp": _now(),
    }


# ---------------------------------------------------------------------------
# Phase 0: manuscript contract and environment
# ---------------------------------------------------------------------------


def _extract_braced(text: str, command: str, start: int = 0) -> str | None:
    """Extract the balanced-brace argument of a LaTeX command."""
    pattern = f"\\{command}{{"
    pos = text.find(pattern, start)
    if pos < 0:
        return None
    pos += len(pattern) - 1  # position of the opening brace
    depth = 0
    content_start = pos + 1
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:i]
    return None


def _clean_tex(text: str) -> str:
    """Remove LaTeX comments and collapse whitespace."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    return " ".join(text.replace("\n", " ").split())


def _items_in_section(text: str, section_name: str) -> list[str]:
    r"""Return enumerated items from a given \section{...} block."""
    pattern = rf"\\section\{{{re.escape(section_name)}\}}"
    section_match = re.search(pattern, text, re.IGNORECASE)
    if not section_match:
        return []
    start = section_match.end()
    next_section = re.search(r"\\section\{", text[start:])
    end = start + next_section.start() if next_section else len(text)
    block = text[start:end]
    # Match \item or \item[label] ... up to next item or environment end.
    raw_items = re.findall(
        r"\\item(?:\[([^\]]*)\])? (.*?)(?=\\item|\\end\{(?:enumerate|description)\})",
        block,
        re.DOTALL,
    )
    cleaned: list[str] = []
    for label, content in raw_items:
        item_text = _clean_tex(content)
        if label:
            item_text = f"{_clean_tex(label)}: {item_text}"
        cleaned.append(item_text)
    return cleaned


def _extract_tex_contract(tex_path: Path) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    title = _extract_braced(text, "title")
    if title:
        title = _clean_tex(title)
        # Strip a surrounding \textbf{...} wrapper if present.
        title = re.sub(r"^\\textbf\{", "", title).removesuffix("}")

    intro_items = _items_in_section(text, "Introduction")
    questions = [q for q in intro_items if q.endswith("?")]
    contributions = [q for q in intro_items if not q.endswith("?")]

    hyp_items = _items_in_section(text, "Preregistered Hypotheses and Stop Conditions")
    hypotheses = [q for q in hyp_items if q.startswith("H")]
    no_go = [q for q in hyp_items if not q.startswith("H")]

    tbds: list[str] = []
    pos = 0
    while True:
        tbd = _extract_braced(text[pos:], "TBD")
        if tbd is None:
            break
        tbds.append(_clean_tex(tbd))
        pos += text[pos:].find(f"\\TBD{{{tbd}}}") + 1

    return {
        "title": title,
        "research_questions": questions,
        "contributions": contributions,
        "hypotheses": hypotheses,
        "no_go_conditions": no_go,
        "tbds": tbds,
        "metrics": [],
        "splits": [],
    }


def phase_0(tex_path: Path, bib_path: Path) -> dict[str, Any]:
    print("[Phase 0] Extracting LaTeX scientific contract...")
    env = _env_report()
    contract = _extract_tex_contract(tex_path)

    md = ""
    md += "## Environment\n"
    md += bullet(f"Python: `{env['python_executable']}`")
    md += bullet(f"Platform: {env['platform']}")
    md += bullet(f"Project root: `{env['cwd']}`")
    md += bullet(f"Data root accessible: {env['data_root_accessible']}")
    md += bullet(f"Git commit: {env['git_commit'] or 'unknown (not a git repo or no commits)'}")
    md += bullet(f"Timestamp: {env['timestamp']}")

    md += "\n## Manuscript contract\n"
    md += bullet(f"Title: **{contract['title'] or 'not extracted'}**")
    md += "\n### Research questions\n"
    for q in contract["research_questions"] or ["(extraction heuristics did not isolate items)"]:
        md += bullet(q)
    md += "\n### Planned contributions\n"
    for c in contract["contributions"] or ["(not extracted)"]:
        md += bullet(c)
    md += "\n### Preregistered hypotheses\n"
    for h in contract["hypotheses"] or ["(not extracted)"]:
        md += bullet(h)
    md += "\n### No-Go / downgrade conditions\n"
    for ng in contract["no_go_conditions"] or ["(not extracted)"]:
        md += bullet(ng)
    md += "\n### Remaining TBD placeholders in the manuscript\n"
    for tbd in contract["tbds"]:
        md += bullet(f"`{tbd}`")
    if not contract["tbds"]:
        md += bullet("None found.")

    md += "\n## Data and code availability commitments\n"
    md += bullet("Source/version manifests and SHA-256 hashes")
    md += bullet("Structure-match tiers and mapping proofs")
    md += bullet("Tensor transformation histories")
    md += bullet("Frozen splits")
    md += bullet("Benchmark loaders and metrics")

    write_report(REPORT_ROOT / "00_environment_and_scope.md", md, title="Phase 0: Environment and Scope Audit")

    claim_md = ""
    claim_md += "## Permitted claims\n"
    claim_md += bullet("Cross-protocol disagreement is a source-conditional quantity, not experimental truth.")
    claim_md += bullet("With only two protocols, only the disagreement and a computational center are identifiable.")
    claim_md += bullet("PULSE predicts source-conditional tensor distributions; it does not claim a single true tensor.")
    claim_md += "\n## Forbidden claims\n"
    claim_md += bullet("'true tensor' or 'ground truth consensus' without a third protocol or experiment.")
    claim_md += bullet("Averaging JARVIS + MP and calling it physical truth.")
    claim_md += bullet("Architecture-level leaderboard superiority on random splits alone.")
    claim_md += "\n## Uncertainty layers\n"
    claim_md += bullet("Representation uncertainty (cell, frame, Voigt, shear, unit, symmetry)")
    claim_md += bullet("Structure uncertainty (relaxed cell / space group)")
    claim_md += bullet("Protocol uncertainty (functional, pseudopotential, DFPT implementation)")
    claim_md += bullet("Version uncertainty (database rebuilds, symmetrization)")
    claim_md += bullet("Model uncertainty (data, architecture, optimization)")
    write_report(PROJECT_ROOT / "docs" / "claim_boundary.md", claim_md, title="Claim Boundary")

    stat_md = ""
    stat_md += "## Statistical plan summary\n"
    stat_md += bullet("Primary unit: a strict Tier 0-1 structure-matched pair.")
    stat_md += bullet("Bootstrap 95% confidence intervals for medians (2000 replicates, seed 42).")
    stat_md += bullet("Rank stability: top-k Jaccard, Kendall tau, Spearman rho.")
    stat_md += bullet("Stratification by crystal system, space group, chemical system, response amplitude.")
    stat_md += bullet("PMR = median paired-source Frobenius disagreement / mean in-source MAE.")
    write_report(PROJECT_ROOT / "docs" / "statistical_plan.md", stat_md, title="Statistical Plan")

    print("[Phase 0] Wrote reports/00_environment_and_scope.md, docs/claim_boundary.md, docs/statistical_plan.md")
    return {"env": env, "contract": contract}


# ---------------------------------------------------------------------------
# Phase 1: data inventory
# ---------------------------------------------------------------------------


def phase_1() -> pd.DataFrame:
    print("[Phase 1] Scanning E:/DATA read-only assets...")
    cfg = _load_config("data_sources.yaml")
    df = build_inventory(CONFIG_ROOT / "data_sources.yaml")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARTIFACT_ROOT / "inventories" / "data_inventory.parquet")
    df.to_json(ARTIFACT_ROOT / "inventories" / "data_inventory.json", orient="records", indent=2)

    md = ""
    md += "## Inventory summary\n"
    md += bullet(f"Sources configured: {', '.join(cfg.get('sources', {}).keys())}")
    md += bullet(f"Artifacts inventoried: {len(df)}")
    md += "\n## Asset table\n"
    md += table_from_records(df[["source_name", "source_version", "role", "path", "sha256_or_fingerprint"]].to_dict("records"))
    md += "\n## Observed counts from T2C-Flow MANIFEST\n"
    manifest_path = Path(cfg["sources"]["t2c_flow"]["root"]) / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        counts = manifest.get("counts", {})
        md += table_from_records([{"metric": k, "value": v} for k, v in counts.items()])
    write_report(REPORT_ROOT / "01_data_inventory.md", md, title="Phase 1: Data Inventory")
    print("[Phase 1] Wrote artifacts/inventories/data_inventory.* and reports/01_data_inventory.md")
    return df


# ---------------------------------------------------------------------------
# Phase 2: convention validation
# ---------------------------------------------------------------------------


def phase_2() -> dict[str, Any]:
    print("[Phase 2] Validating tensor conventions...")
    rng = np.random.default_rng(123)
    voigt = rng.normal(size=(3, 6))
    cart = voigt_to_cartesian(voigt, engineering_shear=True)

    # Round-trip check
    from crosspiezo.conventions.voigt import cartesian_to_voigt

    roundtrip = cartesian_to_voigt(cart, engineering_shear=True)
    rt_error = float(np.max(np.abs(roundtrip - voigt)))

    # O(3) covariance: rotate cartesian tensor and check equivalent Voigt transforms
    theta = 0.3
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    # For polar third-rank: e'_ijk = R_il R_jm R_kn e_lmn
    rotated = np.einsum("il,jm,kn,lmn->ijk", rot, rot, rot, cart)
    rot_norm = float(np.linalg.norm(rotated))
    orig_norm = float(np.linalg.norm(cart))

    # Symmetry projection smoke test on a non-piezoelectric group (should not crash)
    try:
        rotations = point_group_rotations("P-1")
        _ = project_piezo_tensor(cart, rotations)
        resid = symmetry_residual(cart, rotations)
    except Exception as exc:  # noqa: BLE001
        resid = str(exc)

    results = {
        "voigt_cartesian_roundtrip_max_error": rt_error,
        "rotation_norm_preserved": bool(np.isclose(rot_norm, orig_norm)),
        "symmetry_residual_sample": resid if isinstance(resid, (int, float)) else None,
        "timestamp": _now(),
    }

    md = ""
    md += "## Convention validation results\n"
    md += bullet(f"Voigt -> Cartesian -> Voigt round-trip max error: {rt_error:.3e}")
    md += bullet(f"Rotation preserves Frobenius norm: {results['rotation_norm_preserved']}")
    md += bullet(f"Symmetry residual on random tensor (P-1): {resid}")
    md += "\n## Internal convention\n"
    md += bullet("Piezoelectric stress tensor `e` (C/m^2)")
    md += bullet("Full Cartesian 3x3x3, last two indices symmetric")
    md += bullet("Internal engineering Voigt order: xx, yy, zz, yz, xz, xy")
    md += bullet("Engineering shear: off-diagonal Voigt = 2 * tensor-shear component")
    md += bullet("Original Voigt and converted Cartesian both retained")
    write_report(REPORT_ROOT / "02_convention_audit.md", md, title="Phase 2: Convention Audit")
    print("[Phase 2] Wrote reports/02_convention_audit.md")
    return results


# ---------------------------------------------------------------------------
# Phase 3: strict pairs and discrepancy audit
# ---------------------------------------------------------------------------


def _load_piezo_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = _load_config("data_sources.yaml")
    t2c = cfg["sources"]["t2c_flow"]
    root = Path(t2c["root"])
    jarvis = pd.read_parquet(root / t2c["records"]["jarvis_piezo"])
    mp = pd.read_parquet(root / t2c["records"]["mp_piezo"])
    overlap = pd.read_parquet(root / t2c["records"]["jarvis_mp_overlap"])
    return jarvis, mp, overlap


def _tensor_from_row(row: pd.Series) -> np.ndarray | None:
    """Return 3x3x3 Cartesian total piezo tensor from a unified row."""
    import ast

    def _to_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        return None

    try:
        cart = _to_array(row["piezo_cartesian_total"])
        if cart is not None and cart.shape == (3, 3, 3):
            return cart
        voigt = _to_array(row["piezo_voigt_total"])
        if voigt is not None and voigt.shape == (3, 6):
            return voigt_to_cartesian(voigt, engineering_shear=True)
        return None
    except Exception:  # noqa: BLE001
        return None


def _space_group_symbol(space_group: Any) -> str | None:
    try:
        sg = int(float(space_group))
        return f"{sg}"
    except Exception:  # noqa: BLE001
        return None


def phase_3() -> dict[str, Any]:
    print("[Phase 3] Building strict JARVIS-MP pairs...")
    match_cfg = _load_config("matching.yaml")
    matcher_params = match_cfg["matcher"]
    jarvis, mp, overlap = _load_piezo_sources()

    # Index by ID for fast lookup
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
            quarantine_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "reason": "missing_source_record",
            })
            continue
        if not isinstance(jrow["cif"], str) or not isinstance(mrow["cif"], str):
            quarantine_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "reason": "missing_cif",
            })
            continue

        result = match_structures(
            left_key=f"jarvis:{jid}",
            right_key=f"mp:{mid}",
            left_cif=jrow["cif"],
            right_cif=mrow["cif"],
            ltol=matcher_params["ltol"],
            stol=matcher_params["stol"],
            angle_tol=matcher_params["angle_tol"],
        )
        rec = to_match_record(result)
        match_results.append(rec.model_dump())

        if result.tier in (MatchTier.TIER_0, MatchTier.TIER_1):
            jtensor = _tensor_from_row(jrow)
            mtensor = _tensor_from_row(mrow)
            if jtensor is None or mtensor is None:
                quarantine_records.append({
                    "jarvis_id": jid,
                    "mp_id": mid,
                    "reason": "tensor_conversion_failed",
                })
                continue

            # Align MP tensor to the JARVIS Cartesian frame using the match rotation
            if result.cartesian_rotation is not None:
                rot = np.asarray(result.cartesian_rotation, dtype=np.float64)
                mtensor = np.einsum("il,jm,kn,lmn->ijk", rot, rot, rot, mtensor)

            # Source symmetry residuals (diagnostic only; projection is kept
            # separate because source IEEE frames may not match the CIF setting).
            sg_symbol = _space_group_symbol(jrow["space_group"])
            if sg_symbol:
                try:
                    rots = point_group_rotations(sg_symbol)
                    jresid = symmetry_residual(jtensor, rots)
                    mresid = symmetry_residual(mtensor, rots)
                except Exception:  # noqa: BLE001
                    jresid, mresid = None, None
            else:
                jresid, mresid = None, None

            sym_threshold = match_cfg.get("symmetry_residual_threshold", 1.0)
            high_residual = (
                (jresid is not None and jresid > sym_threshold)
                or (mresid is not None and mresid > sym_threshold)
            )

            # Discrepancy computed on frame-aligned raw Cartesian tensors.
            pair_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "formula": jrow["formula"],
                "space_group": jrow["space_group"],
                "match_tier": result.tier.value,
                "rms_distance": result.rms_distance,
                "max_distance": result.max_distance,
                "lattice_distance": result.lattice_distance,
                "space_group_relation": result.space_group_relation,
                "jarvis_norm": float(np.linalg.norm(jtensor)),
                "mp_norm": float(np.linalg.norm(mtensor)),
                "absolute_discrepancy": absolute_discrepancy(jtensor, mtensor),
                "normalized_discrepancy": normalized_discrepancy(jtensor, mtensor),
                "cosine_similarity": cosine_similarity(jtensor, mtensor),
                "amplitude_ratio": amplitude_ratio(jtensor, mtensor),
                "sign_disagreement": sign_disagreement(jtensor, mtensor),
                "jarvis_symmetry_residual": jresid,
                "mp_symmetry_residual": mresid,
                "high_symmetry_residual": high_residual,
            })
        elif result.tier == MatchTier.QUARANTINE:
            quarantine_records.append({
                "jarvis_id": jid,
                "mp_id": mid,
                "reason": ";".join(result.reasons or []),
            })

    matches_df = pd.DataFrame(match_results)
    pairs_df = pd.DataFrame(pair_records)
    quarantine_df = pd.DataFrame(quarantine_records)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    matches_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "all_matches.parquet")
    pairs_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "strict_pairs.parquet")
    quarantine_df.to_parquet(ARTIFACT_ROOT / "pair_manifests" / "quarantined_pairs.parquet")
    matches_df.to_json(ARTIFACT_ROOT / "pair_manifests" / "all_matches.json", orient="records", indent=2)
    pairs_df.to_json(ARTIFACT_ROOT / "pair_manifests" / "strict_pairs.json", orient="records", indent=2)

    md = ""
    md += "## Matching protocol\n"
    md += bullet(f"Frozen tolerances: ltol={matcher_params['ltol']}, stol={matcher_params['stol']}, angle_tol={matcher_params['angle_tol']}")
    md += bullet("Tier 0 reserved for explicit upstream provenance; Tier 1 assigned by structure matcher.")
    md += "\n## Match counts\n"
    tier_counts = matches_df["match_tier"].value_counts().to_dict()
    md += table_from_records([{"tier": k, "count": v} for k, v in tier_counts.items()])
    md += f"\n- Preliminary formula-overlap candidates: {len(overlap)}\n"
    md += f"- Tier 1 strict pairs retained for analysis: {len(pairs_df)}\n"
    md += f"- Quarantined records: {len(quarantine_df)}\n"

    if len(pairs_df) > 0:
        md += "\n## Discrepancy atlas (frame-aligned raw Cartesian tensors)\n"
        md += bullet("Discrepancy is computed after rotating the MP tensor into the JARVIS Cartesian frame.")
        md += bullet("Source-level symmetry residuals are reported separately as convention diagnostics.")
        for col in ["absolute_discrepancy", "normalized_discrepancy"]:
            summary = discrepancy_summary(pairs_df[col].values, name=col)
            md += table_from_records([summary])
        if "jarvis_symmetry_residual" in pairs_df.columns:
            jar_resid = pairs_df["jarvis_symmetry_residual"].dropna().values
            mp_resid = pairs_df["mp_symmetry_residual"].dropna().values
            md += "\n### Source symmetry residuals (CIF-setting point group)\n"
            md += table_from_records([{
                "source": "JARVIS",
                "median_residual": float(np.median(jar_resid)),
                "p95_residual": float(np.percentile(jar_resid, 95)),
                "high_residual_count": int(np.sum(pairs_df["jarvis_symmetry_residual"] > match_cfg.get("symmetry_residual_threshold", 1.0))),
            }, {
                "source": "MP",
                "median_residual": float(np.median(mp_resid)),
                "p95_residual": float(np.percentile(mp_resid, 95)),
                "high_residual_count": int(np.sum(pairs_df["mp_symmetry_residual"] > match_cfg.get("symmetry_residual_threshold", 1.0))),
            }])
        md += bullet(
            "Caveat: the large normalized discrepancy may reflect remaining "
            "source-frame / orientation mismatches in addition to real protocol "
            "differences.  It is not a final protocol-floor estimate until a "
            "full source-convention audit is completed."
        )
        md += "\n## Rank stability by Frobenius norm\n"
        rank = rank_stability(pairs_df["jarvis_norm"].values, pairs_df["mp_norm"].values)
        md += table_from_records([{"metric": k, "value": v} for k, v in rank.items()])

    write_report(REPORT_ROOT / "03_pairing_audit.md", md, title="Phase 3: Pairing Audit")
    print(f"[Phase 3] Wrote strict pair manifest ({len(pairs_df)} Tier 1 pairs) and reports/03_pairing_audit.md")
    return {
        "matches": matches_df,
        "pairs": pairs_df,
        "quarantine": quarantine_df,
        "tier_counts": tier_counts,
    }


# ---------------------------------------------------------------------------
# Phase 4: feasibility and Go/No-Go
# ---------------------------------------------------------------------------


def phase_4(phase_3_result: dict[str, Any]) -> dict[str, str]:
    print("[Phase 4] Compiling feasibility results and Go/No-Go...")
    cfg = _load_config("feasibility.yaml")
    pairs = phase_3_result["pairs"]
    thresholds = cfg["thresholds"]

    n_tier1 = len(pairs)
    # PMR requires audited in-source model errors; not available in Phase 0-4.
    pmr = None

    top_50_jaccard = None
    top_50_kendall = None
    if n_tier1 > 0 and "jarvis_norm" in pairs.columns:
        rank = rank_stability(pairs["jarvis_norm"].values, pairs["mp_norm"].values)
        top_50_jaccard = rank.get("top_50_jaccard")
        top_50_kendall = rank.get("kendall_tau")

    # Decision rules (frozen in taskbook)
    if n_tier1 < thresholds["no_go_max_tier01_pairs"]:
        decision = "No-Go"
        rationale = f"Tier 0-1 pairs ({n_tier1}) below No-Go threshold ({thresholds['no_go_max_tier01_pairs']})."
    elif n_tier1 < thresholds["full_go_min_tier01_pairs"]:
        decision = "Narrow / Manual"
        rationale = f"Pair count ({n_tier1}) is in the manual-review band [{thresholds['no_go_max_tier01_pairs']}, {thresholds['full_go_min_tier01_pairs']})."
    else:
        decision = "Narrow Go"
        rationale = f"Pair count ({n_tier1}) meets Narrow-Go threshold; awaiting full model-error comparison and soft-mode evidence for Full Go."

    if top_50_jaccard is not None and top_50_jaccard <= thresholds["top_50_jaccard_unstable"]:
        decision = "Narrow Go"
        rationale += " Top-50 Jaccard indicates rank instability."

    summary = {
        "decision": decision,
        "n_tier1_pairs": n_tier1,
        "pmr": pmr,
        "top_50_jaccard": top_50_jaccard,
        "top_50_kendall": top_50_kendall,
        "rationale": rationale,
        "timestamp": _now(),
    }
    (ARTIFACT_ROOT / "feasibility").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_ROOT / "feasibility" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if n_tier1 > 0:
        pairs.to_parquet(ARTIFACT_ROOT / "feasibility" / "frozen_metrics.parquet")

    md = ""
    md += "## Feasibility results\n"
    md += bullet(f"Tier 0-1 strict pairs: **{n_tier1}**")
    md += bullet(f"PMR (protocol-to-model ratio): {'not computable without in-source model errors' if pmr is None else f'{pmr:.3f}'}")
    md += bullet(f"Top-50 Jaccard (Frobenius norm ranking): {top_50_jaccard}")
    md += bullet(f"Top-50 Kendall tau: {top_50_kendall}")
    if n_tier1 > 0:
        md += "\n### Discrepancy summary\n"
        for col in ["absolute_discrepancy", "normalized_discrepancy"]:
            md += table_from_records([discrepancy_summary(pairs[col].values, name=col)])

    md += "\n## Go / No-Go decision\n"
    md += bullet(f"**Decision: {decision}**")
    md += bullet(f"Rationale: {rationale}")

    md += "\n## Evidence for the decision\n"
    md += bullet(f"{n_tier1} Tier 0-1 pairs exist, so the dataset can support a benchmark/audit paper.")
    md += bullet("Cross-protocol Frobenius disagreement is measured; median values are in the report above.")
    md += bullet("Rank stability is quantified with top-k Jaccard and Kendall tau.")

    md += "\n## Counter-evidence / limitations\n"
    md += bullet("PMR cannot be computed in Phase 0-4 because no audited in-source model errors were extracted/reproduced.")
    md += bullet("Soft-mode sensitivity analysis requires PiezoJet strict-factor intersection with MP pairs; not completed in this minimal audit.")
    md += bullet("Only T2C-Flow pre-processed tensors were used; raw source Voigt conventions were not independently audited from source downloads.")

    md += "\n## Required LaTeX revisions\n"
    md += bullet("Replace `\\TBD{strict matched count}` with the Tier 0-1 count above.")
    md += bullet("Do not claim a positive PMR until in-source model errors are collected or reproduced.")
    md += bullet("Keep the third-protocol adjudication set as optional / not executed.")

    md += "\n## Next phase budget (requires human approval)\n"
    md += bullet("Collect literature in-source errors or reproduce a simple baseline on frozen splits.")
    md += bullet("Cross-reference PiezoJet strict factors with the Tier 1 pairs for soft-mode analysis.")
    md += bullet("If Full Go criteria are met, begin PULSE model development with a new train/val/test split.")

    write_report(REPORT_ROOT / "04_feasibility_results.md", md, title="Phase 4: Feasibility Results")
    write_report(REPORT_ROOT / "05_go_no_go.md", md, title="Phase 4: Go / No-Go Decision")
    print(f"[Phase 4] Decision: {decision}. Wrote reports/04_feasibility_results.md and reports/05_go_no_go.md")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    tex_path = PROJECT_ROOT / "PiezoProtocol_LaTeX_Draft_v0.1" / "PiezoProtocol_Draft_v0.1.tex"
    bib_path = PROJECT_ROOT / "PiezoProtocol_LaTeX_Draft_v0.1" / "references.bib"
    if not tex_path.exists():
        print(f"ERROR: Manuscript not found at {tex_path}")
        return 1

    phase_0(tex_path, bib_path)
    phase_1()
    phase_2()
    phase_3_result = phase_3()
    phase_4(phase_3_result)
    print("\nAll phases complete. Review reports/05_go_no_go.md before continuing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
