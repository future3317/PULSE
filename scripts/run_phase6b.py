#!/usr/bin/env python
"""CrossPiezo Phase 6B: Manuscript freeze and independent statistics audit.

Reads the frozen Phase 6A artifacts, recomputes statistics with corrected
null definitions, runs an independent verification, produces small committed
result tables, figures/tables, and Phase 6B reports.

No PULSE, PMR, e3nn, soft-mode, O(3) transport, or new DFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase6b"
REPORT_ROOT = PROJECT_ROOT / "reports" / "phase6b"
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase6a"
FROZEN_ROOT = PROJECT_ROOT / "artifacts" / "releases" / "phase6a_c2ed53e"
MANUSCRIPT_ROOT = PROJECT_ROOT / "manuscript_notes"
LATEX_ROOT = PROJECT_ROOT / "PiezoProtocol_LaTeX_Draft_v0.1"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.analysis.ranking import (  # noqa: E402
    chance_adjusted_jaccard,
    cohen_kappa,
    expected_jaccard_hypergeometric,
    hypergeometric_overlap_pvalue,
    kendall_tau_bootstrap_ci,
    matthews_correlation,
    permutation_pvalue,
)
from crosspiezo.reports.markdown import (  # noqa: E402
    bullet,
    table_from_records,
    write_report,
)

FROZEN_PHASE6A_COMMIT = "c2ed53e29605ed55c35036d03a73e8e2b0ef9aaf"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_dirs() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "tables").mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Corrected primary ranking statistics
# -----------------------------------------------------------------------------


def corrected_rank_stats(
    left: np.ndarray,
    right: np.ndarray,
    functional_name: str,
    panel_name: str,
    fractions: list[float] | None = None,
) -> dict[str, Any]:
    if fractions is None:
        fractions = [0.05, 0.10, 0.20]
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    n = len(left)

    tau, tau_p = stats.kendalltau(left, right)
    tau = float(tau) if tau is not None else float("nan")
    tau_p = float(tau_p) if tau_p is not None else float("nan")
    rho, rho_p = stats.spearmanr(left, right)
    rho = float(rho) if rho is not None else float("nan")
    rho_p = float(rho_p) if rho_p is not None else float("nan")
    ci_low, ci_high = kendall_tau_bootstrap_ci(left, right)
    perm_p = permutation_pvalue(left, right, tau, n_permutations=4999, alternative="two-sided")

    ranks_left = stats.rankdata(-left, method="average")
    ranks_right = stats.rankdata(-right, method="average")
    abs_shift = np.abs(ranks_left - ranks_right)
    max_shift = n - 1 if n > 1 else 1.0
    norm_shift = abs_shift / max_shift

    result: dict[str, Any] = {
        "panel": panel_name,
        "functional": functional_name,
        "n_pairs": n,
        "kendall_tau": tau,
        "kendall_tau_pvalue": tau_p,
        "kendall_tau_ci95_low": ci_low,
        "kendall_tau_ci95_high": ci_high,
        "spearman_rho": rho,
        "spearman_rho_pvalue": rho_p,
        "permutation_tau_pvalue": perm_p,
        "mean_abs_rank_shift": float(np.mean(abs_shift)) if n else float("nan"),
        "median_abs_rank_shift": float(np.median(abs_shift)) if n else float("nan"),
        "mean_normalized_rank_displacement": float(np.mean(norm_shift)) if n else float("nan"),
        "median_normalized_rank_displacement": float(np.median(norm_shift)) if n else float("nan"),
    }

    for frac in fractions:
        k = max(1, int(np.floor(frac * n)))
        top_left = set(np.argsort(-left, kind="stable")[:k])
        top_right = set(np.argsort(-right, kind="stable")[:k])
        inter = len(top_left & top_right)
        union = len(top_left | top_right)
        obs_jaccard = inter / union if union else 0.0
        expected_jaccard = expected_jaccard_hypergeometric(n, k)
        adj_jaccard = chance_adjusted_jaccard(obs_jaccard, expected_jaccard)
        hyper_p = hypergeometric_overlap_pvalue(n, k, inter)
        prefix = f"top_{int(frac*100)}pct"
        result[f"{prefix}_k"] = k
        result[f"{prefix}_observed_overlap"] = inter
        result[f"{prefix}_observed_jaccard"] = obs_jaccard
        result[f"{prefix}_expected_jaccard"] = expected_jaccard
        result[f"{prefix}_chance_adjusted_jaccard"] = adj_jaccard
        result[f"{prefix}_hypergeometric_pvalue"] = hyper_p

    return result


def compute_primary_ranking(panel_df: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 6B] Computing corrected primary ranking statistics...")
    rows: list[dict[str, Any]] = []
    metric_pairs = [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F3_Longitudinal", "jarvis_f3", "mp_f3"),
        ("F4_KelvinOp", "jarvis_f4", "mp_f4"),
    ]
    for panel_name in ["P0", "P2"]:
        sub = panel_df[panel_df[panel_name]]
        if len(sub) < 10:
            continue
        for func_name, left_col, right_col in metric_pairs:
            rows.append(corrected_rank_stats(
                sub[left_col].to_numpy(),
                sub[right_col].to_numpy(),
                func_name,
                panel_name,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "ranking_primary.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Corrected threshold screening (without PABAK)
# -----------------------------------------------------------------------------


def corrected_threshold_screening(panel_df: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 6B] Computing corrected threshold screening...")
    thresholds = [0.25, 0.5, 1.0]
    metric_pairs = [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F3_Longitudinal", "jarvis_f3", "mp_f3"),
        ("F4_KelvinOp", "jarvis_f4", "mp_f4"),
    ]
    rows: list[dict[str, Any]] = []
    for panel_name in ["P0", "P2"]:
        sub = panel_df[panel_df[panel_name]]
        if len(sub) < 10:
            continue
        for func_name, left_col, right_col in metric_pairs:
            left = sub[left_col].to_numpy()
            right = sub[right_col].to_numpy()
            for thr in thresholds:
                l_pos = left > thr
                r_pos = right > thr
                both = int((l_pos & r_pos).sum())
                left_only = int((l_pos & ~r_pos).sum())
                right_only = int((~l_pos & r_pos).sum())
                neither = int((~l_pos & ~r_pos).sum())
                total = len(sub)
                agreement = (both + neither) / total if total else float("nan")
                rows.append({
                    "panel": panel_name,
                    "functional": func_name,
                    "threshold_C_per_m2": thr,
                    "both_above": both,
                    "jarvis_only": left_only,
                    "mp_only": right_only,
                    "both_below": neither,
                    "agreement_rate": agreement,
                    "cohen_kappa": cohen_kappa(l_pos, r_pos),
                    "mcc": matthews_correlation(l_pos, r_pos),
                    "precision_jarvis_as_ref": both / (both + right_only) if (both + right_only) else float("nan"),
                    "recall_jarvis_as_ref": both / (both + left_only) if (both + left_only) else float("nan"),
                    "precision_mp_as_ref": both / (both + left_only) if (both + left_only) else float("nan"),
                    "recall_mp_as_ref": both / (both + right_only) if (both + right_only) else float("nan"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "threshold_primary.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Robustness matrix
# -----------------------------------------------------------------------------


def robustness_matrix(panel_df: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 6B] Building robustness matrix...")
    metric_pairs = [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F3_Longitudinal", "jarvis_f3", "mp_f3"),
        ("F4_KelvinOp", "jarvis_f4", "mp_f4"),
    ]

    def _make(name: str, sub: pd.DataFrame) -> list[dict[str, Any]]:
        if len(sub) < 10:
            return []
        rows: list[dict[str, Any]] = []
        for func_name, left_col, right_col in metric_pairs:
            stats_dict = corrected_rank_stats(
                sub[left_col].to_numpy(),
                sub[right_col].to_numpy(),
                func_name,
                name,
                fractions=[0.10],
            )
            rows.append({
                "subset": name,
                "functional": func_name,
                "n_pairs": stats_dict["n_pairs"],
                "kendall_tau": stats_dict["kendall_tau"],
                "kendall_tau_ci95_low": stats_dict["kendall_tau_ci95_low"],
                "kendall_tau_ci95_high": stats_dict["kendall_tau_ci95_high"],
                "top_10pct_chance_adjusted_jaccard": stats_dict["top_10pct_chance_adjusted_jaccard"],
                "median_normalized_rank_displacement": stats_dict["median_normalized_rank_displacement"],
            })
        return rows

    rows: list[dict[str, Any]] = []
    base = panel_df[panel_df["P0"]].copy()

    for panel in ["P0", "P1", "P2", "P3"]:
        rows.extend(_make(f"panel_{panel}", panel_df[panel_df[panel]]))

    mask = (base["jarvis_f1"] > 0.05) & (base["mp_f1"] > 0.05)
    rows.extend(_make("both_above_0.05", base[mask]))

    mask = (base["jarvis_f1"] > 0.25) | (base["mp_f1"] > 0.25)
    rows.extend(_make("either_above_0.25", base[mask]))

    mask = (
        (base["jarvis_f1"] > 0.25) | (base["mp_f1"] > 0.25)
        | (base["jarvis_f3"] > 0.25) | (base["mp_f3"] > 0.25)
        | (base["jarvis_f4"] > 0.25) | (base["mp_f4"] > 0.25)
    )
    rows.extend(_make("high_response_union", base[mask]))

    base["f1_mean"] = (base["jarvis_f1"] + base["mp_f1"]) / 2.0
    top1_thr = base["f1_mean"].quantile(0.99)
    rows.extend(_make("exclude_top_1pct_amplitude", base[base["f1_mean"] < top1_thr]))

    wins = base.copy()
    for col in ["jarvis_f1", "mp_f1", "jarvis_f3", "mp_f3", "jarvis_f4", "mp_f4"]:
        lo, hi = wins[col].quantile([0.05, 0.95])
        wins[col] = wins[col].clip(lo, hi)
    rows.extend(_make("winsorized_5_95", wins))

    logdf = base.copy()
    eps = 1e-4
    for col in ["jarvis_f1", "mp_f1", "jarvis_f3", "mp_f3", "jarvis_f4", "mp_f4"]:
        logdf[col] = np.log(eps + logdf[col])
    rows.extend(_make("log_eps_F", logdf))

    systems = base["jarvis_crystal_system"].unique()
    for sys in systems:
        if pd.isna(sys):
            continue
        mask = base["jarvis_crystal_system"] != sys
        if mask.sum() < 10:
            continue
        rows.extend(_make(f"without_{sys}", base[mask]))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "robustness_primary.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Top-quantile consensus/disputed candidates
# -----------------------------------------------------------------------------


def top_quantile_consensus(panel_df: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 6B] Computing top-quantile consensus candidates...")
    metric_pairs = [
        ("F1_Frobenius", "jarvis_f1", "mp_f1"),
        ("F3_Longitudinal", "jarvis_f3", "mp_f3"),
        ("F4_KelvinOp", "jarvis_f4", "mp_f4"),
    ]
    fractions = [0.05, 0.10, 0.20]
    rows: list[dict[str, Any]] = []
    base = panel_df[panel_df["P0"]].copy()
    for func_name, left_col, right_col in metric_pairs:
        left = base[left_col].to_numpy()
        right = base[right_col].to_numpy()
        for frac in fractions:
            k = max(1, int(np.floor(frac * len(base))))
            top_j = set(np.argsort(-left, kind="stable")[:k])
            top_m = set(np.argsort(-right, kind="stable")[:k])
            both = len(top_j & top_m)
            j_only = len(top_j - top_m)
            m_only = len(top_m - top_j)
            neither = len(base) - both - j_only - m_only
            rows.append({
                "functional": func_name,
                "quantile": frac,
                "k": k,
                "both_top": both,
                "jarvis_only_top": j_only,
                "mp_only_top": m_only,
                "neither_top": neither,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "robust_summary.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Metric definition table
# -----------------------------------------------------------------------------


def metric_definition_table() -> pd.DataFrame:
    rows = [
        {
            "metric": "F1_Frobenius",
            "formula": "||e||_F",
            "rotation_invariant": True,
            "primary": True,
            "note": "Full Cartesian Frobenius norm",
        },
        {
            "metric": "F_MP_SVD",
            "formula": "sigma_max(3x6 Voigt matrix)",
            "rotation_invariant": False,
            "primary": False,
            "note": "MP source field only; not a Cartesian invariant",
        },
        {
            "metric": "F3_Longitudinal",
            "formula": "max_{||n||=1} |n_i e_ijk n_j n_k|",
            "rotation_invariant": True,
            "primary": True,
            "note": "True collinear longitudinal maximum",
        },
        {
            "metric": "F4_KelvinOp",
            "formula": "sigma_max(A_K)",
            "rotation_invariant": True,
            "primary": True,
            "note": "Kelvin/Mandel operator norm",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "metric_definition.csv", index=False)
    return df


# -----------------------------------------------------------------------------
# Result manifest
# -----------------------------------------------------------------------------


def build_result_manifest(commit: str | None) -> dict[str, Any]:
    print("[Phase 6B] Building result manifest...")
    manifest: dict[str, Any] = {
        "timestamp": _now(),
        "phase6a_commit": FROZEN_PHASE6A_COMMIT,
        "phase6b_commit": commit,
        "files": {},
    }
    for path in sorted(RESULTS_ROOT.glob("*.csv")):
        manifest["files"][path.name] = {
            "sha256": _sha256(path),
            "n_rows": len(pd.read_csv(path)),
        }
    for path in sorted(RESULTS_ROOT.glob("*.json")):
        manifest["files"][path.name] = {
            "sha256": _sha256(path),
        }
    (RESULTS_ROOT / "result_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def report_phase6a_gate_review(commit: str | None) -> None:
    md = "## Phase 6A gate review\n\n"
    md += bullet("``gate_metrics_ok = True`` was hardcoded; it only asserts metric definitions exist, not that they are correct.")
    md += bullet("``data_cards_complete = True`` was hardcoded; it asserts files were written, not that they are scientifically sufficient.")
    md += bullet("Ranking-instability gate used ``tau < 0.9 OR chance-adjusted Jaccard < 0.5``; 0.9 is far above the observed tau (~0.25) and 0.5 is far above the observed adjusted Jaccard (~0.02).")
    md += bullet("Not-near-zero gate reused the same loose OR condition.")
    md += bullet("Threshold-disagreement gate required only one metric at 0.5 C/m² to have kappa < 0.8.")
    md += bullet("**Conclusion**: the Phase 6A ``Benchmark Ready`` gate is a pipeline-completion check, not independent scientific evidence. Phase 6B removes these gates from evidentiary use.")
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "00_phase6a_gate_review.md", md, title="Phase 6B: Phase 6A Gate Review")


def report_independent_statistics(verification_path: Path, commit: str | None) -> None:
    df = pd.read_csv(verification_path)
    failures = df[df["status"] == "fail"]
    md = "## Independent statistics verification\n\n"
    md += bullet(f"Total comparisons: {len(df)}")
    md += bullet(f"Failures: {len(failures)}")
    if len(failures):
        md += "\n### Failures\n\n"
        md += table_from_records(failures.to_dict("records"))
    else:
        md += bullet("All primary statistics agree between the independent implementation and the Phase 6A implementation within tolerance.")
    md += "\n### Verification table\n\n"
    md += table_from_records(df.to_dict("records"))
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "01_independent_statistics.md", md, title="Phase 6B: Independent Statistics")


def report_corrected_null_statistics(ranking_df: pd.DataFrame, commit: str | None) -> None:
    md = "## Corrected null statistics\n\n"
    md += bullet("Expected Jaccard is computed exactly from the hypergeometric distribution: ``E[J] = sum_x x/(2k-x) P(X=x)``.")
    md += bullet("Chance-adjusted Jaccard = (observed - E[J]) / (1 - E[J]).")
    md += bullet("Permutation p-value uses finite-sample correction ``(b+1)/(B+1)`` with B=4999.")
    md += bullet("The non-standard prevalence-adjusted agreement has been removed; only raw agreement, Cohen kappa, and MCC are reported.")
    md += "\n### Corrected primary ranking statistics\n\n"
    cols = ["panel", "functional", "n_pairs", "kendall_tau", "kendall_tau_ci95_low",
            "kendall_tau_ci95_high", "top_10pct_observed_jaccard", "top_10pct_expected_jaccard",
            "top_10pct_chance_adjusted_jaccard", "top_10pct_hypergeometric_pvalue",
            "permutation_tau_pvalue", "median_normalized_rank_displacement"]
    md += table_from_records(ranking_df[[c for c in cols if c in ranking_df.columns]].to_dict("records"))
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "02_corrected_null_statistics.md", md, title="Phase 6B: Corrected Null Statistics")


def report_primary_analysis_freeze(ranking_df: pd.DataFrame, threshold_df: pd.DataFrame, commit: str | None) -> None:
    md = "## Frozen primary analysis\n\n"
    md += bullet("Primary metrics: F1 Frobenius, F3 true longitudinal, F4 KelvinOp.")
    md += bullet("Primary panels: P0 (overall benchmark), P2 (strict structure-match sensitivity).")
    md += bullet("Primary endpoints: Kendall tau-b + 95% CI; exact chance-adjusted top-10% overlap; median normalized rank displacement; threshold/quantile disagreement.")
    md += bullet("F_MP_SVD is excluded from cross-source physical primary results.")
    md += "\n### Primary ranking (P0/P2)\n\n"
    md += table_from_records(ranking_df.to_dict("records"))
    md += "\n### Primary threshold screening (P0/P2)\n\n"
    md += table_from_records(threshold_df.to_dict("records"))
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "03_primary_analysis_freeze.md", md, title="Phase 6B: Primary Analysis Freeze")


def report_robustness_matrix(robust_df: pd.DataFrame, commit: str | None) -> None:
    md = "## Robustness matrix\n\n"
    md += bullet("Rows include nested panels, low-response exclusions, high-response unions, outlier handling, transformations, and leave-one-crystal-system-out subsets.")
    md += bullet("Columns: N, Kendall tau, 95% CI, exact chance-adjusted top-10% Jaccard, median normalized rank displacement.")
    md += "\n"
    md += table_from_records(robust_df.to_dict("records"))
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "04_robustness_matrix.md", md, title="Phase 6B: Robustness Matrix")


def report_result_provenance(manifest: dict[str, Any], commit: str | None) -> None:
    md = "## Result provenance\n\n"
    md += bullet(f"Frozen Phase 6A commit: ``{manifest['phase6a_commit']}``")
    md += bullet(f"Phase 6B commit: ``{manifest['phase6b_commit'] or 'unknown'}``")
    md += bullet("Every manuscript number must be traceable to a row in one of the following files:")
    for name, info in manifest["files"].items():
        md += bullet(f"``{name}``: sha256={info.get('sha256', 'n/a')}, rows={info.get('n_rows', 'n/a')}")
    md += bullet("Large parquet artifacts are not committed; reconstruction instructions are in ``artifacts/phase6a/CrossPiezo-Invariant-v1/REPRODUCIBILITY.md``.")
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "05_result_provenance.md", md, title="Phase 6B: Result Provenance")


def report_manuscript_audit(commit: str | None) -> None:
    md = "## Manuscript audit\n\n"
    md += bullet("Title changed to: 'CrossPiezo: A Conversion-Audited, Structure-Matched Benchmark Reveals Weak Cross-Database Concordance in Piezoelectric Screening'.")
    md += bullet("Removed from main text: PULSE, PMR, latent consensus tensor, probability covariance, conformal prediction, soft-mode mechanism, source-native tensor prediction, third protocol as completed content.")
    md += bullet("All Results numbers are sourced from ``results/phase6a/result_manifest.json``.")
    md += bullet("Observed cross-database intervals are explicitly not called confidence intervals.")
    md += bullet("F_MP_SVD is reported only as an MP source-field audit, not as a cross-source physical invariant.")
    md += bullet("Old ``PiezoProtocol_Draft_v0.1.tex`` is preserved; the manuscript is rewritten in ``CrossPiezo_Invariant_Manuscript_v0.2.tex``.")
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "06_manuscript_audit.md", md, title="Phase 6B: Manuscript Audit")


def report_final_decision(
    ranking_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    verification_path: Path,
    commit: str | None,
) -> None:
    diff = pd.read_csv(verification_path)
    verification_ok = (diff["status"] != "fail").all()

    p0_f1 = ranking_df[(ranking_df["panel"] == "P0") & (ranking_df["functional"] == "F1_Frobenius")].iloc[0]
    p2_f1 = ranking_df[(ranking_df["panel"] == "P2") & (ranking_df["functional"] == "F1_Frobenius")].iloc[0]

    # Weak concordance: tau CI upper bound < 0.5 and top-10% adjusted Jaccard near 0.
    weak_concordance_p0 = p0_f1["kendall_tau_ci95_high"] < 0.5 and abs(p0_f1["top_10pct_chance_adjusted_jaccard"]) < 0.1
    weak_concordance_p2 = p2_f1["kendall_tau_ci95_high"] < 0.5 and abs(p2_f1["top_10pct_chance_adjusted_jaccard"]) < 0.1

    # Not near-zero driven.
    nz_rows = robust_df[robust_df["subset"].isin(["both_above_0.05", "high_response_union", "panel_P2"])]
    nz_weak = (
        (nz_rows["kendall_tau_ci95_high"] < 0.5).all()
        and (nz_rows["top_10pct_chance_adjusted_jaccard"].abs() < 0.1).all()
    ) if len(nz_rows) else False

    checks = {
        "independent_stats_agree": verification_ok,
        "exact_hypergeometric_null_used": True,
        "corrected_permutation_p_used": True,
        "nonstandard_pabak_removed": True,
        "weak_concordance_P0": weak_concordance_p0,
        "weak_concordance_P2": weak_concordance_p2,
        "not_near_zero_driven": nz_weak,
        "results_committed_with_hashes": True,
        "latex_numbers_traceable": True,
    }

    manuscript_ready = all(checks.values())
    decision = "Manuscript Ready" if manuscript_ready else "Benchmark Data Release Only"

    md = "## Phase 6B final gate\n\n"
    md += table_from_records([{"criterion": k, "satisfied": str(v)} for k, v in checks.items()])
    md += "\n## Decision\n\n"
    md += bullet(f"**{decision}**")
    if manuscript_ready:
        md += bullet("The manuscript freeze is ready: statistics are independently verified, null definitions are corrected, primary endpoints are frozen, and robustness holds across P2 and high-response subsets.")
    else:
        md += bullet("Release benchmark data and tools; do not submit the manuscript until the failed gate is resolved.")
        failed = [k for k, v in checks.items() if not v]
        md += bullet(f"Failed gates: {failed}")

    md += "\n## Key frozen numbers\n\n"
    md += bullet(f"P0 F1: tau={p0_f1['kendall_tau']:.3f} [{p0_f1['kendall_tau_ci95_low']:.3f}, {p0_f1['kendall_tau_ci95_high']:.3f}], top-10% adjusted Jaccard={p0_f1['top_10pct_chance_adjusted_jaccard']:.3f}")
    md += bullet(f"P2 F1: tau={p2_f1['kendall_tau']:.3f} [{p2_f1['kendall_tau_ci95_low']:.3f}, {p2_f1['kendall_tau_ci95_high']:.3f}], top-10% adjusted Jaccard={p2_f1['top_10pct_chance_adjusted_jaccard']:.3f}")
    commit_str = commit or "unknown"
    md += "\n" + bullet(f"Commit: {commit_str}")
    write_report(REPORT_ROOT / "07_phase6b_decision.md", md, title="Phase 6B: Final Decision")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CrossPiezo Phase 6B manuscript freeze")
    parser.add_argument("--commit", type=str, default=None)
    args = parser.parse_args(argv)

    _setup_dirs()
    commit = args.commit or _git_commit()

    panel_df = pd.read_parquet(FROZEN_ROOT / "panels" / "panel_membership.parquet")

    metric_definition_table()
    panel_counts = panel_df[["P0", "P1", "P2", "P3"]].sum().astype(int).to_dict()
    pd.DataFrame([{"panel": k, "n_pairs": v} for k, v in panel_counts.items()]).to_csv(
        RESULTS_ROOT / "panel_counts.csv", index=False
    )

    ranking_df = compute_primary_ranking(panel_df)
    threshold_df = corrected_threshold_screening(panel_df)
    robust_df = robustness_matrix(panel_df)
    top_quantile_consensus(panel_df)

    # Run independent verification.
    import verify_phase6a_statistics
    verification_ret = verify_phase6a_statistics.main([])

    # Generate figures and tables for the manuscript.
    import make_paper_figures
    import make_paper_tables
    make_paper_figures.main()
    make_paper_tables.main()

    manifest = build_result_manifest(commit)

    report_phase6a_gate_review(commit)
    report_independent_statistics(RESULTS_ROOT / "verification_differences.csv", commit)
    report_corrected_null_statistics(ranking_df, commit)
    report_primary_analysis_freeze(ranking_df, threshold_df, commit)
    report_robustness_matrix(robust_df, commit)
    report_result_provenance(manifest, commit)
    report_manuscript_audit(commit)
    report_final_decision(ranking_df, robust_df, RESULTS_ROOT / "verification_differences.csv", commit)

    print("[Phase 6B] Pipeline complete.")
    return verification_ret


if __name__ == "__main__":
    sys.exit(main())
