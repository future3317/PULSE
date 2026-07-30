#!/usr/bin/env python
"""Compile Phase 5B model-validated reports (19-22) and final decision (25).

Run this locally after copying `artifacts/phase5b/model_metrics.parquet` and
prediction NPZ files from the remote training host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crosspiezo.phase5b.calibration import conformal_coverage, residual_scale_calibration
from crosspiezo.phase5b.leaderboard import (
    aggregate_leaderboard,
    rank_inversion_table,
    source_token_gain,
)
from crosspiezo.phase5b.pmr import compute_pmr_table, compute_spg, compute_valid_models
from crosspiezo.reports.markdown import bullet, table_from_records, write_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5B_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase5b"
REPORT_ROOT = PROJECT_ROOT / "reports"
MANUSCRIPT_NOTES = PROJECT_ROOT / "manuscript_notes"


def _load_metrics() -> pd.DataFrame | None:
    path = PHASE5B_ARTIFACT / "model_metrics.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_hierarchy() -> pd.DataFrame | None:
    path = PHASE5B_ARTIFACT / "discrepancy_hierarchy.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _paired_discrepancies(hierarchy: pd.DataFrame, extended: pd.DataFrame, core: pd.DataFrame) -> dict[str, np.ndarray]:
    exact = hierarchy[hierarchy["variant"] == "exact_transported"]
    core_keys = set(zip(core["jarvis_id"], core["mp_id"], strict=False))

    def _mask(keys: set[tuple[str, str]]) -> pd.Series:
        return exact.apply(lambda r: (r["jarvis_id"], r["mp_id"]) in keys, axis=1)

    # High response: top half by maximum source norm on Extended.
    ext_norm = extended.copy()
    ext_norm["max_norm"] = ext_norm[["jarvis_norm", "mp_norm"]].max(axis=1)
    high_threshold = ext_norm["max_norm"].median()
    high_keys = set(zip(
        ext_norm[ext_norm["max_norm"] >= high_threshold]["jarvis_id"],
        ext_norm[ext_norm["max_norm"] >= high_threshold]["mp_id"],
        strict=False,
    ))

    # T1a keys.
    t1a_keys = set(zip(
        extended[extended["sublayer"] == "T1a"]["jarvis_id"],
        extended[extended["sublayer"] == "T1a"]["mp_id"],
        strict=False,
    ))

    scopes: dict[str, np.ndarray] = {
        "all": exact["absolute"].dropna().values,
        "core": exact[_mask(core_keys)]["absolute"].dropna().values,
        "T1a": exact[_mask(t1a_keys)]["absolute"].dropna().values,
        "high_response": exact[_mask(high_keys)]["absolute"].dropna().values,
    }
    return scopes


def report_equivariant_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 5B.5] Compiling equivariant baseline report...")
    valid = compute_valid_models(metrics)
    valid.to_parquet(PHASE5B_ARTIFACT / "valid_models.parquet")

    md = ""
    md += "## Model skill gate\n"
    md += bullet("A model is valid only if its mean in-source MAE is lower than both the zero and composition-mean baselines on the same eval source and split.")
    md += bullet("All reported models use the same test universe, unit (C/m²), total piezo stress tensor, and Frobenius metric.")
    md += "\n## Valid model summary\n"
    md += table_from_records(valid[["model_name", "train_source", "split_type", "absolute_frobenius_mae_mean", "absolute_frobenius_mae_std", "valid"]].to_dict("records"))

    md += "\n## Baseline comparison (in-source)\n"
    insource = metrics[metrics["split_type"].isin(["in_source_jarvis", "in_source_mp"])]
    md += table_from_records(insource.groupby(["model_name", "eval_source"])["absolute_frobenius_mae"].mean().reset_index().to_dict("records"))

    md += "\n## Notes\n"
    md += bullet("If no e3nn model passes the skill gate, the benchmark proceeds as Data-Only Benchmark Go.")
    write_report(REPORT_ROOT / "19_equivariant_baselines.md", md, title="Phase 5B.4: Equivariant Baselines")
    print("[Phase 5B.5] Wrote reports/19_equivariant_baselines.md")
    return valid


def report_valid_pmr(metrics: pd.DataFrame, hierarchy: pd.DataFrame, extended: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 5B.6] Computing valid PMR...")
    valid = compute_valid_models(metrics)
    scopes = _paired_discrepancies(hierarchy, extended, core)
    pmr = compute_pmr_table(metrics, scopes, valid_models=valid)
    pmr.to_parquet(PHASE5B_ARTIFACT / "valid_pmr.parquet")

    md = ""
    md += "## PMR protocol\n"
    md += bullet("PMR_mean_absolute = mean paired Frobenius discrepancy / mean in-source MAE.")
    md += bullet("PMR_median_absolute = median paired Frobenius discrepancy / median in-source MAE.")
    md += bullet("Paired bootstrap 95% CI is computed on the discrepancy numerator; denominator is the observed in-source error.")
    md += bullet("SPG (smallest protocol gap) = median discrepancy / (best valid in-source MAE + epsilon).")
    md += "\n## Valid PMR table\n"
    md += table_from_records(pmr.to_dict("records"))

    # SPG.
    best_errors: list[float] = []
    for src in ["jarvis", "mp"]:
        sub = valid[(valid["eval_source"] == src) & (valid["split_type"] == f"in_source_{src}") & valid["valid"]]["absolute_frobenius_mae_mean"]
        if len(sub):
            best_errors.append(float(sub.min()))
    spg = compute_spg(scopes, best_errors)
    md += "\n## Smallest Protocol Gap (SPG)\n"
    md += table_from_records([{"scope": k, "SPG": v} for k, v in spg.items()])
    write_report(REPORT_ROOT / "20_valid_pmr.md", md, title="Phase 5B.5: Valid PMR")
    print("[Phase 5B.6] Wrote reports/20_valid_pmr.md")
    return pmr


def report_leaderboard(metrics: pd.DataFrame) -> None:
    print("[Phase 5B.7] Compiling leaderboard stress test...")
    agg = aggregate_leaderboard(metrics)
    agg.to_parquet(PHASE5B_ARTIFACT / "leaderboard.parquet")
    inv = rank_inversion_table(agg)
    inv.to_parquet(PHASE5B_ARTIFACT / "rank_inversion.parquet")
    gain = source_token_gain(agg)

    md = ""
    md += "## Leaderboard summary\n"
    md += table_from_records(agg[["model_name", "split_type", "n_seeds", "n_test", "absolute_frobenius_mae_mean", "absolute_frobenius_mae_std", "normalized_frobenius_mae_mean"]].to_dict("records"))
    md += "\n## Rank inversion (in-source vs source-held-out)\n"
    md += table_from_records(inv.to_dict("records"))
    md += "\n## Source-token gain on cross-source splits\n"
    md += table_from_records(gain.to_dict("records"))
    md += "\n## Interpretation\n"
    md += bullet("If source-held-out MAE is systematically higher than in-source MAE, leaderboard rankings are not protocol-robust.")
    md += bullet("A positive source-token gain indicates that the source label helps the model adapt to protocol shift.")
    write_report(REPORT_ROOT / "21_leaderboard_stress_test.md", md, title="Phase 5B.6: Leaderboard Stress Test")
    print("[Phase 5B.7] Wrote reports/21_leaderboard_stress_test.md")


def report_calibration(metrics: pd.DataFrame) -> None:
    print("[Phase 5B.8] Lightweight calibration audit...")
    valid = compute_valid_models(metrics)
    valid_model_names = set(valid[valid["valid"]]["model_name"])
    if not valid_model_names:
        md = "## Calibration audit skipped\nNo e3nn model passed the valid-model gate; calibration is not reported."
        write_report(REPORT_ROOT / "22_lightweight_calibration.md", md, title="Phase 5B.7: Lightweight Calibration")
        print("[Phase 5B.8] No valid models; skipped calibration")
        return

    md = ""
    md += "## Calibration methods\n"
    md += bullet("Deep ensemble: mean and standard deviation across 3 training seeds (already saved in predictions).")
    md += bullet("Source-stratified residual scaling: per-source factor to hit a target marginal coverage.")
    md += bullet("Conservative conformal: symmetric absolute-error quantiles.")

    rows: list[dict[str, Any]] = []
    for model_name in valid_model_names:
        # Prefer paired counterfactual predictions; fall back to in-source.
        split_pref = f"{model_name}_paired_counterfactual_jarvis_predictions.npz"
        path = PHASE5B_ARTIFACT / split_pref
        if not path.exists():
            path = PHASE5B_ARTIFACT / f"{model_name}_in_source_jarvis_predictions.npz"
        if not path.exists():
            continue
        data = np.load(path)
        preds = data["preds_mean"]
        ys = data["ys"]
        sources = np.array(["jarvis"] * len(ys))  # placeholder; could be loaded from eval records
        cal = residual_scale_calibration(preds, ys, sources, target_coverage=0.9)
        conf = conformal_coverage(preds, ys, alphas=[0.1, 0.2, 0.3])
        rows.append({
            "model_name": model_name,
            "global_median_error": cal["global_median_error"],
            "global_mean_error": cal["global_mean_error"],
            "scale_factor_jarvis": cal["source_scale_factors"].get("jarvis", float("nan")),
            "coverage_90": cal["achieved_coverage"].get("jarvis", float("nan")),
            "conformal_alpha0.10_coverage": conf["alpha_0.10"]["coverage"],
        })

    md += "\n## Calibration results\n"
    md += table_from_records(rows)
    md += "\n## Notes\n"
    md += bullet("Coverage is evaluated on the same test panel used for model selection; it is not a true holdout guarantee.")
    md += bullet("Calibration does not change the Data-Only / Model-Validated decision.")
    write_report(REPORT_ROOT / "22_lightweight_calibration.md", md, title="Phase 5B.7: Lightweight Calibration")
    print("[Phase 5B.8] Wrote reports/22_lightweight_calibration.md")


def report_decision(
    metrics: pd.DataFrame,
    valid_models: pd.DataFrame,
    pmr: pd.DataFrame,
    extended: pd.DataFrame,
    core: pd.DataFrame,
    lambda_audit: dict[str, Any],
) -> None:
    print("[Phase 5B.11] Compiling final decision...")
    valid_names = set(valid_models[valid_models["valid"]]["model_name"])
    has_valid_model = len(valid_names) > 0
    n_core = len(core)

    # Ranking instability from Phase 0-4 / 5A.
    top50_jaccard = 0.07526881720430108
    kendall_tau = 0.2568831475074093

    decision = "Data-Only Benchmark Go"
    rationale = (
        "Cross-protocol discrepancy and rank instability are established, but no e3nn model passed the valid-model gate. "
        "The benchmark proceeds as a data/leaderboard paper without claiming protocol gap exceeds competent model error."
    )
    if has_valid_model:
        decision = "Model-Validated Benchmark Go"
        rationale = (
            "At least one equivariant model beats zero/mean in-source, a valid PMR is reported, and rank instability persists. "
            "The benchmark can include model-validated PMR and leaderboard stress test."
        )
    if n_core < 5:
        rationale += f" Core panel is very small (N={n_core}), so componentwise/cosine claims are avoided."

    md = ""
    md += "## Phase 5B decision\n"
    md += bullet(f"**Decision: {decision}**")
    md += bullet(f"Rationale: {rationale}")
    md += "\n## Summary\n"
    md += bullet(f"Core panel: {n_core} pairs; Extended panel: {len(extended)} pairs.")
    md += bullet(f"Valid equivariant models: {', '.join(sorted(valid_names)) if valid_names else 'none'}.")
    md += bullet(f"Top-50 Jaccard (Frobenius norm): {top50_jaccard:.3f}")
    md += bullet(f"Kendall tau: {kendall_tau:.3f}")
    if not pmr.empty:
        md += bullet(f"Best PMR (all, median absolute): {pmr['PMR_median_absolute'].max():.3f}")
    md += bullet(f"Full atom-resolved Λ recovered: {lambda_audit.get('extended', {}).get('full_lambda_candidate_count', 0)} records.")
    md += "\n## Adjudication recommendation\n"
    md += bullet("Proceed with the pre-registered third-protocol DFPT plan only after manuscript revision; do not run new DFT in this phase.")
    md += "\n## Revised paper positioning\n"
    md += bullet("Title candidate: 'CrossPiezo: Cross-Protocol Evaluation Reveals Unstable Rankings in AI Screening of Piezoelectric Materials'.")
    md += bullet("Do not claim a soft-mode mechanism; leave it as an untested hypothesis in Discussion.")
    md += bullet("Do not claim PMR > 1 unless a valid model gate is met and reported.")
    md += "\n## Not executed\n"
    md += bullet("No new DFT/DFPT calculations.")
    md += bullet("No full PULSE model development.")
    md += bullet("No LaTeX results section edits.")
    write_report(REPORT_ROOT / "25_phase5b_decision.md", md, title="Phase 5B.10: Final Decision")
    print("[Phase 5B.11] Wrote reports/25_phase5b_decision.md")


def write_manuscript_notes() -> None:
    MANUSCRIPT_NOTES.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT_NOTES / "phase5b_revised_title_and_abstract.md").write_text(
        "# Revised title and abstract\n\n"
        "Title: CrossPiezo: Cross-Protocol Evaluation Reveals Unstable Rankings in AI Screening of Piezoelectric Materials\n\n"
        "Abstract points:\n"
        "- 538 strict JARVIS-MP structure-matched pairs.\n"
        "- Source-native frame audit shows JARVIS tensors are not aligned with the CIF-setting point group; only 15 pairs enter the Core panel.\n"
        "- Ranking by Frobenius norm is unstable (top-50 Jaccard 0.075).\n"
        "- Valid PMR is reported only from models that beat zero/mean baselines; otherwise the paper stays data-only.\n"
        "- Soft-mode mechanism is withdrawn pending full atom-resolved Λ recovery or third-protocol adjudication.\n",
        encoding="utf-8",
    )
    (MANUSCRIPT_NOTES / "phase5b_claim_matrix.md").write_text(
        "# Claim matrix after Phase 5B\n\n"
        "| Claim | Status | Evidence |\n"
        "|---|---|---|\n"
        "| Cross-protocol disagreement exists | Go | Extended median normalized discrepancy ~1.6 |\n"
        "| Rankings are unstable | Go | top-50 Jaccard 0.075, Kendall tau 0.26 |\n"
        "| Protocol gap exceeds competent model error | Conditional | Valid PMR only if model passes skill gate |\n"
        "| Soft-mode mechanism explains disagreement | Withdrawn | grouped CV negative, no full Λ |\n",
        encoding="utf-8",
    )
    print("[Phase 5B.11] Wrote manuscript_notes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-pmr", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-decision", action="store_true")
    args = parser.parse_args()

    metrics = _load_metrics()
    if metrics is None:
        print("ERROR: artifacts/phase5b/model_metrics.parquet not found. Train models first.")
        return 1

    hierarchy = _load_hierarchy()
    if hierarchy is None:
        print("ERROR: discrepancy hierarchy artifact missing. Run run_phase5b.py first.")
        return 1

    extended = pd.read_parquet(PHASE5B_ARTIFACT / "extended_pairs.parquet")
    core = pd.read_parquet(PHASE5B_ARTIFACT / "core_pairs.parquet")

    if not args.skip_baselines:
        valid_models = report_equivariant_baselines(metrics)
    else:
        valid_models = compute_valid_models(metrics)

    if not args.skip_pmr:
        pmr = report_valid_pmr(metrics, hierarchy, extended, core)
    else:
        pmr = pd.DataFrame()

    if not args.skip_leaderboard:
        report_leaderboard(metrics)

    if not args.skip_calibration:
        report_calibration(metrics)

    if not args.skip_decision:
        lambda_audit = {"extended": {"full_lambda_candidate_count": 0}}
        report_decision(metrics, valid_models, pmr, extended, core, lambda_audit)
        write_manuscript_notes()

    print("\nPhase 5B model/decision reports complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
