#!/usr/bin/env python
"""Render Phase 7B result CSVs into minimal Markdown reports.

This script reads the CSV/JSON artifacts produced by ``run_phase7b.py`` and
writes report markdown under ``reports/phase7b/``.  It performs no new
computations and does not modify the frozen panel or source data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULT_ROOT = PROJECT_ROOT / "results" / "phase7b"
REPORT_ROOT = PROJECT_ROOT / "reports" / "phase7b"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from crosspiezo.reports.markdown import (  # noqa: E402
    bullet,
    header,
    table_from_records,
    write_report,
)


def _load_config() -> dict[str, Any]:
    with open(CONFIG_ROOT / "phase7b.yaml") as f:
        return yaml.safe_load(f)


def _read_csv(name: str) -> pd.DataFrame | None:
    path = RESULT_ROOT / name
    if path.exists():
        return pd.read_csv(path)
    return None


def _records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def render_01_screening_resolution() -> None:
    df = _read_csv("concordance_summary.csv")
    body = header("Work Package A: Screening-resolution curves", 1)
    body += "Screening-resolution summary for P0, P2 and high-response subsets.\n\n"
    body += table_from_records(_records(df))
    body += bullet("nAUCC = normalised area under the chance-adjusted concordance curve.")
    body += bullet("persistent_onset = first q with at least 5 consecutive simultaneous LCB values above delta.")
    write_report(REPORT_ROOT / "01_screening_resolution.md", body, title="Phase 7B WP-A: Screening resolution")


def render_02_scale_order_tail() -> None:
    df = _read_csv("scale_order_tail.csv")
    body = header("Work Package B: Scale / order / tail decomposition", 1)
    body += table_from_records(_records(df))
    body += bullet("Quantile-normalised Kendall tau equals raw tau by monotonic invariance.")
    body += bullet("Threshold crossings and ECDF/quantile-mapping residuals quantify tail disagreement.")
    write_report(REPORT_ROOT / "02_scale_order_tail.md", body, title="Phase 7B WP-B: Scale/order/tail")


def render_03_property_controls() -> None:
    summary = _read_csv("property_controls.csv")
    comp = _read_csv("property_control_comparison.csv")
    body = header("Work Package C: Property controls", 1)
    body += header("Summary", 2)
    body += table_from_records(_records(summary))
    body += header("FDR across control comparisons", 2)
    body += table_from_records(_records(comp))
    body += bullet("Positive Delta_tau / Delta_nAUCC means the control is more cross-source consistent.")
    write_report(REPORT_ROOT / "03_property_controls.md", body, title="Phase 7B WP-C: Property controls")


def render_04_electronic_ionic() -> None:
    df = _read_csv("electronic_ionic_decomposition.csv")
    body = header("Work Package D: Electronic/ionic decomposition", 1)
    body += table_from_records(_records(df))
    body += bullet("Sub-analyses with N < 100 are marked insufficient_N.")
    write_report(REPORT_ROOT / "04_electronic_ionic_decomposition.md", body, title="Phase 7B WP-D: Electronic/ionic")


def render_05_heterogeneity() -> None:
    df = _read_csv("heterogeneity.csv")
    cont = _read_csv("heterogeneity_continuous.csv")
    body = header("Work Package E: Heterogeneity", 1)
    body += table_from_records(_records(df))
    body += header("Continuous covariates", 2)
    body += table_from_records(_records(cont))
    body += bullet("Subgroups with N < 30 are excluded from strong conclusions.")
    write_report(REPORT_ROOT / "05_heterogeneity.md", body, title="Phase 7B WP-E: Heterogeneity")


def render_06_robust_portfolio() -> None:
    bench = _read_csv("portfolio_benchmark.csv")
    pareto = _read_csv("portfolio_pareto.csv")
    body = header("Work Package F: Robust portfolio", 1)
    body += header("Benchmark", 2)
    body += table_from_records(_records(bench))
    body += header("Pareto frontier (coverage vs worst-source recall)", 2)
    body += table_from_records(_records(pareto))
    body += bullet("Evaluation on the frozen holdout fold; disagreement-abstention lambda tuned on dev.")
    write_report(REPORT_ROOT / "06_robust_portfolio.md", body, title="Phase 7B WP-F: Robust portfolio")


def render_07_manuscript() -> None:
    manifest_path = RESULT_ROOT / "phase7b_manifest.json"
    key_numbers = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            key_numbers = json.load(f).get("key_numbers", {})
    body = header("Work Package G: Manuscript", 1)
    body += "Key numbers from the result manifest:\n\n"
    body += table_from_records([dict(key_numbers.items())])
    body += bullet("Full v0.5 LaTeX manuscript and references are written to the project root.")
    write_report(REPORT_ROOT / "07_manuscript.md", body, title="Phase 7B WP-G: Manuscript")


def render_08_decision() -> None:
    cfg = _load_config()
    conc = _read_csv("concordance_summary.csv")
    prop = _read_csv("property_controls.csv")
    port = _read_csv("portfolio_benchmark.csv")

    q1 = False
    if conc is not None and not conc.empty:
        p2 = conc[conc["panel"] == cfg["panels"]["sensitivity"]]
        q1 = bool((p2["nAUCC"] < 0.5).any()) if not p2.empty else False

    q2 = False
    if prop is not None and not prop.empty:
        ok = prop[prop["status"] == "ok"]
        q2 = bool(
            (ok["Delta_tau"].fillna(-np.inf) > 0).any()
            or (ok["Delta_nAUCC"].fillna(-np.inf) > 0).any()
        )

    q3 = False
    if port is not None and not port.empty:
        for (_panel, _metric), g in port.groupby(["panel", "metric"]):
            for bf in g["budget_factor"].unique():
                gk = g[g["budget_factor"] == bf]
                jarvis_rec = gk[gk["strategy"] == "jarvis_only"]["worst_source_recall"].mean()
                mp_rec = gk[gk["strategy"] == "mp_only"]["worst_source_recall"].mean()
                robust_rec = gk[gk["strategy"].isin(["maximin_percentile", "average_percentile", "borda_count"])][
                    "worst_source_recall"
                ].max()
                if robust_rec > max(jarvis_rec, mp_rec):
                    q3 = True
                    break
            if q3:
                break

    if q1 and q2 and q3:
        decision = "Material Go (proceed with screening-resolution manuscript)"
    elif q1:
        decision = "Benchmark Paper Only"
    else:
        decision = "No-Go for material claim; retain benchmark-only scope"

    rows = [
        {"criterion": "elite_tail_gap_stable", "passed": q1},
        {"criterion": "control_property_contrast", "passed": q2},
        {"criterion": "robust_portfolio_beats_single_source", "passed": q3},
    ]
    body = header("Phase 7B Final Decision", 1)
    body += f"**Decision:** {decision}\n\n"
    body += table_from_records(rows)
    body += bullet("This is a two-source robustness benchmark, not independent physical validation.")
    write_report(REPORT_ROOT / "08_phase7b_decision.md", body, title="Phase 7B Final Decision")


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    render_01_screening_resolution()
    render_02_scale_order_tail()
    render_03_property_controls()
    render_04_electronic_ionic()
    render_05_heterogeneity()
    render_06_robust_portfolio()
    render_07_manuscript()
    render_08_decision()
    print(f"[Phase 7B] Rendered reports under {REPORT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
