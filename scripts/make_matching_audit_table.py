#!/usr/bin/env python
"""Build the structure-matching audit summary table.

Reads the frozen panel membership, panel counts, and matcher configuration,
then writes ``results/phase7c/matching_audit_summary.csv``.  No hand-entered
numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "phase7c"
CONFIG_PATH = PROJECT_ROOT / "configs" / "matching.yaml"
PANEL_COUNTS_PATH = PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_counts.json"
PANEL_MEMBERSHIP_PATH = (
    PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_membership.parquet"
)
OUTPUT = RESULTS_ROOT / "matching_audit_summary.csv"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_panel_counts() -> dict:
    with PANEL_COUNTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    config = _load_config()
    matcher = config.get("matcher", {})
    counts = _load_panel_counts()

    pm = pd.read_parquet(PANEL_MEMBERSHIP_PATH)

    rows = []
    for panel in ["P0", "P2"]:
        sub = pm[pm[panel]]
        row = {
            "panel": panel,
            "count": int(len(sub)),
            "expected_count": int(counts.get(panel, len(sub))),
            "mean_rms_distance": float(sub["rms_distance"].mean()),
            "median_rms_distance": float(sub["rms_distance"].median()),
            "max_rms_distance": float(sub["rms_distance"].max()),
            "mean_max_distance": float(sub["max_distance"].mean()),
            "median_max_distance": float(sub["max_distance"].median()),
            "max_max_distance": float(sub["max_distance"].max()),
            "mean_lattice_distance": float(sub["lattice_distance"].mean()),
            "median_lattice_distance": float(sub["lattice_distance"].median()),
            "max_lattice_distance": float(sub["lattice_distance"].max()),
            "ltol": float(matcher.get("ltol", float("nan"))),
            "stol": float(matcher.get("stol", float("nan"))),
            "angle_tol": float(matcher.get("angle_tol", float("nan"))),
            "primitive": bool(matcher.get("primitive", True)),
            "scale": bool(matcher.get("scale", True)),
        }
        if panel == "P2":
            row["rms_threshold"] = float(counts.get("P2_rms_threshold", float("nan")))
            row["lattice_threshold"] = float(
                counts.get("P2_lattice_threshold", float("nan"))
            )
        else:
            row["rms_threshold"] = float("nan")
            row["lattice_threshold"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print(f"[make_matching_audit_table] Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
