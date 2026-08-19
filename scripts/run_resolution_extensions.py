#!/usr/bin/env python
"""Compute ranking-resolution extensions from the frozen P0/P2 panel.

This is a read-only ranking analysis.  It does not refit a model or alter the
frozen panels.  Outputs are written to ``results/phase9`` so the paper package
can consume one versioned result layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_membership.parquet"
RESULT_ROOT = PROJECT_ROOT / "results" / "phase9"
Q_LEVELS = (1, 5, 10, 20, 50)
BOUNDARY_BINS = (-np.inf, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, np.inf)


def _top_indices(values: np.ndarray, q_percent: float) -> np.ndarray:
    k = max(1, int(np.floor(q_percent / 100.0 * len(values))))
    return np.argsort(-values, kind="stable")[:k]


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return 100.0 * ranks / len(values)


def retention_map(panel: pd.DataFrame, left_col: str, right_col: str) -> pd.DataFrame:
    left = panel[left_col].to_numpy(float)
    right = panel[right_col].to_numpy(float)
    rows: list[dict[str, float | int | str]] = []
    for q_source in Q_LEVELS:
        source_idx = set(_top_indices(left, q_source).tolist())
        for q_target in Q_LEVELS:
            target_idx = set(_top_indices(right, q_target).tolist())
            rows.append(
                {
                    "q_source_percent": q_source,
                    "q_target_percent": q_target,
                    "source_k": len(source_idx),
                    "target_k": len(target_idx),
                    "overlap": len(source_idx & target_idx),
                    "retention": len(source_idx & target_idx) / len(source_idx),
                }
            )
    return pd.DataFrame(rows)


def boundary_risk(panel: pd.DataFrame, left_col: str, right_col: str) -> pd.DataFrame:
    left = panel[left_col].to_numpy(float)
    right = panel[right_col].to_numpy(float)
    left_pct = _rank_percentile(left)
    right_pct = _rank_percentile(right)
    rows: list[dict[str, float | int | str]] = []
    labels = ["(-inf,-20]", "(-20,-10]", "(-10,-5]", "(-5,-2]", "(-2,0]",
              "(0,2]", "(2,5]", "(5,10]", "(10,20]", "(20,inf)"]
    for q in (5, 10, 20):
        k = max(1, int(np.floor(q / 100.0 * len(panel))))
        left_elite = np.zeros(len(panel), dtype=bool)
        right_elite = np.zeros(len(panel), dtype=bool)
        left_elite[_top_indices(left, q)] = True
        right_elite[_top_indices(right, q)] = True
        disagreement = left_elite != right_elite
        distance = left_pct - q
        category = pd.cut(distance, bins=BOUNDARY_BINS, labels=labels, include_lowest=True)
        for label in labels:
            mask = category.astype(object) == label
            n = int(mask.sum())
            rows.append(
                {
                    "q_percent": q,
                    "distance_bin": label,
                    "distance_lower": BOUNDARY_BINS[labels.index(label)],
                    "distance_upper": BOUNDARY_BINS[labels.index(label) + 1],
                    "n": n,
                    "disagreements": int(disagreement[mask].sum()),
                    "disagreement_rate": float(disagreement[mask].mean()) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def global_rank_blindness(n: int = 100) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in range(1, n // 4 + 1):
        rows.append(
            {
                "n": n,
                "block_size": k,
                "top_fraction_percent": 100.0 * k / n,
                "top_k_overlap": 0,
                "kendall_tau": 1.0 - 4.0 * k * k / (n * (n - 1)),
                "spearman_rho": 1.0 - 12.0 * k**3 / (n * (n**2 - 1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH)
    outputs: dict[str, str] = {}
    retention_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    for panel_name in ("P0", "P2"):
        sub = panel.loc[panel[panel_name].astype(bool)].reset_index(drop=True)
        for direction, left_source, right_source in (
            ("JARVIS_to_MP", "jarvis_f1", "mp_f1"),
            ("MP_to_JARVIS", "mp_f1", "jarvis_f1"),
        ):
            ret = retention_map(sub, left_source, right_source)
            ret.insert(0, "direction", direction)
            ret.insert(0, "panel", panel_name)
            retention_frames.append(ret)
            risk = boundary_risk(sub, left_source, right_source)
            risk.insert(0, "direction", direction)
            risk.insert(0, "panel", panel_name)
            boundary_frames.append(risk)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    retention_path = RESULT_ROOT / "cross_quantile_retention.csv"
    boundary_path = RESULT_ROOT / "boundary_risk_by_distance.csv"
    theorem_path = RESULT_ROOT / "global_rank_blindness_construction.csv"
    pd.concat(retention_frames, ignore_index=True).to_csv(retention_path, index=False)
    pd.concat(boundary_frames, ignore_index=True).to_csv(boundary_path, index=False)
    global_rank_blindness().to_csv(theorem_path, index=False)
    manifest = {
        "status": "completed",
        "panel_source": str(PANEL_PATH.relative_to(PROJECT_ROOT)),
        "panels": {name: int(panel[name].sum()) for name in ("P0", "P2")},
        "q_levels_percent": list(Q_LEVELS),
        "retention_definition": "|T_A(q_source) intersect T_B(q_target)| / |T_A(q_source)|",
        "boundary_definition": "source rank percentile minus q; disagreement is source-vs-target top-q membership mismatch",
        "outputs": {
            "retention": str(retention_path.relative_to(PROJECT_ROOT)),
            "boundary_risk": str(boundary_path.relative_to(PROJECT_ROOT)),
            "global_rank_blindness": str(theorem_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = RESULT_ROOT / "resolution_extensions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
