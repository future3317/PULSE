#!/usr/bin/env python
"""Run the post-Version-A statistical upgrade and third-protocol preflight.

This script reads the frozen Phase 6A panel and the read-only source data.  It
writes new results under ``results/phase9`` and never overwrites frozen
Phase 7C artifacts.  The DFT/DFPT part is intentionally a preflight: it is
marked blocked unless a real third-protocol executable, input pipeline and
documented numerical environment are available.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from pymatgen.core.composition import Composition
except ImportError:  # pragma: no cover - only used by the light paper environment
    Composition = None  # type: ignore[assignment,misc]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PULSE_CONFIG = PROJECT_ROOT / "configs" / "third_protocol_phase7c.yaml"
PAPER_CONFIG = Path(
    os.environ.get("CROSSPIEZO_PAPER_CONFIG", str(PULSE_CONFIG))
)
DATA_ROOT = Path(os.environ.get("CROSSPIEZO_DATA_ROOT", r"E:\DATA"))
PANEL_DEFAULT_PATH = PROJECT_ROOT / "artifacts" / "phase6a" / "panels" / "panel_membership.parquet"
PANEL_PATH = Path(os.environ.get("CROSSPIEZO_PANEL_PATH", str(PANEL_DEFAULT_PATH)))
PANEL_PROVENANCE_PATH = Path(
    os.environ.get("CROSSPIEZO_PANEL_PROVENANCE_PATH", str(PANEL_DEFAULT_PATH))
)
CONTROL_SNAPSHOT_ROOT = os.environ.get("CROSSPIEZO_CONTROL_SNAPSHOT_ROOT")
RESULT_ROOT = PROJECT_ROOT / "results" / "phase9"
REPORT_ROOT = PROJECT_ROOT / "reports" / "phase9"
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from crosspiezo.analysis.phase7b_stats import naucc  # noqa: E402
from crosspiezo.analysis.phase7c_stats import partial_naucc as phase7c_partial_naucc  # noqa: E402
from crosspiezo.analysis.scientific_upgrade import (  # noqa: E402
    cluster_bootstrap_screening,
    cutoff_diagnostics,
)
from crosspiezo.analysis.ranking import expected_jaccard_hypergeometric  # noqa: E402


METRICS = {
    "F1_Frobenius": ("jarvis_f1", "mp_f1"),
    "F3_Longitudinal": ("jarvis_f3", "mp_f3"),
    "F4_KelvinOp": ("jarvis_f4", "mp_f4"),
}
Q_GRID = np.arange(1.0, 51.0, 1.0)
DISTANCE_FIELDS = ("rms_distance", "max_distance", "lattice_distance")
STRATA_ORDER = (
    "consensus_elite",
    "jarvis_only_elite",
    "mp_only_elite",
    "consensus_low",
    "diversity_balanced",
)
CONTROL_TIE_ATTRIBUTES = ("band_gap", "energy_above_hull", "dielectric_total_trace")
CONTROL_TIE_BOOTSTRAP = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Represent repository paths portably while retaining external provenance."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_csv(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(data: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default, allow_nan=True), encoding="utf-8")
    return {"path": _portable_path(path), "sha256": _sha256(path)}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _load_panel() -> pd.DataFrame:
    panel = (pd.read_csv(PANEL_PATH) if PANEL_PATH.suffix.lower() == ".csv" else pd.read_parquet(PANEL_PATH)).copy()
    panel["pair_id"] = panel["pair_id"].astype(str)
    panel["reduced_formula"] = panel["formula"].map(_reduced_formula)
    panel["atom_count_reduced"] = panel["formula"].map(_atom_count)
    panel["crystal_system"] = np.where(
        panel["jarvis_crystal_system"].eq(panel["mp_crystal_system"]),
        panel["jarvis_crystal_system"],
        "cross_source_mismatch",
    )
    return panel


def _reduced_formula(formula: Any) -> str | None:
    if not formula or not isinstance(formula, str):
        return None
    if Composition is not None:
        try:
            return Composition(formula).reduced_formula
        except Exception:  # noqa: BLE001
            pass
    # The paper-only environment intentionally has no pymatgen.  This small
    # fallback covers the ordinary element-count formulas in the frozen panel;
    # complex formulas remain isolated as their literal formula rather than
    # silently being assigned to an incorrect cluster.
    import re

    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens or "".join(element + amount for element, amount in tokens) != formula:
        return formula
    counts: dict[str, float] = {}
    for element, amount in tokens:
        counts[element] = counts.get(element, 0.0) + (float(amount) if amount else 1.0)
    scale = min(value for value in counts.values() if value > 0)
    ratios = {element: value / scale for element, value in counts.items()}
    return "".join(
        element + (str(int(round(value))) if not math.isclose(value, 1.0) else "")
        for element, value in sorted(ratios.items())
    )


def _atom_count(formula: Any) -> float | None:
    if not formula or not isinstance(formula, str):
        return None
    if Composition is not None:
        try:
            return float(sum(Composition(formula).reduced_composition.values()))
        except Exception:  # noqa: BLE001
            pass
    reduced = _reduced_formula(formula)
    if reduced is None:
        return None
    import re

    counts = re.findall(r"[A-Z][a-z]?([0-9]*)", reduced)
    return float(sum(float(value) if value else 1.0 for value in counts))


def _stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2:
        return {"n": int(len(left)), "tau": float("nan"), "rho": float("nan"), "nAUCC": float("nan")}
    tau, _ = stats.kendalltau(left, right)
    # Avoid SciPy/OpenBLAS floating-point exceptions observed for perfectly
    # ranked or nearly constant arrays in this Windows environment.  Pearson
    # correlation of average ranks is exactly Spearman's rho.
    rank_left = stats.rankdata(left, method="average")
    rank_right = stats.rankdata(right, method="average")
    sd_left = float(np.std(rank_left))
    sd_right = float(np.std(rank_right))
    if sd_left == 0.0 or sd_right == 0.0:
        rho = float("nan")
    else:
        centered_left = rank_left - float(np.mean(rank_left))
        centered_right = rank_right - float(np.mean(rank_right))
        rho = float(np.mean(centered_left * centered_right) / (sd_left * sd_right))
    # The point curve is reconstructed by cluster_bootstrap_screening; use a
    # one-replicate-free call only for the compact matching table.
    _, summary = cluster_bootstrap_screening(
        left, right, np.arange(len(left)), Q_GRID, n_boot=0, seed=42
    )
    return {
        "n": int(len(left)),
        "tau": float(tau) if tau is not None else float("nan"),
        "rho": float(rho) if rho is not None else float("nan"),
        "nAUCC": float(summary["nAUCC"]),
    }


def run_cluster_bootstrap(panel: pd.DataFrame) -> dict[str, Any]:
    curve_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for panel_name in ("P0", "P2"):
        subset = panel[panel[panel_name].astype(bool)].reset_index(drop=True)
        for metric, (left_col, right_col) in METRICS.items():
            curve, summary = cluster_bootstrap_screening(
                subset[left_col].to_numpy(float),
                subset[right_col].to_numpy(float),
                subset["reduced_formula"].to_numpy(object),
                Q_GRID,
                n_boot=2000,
                seed=42,
                alpha=0.05,
                min_consecutive=5,
            )
            curve["panel"] = panel_name
            curve["metric"] = metric
            curve_rows.append(curve)
            summary_rows.append({"panel": panel_name, "metric": metric, **summary})
    curve_df = pd.concat(curve_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return {
        "curve": _write_csv(curve_df, RESULT_ROOT / "cluster_bootstrap_curve.csv"),
        "summary": _write_csv(summary_df, RESULT_ROOT / "cluster_bootstrap_summary.csv"),
        "summary_df": summary_df,
    }


def run_cutoff_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    for panel_name in ("P0", "P2"):
        subset = panel[panel[panel_name].astype(bool)]
        for metric, (left_col, right_col) in METRICS.items():
            table = cutoff_diagnostics(
                subset[left_col].to_numpy(float), subset[right_col].to_numpy(float), Q_GRID
            )
            table["panel"] = panel_name
            table["metric"] = metric
            rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    return {"cutoff": _write_csv(result, RESULT_ROOT / "cutoff_gap_tie_diagnostics.csv"), "df": result}


def run_raw_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    quantiles = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
    for panel_name in ("P0", "P2"):
        subset = panel[panel[panel_name].astype(bool)]
        for metric, (left_col, right_col) in METRICS.items():
            left = subset[left_col].to_numpy(float)
            right = subset[right_col].to_numpy(float)
            for source, values in (("JARVIS", left), ("MP", right)):
                finite = values[np.isfinite(values)]
                row: dict[str, Any] = {
                    "panel": panel_name,
                    "metric": metric,
                    "source": source,
                    "n": len(finite),
                    "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                    "zero_count": int(np.sum(finite == 0.0)),
                }
                for q in quantiles:
                    row[f"q{int(q * 100):02d}"] = float(np.quantile(finite, q)) if len(finite) else float("nan")
                raw_rows.append(row)

            valid = np.isfinite(left) & np.isfinite(right)
            l, r = left[valid], right[valid]
            absolute = np.abs(l - r)
            denominator = np.abs(l) + np.abs(r)
            symmetric_relative = np.full_like(absolute, np.nan, dtype=float)
            np.divide(2.0 * absolute, denominator, out=symmetric_relative, where=denominator > 0)
            nonzero = (np.abs(l) > 0) & (np.abs(r) > 0)
            log_ratio = np.log10(np.abs(l[nonzero]) / np.abs(r[nonzero])) if np.any(nonzero) else np.array([])
            pair_rows.append(
                {
                    "panel": panel_name,
                    "metric": metric,
                    "n": int(len(l)),
                    "absolute_difference_mean": float(np.mean(absolute)) if len(absolute) else float("nan"),
                    "absolute_difference_median": float(np.median(absolute)) if len(absolute) else float("nan"),
                    "absolute_difference_q95": float(np.quantile(absolute, 0.95)) if len(absolute) else float("nan"),
                    "symmetric_relative_difference_median": float(np.nanmedian(symmetric_relative)),
                    "symmetric_relative_difference_q95": float(np.nanquantile(symmetric_relative, 0.95)),
                    "log10_abs_ratio_median": float(np.median(log_ratio)) if len(log_ratio) else float("nan"),
                    "log10_abs_ratio_q05": float(np.quantile(log_ratio, 0.05)) if len(log_ratio) else float("nan"),
                    "log10_abs_ratio_q95": float(np.quantile(log_ratio, 0.95)) if len(log_ratio) else float("nan"),
                }
            )
    return {
        "raw": _write_csv(pd.DataFrame(raw_rows), RESULT_ROOT / "raw_value_summary.csv"),
        "pair": _write_csv(pd.DataFrame(pair_rows), RESULT_ROOT / "raw_pairwise_diagnostics.csv"),
    }


def _rank_percentile(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=True, pct=True)


def _ordered_selection(
    panel: pd.DataFrame,
    indices: set[int],
    score: pd.Series,
    n: int,
    ascending: bool = False,
) -> list[int]:
    candidates = panel.loc[sorted(indices), ["pair_id"]].copy()
    candidates["score"] = score.loc[candidates.index]
    candidates = candidates.sort_values(
        ["score", "pair_id"], ascending=[ascending, True], kind="mergesort"
    )
    return [int(i) for i in candidates.head(n).index]


def build_third_protocol_candidates(panel: pd.DataFrame) -> dict[str, Any]:
    """Materialize the 48 deterministic candidates without touching P0."""
    p0 = panel[panel["P0"].astype(bool)].reset_index(drop=True).copy()
    n = len(p0)
    k10 = max(1, int(np.floor(0.10 * n)))
    k20 = max(1, int(np.floor(0.20 * n)))
    endpoint_names = tuple(METRICS)
    source_agg: dict[str, pd.Series] = {}
    top_union: dict[str, set[int]] = {}
    bottom_union: dict[str, set[int]] = {}
    top_aggregate: dict[str, set[int]] = {}
    bottom_aggregate: dict[str, set[int]] = {}
    for source in ("jarvis", "mp"):
        percentiles = []
        source_top: set[int] = set()
        source_bottom: set[int] = set()
        for metric, (left_col, right_col) in METRICS.items():
            col = left_col if source == "jarvis" else right_col
            percentiles.append(_rank_percentile(p0[col]))
            source_top.update(p0[col].nlargest(k10, keep="first").index.tolist())
            source_bottom.update(p0[col].nsmallest(k20, keep="first").index.tolist())
        aggregate = pd.concat(percentiles, axis=1).mean(axis=1)
        source_agg[source] = aggregate
        top_union[source] = set(int(i) for i in source_top)
        bottom_union[source] = set(int(i) for i in source_bottom)
        top_aggregate[source] = set(_ordered_selection(p0, set(p0.index), aggregate, k10))
        bottom_aggregate[source] = set(_ordered_selection(p0, set(p0.index), aggregate, k20, ascending=True))

    selected: dict[str, list[int]] = {}
    used: set[int] = set()
    combined = source_agg["jarvis"] + source_agg["mp"]
    consensus_pool = top_union["jarvis"] & top_union["mp"]
    selected["consensus_elite"] = _ordered_selection(p0, consensus_pool, combined, 10)
    used.update(selected["consensus_elite"])
    jarvis_pool = (top_aggregate["jarvis"] - top_union["mp"]) - used
    selected["jarvis_only_elite"] = _ordered_selection(p0, jarvis_pool, source_agg["jarvis"], 10)
    used.update(selected["jarvis_only_elite"])
    mp_pool = (top_aggregate["mp"] - top_union["jarvis"]) - used
    selected["mp_only_elite"] = _ordered_selection(p0, mp_pool, source_agg["mp"], 10)
    used.update(selected["mp_only_elite"])

    response_columns = [column for pair in METRICS.values() for column in pair]
    nonzero = p0[response_columns].abs().max(axis=1) > 0
    low_pool = (bottom_aggregate["jarvis"] & bottom_aggregate["mp"]) - used
    low_pool = {i for i in low_pool if bool(nonzero.loc[i])}
    selected["consensus_low"] = _ordered_selection(p0, low_pool, combined, 8, ascending=True)
    used.update(selected["consensus_low"])

    remaining = p0.index.difference(sorted(used))
    p0["atom_bin"] = pd.qcut(p0["atom_count_reduced"], q=3, labels=False, duplicates="drop").fillna(-1).astype(int)
    diversity: list[int] = []
    counts_crystal: dict[str, int] = {}
    counts_atom: dict[int, int] = {}
    counts_formula: dict[str | None, int] = {}
    for i in used:
        c = str(p0.loc[i, "crystal_system"])
        a = int(p0.loc[i, "atom_bin"])
        f = p0.loc[i, "reduced_formula"]
        counts_crystal[c] = counts_crystal.get(c, 0) + 1
        counts_atom[a] = counts_atom.get(a, 0) + 1
        counts_formula[f] = counts_formula.get(f, 0) + 1
    base_true = sum(bool(p0.loc[i, "P2"]) for i in used)
    target_true = max(0, min(10, (48 // 2) - base_true))
    target_counts = {True: target_true, False: 10 - target_true}
    # The diversity stratum is used to balance the complete 48-material set,
    # not merely to force a 5/5 split inside this stratum.  This also respects
    # the crystal-system cap by letting the least represented systems win
    # deterministic greedy tie-breaks.
    for p2_value in (False, True):
        pool = [int(i) for i in remaining if bool(p0.loc[i, "P2"]) == p2_value]
        for _ in range(target_counts[p2_value]):
            if not pool:
                break
            pool.sort(
                key=lambda i: (
                    counts_crystal.get(str(p0.loc[i, "crystal_system"]), 0),
                    counts_atom.get(int(p0.loc[i, "atom_bin"]), 0),
                    counts_formula.get(p0.loc[i, "reduced_formula"], 0),
                    str(p0.loc[i, "pair_id"]),
                )
            )
            i = pool.pop(0)
            diversity.append(i)
            c = str(p0.loc[i, "crystal_system"])
            a = int(p0.loc[i, "atom_bin"])
            f = p0.loc[i, "reduced_formula"]
            counts_crystal[c] = counts_crystal.get(c, 0) + 1
            counts_atom[a] = counts_atom.get(a, 0) + 1
            counts_formula[f] = counts_formula.get(f, 0) + 1
    selected["diversity_balanced"] = diversity

    rows: list[dict[str, Any]] = []
    for stratum in STRATA_ORDER:
        for order, i in enumerate(selected[stratum], start=1):
            row = p0.loc[i]
            rows.append(
                {
                    "stratum": stratum,
                    "selection_order": order,
                    "pair_id": row["pair_id"],
                    "jarvis_id": row["jarvis_id"],
                    "mp_id": row["mp_id"],
                    "formula": row["formula"],
                    "reduced_formula": row["reduced_formula"],
                    "crystal_system": row["crystal_system"],
                    "jarvis_crystal_system": row["jarvis_crystal_system"],
                    "mp_crystal_system": row["mp_crystal_system"],
                    "atom_count_reduced": row["atom_count_reduced"],
                    "P2": bool(row["P2"]),
                    "jarvis_f1": row["jarvis_f1"],
                    "mp_f1": row["mp_f1"],
                    "jarvis_f3": row["jarvis_f3"],
                    "mp_f3": row["mp_f3"],
                    "jarvis_f4": row["jarvis_f4"],
                    "mp_f4": row["mp_f4"],
                    "rms_distance": row["rms_distance"],
                    "max_distance": row["max_distance"],
                    "lattice_distance": row["lattice_distance"],
                    "space_group_relation": row["space_group_relation"],
                }
            )
    candidates = pd.DataFrame(rows)
    balance = {
        "n_candidates": int(len(candidates)),
        "n_unique_pair_ids": int(candidates["pair_id"].nunique()),
        "stratum_counts": candidates["stratum"].value_counts().reindex(STRATA_ORDER, fill_value=0).to_dict(),
        "crystal_system_counts": candidates["crystal_system"].value_counts(dropna=False).to_dict(),
        "P2_counts": {str(k): int(v) for k, v in candidates["P2"].value_counts().to_dict().items()},
        "reduced_formula_max_count": int(candidates["reduced_formula"].value_counts(dropna=False).max()),
        "crystal_system_mismatch_count": int(candidates["crystal_system"].eq("cross_source_mismatch").sum()),
        "p2_balance_pass": bool(candidates["P2"].sum() == len(candidates) / 2),
        "max_crystal_system_pass": bool(candidates["crystal_system"].value_counts().max() <= 10),
        "max_reduced_formula_pass": bool(candidates["reduced_formula"].value_counts(dropna=False).max() <= 3),
    }
    return {
        "candidates": candidates,
        "balance": balance,
        "selection_policy": {
            "panel": "P0",
            "panel_n": n,
            "top_k": k10,
            "bottom_k": k20,
            "endpoint_aggregation": "source-wise mean percentile across F1/F3/F4; stable pair_id tie-break",
            "elite_pool": "union of endpoint-specific source top-k sets; consensus is the intersection of the two source pools",
            "source_only": "source aggregate top-k candidates outside the other source endpoint-union pool",
            "low_pool": "intersection of source aggregate bottom-k sets, restricted to non-zero response",
            "diversity_balanced": "remaining P0 rows; choose the P2/non-P2 counts needed to balance the complete 48-row set, then greedily minimize crystal/atom/formula over-representation with pair_id tie-break",
            "ambiguity_note": "The pre-registration did not specify endpoint aggregation or tie policy; these operational rules are recorded here and must be accepted before external DFT execution.",
        },
    }


def run_matching_sensitivity(panel: pd.DataFrame) -> dict[str, Any]:
    threshold_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for panel_name in ("P0", "P2"):
        base = panel[panel[panel_name].astype(bool)].copy()
        for distance in DISTANCE_FIELDS:
            values = base[distance].to_numpy(float)
            grid = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0.50, 1.00, 11)))
            for q, threshold in zip(np.linspace(0.50, 1.00, len(grid)), grid):
                sub = base[base[distance] <= threshold + 1e-15]
                for metric, (left_col, right_col) in METRICS.items():
                    summary = _stats(sub[left_col].to_numpy(float), sub[right_col].to_numpy(float))
                    threshold_rows.append(
                        {
                            "analysis": "continuous_threshold",
                            "panel": panel_name,
                            "distance_metric": distance,
                            "threshold_quantile": float(q),
                            "threshold_value": float(threshold),
                            "metric": metric,
                            **summary,
                            "n_excluded": int(len(base) - len(sub)),
                            "boundary_count": int(np.isclose(values, threshold, rtol=0.0, atol=1e-12).sum()),
                            "nonidentical_space_group_count": int(sub["space_group_relation"].ne("identical").sum()),
                        }
                    )
            # Equal-frequency distance strata provide a complementary view and
            # keep all rows in exactly one bin, including boundary ties.
            ranks = base[distance].rank(method="first", pct=True)
            bins = pd.cut(
                ranks,
                bins=[0.0, 0.20, 0.40, 0.60, 0.80, 1.0],
                labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                include_lowest=True,
            )
            for stratum, idx in bins.groupby(bins, observed=True).groups.items():
                sub = base.loc[idx]
                for metric, (left_col, right_col) in METRICS.items():
                    summary = _stats(sub[left_col].to_numpy(float), sub[right_col].to_numpy(float))
                    stratum_rows.append(
                        {
                            "analysis": "distance_stratum",
                            "panel": panel_name,
                            "distance_metric": distance,
                            "distance_stratum": str(stratum),
                            "metric": metric,
                            **summary,
                            "n_excluded": int(len(base) - len(sub)),
                            "distance_min": float(sub[distance].min()),
                            "distance_max": float(sub[distance].max()),
                            "boundary_count": int(sub[distance].duplicated(keep=False).sum()),
                            "nonidentical_space_group_count": int(sub["space_group_relation"].ne("identical").sum()),
                        }
                    )
    return {
        "threshold": _write_csv(pd.DataFrame(threshold_rows), RESULT_ROOT / "matching_distance_threshold_sensitivity.csv"),
        "strata": _write_csv(pd.DataFrame(stratum_rows), RESULT_ROOT / "matching_distance_strata.csv"),
    }


def _top_k_overlap(left: np.ndarray, right: np.ndarray, q_percent: float) -> dict[str, float]:
    """Return deterministic top-k overlap diagnostics for a paired score vector."""
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = np.asarray(left)[valid], np.asarray(right)[valid]
    n = len(left)
    if n == 0:
        return {"k": 0, "overlap": float("nan"), "jaccard": float("nan")}
    k = max(1, int(math.floor(q_percent / 100.0 * n)))
    left_top = set(np.argsort(-left, kind="stable")[:k].tolist())
    right_top = set(np.argsort(-right, kind="stable")[:k].tolist())
    overlap = len(left_top & right_top)
    union = 2 * k - overlap
    return {
        "k": k,
        "overlap": float(overlap),
        "jaccard": float(overlap / union) if union else float("nan"),
    }


def run_matching_trim_sensitivity(panel: pd.DataFrame) -> dict[str, Any]:
    """Recompute P0 F1 resolution after removing the largest distance pairs."""
    fractions = (0.0, 0.01, 0.05, 0.10, 0.20)
    rows: list[dict[str, Any]] = []
    base = panel[panel["P0"].astype(bool)].reset_index(drop=True).copy()
    for distance in DISTANCE_FIELDS:
        finite = base[np.isfinite(base[distance].to_numpy(float))].copy()
        finite = finite.sort_values([distance, "pair_id"], ascending=[False, True], kind="mergesort")
        for fraction in fractions:
            n_remove = int(math.floor(fraction * len(finite)))
            removed = set(finite.head(n_remove)["pair_id"].astype(str))
            sub = base[~base["pair_id"].astype(str).isin(removed)].copy()
            left = sub["jarvis_f1"].to_numpy(float)
            right = sub["mp_f1"].to_numpy(float)
            _, summary = cluster_bootstrap_screening(
                left,
                right,
                sub["reduced_formula"].to_numpy(object),
                Q_GRID,
                n_boot=0,
                seed=42,
            )
            overlap = _top_k_overlap(left, right, 10.0)
            rows.append(
                {
                    "analysis": "largest_distance_trim",
                    "panel": "P0",
                    "metric": "F1_Frobenius",
                    "distance_metric": distance,
                    "trim_fraction": fraction,
                    "n_removed": n_remove,
                    "n_retained": int(len(sub)),
                    "distance_cutoff": float(finite.head(n_remove)[distance].min()) if n_remove else float("nan"),
                    "tau": _stats(left, right)["tau"],
                    "rho": _stats(left, right)["rho"],
                    "nAUCC": summary["nAUCC"],
                    "elite_partial_nAUCC": summary["partial_nAUCC_elite"],
                    "top10_k": overlap["k"],
                    "top10_overlap": overlap["overlap"],
                    "top10_jaccard": overlap["jaccard"],
                }
            )
    result = pd.DataFrame(rows)
    return {"trim": _write_csv(result, RESULT_ROOT / "matching_distance_trim_sensitivity.csv"), "df": result}


def _parse_trace(value: Any) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float("nan")
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        arr = np.asarray(parsed, dtype=float)
    except (ValueError, SyntaxError, TypeError):
        return float("nan")
    if arr.shape == (3, 3):
        return float(np.trace(arr))
    if arr.shape == (6,):
        return float(arr[0] + arr[1] + arr[2])
    return float("nan")


def _random_tie_order(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank descending while independently shuffling exact-value ties."""
    values = np.asarray(values, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if not len(valid_idx):
        return valid_idx
    unique = np.unique(values[valid_idx])[::-1]
    chunks: list[np.ndarray] = []
    for value in unique:
        tied = np.flatnonzero(values == value)
        rng.shuffle(tied)
        chunks.append(tied)
    return np.concatenate(chunks)


def _curve_from_orders(order_left: np.ndarray, order_right: np.ndarray, q_grid: np.ndarray) -> pd.DataFrame:
    n = len(order_left)
    pos_left = np.empty(n, dtype=int)
    pos_right = np.empty(n, dtype=int)
    pos_left[order_left] = np.arange(n)
    pos_right[order_right] = np.arange(n)
    rows: list[dict[str, Any]] = []
    for q in q_grid:
        k = max(1, int(math.floor(q / 100.0 * n)))
        overlap = int(((pos_left < k) & (pos_right < k)).sum())
        union = 2 * k - overlap
        jaccard = overlap / union if union else 0.0
        expected = expected_jaccard_hypergeometric(n, k)
        rows.append(
            {
                "q_percentile": float(q),
                "chance_adjusted_jaccard": (jaccard - expected) / (1.0 - expected),
            }
        )
    return pd.DataFrame(rows)


def _read_control_source(filename: str, columns: list[str]) -> pd.DataFrame:
    if CONTROL_SNAPSHOT_ROOT:
        return pd.read_csv(Path(CONTROL_SNAPSHOT_ROOT) / filename, usecols=columns)
    return pd.read_parquet(DATA_ROOT / "T2C-Flow" / "processed" / filename.replace("_controls.csv", ".parquet"), columns=columns)


def run_control_tie_audit(panel: pd.DataFrame) -> dict[str, Any]:
    """Quantify control nAUCC variation caused solely by random tie-breaking."""
    columns = ["material_id", "band_gap", "energy_above_hull", "dielectric_total"]
    if CONTROL_SNAPSHOT_ROOT:
        jarvis = _read_control_source("jarvis_piezo_controls.csv", columns)
        mp = _read_control_source("materials_project_piezo_controls.csv", columns)
    else:
        processed = DATA_ROOT / "T2C-Flow" / "processed"
        jarvis = pd.read_parquet(processed / "jarvis_piezo.parquet", columns=columns)
        mp = pd.read_parquet(processed / "materials_project_piezo.parquet", columns=columns)
    jarvis = jarvis.set_index(jarvis["material_id"].astype(str))
    mp = mp.set_index(mp["material_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for panel_name in ("P0", "P2"):
        sub = panel[panel[panel_name].astype(bool)].reset_index(drop=True)
        for attr in CONTROL_TIE_ATTRIBUTES:
            if attr == "dielectric_total_trace":
                left = sub["jarvis_id"].astype(str).map(jarvis["dielectric_total"].map(_parse_trace)).to_numpy(float)
                right = sub["mp_id"].astype(str).map(mp["dielectric_total"].map(_parse_trace)).to_numpy(float)
            else:
                left = sub["jarvis_id"].astype(str).map(jarvis[attr]).to_numpy(float)
                right = sub["mp_id"].astype(str).map(mp[attr]).to_numpy(float)
            valid = np.isfinite(left) & np.isfinite(right)
            left, right = left[valid], right[valid]
            if len(left) < 2:
                continue
            left_counts = pd.Series(left).value_counts()
            right_counts = pd.Series(right).value_counts()
            stable_seed = 4200 + sum(ord(char) for char in f"{panel_name}:{attr}")
            rng = np.random.default_rng(stable_seed)
            reps: list[dict[str, float]] = []
            for _ in range(CONTROL_TIE_BOOTSTRAP):
                order_left = _random_tie_order(left, rng)
                order_right = _random_tie_order(right, rng)
                curve = _curve_from_orders(order_left, order_right, Q_GRID)
                pos_left = np.empty(len(order_left), dtype=int)
                pos_left[order_left] = np.arange(len(order_left))
                pos_right = np.empty(len(order_right), dtype=int)
                pos_right[order_right] = np.arange(len(order_right))
                top10 = _top_k_overlap(-pos_left, -pos_right, 10.0)
                reps.append(
                    {
                        "nAUCC": naucc(curve),
                        "elite_partial_nAUCC": phase7c_partial_naucc(curve, 1.0, 10.0),
                        "top10_overlap": top10["overlap"],
                    }
                )
            rep_df = pd.DataFrame(reps)
            rows.append(
                {
                    "panel": panel_name,
                    "attribute": attr,
                    "n": int(len(left)),
                    "n_tied_rows_jarvis": int(left_counts[left_counts > 1].sum()),
                    "n_tied_rows_mp": int(right_counts[right_counts > 1].sum()),
                    "tie_break_replicates": CONTROL_TIE_BOOTSTRAP,
                    "nAUCC_stable": float(naucc(_curve_from_orders(np.argsort(-left, kind="stable"), np.argsort(-right, kind="stable"), Q_GRID))),
                    "nAUCC_min": float(rep_df["nAUCC"].min()),
                    "nAUCC_max": float(rep_df["nAUCC"].max()),
                    "nAUCC_ci95_low": float(rep_df["nAUCC"].quantile(0.025)),
                    "nAUCC_ci95_high": float(rep_df["nAUCC"].quantile(0.975)),
                    "elite_partial_nAUCC_min": float(rep_df["elite_partial_nAUCC"].min()),
                    "elite_partial_nAUCC_max": float(rep_df["elite_partial_nAUCC"].max()),
                    "top10_overlap_min": float(rep_df["top10_overlap"].min()),
                    "top10_overlap_max": float(rep_df["top10_overlap"].max()),
                }
            )
    result = pd.DataFrame(rows)
    provenance_path = PROJECT_ROOT / "results" / "phase7c" / "control_provenance.csv"
    clean_provenance = pd.read_csv(provenance_path).rename(
        columns={"same_field_copy_flag": "high_identical_value_fraction_flag"}
    )
    return {
        "ties": _write_csv(result, RESULT_ROOT / "control_tie_sensitivity.csv"),
        "provenance": _write_csv(clean_provenance, RESULT_ROOT / "control_provenance_clean.csv"),
        "df": result,
    }


def run_third_protocol_preflight(candidate_info: dict[str, Any]) -> dict[str, Any]:
    executable_names = ("abinit", "pw.x", "vasp_std", "qvasp")
    executables = {name: shutil.which(name) for name in executable_names}
    runner_files = [
        _portable_path(path)
        for root in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src")
        for path in root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("dfpt", "abinit", "espresso", "vasp"))
    ]
    status = {
        "status": "blocked_preflight",
        "candidate_manifest_generated": True,
        "candidate_count": int(len(candidate_info["candidates"])),
        "third_protocol_config": _portable_path(PAPER_CONFIG),
        "third_protocol_config_sha256": _sha256(PAPER_CONFIG) if PAPER_CONFIG.exists() else None,
        "executables": executables,
        "project_runner_files": runner_files,
        "independent_tensor_outputs": False,
        "raw_source_data_read_only": True,
        "blockers": [
            "No ABINIT, Quantum ESPRESSO, VASP or equivalent third-protocol executable is available on PATH.",
            "No project input-generation/relaxation/DFPT runner or scheduler submission entry point is available.",
            "No documented third-protocol functional, pseudopotential files, code version, k-point generator or resource allocation is available.",
            "The existing MP/JARVIS tensors cannot be relabeled as an independent third protocol.",
            "No new data download or external resource provisioning was authorized in this run.",
        ],
        "required_to_resume": [
            "Confirm the endpoint aggregation/tie policy recorded in the candidate manifest.",
            "Provide a runnable ABINIT or Quantum ESPRESSO environment with exact version and pseudopotentials.",
            "Provide a scheduler/resource allocation and a runner that emits all pre-registered outputs.",
        ],
    }
    return {"status": _write_json(status, RESULT_ROOT / "third_protocol_execution_status.json"), "data": status}


def write_report(
    bootstrap: pd.DataFrame,
    cutoff: pd.DataFrame,
    candidates: pd.DataFrame,
    balance: dict[str, Any],
    preflight: dict[str, Any],
    trim: pd.DataFrame,
    ties: pd.DataFrame,
) -> dict[str, Any]:
    p0 = bootstrap[bootstrap["panel"].eq("P0")].copy()
    lines = [
        "<!-- Generated by scripts/run_scientific_upgrade.py; provenance is in the result manifest. -->",
        "# Phase 9 scientific upgrade",
        "",
        "## Scope",
        "",
        "This report adds a reduced-formula cluster bootstrap, tie-aware cutoff diagnostics, raw-value diagnostics, matching-distance trimming sensitivity and control tie-breaking sensitivity to the frozen P0/P2 panels. It does not modify the Phase 7C result layer.",
        "",
        "## Unified cluster-bootstrap summary",
        "",
        "| Panel | Metric | n | Groups | Kendall tau (95% CI) | nAUCC (95% CI) | Elite partial nAUCC (95% CI) | persistent onset q at delta=0.05 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bootstrap.sort_values(["panel", "metric"]).iterrows():
        onset = row["persistent_onset_delta0.05"]
        onset_text = "not reached" if pd.isna(onset) else f"{onset:.0f}%"
        lines.append(
            f"| {row['panel']} | {row['metric']} | {int(row['n'])} | {int(row['n_groups'])} | "
            f"{row['tau']:.4f} [{row['tau_ci95_low']:.4f}, {row['tau_ci95_high']:.4f}] | "
            f"{row['nAUCC']:.4f} [{row['nAUCC_ci95_low']:.4f}, {row['nAUCC_ci95_high']:.4f}] | "
            f"{row['partial_nAUCC_elite']:.4f} [{row['partial_nAUCC_elite_ci95_low']:.4f}, {row['partial_nAUCC_elite_ci95_high']:.4f}] | {onset_text} |"
        )
    p0_cut = cutoff[(cutoff["panel"] == "P0") & (cutoff["metric"] == "F1_Frobenius") & cutoff["q_percentile"].isin([1.0, 5.0, 10.0])]
    lines += [
        "",
        "## P0 F1 cutoff diagnostics",
        "",
        "| q | k | overlap | tie flag | JARVIS relative gap | MP relative gap | tie-aware overlap bounds |",
        "|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for _, row in p0_cut.iterrows():
        lines.append(
            f"| {row['q_percentile']:.0f} | {int(row['k'])} | {int(row['observed_overlap'])} | "
            f"{'yes' if row['tie_ambiguous'] else 'no'} | {row['left_relative_gap']:.4g} | {row['right_relative_gap']:.4g} | "
            f"[{int(row['min_overlap'])}, {int(row['max_overlap'])}] |"
        )
    raw = pd.read_csv(RESULT_ROOT / "raw_pairwise_diagnostics.csv")
    raw_f1 = raw[(raw["panel"] == "P0") & (raw["metric"] == "F1_Frobenius")].iloc[0]
    threshold = pd.read_csv(RESULT_ROOT / "matching_distance_threshold_sensitivity.csv")
    threshold_f1 = threshold[(threshold["panel"] == "P0") & (threshold["metric"] == "F1_Frobenius")]
    lines += [
        "",
        "## Raw-value and matching-distance diagnostics",
        "",
        f"For P0 F1, the median absolute source difference is {raw_f1['absolute_difference_median']:.3f} C/m$^2$, the 95th percentile is {raw_f1['absolute_difference_q95']:.3f} C/m$^2$, and the median symmetric relative difference is {raw_f1['symmetric_relative_difference_median']:.3f}.",
        "",
        "| Distance field | P0 F1 tau range over 50--100% retained thresholds | P0 F1 nAUCC range |",
        "|---|---:|---:|",
    ]
    for distance, group in threshold_f1.groupby("distance_metric", sort=True):
        lines.append(
            f"| {distance} | [{group['tau'].min():.3f}, {group['tau'].max():.3f}] | "
            f"[{group['nAUCC'].min():.3f}, {group['nAUCC'].max():.3f}] |"
        )
    trim_f1 = trim[trim["metric"].eq("F1_Frobenius")]
    lines += [
        "",
        "### Largest-distance trimming (P0 F1)",
        "",
        "The primary matching panel was reanalysed after removing the largest 1%, 5%, 10% and 20% distances. The table reports point estimates only; it is a sensitivity analysis rather than a redefinition of P0.",
        "",
        "| Distance | Removed | Retained | nAUCC | Elite partial nAUCC | Top-10 overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in trim_f1[trim_f1["trim_fraction"].isin([0.0, 0.10, 0.20])].iterrows():
        lines.append(
            f"| {row['distance_metric']} {row['trim_fraction']:.0%} | {int(row['n_removed'])} | {int(row['n_retained'])} | "
            f"{row['nAUCC']:.3f} | {row['elite_partial_nAUCC']:.3f} | {int(row['top10_overlap'])}/{int(row['top10_k'])} |"
        )
    lines += [
        "",
        "### Control tie-breaking sensitivity",
        "",
        "Exact score ties were independently shuffled 1,000 times for the available control properties. The resulting ranges are reported as tie-induced sensitivity, not as sampling confidence intervals.",
        "",
        "| Panel | Attribute | Tied rows (J/M) | nAUCC range | Elite partial nAUCC range | Top-10 overlap range |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in ties.iterrows():
        lines.append(
            f"| {row['panel']} | {row['attribute']} | {int(row['n_tied_rows_jarvis'])}/{int(row['n_tied_rows_mp'])} | "
            f"[{row['nAUCC_min']:.3f}, {row['nAUCC_max']:.3f}] | [{row['elite_partial_nAUCC_min']:.3f}, {row['elite_partial_nAUCC_max']:.3f}] | "
            f"[{int(row['top10_overlap_min'])}, {int(row['top10_overlap_max'])}] |"
        )
    lines += [
        "",
        "## Third-protocol candidate set",
        "",
        f"The deterministic reconstruction contains {len(candidates)} unique candidates with stratum counts "
        + ", ".join(f"{k}={v}" for k, v in balance["stratum_counts"].items())
        + ". The endpoint aggregation and tie rules were not explicit in the pre-registration; they are recorded in `third_protocol_candidate_manifest.json` and must be accepted before external DFT execution.",
        "",
        f"P2 counts are {balance['P2_counts']}; maximum reduced-formula multiplicity is {balance['reduced_formula_max_count']}; crystal-system mismatch rows are {balance['crystal_system_mismatch_count']}.",
        "",
        "## Third-protocol execution status",
        "",
        "The third protocol is **deferred after preflight**. No independent tensor result is reported. The candidate manifest and execution status are in `third_protocol_execution_status.json` for a future server-backed run.",
        "",
        "## Result files",
        "",
        "- `cluster_bootstrap_summary.csv` and `cluster_bootstrap_curve.csv`",
        "- `cutoff_gap_tie_diagnostics.csv`",
        "- `raw_value_summary.csv` and `raw_pairwise_diagnostics.csv`",
        "- `matching_distance_threshold_sensitivity.csv` and `matching_distance_strata.csv`",
        "- `matching_distance_trim_sensitivity.csv` and `control_tie_sensitivity.csv`",
        "- `control_provenance_clean.csv` (renamed high-identical-value flag)",
        "- `third_protocol_candidate_list.csv`, `third_protocol_candidate_manifest.json`, and `third_protocol_execution_status.json`",
    ]
    path = REPORT_ROOT / "01_scientific_upgrade.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": _portable_path(path), "sha256": _sha256(path)}


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    bootstrap = run_cluster_bootstrap(panel)
    cutoff = run_cutoff_diagnostics(panel)
    raw = run_raw_diagnostics(panel)
    matching = run_matching_sensitivity(panel)
    trim = run_matching_trim_sensitivity(panel)
    ties = run_control_tie_audit(panel)
    candidate_info = build_third_protocol_candidates(panel)
    candidates = candidate_info["candidates"]
    candidate_artifact = _write_csv(candidates, RESULT_ROOT / "third_protocol_candidate_list.csv")
    manifest = {
        "status": "candidate_list_generated_third_protocol_blocked",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_commit": _git_commit(),
        "panel_path": _portable_path(PANEL_PROVENANCE_PATH),
        "panel_sha256": _sha256(PANEL_PROVENANCE_PATH),
        "panel_runtime_path": _portable_path(PANEL_PATH),
        "paper_config_path": _portable_path(PAPER_CONFIG),
        "paper_config_sha256": _sha256(PAPER_CONFIG) if PAPER_CONFIG.exists() else None,
        "candidate_list_artifact": candidate_artifact,
        "balance": candidate_info["balance"],
        "selection_policy": candidate_info["selection_policy"],
        "strata": {
            stratum: candidates[candidates["stratum"].eq(stratum)]["pair_id"].tolist()
            for stratum in STRATA_ORDER
        },
    }
    manifest_artifact = _write_json(manifest, RESULT_ROOT / "third_protocol_candidate_manifest.json")
    preflight = run_third_protocol_preflight(candidate_info)
    report = write_report(
        bootstrap["summary_df"],
        cutoff["df"],
        candidates,
        candidate_info["balance"],
        preflight["data"],
        trim["df"],
        ties["df"],
    )
    output_manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_commit": _git_commit(),
        "panel_sha256": _sha256(PANEL_PROVENANCE_PATH),
        "artifacts": {
            "cluster_bootstrap": bootstrap["summary"],
            "cluster_bootstrap_curve": bootstrap["curve"],
            "cutoff": cutoff["cutoff"],
            "raw_value_summary": raw["raw"],
            "raw_pairwise_diagnostics": raw["pair"],
            "matching_distance_thresholds": matching["threshold"],
            "matching_distance_strata": matching["strata"],
            "matching_distance_trim": trim["trim"],
            "control_tie_sensitivity": ties["ties"],
            "control_provenance_clean": ties["provenance"],
            "candidate_manifest": manifest_artifact,
            "third_protocol_preflight": preflight["status"],
            "report": report,
        },
    }
    _write_json(output_manifest, RESULT_ROOT / "phase9_manifest.json")
    print(json.dumps({"status": "completed_statistics_blocked_third_protocol", "balance": candidate_info["balance"], "report": report}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
