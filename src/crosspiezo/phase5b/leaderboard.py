"""Leaderboard stress-test aggregation for Phase 5B."""

from __future__ import annotations

from typing import Any

import pandas as pd

SPLIT_ORDER = [
    "in_source_jarvis",
    "in_source_mp",
    "formula_disjoint_jarvis",
    "formula_disjoint_mp",
    "prototype_disjoint_jarvis",
    "prototype_disjoint_mp",
    "source_held_out_jarvis",
    "source_held_out_mp",
    "paired_counterfactual_jarvis",
    "paired_counterfactual_mp",
]


def aggregate_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics by model/split over seeds."""
    return metrics.groupby(["model_name", "train_source", "eval_source", "split_type"]).agg(
        n_seeds=("seed", "nunique"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        absolute_frobenius_mae_mean=("absolute_frobenius_mae", "mean"),
        absolute_frobenius_mae_std=("absolute_frobenius_mae", "std"),
        normalized_frobenius_mae_mean=("normalized_frobenius_mae", "mean"),
        normalized_frobenius_mae_std=("normalized_frobenius_mae", "std"),
        cosine_similarity_mean=("cosine_similarity", "mean"),
        component_mae_mean=("component_mae", "mean"),
    ).reset_index()


def rank_inversion_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Compare in-source and source-held-out rankings per model."""
    rows: list[dict[str, Any]] = []
    in_src = agg[agg["split_type"].isin(["in_source_jarvis", "in_source_mp"])]
    held = agg[agg["split_type"].isin(["source_held_out_jarvis", "source_held_out_mp"])]
    for model_name in agg["model_name"].unique():
        for eval_src in ["jarvis", "mp"]:
            in_key = f"in_source_{eval_src}"
            held_key = f"source_held_out_{eval_src}"
            in_val = in_src[(in_src["model_name"] == model_name) & (in_src["split_type"] == in_key)]["absolute_frobenius_mae_mean"]
            held_val = held[(held["model_name"] == model_name) & (held["split_type"] == held_key)]["absolute_frobenius_mae_mean"]
            if len(in_val) == 0 or len(held_val) == 0:
                continue
            rows.append({
                "model_name": model_name,
                "eval_source": eval_src,
                "in_source_mae": float(in_val.iloc[0]),
                "source_held_out_mae": float(held_val.iloc[0]),
                "degradation_ratio": float(held_val.iloc[0] / (in_val.iloc[0] + 1e-12)),
                "rank_inversion": bool(held_val.iloc[0] > in_val.iloc[0]),
            })
    return pd.DataFrame(rows)


def source_token_gain(agg: pd.DataFrame) -> pd.DataFrame:
    """Compare pooled vs source-token models on cross-source splits."""
    rows: list[dict[str, Any]] = []
    for split_type in ["source_held_out_jarvis", "source_held_out_mp", "paired_counterfactual_jarvis", "paired_counterfactual_mp"]:
        pooled = agg[(agg["model_name"] == "e3nn_pooled") & (agg["split_type"] == split_type)]
        token = agg[(agg["model_name"] == "e3nn_source_token") & (agg["split_type"] == split_type)]
        if len(pooled) == 0 or len(token) == 0:
            continue
        rows.append({
            "split_type": split_type,
            "pooled_mae": float(pooled["absolute_frobenius_mae_mean"].iloc[0]),
            "source_token_mae": float(token["absolute_frobenius_mae_mean"].iloc[0]),
            "token_improvement": float(pooled["absolute_frobenius_mae_mean"].iloc[0] - token["absolute_frobenius_mae_mean"].iloc[0]),
        })
    return pd.DataFrame(rows)
