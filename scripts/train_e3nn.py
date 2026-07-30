#!/usr/bin/env python
"""Train e3nn piezo baselines for Phase 5B on a GPU host.

This script is meant to run on the remote `equivcompiler` environment after the
Phase 5B data-pipeline has produced artifacts/phase5b/extended_pairs.parquet.
If that artifact is missing, it falls back to the Phase 0-4 strict pair manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crosspiezo.analysis.baselines import (
    _prepare_records,
    composition_mean_baseline,
    mlp_invariant_baseline,
    source_token_baseline,
    structural_ridge_baseline,
    zero_baseline,
)
from crosspiezo.models.piezo_e3nn import PiezoE3NN
from crosspiezo.models.trainer import (
    PiezoGraphDataset,
    PiezoRecord,
    _collate_records,
    _formula_to_prototype,
    build_paired_counterfactual_eval,
    evaluate_model,
    set_seed,
    train_one_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(name: str) -> dict[str, Any]:
    with open(PROJECT_ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def _data_root(cfg: dict[str, Any]) -> Path:
    env_root = cfg.get("data_root")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    remote = Path("/home/workspace/lrh/DATA")
    if remote.exists():
        return remote
    raise FileNotFoundError("Cannot locate data root")


def _load_dataframes(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _load_config("data_sources.yaml")
    sources = cfg["sources"]["t2c_flow"]
    root = data_root / "T2C-Flow"
    jarvis_df = pd.read_parquet(root / sources["records"]["jarvis_piezo"])
    mp_df = pd.read_parquet(root / sources["records"]["mp_piezo"])
    return jarvis_df, mp_df


def _load_records(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jarvis_df, mp_df = _load_dataframes(data_root)
    return _prepare_records(jarvis_df, mp_df)


def _load_test_panel(data_root: Path, seed: int = 42) -> pd.DataFrame:
    panel_path = PROJECT_ROOT / "artifacts" / "phase5b" / "extended_pairs.parquet"
    if panel_path.exists():
        pairs = pd.read_parquet(panel_path)
    else:
        pairs = pd.read_parquet(PROJECT_ROOT / "artifacts" / "pair_manifests" / "strict_pairs.parquet")
    pairs = pairs.copy()
    pairs["prototype"] = pairs["formula"].apply(_formula_to_prototype)
    prototypes = pairs["prototype"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(prototypes)
    n_test = max(1, int(round(len(prototypes) * 0.2)))
    test_prototypes = set(prototypes[:n_test])
    test = pairs[pairs["prototype"].isin(test_prototypes)].copy()
    test["split"] = "test"
    return test


def _build_eval_records(
    jarvis_records: list[dict[str, Any]],
    mp_records: list[dict[str, Any]],
    test_panel: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    test_jids = set(test_panel["jarvis_id"])
    test_mids = set(test_panel["mp_id"])
    test_formulas = set(test_panel["formula"])
    test_prototypes = set(test_panel["prototype"])

    eval_sets: dict[str, list[dict[str, Any]]] = {}
    eval_sets["jarvis_insource"] = [r for r in jarvis_records if r["id"] in test_jids]
    eval_sets["mp_insource"] = [r for r in mp_records if r["id"] in test_mids]
    # With only two sources, a pooled model cannot be "source-held-out".
    # Cross-source evaluation is reported as cross_source, not source_held_out.
    eval_sets.update(build_paired_counterfactual_eval(test_panel, jarvis_records, mp_records))

    eval_sets["jarvis_formula_disjoint"] = [r for r in jarvis_records if r["formula"] in test_formulas]
    eval_sets["mp_formula_disjoint"] = [r for r in mp_records if r["formula"] in test_formulas]

    eval_sets["jarvis_prototype_disjoint"] = [
        r for r in jarvis_records if _formula_to_prototype(r["formula"]) in test_prototypes
    ]
    eval_sets["mp_prototype_disjoint"] = [
        r for r in mp_records if _formula_to_prototype(r["formula"]) in test_prototypes
    ]
    return eval_sets


def _train_val_split(records: list[PiezoRecord], val_frac: float, seed: int) -> tuple[list[PiezoRecord], list[PiezoRecord]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(records) * val_frac)))
    train = [records[i] for i in idx[n_val:]]
    val = [records[i] for i in idx[:n_val]]
    return train, val


def _make_loader(records: list[PiezoRecord], batch_size: int, source_token_dim: int, shuffle: bool) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    class _SimpleDS(torch.utils.data.Dataset[PiezoRecord]):
        def __init__(self, recs: list[PiezoRecord]) -> None:
            self.recs = recs
        def __len__(self) -> int:
            return len(self.recs)
        def __getitem__(self, i: int) -> PiezoRecord:
            return self.recs[i]
    return torch.utils.data.DataLoader(
        _SimpleDS(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: _collate_records(batch, source_token_dim),
        num_workers=0,
    )


def _run_baseline_row(
    name: str,
    train_recs: list[dict[str, Any]],
    eval_recs: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    split_type: str,
    seed: int,
) -> dict[str, Any]:
    if name == "zero":
        result = zero_baseline(eval_recs, train_source, eval_source, seed=seed)
    elif name == "composition_mean":
        result = composition_mean_baseline(train_recs, eval_recs, train_source, eval_source, seed=seed)
    elif name == "source_specific_composition_mean":
        result = composition_mean_baseline(train_recs, eval_recs, train_source, eval_source, seed=seed, source_specific=True)
    elif name == "source_token_ridge":
        result = source_token_baseline(train_recs, eval_recs, train_source, eval_source, seed=seed)
    elif name == "structural_ridge":
        result = structural_ridge_baseline(train_recs, eval_recs, train_source, eval_source, feature_mode="structure", seed=seed)
    elif name == "mlp_invariant":
        result = mlp_invariant_baseline(train_recs, eval_recs, train_source, eval_source, seed=seed)
    else:
        raise ValueError(name)
    return {
        "model_name": result.baseline_name,
        "train_source": train_source,
        "eval_source": eval_source,
        "split_type": f"{split_type}_{eval_source}",
        "seed": seed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "absolute_frobenius_mae": result.absolute_frobenius_mae,
        "normalized_frobenius_mae": result.normalized_frobenius_mae,
        "component_mae": result.component_mae,
        "cosine_similarity": result.cosine_similarity,
        "amplitude_ratio": result.amplitude_ratio,
    }


def _baseline_splits(
    train_jarvis: list[dict[str, Any]],
    train_mp: list[dict[str, Any]],
    eval_jarvis_recs: list[dict[str, Any]],
    eval_mp_recs: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], str, str, str]]:
    """Return the baseline evaluation split definitions.

    With only two sources, a pooled model cannot be "source-held-out"; those
    evaluations are reported as ``cross_source`` instead.
    """
    return [
        ("zero", [], eval_jarvis_recs, "none", "jarvis", "in_source"),
        ("zero", [], eval_mp_recs, "none", "mp", "in_source"),
        ("composition_mean", train_jarvis, eval_jarvis_recs, "jarvis", "jarvis", "in_source"),
        ("composition_mean", train_mp, eval_mp_recs, "mp", "mp", "in_source"),
        ("source_specific_composition_mean", train_jarvis + train_mp, eval_jarvis_recs, "pooled", "jarvis", "in_source"),
        ("source_specific_composition_mean", train_jarvis + train_mp, eval_mp_recs, "pooled", "mp", "in_source"),
        ("structural_ridge", train_jarvis, eval_jarvis_recs, "jarvis", "jarvis", "in_source"),
        ("structural_ridge", train_mp, eval_mp_recs, "mp", "mp", "in_source"),
        ("structural_ridge", train_jarvis, eval_mp_recs, "jarvis", "mp", "source_held_out"),
        ("structural_ridge", train_mp, eval_jarvis_recs, "mp", "jarvis", "source_held_out"),
        ("mlp_invariant", train_jarvis, eval_jarvis_recs, "jarvis", "jarvis", "in_source"),
        ("mlp_invariant", train_mp, eval_mp_recs, "mp", "mp", "in_source"),
        ("mlp_invariant", train_jarvis + train_mp, eval_jarvis_recs, "pooled", "jarvis", "in_source"),
        ("mlp_invariant", train_jarvis + train_mp, eval_mp_recs, "pooled", "mp", "in_source"),
        ("mlp_invariant", train_jarvis, eval_mp_recs, "jarvis", "mp", "cross_source"),
        ("mlp_invariant", train_mp, eval_jarvis_recs, "mp", "jarvis", "cross_source"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=200, help="skip structures with more atoms")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-e3nn", action="store_true", help="train only fast baselines, skip e3nn")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    cfg = _load_config("data_sources.yaml")
    data_root = _data_root(cfg)
    jarvis_df, mp_df = _load_dataframes(data_root)
    jarvis_records, mp_records = _load_records(data_root)
    test_panel = _load_test_panel(data_root)
    test_panel.to_parquet(PROJECT_ROOT / "artifacts" / "phase5b" / "test_panel.parquet")
    test_jids = set(test_panel["jarvis_id"])
    test_mids = set(test_panel["mp_id"])

    # Exclude test panel from training (and also formula/prototype for strict splits).
    def _train_pool(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in recs if r["id"] not in test_jids and r["id"] not in test_mids]

    train_jarvis = _train_pool(jarvis_records)
    train_mp = _train_pool(mp_records)

    # Eval record sets for baselines.
    eval_jarvis_recs = [r for r in jarvis_records if r["id"] in test_jids]
    eval_mp_recs = [r for r in mp_records if r["id"] in test_mids]

    # Build PyTorch datasets only if e3nn models are requested.
    if not args.skip_e3nn:
        def _build_dataset(df: pd.DataFrame, source: str) -> list[PiezoRecord]:
            ds = PiezoGraphDataset(df, source=source)
            # Filter oversized structures to keep memory reasonable.
            return [r for r in ds.records if len(r.z) <= args.max_atoms]

        jarvis_all = _build_dataset(jarvis_df, "jarvis")
        mp_all = _build_dataset(mp_df, "mp")
        train_jarvis_ds = [r for r in jarvis_all if r.material_id not in test_jids]
        train_mp_ds = [r for r in mp_all if r.material_id not in test_mids]
        pooled_all = train_jarvis_ds + train_mp_ds

        # Eval sets.
        eval_jarvis = [r for r in jarvis_all if r.material_id in test_jids]
        eval_mp = [r for r in mp_all if r.material_id in test_mids]
    else:
        # Dummy containers so the e3nn loop can be skipped cleanly.
        jarvis_all = []
        mp_all = []
        train_jarvis_ds = []
        train_mp_ds = []
        pooled_all = []
        eval_jarvis = []
        eval_mp = []

    metrics_rows: list[dict[str, Any]] = []

    # Baselines.
    baseline_splits = _baseline_splits(
        train_jarvis, train_mp, eval_jarvis_recs, eval_mp_recs
    )
    for name, tr, ev, ts, es, st in baseline_splits:
        for seed in range(42, 42 + args.n_seeds):
            try:
                row = _run_baseline_row(name, tr, ev, ts, es, st, seed)
                metrics_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"Baseline {name} {ts}->{es} seed {seed} failed: {exc}")

    # e3nn models.
    e3nn_configs: list[tuple[str, list[PiezoRecord], int]] = [
        ("e3nn_jarvis", train_jarvis_ds, 0),
        ("e3nn_mp", train_mp_ds, 0),
        ("e3nn_pooled", pooled_all, 0),
        ("e3nn_source_token", pooled_all, 2),
    ]

    out_dir = PROJECT_ROOT / "artifacts" / "phase5b"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, train_pool, token_dim in e3nn_configs:
        if not train_pool:
            continue
        eval_map: dict[str, list[PiezoRecord]] = {
            "in_source_jarvis": eval_jarvis,
            "in_source_mp": eval_mp,
            "cross_source_jarvis": eval_jarvis,
            "cross_source_mp": eval_mp,
            "paired_counterfactual_jarvis": eval_jarvis,
            "paired_counterfactual_mp": eval_mp,
        }
        # Source-specific models only evaluate on their own source.
        if model_name == "e3nn_jarvis":
            eval_map = {"in_source_jarvis": eval_jarvis}
        elif model_name == "e3nn_mp":
            eval_map = {"in_source_mp": eval_mp}

        seed_results: dict[str, list[dict[str, Any]]] = {k: [] for k in eval_map}
        for seed in range(42, 42 + args.n_seeds):
            set_seed(seed)
            train_recs, val_recs = _train_val_split(train_pool, val_frac=0.1, seed=seed)
            train_loader = _make_loader(train_recs, args.batch_size, token_dim, shuffle=True)
            val_loader = _make_loader(val_recs, args.batch_size, token_dim, shuffle=False)

            model = PiezoE3NN(
                num_atom_types=100,
                hidden_dim=args.hidden,
                num_layers=args.layers,
                lmax=args.lmax,
                source_token_dim=token_dim,
            )
            print(f"\nTraining {model_name} seed {seed} (train={len(train_recs)}, val={len(val_recs)})")
            info = train_one_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
            print(f"  best val MAE: {info['best_val_mae']:.4f}")

            for split_name, eval_recs in eval_map.items():
                if not eval_recs:
                    continue
                loader = _make_loader(eval_recs, args.batch_size, token_dim, shuffle=False)
                mets, preds, ys = evaluate_model(model, loader, device)
                ts = "jarvis" if model_name == "e3nn_jarvis" else ("mp" if model_name == "e3nn_mp" else "pooled")
                es = "jarvis" if "jarvis" in split_name else "mp"
                metrics_rows.append({
                    "model_name": model_name,
                    "train_source": ts,
                    "eval_source": es,
                    "split_type": split_name,
                    "seed": seed,
                    "n_train": len(train_recs),
                    "n_test": len(eval_recs),
                    **mets,
                })
                seed_results[split_name].append({"preds": preds, "ys": ys})

        # Save ensemble mean predictions across seeds for the first eval split.
        for split_name, results in seed_results.items():
            if not results:
                continue
            preds = np.stack([r["preds"] for r in results], axis=0)
            ys = results[0]["ys"]
            np.savez(
                out_dir / f"{model_name}_{split_name}_predictions.npz",
                preds_mean=preds.mean(axis=0),
                preds_std=preds.std(axis=0),
                ys=ys,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_parquet(out_dir / "model_metrics.parquet")
    metrics_df.to_json(out_dir / "model_metrics.json", orient="records", indent=2)
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nWrote {len(metrics_df)} metric rows to {out_dir / 'model_metrics.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
