"""Training and evaluation utilities for the e3nn piezo baseline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, Dataset

from crosspiezo.conventions.voigt import voigt_to_cartesian


def _to_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        import ast
        return np.asarray(ast.literal_eval(value), dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    return None


def _tensor_from_row(row: pd.Series) -> np.ndarray | None:
    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        return voigt_to_cartesian(voigt, engineering_shear=True)
    return None


def _formula_to_prototype(formula: str) -> str:
    """Return an anonymous-formula prototype (e.g. AB2), sensitive to stoichiometry.

    Unlike ``Composition.anonymized_formula`` this keeps the integer
    stoichiometric ratio, so Na2Cl2 (A2B2) and NaCl (AB) are different prototypes.
    """
    from pymatgen.core.composition import Composition

    comp = Composition(formula)
    items = sorted(comp.items(), key=lambda item: (-item[1], str(item[0])))
    return "".join(
        f"{chr(ord('A') + i)}{int(amount)}" for i, (_, amount) in enumerate(items)
    )


@dataclass(frozen=True)
class PiezoRecord:
    material_id: str
    formula: str
    source: str  # "jarvis" or "mp"
    prototype: str
    z: torch.Tensor
    pos: torch.Tensor
    lattice: torch.Tensor
    edge_index: torch.Tensor
    edge_vec: torch.Tensor
    edge_len: torch.Tensor
    target: torch.Tensor


def _build_periodic_graph(
    struct: Structure, cutoff: float = 4.0, max_neighbors: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a radius graph with periodic boundary conditions via pymatgen."""
    centers: list[int] = []
    neighbors: list[int] = []
    vectors: list[np.ndarray] = []
    for i, site in enumerate(struct):
        nbrs = struct.get_neighbors(site, r=cutoff)
        # ``get_neighbors`` returns neighbor objects directly in newer pymatgen.
        def _dist(nbr: Any) -> float:
            return float(getattr(nbr, "nn_distance", getattr(nbr, "distance", 0.0)))

        nbrs = sorted(nbrs, key=_dist)[:max_neighbors]
        for nbr in nbrs:
            j = int(getattr(nbr, "index", nbr))
            vec = np.asarray(nbr.coords, dtype=np.float64) - np.asarray(site.coords, dtype=np.float64)
            centers.append(i)
            neighbors.append(j)
            vectors.append(vec)
    if not centers:
        centers = [0]
        neighbors = [0]
        vectors = [np.zeros(3)]
    edge_index = torch.tensor([centers, neighbors], dtype=torch.long)
    edge_vec = torch.tensor(np.stack(vectors), dtype=torch.float32)
    edge_len = torch.linalg.norm(edge_vec, dim=1)
    return edge_index, edge_vec, edge_len


class PiezoGraphDataset(Dataset[PiezoRecord]):
    """Lightweight dataset for the e3nn radius-graph model.

    The record now also carries lattice and periodic-edge information.  The
    current ``PiezoE3NN`` model does not consume these edges and is therefore
    marked as an invalid crystal baseline until a periodic-graph equivariant
    model is implemented.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source: str,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
    ) -> None:
        self.records: list[PiezoRecord] = []
        for _, row in df.iterrows():
            cif = row.get("cif")
            if not isinstance(cif, str):
                continue
            try:
                struct = Structure.from_str(cif, fmt="cif")
            except Exception:  # noqa: BLE001
                continue
            tensor = _tensor_from_row(row)
            if tensor is None or len(struct) == 0:
                continue
            edge_index, edge_vec, edge_len = _build_periodic_graph(struct, cutoff, max_neighbors)
            self.records.append(PiezoRecord(
                material_id=str(row["material_id"]),
                formula=str(row["formula"]),
                source=source,
                prototype=_formula_to_prototype(str(row["formula"])),
                z=torch.tensor(struct.atomic_numbers, dtype=torch.long),
                pos=torch.tensor(struct.cart_coords, dtype=torch.float32),
                lattice=torch.tensor(struct.lattice.matrix, dtype=torch.float32),
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_len=edge_len,
                target=torch.tensor(tensor, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PiezoRecord:
        return self.records[idx]


def _collate_records(batch: list[PiezoRecord], source_token_dim: int = 0) -> dict[str, torch.Tensor]:
    pos_chunks: list[torch.Tensor] = []
    z_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    batch_idx: list[torch.Tensor] = []
    edge_index_chunks: list[torch.Tensor] = []
    edge_vec_chunks: list[torch.Tensor] = []
    edge_len_chunks: list[torch.Tensor] = []
    lattice_chunks: list[torch.Tensor] = []
    offset = 0
    for i, rec in enumerate(batch):
        pos_chunks.append(rec.pos)
        z_chunks.append(rec.z)
        if source_token_dim > 0:
            token = torch.zeros(len(rec.z), source_token_dim, dtype=torch.float32)
            token[:, 0 if rec.source == "jarvis" else 1] = 1.0
            token_chunks.append(token)
        y_chunks.append(rec.target)
        batch_idx.append(torch.full((len(rec.z),), i, dtype=torch.long))
        edge_index_chunks.append(rec.edge_index + offset)
        edge_vec_chunks.append(rec.edge_vec)
        edge_len_chunks.append(rec.edge_len)
        lattice_chunks.append(rec.lattice)
        offset += len(rec.z)
    out: dict[str, torch.Tensor] = {
        "pos": torch.cat(pos_chunks, dim=0),
        "z": torch.cat(z_chunks, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "y": torch.stack(y_chunks, dim=0),
        "edge_index": torch.cat(edge_index_chunks, dim=1),
        "edge_vec": torch.cat(edge_vec_chunks, dim=0),
        "edge_len": torch.cat(edge_len_chunks, dim=0),
        "lattice": torch.stack(lattice_chunks, dim=0),
    }
    if source_token_dim > 0:
        out["source_token"] = torch.cat(token_chunks, dim=0)
    return out


def build_paired_counterfactual_eval(
    test_panel: pd.DataFrame,
    jarvis_records: list[dict[str, Any]],
    mp_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return the paired-counterfactual eval sets for the two sources.

    A "paired counterfactual" prediction for a JARVIS test material is made on
    its MP mate (and vice-versa), never on a same-source record.  This prevents
    the cross-source test set from leaking into the source's own test panel.
    """
    mate_jarvis_ids = set(test_panel["jarvis_id"])
    mate_mp_ids = set(test_panel["mp_id"])
    jarvis_by_id = {r["id"]: r for r in jarvis_records}
    mp_by_id = {r["id"]: r for r in mp_records}
    return {
        "paired_counterfactual_jarvis": [mp_by_id[mid] for mid in mate_mp_ids if mid in mp_by_id],
        "paired_counterfactual_mp": [jarvis_by_id[jid] for jid in mate_jarvis_ids if jid in jarvis_by_id],
    }


def paired_counterfactual_is_disjoint(
    paired_eval: dict[str, list[dict[str, Any]]],
    train_ids: set[str],
) -> bool:
    """Return True if no paired-counterfactual eval record appears in training."""
    for recs in paired_eval.values():
        if any(r["id"] in train_ids for r in recs):
            return False
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """pred, target shape (batch, 3, 3, 3)."""
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    diff = pred - target
    abs_err = torch.linalg.norm(diff.reshape(diff.size(0), -1), dim=1)
    target_norm = torch.linalg.norm(target.reshape(target.size(0), -1), dim=1)
    pred_norm = torch.linalg.norm(pred.reshape(pred.size(0), -1), dim=1)
    norm = 0.5 * (target_norm + pred_norm) + 1e-6
    comp_err = torch.mean(torch.abs(diff.reshape(diff.size(0), -1)), dim=1)
    flat_t = target.reshape(target.size(0), -1)
    flat_p = pred.reshape(pred.size(0), -1)
    cos = (flat_t * flat_p).sum(dim=1) / (flat_t.norm(dim=1) * flat_p.norm(dim=1) + 1e-12)
    amp = pred_norm / (target_norm + 1e-12)
    return {
        "absolute_frobenius_mae": float(abs_err.mean()),
        "normalized_frobenius_mae": float((abs_err / norm).mean()),
        "component_mae": float(comp_err.mean()),
        "cosine_similarity": float(cos.mean()),
        "amplitude_ratio": float(amp.mean()),
    }


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    val_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 10,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, Any] = {}
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * pred.size(0)
        train_loss /= len(train_loader.dataset) if len(train_loader.dataset) else 1

        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {}
        with torch.no_grad():
            all_pred: list[torch.Tensor] = []
            all_y: list[torch.Tensor] = []
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch)
                loss = criterion(pred, batch["y"])
                val_loss += float(loss.item()) * pred.size(0)
                all_pred.append(pred.cpu())
                all_y.append(batch["y"].cpu())
            val_loss /= len(val_loader.dataset) if len(val_loader.dataset) else 1
            all_pred_t = torch.cat(all_pred, dim=0)
            all_y_t = torch.cat(all_y, dim=0)
            val_metrics = _metrics(all_pred_t, all_y_t)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        scheduler.step(val_loss)
        if val_metrics["absolute_frobenius_mae"] < best_val:
            best_val = val_metrics["absolute_frobenius_mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_mae": best_val}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)
            preds.append(pred.cpu())
            ys.append(batch["y"].cpu())
    pred_t = torch.cat(preds, dim=0)
    y_t = torch.cat(ys, dim=0)
    return _metrics(pred_t, y_t), pred_t.numpy(), y_t.numpy()
