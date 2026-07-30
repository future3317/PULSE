"""PyTorch Geometric-style dataset for piezoelectric tensor prediction.

Reads structures from T2C-Flow parquet CIFs and targets from the same rows.
Builds radius graphs with periodic boundary conditions via pymatgen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PiezoData:
    """A single crystal sample for the equivariant model."""

    material_id: str
    formula: str
    source: str
    z: torch.Tensor  # atomic numbers, shape (n_atoms,)
    pos: torch.Tensor  # Cartesian positions, shape (n_atoms, 3)
    lattice: torch.Tensor  # lattice matrix, shape (3, 3)
    edge_index: torch.Tensor  # shape (2, n_edges)
    edge_vec: torch.Tensor  # relative vectors, shape (n_edges, 3)
    edge_len: torch.Tensor  # distances, shape (n_edges,)
    target: torch.Tensor  # Cartesian piezo tensor, shape (3, 3, 3)


def _tensor_from_row(row: pd.Series) -> np.ndarray | None:
    """Extract 3x3x3 Cartesian total piezo tensor from a T2C-Flow row."""
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

    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        from crosspiezo.conventions.voigt import voigt_to_cartesian
        return voigt_to_cartesian(voigt, engineering_shear=True)
    return None


def _build_periodic_graph(
    structure: Structure,
    cutoff: float = 4.0,
    max_neighbors: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a radius graph with periodic boundary conditions.

    Returns edge_index, edge_vec, edge_len.
    """
    centers: list[int] = []
    neighbors: list[int] = []
    vectors: list[np.ndarray] = []
    for i, site in enumerate(structure):
        nbrs = structure.get_neighbors(site, r=cutoff)
        # Sort by distance and cap neighbors.
        nbrs = sorted(nbrs, key=lambda x: x[1])[:max_neighbors]
        for nbr, _dist in nbrs:
            j = nbr.index
            vec = nbr.coords - site.coords
            centers.append(i)
            neighbors.append(j)
            vectors.append(vec)
    if not centers:
        # Single atom or no neighbors: self-loop.
        centers = [0]
        neighbors = [0]
        vectors = [np.zeros(3)]
    edge_index = torch.tensor([centers, neighbors], dtype=torch.long)
    edge_vec = torch.tensor(np.stack(vectors), dtype=torch.float32)
    edge_len = torch.linalg.norm(edge_vec, dim=1)
    return edge_index, edge_vec, edge_len


class PiezoDataset(Dataset[PiezoData]):
    """Dataset of crystal structures and piezoelectric tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        source: str,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
    ) -> None:
        self.records: list[PiezoData] = []
        self.source = source
        for _, row in df.iterrows():
            cif = row.get("cif")
            if not isinstance(cif, str):
                continue
            try:
                structure = Structure.from_str(cif, fmt="cif")
            except Exception:  # noqa: BLE001
                continue
            tensor = _tensor_from_row(row)
            if tensor is None:
                continue
            if len(structure) == 0:
                continue
            edge_index, edge_vec, edge_len = _build_periodic_graph(
                structure, cutoff=cutoff, max_neighbors=max_neighbors
            )
            self.records.append(PiezoData(
                material_id=str(row.get("material_id", "")),
                formula=str(row.get("formula", "")),
                source=source,
                z=torch.tensor(structure.atomic_numbers, dtype=torch.long),
                pos=torch.tensor(structure.cart_coords, dtype=torch.float32),
                lattice=torch.tensor(structure.lattice.matrix, dtype=torch.float32),
                edge_index=edge_index,
                edge_vec=edge_vec,
                edge_len=edge_len,
                target=torch.tensor(tensor, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PiezoData:
        return self.records[idx]
