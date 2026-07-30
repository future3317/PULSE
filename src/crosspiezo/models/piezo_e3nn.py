"""Minimal O(3)-equivariant piezoelectric tensor baseline using e3nn."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.nn import FullyConnectedNet
from e3nn.nn.models.gate_points_2101 import Network


class PiezoE3NN(nn.Module):
    """Equivariant GNN predicting a rank-3 polar Cartesian tensor.

    The network outputs the full 3x3x3 tensor; the last two indices are
    symmetrized so that the prediction is a valid piezoelectric stress tensor.
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        hidden_dim: int = 32,
        num_layers: int = 2,
        max_radius: float = 4.0,
        num_basis: int = 8,
        lmax: int = 2,
    ) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.max_radius = max_radius
        self.ct = CartesianTensor("ijk")

        # Atomic number embedding as scalar node attributes.
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)

        irreps_hidden = (o3.Irreps.spherical_harmonics(lmax) * hidden_dim).simplify()
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        irreps_node_attr = o3.Irreps(f"{hidden_dim}x0e")
        irreps_output = CartesianTensor("ijk")  # rank-3 polar tensor

        self.network = Network(
            irreps_in=o3.Irreps("1x0e"),
            irreps_hidden=irreps_hidden,
            irreps_out=irreps_output,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            layers=num_layers,
            max_radius=max_radius,
            number_of_basis=num_basis,
            radial_layers=1,
            radial_neurons=32,
            num_neighbors=12.0,
            num_nodes=20.0,
            reduce_output=True,
        )

        # Project scalar embedding to node_attr.
        self.node_attr_proj = FullyConnectedNet(
            [hidden_dim, hidden_dim, hidden_dim],
            torch.nn.functional.silu,
        )

    def forward(self, data: Any) -> torch.Tensor:
        """Forward pass returning a (batch, 3, 3, 3) Cartesian tensor."""
        z = data.z
        pos = data.pos
        batch = getattr(data, "batch", torch.zeros(z.size(0), dtype=torch.long, device=z.device))

        # Node features: just a single scalar per atom.
        x = torch.ones(z.size(0), 1, device=z.device, dtype=torch.float32)

        # Node attributes from atom type embedding.
        node_attr = self.atom_embed(z)
        node_attr = self.node_attr_proj(node_attr)

        # e3nn.Network builds the radius graph internally and expects a dict/Data
        # with ``pos``, ``x``, ``z``, and ``batch``.
        out = self.network({"pos": pos, "x": x, "z": node_attr, "batch": batch})

        # Convert irrep output to Cartesian tensor and symmetrize last two indices.
        cart = self.ct.to_cartesian(out)
        cart = 0.5 * (cart + cart.transpose(1, 2))
        return cart
