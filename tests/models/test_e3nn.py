"""Red tests for C-10 and C-11: e3nn output symmetry and periodic graph handling.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

e3nn = pytest.importorskip("e3nn")

from crosspiezo.models.piezo_e3nn import PiezoE3NN  # noqa: E402
from crosspiezo.models.trainer import PiezoGraphDataset  # noqa: E402


def _dummy_batch(n_atoms: int = 5) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(601)
    return {
        "z": torch.tensor([1] * n_atoms, dtype=torch.long),
        "pos": torch.tensor(rng.normal(size=(n_atoms, 3)), dtype=torch.float32),
        "batch": torch.zeros(n_atoms, dtype=torch.long),
    }


def test_output_symmetrizes_last_two_indices():
    """Predicted piezo tensor must be symmetric in the last two (strain) indices."""
    model = PiezoE3NN(num_atom_types=10, hidden_dim=8, num_layers=1, lmax=1)
    model.eval()
    with torch.no_grad():
        out = model(_dummy_batch())
    assert out.shape[1:] == (3, 3, 3)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-5), (
        "output is not symmetric in the last two tensor indices"
    )


def test_output_does_not_symmetrize_first_two_indices():
    """The first index (polar response direction) must remain independent."""
    source = inspect.getsource(PiezoE3NN.forward)
    # The forward pass must symmetrize the last two strain indices, not the
    # first two (polar and first strain).
    assert "transpose(-1, -2)" in source or "transpose(2, 3)" in source, (
        "forward pass does not symmetrize the last two tensor indices"
    )
    assert "transpose(1, 2)" not in source, (
        "forward pass incorrectly symmetrizes the first two tensor indices"
    )


def test_dataset_exposes_periodic_graph_fields():
    """The dataset consumed by the model must expose lattice and PBC edges."""
    pytest.importorskip("pymatgen")
    from pymatgen.core.structure import Structure

    lattice = np.eye(3) * 3.0
    struct = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    import pandas as pd

    df = pd.DataFrame({
        "material_id": ["test"],
        "formula": ["NaCl"],
        "cif": [struct.to(fmt="cif")],
        "piezo_cartesian_total": [np.zeros((3, 3, 3), dtype=np.float64).tolist()],
    })
    ds = PiezoGraphDataset(df, source="jarvis")
    assert len(ds) > 0
    rec = ds[0]
    assert hasattr(rec, "lattice"), "record missing lattice"
    assert hasattr(rec, "edge_index"), "record missing edge_index"
    assert hasattr(rec, "edge_vec"), "record missing edge_vec"
