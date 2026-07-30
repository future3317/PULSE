"""Red tests for C-02: polar rank-3 O(3) parity.

A polar tensor of odd rank picks up one factor of det(R) from the three
rotation matrices R \otimes R \otimes R.  It must *not* receive an additional
det(R) factor; that would be the transformation law for an axial/pseudotensor.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core.operations import SymmOp

from crosspiezo.analysis.o3_transport import _transport_tensor


def _random_polar_tensor(rng: np.random.Generator) -> np.ndarray:
    cart = rng.normal(size=(3, 3, 3))
    return 0.5 * (cart + cart.transpose(0, 2, 1))


def _apply_rotation_matrix(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, tensor)


def test_polar_tensor_under_inversion():
    """Inversion (R = -I) must map a polar rank-3 tensor to -tensor."""
    rng = np.random.default_rng(201)
    tensor = _random_polar_tensor(rng)
    rotation = -np.eye(3, dtype=np.float64)
    transported = _transport_tensor(tensor, rotation)
    expected = -tensor
    assert np.allclose(transported, expected, atol=1e-9), (
        "polar tensor sign under inversion is incorrect"
    )


def test_polar_tensor_under_reflection():
    """Reflection must equal R \otimes R \otimes R with no extra det factor."""
    rng = np.random.default_rng(202)
    tensor = _random_polar_tensor(rng)
    rotation = np.diag([-1.0, 1.0, 1.0])

    transported = _transport_tensor(tensor, rotation)
    expected = _apply_rotation_matrix(tensor, rotation)
    assert np.allclose(transported, expected, atol=1e-9), (
        "reflection added an extra det factor"
    )


def test_pymatgen_symmop_oracle():
    """Our transport matches pymatgen SymmOp.transform_tensor for a rotoreflection."""
    rng = np.random.default_rng(203)
    tensor = _random_polar_tensor(rng)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    op = SymmOp.from_rotation_and_translation(rotation)

    transported = _transport_tensor(tensor, rotation)
    expected = op.transform_tensor(tensor)
    assert np.allclose(transported, expected, atol=1e-9), "pymatgen SymmOp oracle mismatch"


def test_polar_and_axial_transforms_are_separate():
    """The module must expose separate polar and axial rank-3 transforms."""
    from crosspiezo.analysis.o3_transport import transform_axial_rank3, transform_polar_rank3

    rng = np.random.default_rng(204)
    tensor = _random_polar_tensor(rng)
    rotation = np.diag([-1.0, 1.0, 1.0])
    polar = transform_polar_rank3(tensor, rotation)
    axial = transform_axial_rank3(tensor, rotation)
    assert not np.allclose(polar, axial, atol=1e-9), "polar and axial transforms coincide"
