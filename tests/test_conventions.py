"""Algebraic tests for tensor conventions."""

from __future__ import annotations

import numpy as np
from pymatgen.core.structure import Structure

from crosspiezo.conventions.symmetry import (
    point_group_rotations,
    project_piezo_tensor,
    symmetry_residual,
)
from crosspiezo.conventions.voigt import (
    round_trip_voigt,
    voigt_to_cartesian,
)


def _cubic_structure() -> Structure:
    return Structure(np.eye(3) * 3.0, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def test_voigt_cartesian_round_trip():
    rng = np.random.default_rng(0)
    voigt = rng.normal(size=(3, 6))
    recovered = round_trip_voigt(voigt)
    assert np.allclose(recovered, voigt)


def test_cartesian_symmetry_last_two_indices():
    rng = np.random.default_rng(1)
    voigt = rng.normal(size=(3, 6))
    cart = voigt_to_cartesian(voigt)
    assert np.allclose(cart, cart.transpose(0, 2, 1))


def test_rotation_covariance():
    rng = np.random.default_rng(2)
    voigt = rng.normal(size=(3, 6))
    cart = voigt_to_cartesian(voigt)
    theta = 0.4
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    rotated = np.einsum("il,jm,kn,lmn->ijk", rot, rot, rot, cart)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(cart))


def test_point_group_projection_idempotent():
    rng = np.random.default_rng(3)
    cart = rng.normal(size=(3, 3, 3))
    cart = 0.5 * (cart + cart.transpose(0, 2, 1))
    rots = point_group_rotations(_cubic_structure())
    proj1 = project_piezo_tensor(cart, rots)
    proj2 = project_piezo_tensor(proj1, rots)
    assert np.allclose(proj1, proj2)


def test_symmetry_residual_nonnegative():
    rng = np.random.default_rng(4)
    cart = rng.normal(size=(3, 3, 3))
    rots = point_group_rotations(_cubic_structure())
    assert symmetry_residual(cart, rots) >= -1e-12
