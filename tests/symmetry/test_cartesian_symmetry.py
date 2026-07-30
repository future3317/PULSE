"""Red tests for C-03: symmetry operations must come from the actual structure.

Using abstract fractional matrices from ``SpaceGroup`` ignores the real lattice
setting, orientation, and centering.  Cartesian point-group operations must be
obtained from ``SpacegroupAnalyzer(structure).get_point_group_operations(cartesian=True)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crosspiezo.conventions.symmetry import point_group_rotations, project_piezo_tensor


def _random_piezo_tensor(rng: np.random.Generator) -> np.ndarray:
    cart = rng.normal(size=(3, 3, 3))
    return 0.5 * (cart + cart.transpose(0, 2, 1))


def _rotated_orthorhombic_structure() -> Structure:
    """Build an orthorhombic Pnma-like structure in a non-standard orientation."""
    rng = np.random.default_rng(301)
    # Start with a conventional orthorhombic lattice.
    lattice = np.diag([4.0, 5.5, 7.2])
    # Rotate the lattice by a random proper rotation.
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    lattice_rot = rot @ lattice
    species = ["Ca", "Ti", "O", "O", "O"]
    # Primitive-ish fractional coordinates; SG may not be exactly Pnma, but the
    # point group from the analyzer must be used, not the conventional symbol.
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ])
    return Structure(lattice_rot, species, coords)


def test_rotations_are_orthogonal_and_proper_or_improper():
    """Cartesian rotations must satisfy R^T R = I and det = +/-1."""
    structure = _rotated_orthorhombic_structure()
    analyzer = SpacegroupAnalyzer(structure)
    rotations = analyzer.get_point_group_operations(cartesian=True)
    for r in rotations:
        rot = r.rotation_matrix
        assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(rot) - 1.0) < 1e-9 or abs(np.linalg.det(rot) + 1.0) < 1e-9


def test_symbol_based_rotations_are_rejected():
    """Abstract symbol/number inputs must be rejected because they cannot produce
    Cartesian rotations for the actual structure setting."""
    structure = _rotated_orthorhombic_structure()
    analyzer = SpacegroupAnalyzer(structure)
    sg_number = analyzer.get_space_group_number()

    with pytest.raises(TypeError):
        point_group_rotations(sg_number)


def test_point_group_rotations_accepts_structure():
    """The helper should accept a pymatgen Structure, not only an int/symbol."""
    structure = _rotated_orthorhombic_structure()
    rots = point_group_rotations(structure)
    assert len(rots) > 0
