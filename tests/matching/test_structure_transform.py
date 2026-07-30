"""Red tests for C-05: structure-match rotation must separate basis change from
physical rotation, and C-06: RMS/max distance order from StructureMatcher.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core.structure import Structure

from crosspiezo.matching.structure_matcher import match_structures


def _test_structure() -> Structure:
    """A low-symmetry triclinic structure for robust rotation tests.

    The lattice is kept nearly orthogonal so that a physical rigid rotation
    stays within the frozen angle tolerance (5°) while still being triclinic.
    """
    lattice = np.array([[2.5, 0.0, 0.0], [0.1, 3.0, 0.0], [0.1, 0.1, 4.0]])
    return Structure(
        lattice,
        ["Na", "Cl", "K", "F"],
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.3, 0.2],
            [0.7, 0.1, 0.6],
            [0.2, 0.8, 0.4],
        ],
    )


def _cif(struct: Structure) -> Structure:
    """Return the structure directly; serializing to CIF discards orientation."""
    return struct


def test_unimodular_basis_change_gives_identity_cartesian_rotation():
    """A pure basis relabeling is not a physical rotation; Q must be I."""
    s1 = _test_structure()
    # Swap a and b lattice vectors and transform fractional coords so that
    # Cartesian coordinates stay unchanged.
    m = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    lattice2 = m @ s1.lattice.matrix
    frac2 = s1.frac_coords @ m
    s2 = Structure(lattice2, s1.species, frac2)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit, "basis-relabeled structures should match"
    assert result.cartesian_rotation is not None
    rotation = np.asarray(result.cartesian_rotation)
    assert np.allclose(rotation, np.eye(3), atol=1e-6), (
        f"basis change returned physical rotation {rotation}"
    )


def test_rigid_rotation_recovery():
    """A true rigid rotation of the Cartesian coordinates must be recovered."""
    s1 = _test_structure()
    theta = 0.1  # small enough to stay within the frozen 5° angle tolerance
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # Rotate both lattice and Cartesian positions; fractional coords unchanged.
    new_lattice = q @ s1.lattice.matrix
    s2 = Structure(new_lattice, s1.species, s1.frac_coords)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    assert result.cartesian_rotation is not None
    recovered = np.asarray(result.cartesian_rotation)
    # Finite-atom Kabsch recovers the rotation to within a small tolerance.
    assert np.linalg.norm(recovered - q) < 5e-2, f"recovered rotation {recovered} != {q}"
    assert result.rotation_class == "proper_rotation"
    assert result.kabsch_rms is not None and result.kabsch_rms < 0.1


def test_improper_transformation_retains_negative_determinant():
    """A mirror/reflection relation must not be forced to det = +1."""
    from crosspiezo.matching.structure_matcher import _kabsch_rotation

    # Explicit mirror of a finite point set.
    left = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.1],
        [0.1, 1.2, 0.3],
        [0.2, 0.1, 1.5],
    ])
    q = np.diag([-1.0, 1.0, 1.0])
    right = left @ q.T

    rot_proper, rot_improper, rms_proper, rms_improper = _kabsch_rotation(left, right)
    assert rms_improper < rms_proper - 1e-6, "Kabsch should prefer improper solution for a mirror"
    det = float(np.linalg.det(rot_improper))
    assert det < 0, f"improper transformation was forced to det={det}"


def test_rms_distance_is_less_than_max_distance():
    """StructureMatcher.get_rms_dist returns (rms, max); fields must match."""
    s1 = _test_structure()
    # Displace one atom slightly.
    frac = s1.frac_coords.copy()
    frac[0, 0] += 0.02
    s2 = Structure(s1.lattice.matrix, s1.species, frac)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    assert result.rms_distance is not None and result.max_distance is not None
    assert result.rms_distance <= result.max_distance + 1e-6, (
        f"rms_distance {result.rms_distance} > max_distance {result.max_distance}"
    )


def test_match_result_includes_unimodular_transform_and_translation():
    """The match record must expose the integer basis matrix and translation."""
    s1 = _test_structure()
    s2 = _test_structure()
    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    record = __import__("crosspiezo.matching.structure_matcher", fromlist=["to_match_record"]).to_match_record(result)
    assert record.unimodular_cell_transform is not None
    assert record.fractional_translation is not None
