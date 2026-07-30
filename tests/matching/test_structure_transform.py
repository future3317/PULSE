"""Red tests for C-05: structure-match rotation must separate basis change from
physical rotation, and C-06: RMS/max distance order from StructureMatcher.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core.structure import Structure

from crosspiezo.matching.structure_matcher import match_structures


def _simple_cubic_structure(a: float = 3.0, species: str = "NaCl") -> Structure:
    """A 2-atom cubic structure."""
    lattice = np.eye(3) * a
    if species == "NaCl":
        return Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return Structure(lattice, ["Si"], [[0.0, 0.0, 0.0]])


def _cif(struct: Structure) -> str:
    return struct.to(fmt="cif")


def test_unimodular_basis_change_gives_identity_cartesian_rotation():
    """Swapping cell axes is a relabeling, not a physical rotation; Q must be I."""
    s1 = _simple_cubic_structure()
    # Swap a and b axes; update fractional coords accordingly.
    lattice2 = np.array([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    s2 = Structure(lattice2, s1.species, s1.frac_coords)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit, "basis-relabeled structures should match"
    assert result.cartesian_rotation is not None
    rotation = np.asarray(result.cartesian_rotation)
    assert np.allclose(rotation, np.eye(3), atol=1e-6), (
        f"basis change returned physical rotation {rotation}"
    )


def test_rigid_rotation_recovery():
    """A true rigid rotation of the Cartesian coordinates must be recovered."""
    s1 = _simple_cubic_structure()
    theta = 0.4
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    new_lattice = q @ s1.lattice.matrix
    new_frac = s1.frac_coords @ np.linalg.inv(q).T
    s2 = Structure(new_lattice, s1.species, new_frac)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    assert result.cartesian_rotation is not None
    recovered = np.asarray(result.cartesian_rotation)
    assert np.allclose(recovered, q, atol=1e-6), f"recovered rotation {recovered} != {q}"


def test_improper_transformation_retains_negative_determinant():
    """A mirror/reflection relation must not be forced to det = +1."""
    s1 = _simple_cubic_structure()
    q = np.diag([-1.0, 1.0, 1.0])
    new_lattice = q @ s1.lattice.matrix
    new_frac = s1.frac_coords @ np.linalg.inv(q).T
    s2 = Structure(new_lattice, s1.species, new_frac)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    assert result.cartesian_rotation is not None
    det = float(np.linalg.det(np.asarray(result.cartesian_rotation)))
    assert det < 0, f"improper transformation was forced to det={det}"


def test_rms_distance_is_less_than_max_distance():
    """StructureMatcher.get_rms_dist returns (rms, max); fields must match."""
    s1 = _simple_cubic_structure()
    # Displace one atom by 0.4 Å.
    frac = s1.frac_coords.copy()
    frac[0, 0] += 0.4 / 3.0
    s2 = Structure(s1.lattice.matrix, s1.species, frac)

    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    assert result.rms_distance is not None and result.max_distance is not None
    assert result.rms_distance <= result.max_distance + 1e-6, (
        f"rms_distance {result.rms_distance} > max_distance {result.max_distance}"
    )


def test_match_result_includes_unimodular_transform_and_translation():
    """The match record must expose the integer basis matrix and translation."""
    s1 = _simple_cubic_structure()
    s2 = _simple_cubic_structure()
    result = match_structures("A", "B", _cif(s1), _cif(s2))
    assert result.fit
    record = __import__("crosspiezo.matching.structure_matcher", fromlist=["to_match_record"]).to_match_record(result)
    assert record.unimodular_cell_transform is not None
    assert len(record.pass_fail_reasons) == 0 or "translation" in str(record.pass_fail_reasons)
