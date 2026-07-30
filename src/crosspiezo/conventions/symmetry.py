"""Point-group symmetry projection for third-rank polar tensors.

All Cartesian symmetry operations are derived from an actual pymatgen
``Structure`` using ``SpacegroupAnalyzer.get_point_group_operations(cartesian=True)``.
Using an abstract space-group symbol or fractional matrices is no longer supported
because they ignore the real lattice setting and orientation.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def structure_point_group_rotations(
    structure: Structure,
    symprec: float = 0.01,
    angle_tolerance: float = 5.0,
) -> list[np.ndarray]:
    """Return the Cartesian point-group rotation matrices for a structure.

    Uses ``SpacegroupAnalyzer(...).get_point_group_operations(cartesian=True)``,
    which returns operations in the Cartesian basis of the input structure.
    No custom lattice conjugation or SVD orthogonalization is applied.

    Parameters
    ----------
    structure:
        The pymatgen ``Structure`` whose actual setting is used.
    symprec, angle_tolerance:
        Spglib precision parameters passed to ``SpacegroupAnalyzer``.

    Returns
    -------
    rotations:
        Unique 3x3 orthogonal matrices (det = +/-1) in Cartesian coordinates.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    operations = analyzer.get_point_group_operations(cartesian=True)
    if not operations:
        return [np.eye(3, dtype=np.float64)]

    seen: list[np.ndarray] = []
    for op in operations:
        rot = np.asarray(op.rotation_matrix, dtype=np.float64)
        det = float(np.linalg.det(rot))
        if not (abs(det - 1.0) < 1e-5 or abs(det + 1.0) < 1e-5):
            continue
        if not any(np.allclose(rot, existing, atol=1e-6) for existing in seen):
            seen.append(rot)
    return seen if seen else [np.eye(3, dtype=np.float64)]


def point_group_rotations(structure: Structure) -> list[np.ndarray]:
    """Return Cartesian point-group rotations for a ``Structure``.

    .. deprecated::
        The old int/symbol path used fractional matrices and has been removed.
        Pass a ``pymatgen.core.structure.Structure``.
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            "point_group_rotations now requires a pymatgen Structure; "
            "abstract space-group symbols cannot yield Cartesian rotations."
        )
    return structure_point_group_rotations(structure)


def project_piezo_tensor(tensor: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray:
    """Average a 3x3x3 piezoelectric tensor over the point-group rotations.

    The projection enforces

        e'_ijk = (1/|G|) sum_R R_il R_jm R_kn e_lmn.

    Returns
    -------
    projected: ndarray of shape (3, 3, 3)
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.shape != (3, 3, 3):
        raise ValueError(f"Expected tensor shape (3, 3, 3), got {tensor.shape}")
    if not rotations:
        return tensor.copy()
    projected = np.zeros_like(tensor)
    for r in rotations:
        projected += transform_polar_rank3(tensor, r)
    projected /= len(rotations)
    # Enforce exact last-index symmetry (numerical noise)
    projected = 0.5 * (projected + projected.transpose(0, 2, 1))
    return np.asarray(projected, dtype=np.float64)


def transform_polar_rank3(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Standard polar rank-3 transform (duplicated here to avoid import cycle)."""
    return np.asarray(
        np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, tensor),
        dtype=np.float64,
    )


def symmetry_residual(tensor: np.ndarray, rotations: list[np.ndarray]) -> float:
    """Frobenius norm of (tensor - projected tensor)."""
    projected = project_piezo_tensor(tensor, rotations)
    return float(np.linalg.norm(tensor - projected))


def allowed_components_mask(structure: Structure) -> np.ndarray:
    """Return a 3x3x3 boolean mask of symmetry-allowed Cartesian components."""
    rotations = point_group_rotations(structure)
    mask = np.zeros((3, 3, 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                probe = np.zeros((3, 3, 3), dtype=np.float64)
                probe[i, j, k] = 1.0
                probe[i, k, j] = 1.0
                proj = project_piezo_tensor(probe, rotations)
                mask[i, j, k] = np.linalg.norm(proj) > 1e-8
    return mask


def structure_point_group(structure: Structure) -> str:
    """Return the point-group symbol for a pymatgen Structure."""
    info = structure.get_space_group_info()
    return str(info[1]) if info else "unknown"
