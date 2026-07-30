"""Point-group symmetry projection for third-rank polar tensors."""

from __future__ import annotations

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup


def point_group_rotations(space_group_symbol: str | int) -> list[np.ndarray]:
    """Return the pure rotation matrices for a space- or point-group symbol.

    Inversion and improper rotations are kept as provided by pymatgen; for
    piezoelectric tensors the direct product of rotation matrices already gives
    the correct sign because a polar third-rank tensor is odd under inversion.
    """
    if isinstance(space_group_symbol, int):
        sg = SpaceGroup.from_int_number(space_group_symbol)
    else:
        try:
            sg = SpaceGroup(space_group_symbol)
        except ValueError:
            # Try interpreting a numeric string as an international number.
            sg = SpaceGroup.from_int_number(int(space_group_symbol))
    rotations: list[np.ndarray] = []
    for op in sg.symmetry_ops:
        rot = np.asarray(op.rotation_matrix, dtype=np.float64)
        rotations.append(rot)
    return rotations


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
        # e'_{ijk} = R_il R_jm R_kn e_lmn
        projected += np.einsum("il,jm,kn,lmn->ijk", r, r, r, tensor)
    projected /= len(rotations)
    # Enforce exact last-index symmetry (numerical noise)
    projected = 0.5 * (projected + projected.transpose(0, 2, 1))
    return np.asarray(projected, dtype=np.float64)


def symmetry_residual(tensor: np.ndarray, rotations: list[np.ndarray]) -> float:
    """Frobenius norm of (tensor - projected tensor)."""
    projected = project_piezo_tensor(tensor, rotations)
    return float(np.linalg.norm(tensor - projected))


def allowed_components_mask(space_group_symbol: str | int) -> np.ndarray:
    """Return a 3x3x3 boolean mask of symmetry-allowed Cartesian components."""
    rotations = point_group_rotations(space_group_symbol)
    mask = np.zeros((3, 3, 3), dtype=bool)
    # A component is allowed if perturbing it yields a non-zero projection.
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
