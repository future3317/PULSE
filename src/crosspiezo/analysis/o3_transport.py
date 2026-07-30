"""O(3) tensor transport and domain-aware discrepancy variants.

This module implements the discrepancy metrics required by Phase 5A:

* exact transported discrepancy using the structure-match rotation;
* proper-orbit discrepancy restricted to source point-group proper rotations;
* domain-aware discrepancy that detects and flags inversion-related polar domains;
* point-group-equivalent discrepancy minimized over the common point group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from crosspiezo.analysis.discrepancy import absolute_discrepancy, normalized_discrepancy
from crosspiezo.conventions.symmetry import point_group_rotations, project_piezo_tensor


def _transport_tensor(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transport a polar third-rank tensor by a proper O(3) rotation.

    For an improper rotation (det = -1) the tensor additionally picks up a
    minus sign because it is polar (odd under inversion).  The caller is
    responsible for deciding whether an improper transformation is physically
    admissible for a given pair.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    transported = np.asarray(np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, tensor), dtype=np.float64)
    if np.linalg.det(rotation) < 0:
        transported = -transported
    return transported


def _polar_domain_tensor(tensor: np.ndarray) -> np.ndarray:
    """Return the inversion-related polar-domain partner of a tensor."""
    return -np.asarray(tensor, dtype=np.float64)


def _nearest_proper_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project an O(3) matrix to the nearest proper rotation (det = +1)."""
    u, _, vh = svd(rotation)
    proper = np.asarray(u @ vh, dtype=np.float64)
    if np.linalg.det(proper) < 0:
        u[:, -1] *= -1
        proper = np.asarray(u @ vh, dtype=np.float64)
    return proper


@dataclass(frozen=True)
class DiscrepancyVariant:
    """A single discrepancy variant for a matched pair."""

    variant_name: str
    absolute: float
    normalized: float
    sign_flip_fraction: float
    cosine_similarity: float
    amplitude_ratio: float
    n_pairs: int


def exact_transported_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    rotation: np.ndarray | None,
) -> dict[str, float]:
    """Discrepancy after transporting ``right`` into ``left`` frame.

    The structure matcher returns a proper rotation (det = +1) by construction.
    If it is None, the raw tensor discrepancy is returned with a warning flag.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if rotation is None:
        return {
            "absolute": absolute_discrepancy(left, right),
            "normalized": normalized_discrepancy(left, right),
            "sign_flip_fraction": _sign_flip(left, right),
            "cosine_similarity": _cosine(left, right),
            "amplitude_ratio": _amplitude_ratio(left, right),
            "rotation_available": 0.0,
        }
    rot = np.asarray(rotation, dtype=np.float64)
    right_transported = _transport_tensor(right, rot)
    return {
        "absolute": absolute_discrepancy(left, right_transported),
        "normalized": normalized_discrepancy(left, right_transported),
        "sign_flip_fraction": _sign_flip(left, right_transported),
        "cosine_similarity": _cosine(left, right_transported),
        "amplitude_ratio": _amplitude_ratio(left, right_transported),
        "rotation_available": 1.0,
    }


def proper_orbit_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    space_group_symbol: str | int,
) -> dict[str, float]:
    """Minimum discrepancy over proper rotations in the source point group.

    Only the proper rotations of the common space group are allowed.  This is a
    sanity check that the reported disagreement is not an artifact of an
    arbitrary Cartesian frame choice.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    try:
        rots = point_group_rotations(space_group_symbol)
    except Exception:  # noqa: BLE001
        return _nan_result()
    proper_rots = [r for r in rots if np.linalg.det(r) > 0]
    if not proper_rots:
        return _nan_result()
    best = None
    for r in proper_rots:
        cand = _transport_tensor(right, r)
        d = absolute_discrepancy(left, cand)
        if best is None or d < best["absolute"]:
            best = {
                "absolute": d,
                "normalized": normalized_discrepancy(left, cand),
                "sign_flip_fraction": _sign_flip(left, cand),
                "cosine_similarity": _cosine(left, cand),
                "amplitude_ratio": _amplitude_ratio(left, cand),
                "rotation_available": 1.0,
            }
    return best if best is not None else _nan_result()


def domain_aware_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    rotation: np.ndarray | None,
) -> dict[str, float]:
    """Discrepancy that treats inversion-related polar domains as equivalent.

    Returns the *domain-equivalent* discrepancy (the smaller of the signed and
    flipped discrepancies) and flags whether a flip was selected.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if rotation is None:
        right_t = right
    else:
        right_t = _transport_tensor(right, np.asarray(rotation, dtype=np.float64))
    signed = absolute_discrepancy(left, right_t)
    flipped = absolute_discrepancy(left, _polar_domain_tensor(right_t))
    use_flip = flipped < signed
    chosen = _polar_domain_tensor(right_t) if use_flip else right_t
    return {
        "absolute": float(min(signed, flipped)),
        "normalized": normalized_discrepancy(left, chosen),
        "sign_flip_fraction": _sign_flip(left, chosen),
        "cosine_similarity": _cosine(left, chosen),
        "amplitude_ratio": _amplitude_ratio(left, chosen),
        "polar_domain_flip": 1.0 if use_flip else 0.0,
        "rotation_available": 0.0 if rotation is None else 1.0,
    }


def point_group_equivalent_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    space_group_symbol: str | int,
) -> dict[str, float]:
    """Minimum discrepancy over the full common point group (proper + improper).

    This is the strongest symmetry-based alignment: it minimizes over all
    crystallographically equivalent settings of the common matched structure.
    It must not be confused with an unconstrained best rotation.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    try:
        rots = point_group_rotations(space_group_symbol)
    except Exception:  # noqa: BLE001
        return _nan_result()
    if not rots:
        return _nan_result()
    best = None
    for r in rots:
        cand = _transport_tensor(right, r)
        d = absolute_discrepancy(left, cand)
        if best is None or d < best["absolute"]:
            best = {
                "absolute": d,
                "normalized": normalized_discrepancy(left, cand),
                "sign_flip_fraction": _sign_flip(left, cand),
                "cosine_similarity": _cosine(left, cand),
                "amplitude_ratio": _amplitude_ratio(left, cand),
                "rotation_available": 1.0,
            }
    return best if best is not None else _nan_result()


def symmetry_projected_discrepancy(
    left: np.ndarray,
    right: np.ndarray,
    space_group_symbol: str | int,
    rotation: np.ndarray | None = None,
) -> dict[str, float]:
    """Discrepancy after Reynolds projection onto the common point group.

    Both tensors are transported to the same frame (when a rotation is supplied)
    and then projected.  This tests whether disagreement survives removal of
    symmetry-forbidden components.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if rotation is not None:
        right = _transport_tensor(right, np.asarray(rotation, dtype=np.float64))
    try:
        rots = point_group_rotations(space_group_symbol)
        left_proj = project_piezo_tensor(left, rots)
        right_proj = project_piezo_tensor(right, rots)
    except Exception:  # noqa: BLE001
        return _nan_result()
    return {
        "absolute": absolute_discrepancy(left_proj, right_proj),
        "normalized": normalized_discrepancy(left_proj, right_proj),
        "sign_flip_fraction": _sign_flip(left_proj, right_proj),
        "cosine_similarity": _cosine(left_proj, right_proj),
        "amplitude_ratio": _amplitude_ratio(left_proj, right_proj),
        "rotation_available": 0.0 if rotation is None else 1.0,
    }


def _sign_flip(left: np.ndarray, right: np.ndarray, threshold: float = 1e-6) -> float:
    mask = (np.abs(left) > threshold) | (np.abs(right) > threshold)
    if not np.any(mask):
        return 0.0
    return float(np.mean((left[mask] * right[mask]) < 0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    lf = left.ravel()
    rf = right.ravel()
    denom = np.linalg.norm(lf) * np.linalg.norm(rf) + 1e-12
    return float(np.dot(lf, rf) / denom)


def _amplitude_ratio(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) + 1e-12
    return float(np.linalg.norm(right) / denom)


def _nan_result() -> dict[str, float]:
    return {
        "absolute": float("nan"),
        "normalized": float("nan"),
        "sign_flip_fraction": float("nan"),
        "cosine_similarity": float("nan"),
        "amplitude_ratio": float("nan"),
        "rotation_available": 0.0,
    }
