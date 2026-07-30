"""Voigt/Cartesian conversions for piezoelectric and elastic tensors.

The internal Voigt order is xx, yy, zz, yz, xz, xy.  For third-rank
piezoelectric tensors the off-diagonal Voigt components are the physical tensor
components; no extra 0.5 or 2.0 factor appears because the engineering strain
convention is already encoded in the strain vector, not in the piezoelectric
coefficients.

Both source Voigt and converted Cartesian tensors are retained in the
``TensorRecord``.

Trusted-library cross-checks are provided via pymatgen's
``PiezoTensor.from_vasp_voigt`` for independent source reconstruction.
"""

from __future__ import annotations

import warnings

import numpy as np

_INTERNAL_VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"]

# Map Voigt index -> (j,k) Cartesian strain indices.
_VOIGT_TO_CART = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (1, 2),
    4: (0, 2),
    5: (0, 1),
}


def _voigt_to_cart_direct(voigt: np.ndarray) -> np.ndarray:
    """Direct 3x6 Voigt -> 3x3x3 Cartesian mapping for piezoelectric tensors."""
    voigt = np.asarray(voigt, dtype=np.float64)
    if voigt.shape != (3, 6):
        raise ValueError(f"Expected Voigt shape (3, 6), got {voigt.shape}")
    cart = np.zeros((3, 3, 3), dtype=np.float64)
    for alpha, (j, k) in _VOIGT_TO_CART.items():
        for i in range(3):
            val = voigt[i, alpha]
            cart[i, j, k] = val
            cart[i, k, j] = val
    return cart


def _cart_to_voigt_direct(cart: np.ndarray) -> np.ndarray:
    """Direct 3x3x3 Cartesian -> 3x6 Voigt mapping for piezoelectric tensors."""
    cart = np.asarray(cart, dtype=np.float64)
    if cart.shape != (3, 3, 3):
        raise ValueError(f"Expected Cartesian shape (3, 3, 3), got {cart.shape}")
    voigt = np.zeros((3, 6), dtype=np.float64)
    for alpha, (j, k) in _VOIGT_TO_CART.items():
        for i in range(3):
            voigt[i, alpha] = cart[i, j, k]
    return voigt


def piezo_stress_voigt_to_cartesian(voigt: np.ndarray) -> np.ndarray:
    """Convert a 3x6 Voigt piezoelectric stress tensor ``e`` to 3x3x3 Cartesian.

    The conversion is direct for all six Voigt components because ``e`` is the
    work-conjugate coefficient of the engineering strain vector.
    """
    return _voigt_to_cart_direct(voigt)


def piezo_stress_cartesian_to_voigt(cart: np.ndarray) -> np.ndarray:
    """Convert a 3x3x3 Cartesian piezoelectric stress tensor ``e`` to 3x6 Voigt."""
    return _cart_to_voigt_direct(cart)


def piezo_strain_voigt_to_cartesian(voigt: np.ndarray) -> np.ndarray:
    """Convert a 3x6 Voigt piezoelectric strain tensor ``d`` to 3x3x3 Cartesian.

    The same direct mapping applies as for ``e``; converting between ``d`` and
    ``e`` requires the elastic stiffness/compliance and is handled elsewhere.
    """
    return _voigt_to_cart_direct(voigt)


def piezo_strain_cartesian_to_voigt(cart: np.ndarray) -> np.ndarray:
    """Convert a 3x3x3 Cartesian piezoelectric strain tensor ``d`` to 3x6 Voigt."""
    return _cart_to_voigt_direct(cart)


def voigt_to_cartesian(voigt: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Deprecated generic wrapper; use the piezo-specific converters instead.

    For historical compatibility this function now performs the direct
    piezoelectric mapping regardless of ``engineering_shear``.  The previous
    0.5/2.0 scaling was inconsistent with the work-conjugacy identity for
    piezoelectric stress/strain tensors.
    """
    if not engineering_shear:
        warnings.warn(
            "engineering_shear=False is no longer meaningful for piezoelectric tensors; "
            "the mapping is direct for both stress and strain Voigt forms.",
            stacklevel=2,
        )
    return _voigt_to_cart_direct(voigt)


def cartesian_to_voigt(cart: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Deprecated generic wrapper; use the piezo-specific converters instead."""
    if not engineering_shear:
        warnings.warn(
            "engineering_shear=False is no longer meaningful for piezoelectric tensors; "
            "the mapping is direct for both stress and strain Voigt forms.",
            stacklevel=2,
        )
    return _cart_to_voigt_direct(cart)


def round_trip_voigt(voigt: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Voigt -> Cartesian -> Voigt, for algebraic tests."""
    return cartesian_to_voigt(voigt_to_cartesian(voigt, engineering_shear), engineering_shear)


# -----------------------------------------------------------------------------
# Trusted-library oracles for independent source reconstruction
# -----------------------------------------------------------------------------

# VASP Voigt order used by pymatgen.analysis.piezo.PiezoTensor.
_VASP_VOIGT_ORDER = [0, 1, 2, 5, 3, 4]  # xx, yy, zz, xy, yz, zx


def _internal_voigt_to_vasp_voigt(internal_voigt: np.ndarray) -> np.ndarray:
    """Convert internal order [xx,yy,zz,yz,xz,xy] to VASP [xx,yy,zz,xy,yz,zx]."""
    internal_voigt = np.asarray(internal_voigt, dtype=np.float64)
    if internal_voigt.shape != (3, 6):
        raise ValueError(f"Expected internal Voigt shape (3, 6), got {internal_voigt.shape}")
    return internal_voigt[:, _VASP_VOIGT_ORDER]


def trusted_piezo_stress_voigt_to_cartesian(voigt: np.ndarray) -> np.ndarray:
    """Trusted Cartesian reconstruction from Voigt using pymatgen.

    This is independent of the project converter and should agree with
    ``piezo_stress_voigt_to_cartesian`` for physically valid input.
    """
    from pymatgen.analysis.piezo import PiezoTensor

    vasp = _internal_voigt_to_vasp_voigt(voigt)
    return np.asarray(PiezoTensor.from_vasp_voigt(vasp), dtype=np.float64)


def tensor_lineage_metrics(
    processed_voigt: np.ndarray,
    trusted_cartesian: np.ndarray,
    project_cartesian: np.ndarray,
    stored_cartesian: np.ndarray,
) -> dict[str, float]:
    """Compare processed-Voigt-derived Cartesian tensors to the stored field.

    All three Cartesian tensors are passed explicitly so that trusted and
    project converters can be audited independently.  No tensor is regenerated
    inside this function.
    """
    processed_voigt = np.asarray(processed_voigt, dtype=np.float64)
    trusted_cartesian = np.asarray(trusted_cartesian, dtype=np.float64)
    project_cartesian = np.asarray(project_cartesian, dtype=np.float64)
    stored_cartesian = np.asarray(stored_cartesian, dtype=np.float64)

    if processed_voigt.shape != (3, 6):
        raise ValueError(f"Expected Voigt shape (3, 6), got {processed_voigt.shape}")
    for arr, name in [(trusted_cartesian, "trusted"), (project_cartesian, "project"), (stored_cartesian, "stored")]:
        if arr.shape != (3, 3, 3):
            raise ValueError(f"Expected {name} Cartesian shape (3, 3, 3), got {arr.shape}")

    def _diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _shear_diff(a: np.ndarray, b: np.ndarray) -> float:
        a_voigt = _cart_to_voigt_direct(a)
        b_voigt = _cart_to_voigt_direct(b)
        return float(np.linalg.norm(a_voigt[:, 3:6] - b_voigt[:, 3:6]))

    denom = max(np.linalg.norm(stored_cartesian), 1e-12)

    return {
        "frobenius_diff_trusted_vs_project": _diff(trusted_cartesian, project_cartesian),
        "frobenius_diff_trusted_vs_stored": _diff(trusted_cartesian, stored_cartesian),
        "frobenius_diff_project_vs_stored": _diff(project_cartesian, stored_cartesian),
        "shear_diff_trusted_vs_stored": _shear_diff(trusted_cartesian, stored_cartesian),
        "shear_diff_project_vs_stored": _shear_diff(project_cartesian, stored_cartesian),
        "relative_diff_trusted_vs_stored": _diff(trusted_cartesian, stored_cartesian) / denom,
        "relative_diff_project_vs_stored": _diff(project_cartesian, stored_cartesian) / denom,
        "stored_norm": float(np.linalg.norm(stored_cartesian)),
        "trusted_norm": float(np.linalg.norm(trusted_cartesian)),
        "project_norm": float(np.linalg.norm(project_cartesian)),
    }
