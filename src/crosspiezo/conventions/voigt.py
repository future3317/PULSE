"""Voigt/Cartesian conversions for piezoelectric and elastic tensors.

The internal Voigt order is xx, yy, zz, yz, xz, xy.  For third-rank
piezoelectric tensors the off-diagonal Voigt components are the physical tensor
components; no extra 0.5 or 2.0 factor appears because the engineering strain
convention is already encoded in the strain vector, not in the piezoelectric
coefficients.

Both source Voigt and converted Cartesian tensors are retained in the
``TensorRecord``.
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
