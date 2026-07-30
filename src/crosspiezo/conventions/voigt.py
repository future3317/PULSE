"""Voigt/Cartesian conversions for third-rank piezoelectric tensors.

The internal Voigt order is xx, yy, zz, yz, xz, xy.  Engineering shear is
assumed: the off-diagonal Voigt components equal twice the tensor-shear
components.  Both source Voigt and converted Cartesian tensors are retained in
the ``TensorRecord``.
"""

from __future__ import annotations

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


def voigt_to_cartesian(voigt: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Convert a 3x6 Voigt piezoelectric tensor to 3x3x3 Cartesian.

    Parameters
    ----------
    voigt: ndarray of shape (3, 6)
        Voigt tensor in internal order.
    engineering_shear: bool
        If True, off-diagonal Voigt entries are engineering shears
        (2 * tensor shear).

    Returns
    -------
    cart: ndarray of shape (3, 3, 3)
        Cartesian tensor with last two indices symmetric.
    """
    voigt = np.asarray(voigt, dtype=np.float64)
    if voigt.shape != (3, 6):
        raise ValueError(f"Expected Voigt shape (3, 6), got {voigt.shape}")
    cart = np.zeros((3, 3, 3), dtype=np.float64)
    shear_factor = 0.5 if engineering_shear else 1.0
    for alpha, (j, k) in _VOIGT_TO_CART.items():
        factor = shear_factor if j != k else 1.0
        for i in range(3):
            cart[i, j, k] = factor * voigt[i, alpha]
            cart[i, k, j] = cart[i, j, k]
    return cart


def cartesian_to_voigt(cart: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Convert a 3x3x3 Cartesian piezoelectric tensor to 3x6 Voigt.

    Parameters
    ----------
    cart: ndarray of shape (3, 3, 3)
        Cartesian tensor with last two indices symmetric.
    engineering_shear: bool
        If True, off-diagonal Voigt entries are engineering shears
        (2 * tensor shear).

    Returns
    -------
    voigt: ndarray of shape (3, 6)
        Voigt tensor in internal order.
    """
    cart = np.asarray(cart, dtype=np.float64)
    if cart.shape != (3, 3, 3):
        raise ValueError(f"Expected Cartesian shape (3, 3, 3), got {cart.shape}")
    voigt = np.zeros((3, 6), dtype=np.float64)
    shear_factor = 2.0 if engineering_shear else 1.0
    for alpha, (j, k) in _VOIGT_TO_CART.items():
        for i in range(3):
            if j == k:
                voigt[i, alpha] = cart[i, j, k]
            else:
                voigt[i, alpha] = shear_factor * cart[i, j, k]
    return voigt


def round_trip_voigt(voigt: np.ndarray, engineering_shear: bool = True) -> np.ndarray:
    """Voigt -> Cartesian -> Voigt, for algebraic tests."""
    return cartesian_to_voigt(voigt_to_cartesian(voigt, engineering_shear), engineering_shear)
