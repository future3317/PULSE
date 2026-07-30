"""Red tests for C-08: orbit/point-group discrepancy must operate after exact-frame
transport.  Minimizing over the common point group on a raw right tensor that is
still in its own source frame is physically meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from crosspiezo.analysis.o3_transport import (
    exact_transported_discrepancy,
    point_group_equivalent_discrepancy,
    proper_orbit_discrepancy,
)


def _random_symmetric_tensor(rng: np.random.Generator) -> np.ndarray:
    cart = rng.normal(size=(3, 3, 3))
    return 0.5 * (cart + cart.transpose(0, 2, 1))


def _apply_rotation(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, tensor)


def test_proper_orbit_after_exact_transport():
    """After transporting right into left frame, proper-orbit min over point group
    should be invariant to the original relative orientation."""
    rng = np.random.default_rng(401)
    left = _random_symmetric_tensor(rng)
    # Arbitrary proper rotation not in the common point group.
    theta = 0.37
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    right = _apply_rotation(left, q)

    # The frame rotation that maps the right tensor back onto the left frame.
    q_inv = q.T
    exact = exact_transported_discrepancy(left, right, q_inv)
    assert exact["absolute"] < 1e-6, "exact transport discrepancy should be zero"

    # The proper-orbit routine must accept the frame rotation and transport first.
    orbit = proper_orbit_discrepancy(left, right, space_group_symbol=1, rotation=q_inv)
    assert orbit["absolute"] < 1e-6, (
        "proper-orbit discrepancy should vanish after exact-frame transport"
    )


def test_point_group_equivalent_after_exact_transport():
    """Point-group-equivalent discrepancy must also use the verified mapping first."""
    rng = np.random.default_rng(402)
    left = _random_symmetric_tensor(rng)
    theta = 0.22
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    right = _apply_rotation(left, q)

    q_inv = q.T
    equiv = point_group_equivalent_discrepancy(left, right, space_group_symbol=1, rotation=q_inv)
    assert equiv["absolute"] < 1e-6, (
        "point-group-equivalent discrepancy should vanish after exact-frame transport"
    )


def test_orbit_discrepancy_refuses_missing_rotation():
    """Without a verified mapping, componentwise orbit discrepancy is unresolved."""
    rng = np.random.default_rng(403)
    left = _random_symmetric_tensor(rng)
    right = _random_symmetric_tensor(rng)
    orbit = proper_orbit_discrepancy(left, right, space_group_symbol=1, rotation=None)
    assert np.isnan(orbit["absolute"])
