"""Red tests for C-09: longitudinal/shear ranking functionals must be rotation
invariant and physically defined, not just coordinate-axis components.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crosspiezo.analysis.ranking import (
    frobenius_norm_score,
    max_longitudinal_response,
    max_shear_response,
)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    return Rotation.from_rotvec(rng.normal(size=3)).as_matrix()


def _apply_rotation(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, tensor)


def _brute_force_longitudinal(tensor: np.ndarray, n_points: int = 4000) -> float:
    """max_{||n||=1} | n_i e_ijk n_j n_k | via uniform sphere sampling."""
    rng = np.random.default_rng(901)
    dirs = rng.normal(size=(n_points, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    vals = np.abs(np.einsum("ni,ijk,nj,nk->n", dirs, tensor, dirs, dirs))
    return float(np.max(vals))


def test_frobenius_norm_is_rotation_invariant():
    rng = np.random.default_rng(501)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    assert np.isclose(frobenius_norm_score(tensor), frobenius_norm_score(rotated), atol=1e-6)


def test_max_longitudinal_is_rotation_invariant():
    rng = np.random.default_rng(502)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    assert np.isclose(
        max_longitudinal_response(tensor),
        max_longitudinal_response(rotated),
        atol=1e-6,
    ), "max_longitudinal_response is not rotation invariant"


def test_max_shear_is_rotation_invariant():
    rng = np.random.default_rng(503)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    assert np.isclose(
        max_shear_response(tensor),
        max_shear_response(rotated),
        atol=1e-6,
    ), "max_shear_response is not rotation invariant"


def test_longitudinal_matches_brute_force_oracle():
    """A tensor with a pure shear-like response must have non-zero longitudinal."""
    tensor = np.zeros((3, 3, 3), dtype=np.float64)
    tensor[0, 1, 2] = 1.0
    tensor[0, 2, 1] = 1.0
    expected = _brute_force_longitudinal(tensor)
    assert expected > 0.1, "brute-force longitudinal oracle is unexpectedly zero"
    ours = max_longitudinal_response(tensor)
    assert np.isclose(ours, expected, atol=1e-2), (
        f"longitudinal functional {ours} != brute-force oracle {expected}"
    )
