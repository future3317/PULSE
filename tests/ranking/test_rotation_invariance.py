"""Red tests for C-09: longitudinal/shear ranking functionals must be rotation
invariant and physically defined, not just coordinate-axis components.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crosspiezo.analysis.ranking import (
    frobenius_norm_score,
    kelvin_operator_norm,
    max_longitudinal_modulus,
    max_longitudinal_response,
    max_shear_response,
    mp_reported_svd_scalar,
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


def test_max_shear_is_withdrawn():
    """The old shear functional was coordinate-axis dependent and is removed."""
    rng = np.random.default_rng(503)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    with pytest.raises(NotImplementedError):
        max_shear_response(tensor)


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


def test_max_longitudinal_modulus_is_rotation_invariant():
    """The deterministic modulus must be invariant under physical rotations."""
    rng = np.random.default_rng(601)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    assert np.isclose(
        max_longitudinal_modulus(tensor),
        max_longitudinal_modulus(rotated),
        atol=1e-6,
    ), "max_longitudinal_modulus is not rotation invariant"


def test_max_longitudinal_modulus_is_deterministic():
    """Same input must yield exactly the same scalar."""
    rng = np.random.default_rng(602)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    a = max_longitudinal_modulus(tensor)
    b = max_longitudinal_modulus(tensor)
    assert a == b, "max_longitudinal_modulus is not deterministic"


def test_max_longitudinal_modulus_analytic_oracle():
    """For a pure e_111 tensor the maximum longitudinal response is |e_111|."""
    tensor = np.zeros((3, 3, 3), dtype=np.float64)
    tensor[0, 0, 0] = 5.0
    assert np.isclose(max_longitudinal_modulus(tensor), 5.0, atol=1e-9)


def test_max_longitudinal_modulus_matches_dense_oracle():
    """The deterministic optimiser must agree with a dense brute-force grid."""
    rng = np.random.default_rng(603)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    ours = max_longitudinal_modulus(tensor, grid_N=5000, n_starts=10)
    expected = _brute_force_longitudinal(tensor, n_points=100000)
    assert np.isclose(ours, expected, rtol=1e-3), (
        f"deterministic modulus {ours} != dense oracle {expected}"
    )


def test_kelvin_operator_norm_is_rotation_invariant():
    """F4 Kelvin/Mandel operator norm must be a coordinate invariant."""
    rng = np.random.default_rng(604)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    assert np.isclose(
        kelvin_operator_norm(tensor),
        kelvin_operator_norm(rotated),
        atol=1e-6,
    ), "kelvin_operator_norm is not rotation invariant"


def test_kelvin_operator_norm_positive_and_bounded():
    """For a non-zero tensor the Kelvin norm is positive and equals its SVD."""
    tensor = np.zeros((3, 3, 3), dtype=np.float64)
    tensor[0, 0, 0] = 3.0
    assert kelvin_operator_norm(tensor) == pytest.approx(3.0, abs=1e-9)


def test_mp_reported_svd_scalar_is_plain_svd():
    """The MP-reported scalar reproduces the largest singular value."""
    rng = np.random.default_rng(605)
    voigt = rng.normal(size=(3, 6))
    expected = np.linalg.svd(voigt, compute_uv=False).max()
    assert np.isclose(mp_reported_svd_scalar(voigt), expected, atol=1e-9)


def test_mp_reported_svd_scalar_is_not_cartesian_rotation_invariant():
    """Plain 3x6 SVD is a property of the Voigt matrix, not the Cartesian tensor.

    Rotating the Cartesian tensor and re-expressing it in Voigt form can change
    the singular values.  This test documents that limitation; the scalar is
    therefore only usable as a source-native field, not as a cross-source
    physical invariant.
    """
    rng = np.random.default_rng(606)
    tensor = rng.normal(size=(3, 3, 3))
    tensor = 0.5 * (tensor + tensor.transpose(0, 2, 1))
    q = _random_rotation_matrix(rng)
    rotated = _apply_rotation(tensor, q)
    from crosspiezo.conventions.voigt import piezo_stress_cartesian_to_voigt

    v1 = mp_reported_svd_scalar(piezo_stress_cartesian_to_voigt(tensor))
    v2 = mp_reported_svd_scalar(piezo_stress_cartesian_to_voigt(rotated))
    # Plain SVD of the Voigt matrix is NOT invariant under arbitrary rotations.
    assert not np.isclose(v1, v2, atol=1e-3), (
        "mp_reported_svd_scalar unexpectedly invariant; re-check test tensor"
    )
