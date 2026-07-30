"""Red tests for C-01: piezoelectric Voigt/Cartesian shear conversion.

The old converter uses a single ``engineering_shear`` flag and scales shear
components by 0.5 / 2.0.  For a piezoelectric stress tensor that is conjugate to
the engineering strain, the off-diagonal Voigt components must equal the
tensor-shear Cartesian components, not half of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.analysis.piezo import PiezoTensor

from crosspiezo.conventions.voigt import (
    cartesian_to_voigt,
    piezo_stress_voigt_to_cartesian,
    trusted_piezo_stress_voigt_to_cartesian,
    voigt_to_cartesian,
)


def _random_symmetric_piezo_tensor(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3x3x3 tensor with symmetry in the last two indices."""
    cart = rng.normal(size=(3, 3, 3))
    cart = 0.5 * (cart + cart.transpose(0, 2, 1))
    return cart


def _engineering_strain_from_tensor(eps: np.ndarray) -> np.ndarray:
    """Voigt engineering strain [xx, yy, zz, 2*yz, 2*xz, 2*xy]."""
    return np.array([
        eps[0, 0],
        eps[1, 1],
        eps[2, 2],
        2.0 * eps[1, 2],
        2.0 * eps[0, 2],
        2.0 * eps[0, 1],
    ])


def test_work_conjugacy_identity():
    """e_voigt · eta_engineering == sum_ijk e_cart[i,j,k] eps[j,k]."""
    rng = np.random.default_rng(101)
    cart = _random_symmetric_piezo_tensor(rng)
    e_voigt = cartesian_to_voigt(cart, engineering_shear=True)

    eps = rng.normal(size=(3, 3))
    eps = 0.5 * (eps + eps.T)
    eta = _engineering_strain_from_tensor(eps)

    lhs = float(np.einsum("ij,j->", e_voigt, eta))
    rhs = float(np.einsum("ijk,jk->", cart, eps))
    assert np.isclose(lhs, rhs, atol=1e-9), f"work conjugacy fails: {lhs} != {rhs}"


def test_shear_basis_component_value():
    """A pure yz Voigt component must map to e_i,1,2 = voigt[i,3], not half."""
    voigt = np.zeros((3, 6), dtype=np.float64)
    voigt[:, 3] = 2.0  # yz engineering component
    cart = voigt_to_cartesian(voigt, engineering_shear=True)
    for i in range(3):
        assert np.isclose(cart[i, 1, 2], 2.0), (
            f"cart[{i},1,2] = {cart[i,1,2]}, expected 2.0 for engineering shear"
        )


def test_pymatgen_voigt_oracle():
    """Our Cartesian tensor agrees with pymatgen's PiezoTensor.from_vasp_voigt."""
    rng = np.random.default_rng(102)
    # Build a random symmetric Cartesian tensor and convert to VASP Voigt order:
    # pymatgen uses xx, yy, zz, xy, yz, zx.  Internal order is xx, yy, zz, yz, xz, xy.
    cart = _random_symmetric_piezo_tensor(rng)
    internal = cartesian_to_voigt(cart, engineering_shear=True)
    # internal -> vasp: [xx,yy,zz,yz,xz,xy] -> [xx,yy,zz,xy,yz,zx]
    vasp = internal[:, [0, 1, 2, 5, 3, 4]]
    expected = PiezoTensor.from_vasp_voigt(vasp).voigt
    # Compare in internal order after converting back.
    ours = cartesian_to_voigt(cart, engineering_shear=True)
    assert np.allclose(ours, expected, atol=1e-9), "pymatgen oracle mismatch"


def test_trusted_converter_agrees_with_project_converter():
    """pymatgen PiezoTensor.from_vasp_voigt agrees with our piezo_stress converter."""
    rng = np.random.default_rng(104)
    cart = _random_symmetric_piezo_tensor(rng)
    internal = cartesian_to_voigt(cart, engineering_shear=True)
    project = piezo_stress_voigt_to_cartesian(internal)
    trusted = trusted_piezo_stress_voigt_to_cartesian(internal)
    assert np.allclose(project, trusted, atol=1e-9), "trusted/project converter mismatch"


def test_round_trip_does_not_imply_correctness():
    """A self-consistent but wrong converter can still round-trip; test identity."""
    rng = np.random.default_rng(103)
    cart = _random_symmetric_piezo_tensor(rng)
    voigt = cartesian_to_voigt(cart, engineering_shear=True)
    recovered = voigt_to_cartesian(voigt, engineering_shear=True)
    # This assertion is expected to pass even before the fix; the real oracle is
    # work conjugacy above.  We keep it as a sanity check.
    assert np.allclose(cart, recovered)


def test_tensor_lineage_reports_zero_diff_for_consistent_tensors():
    """When stored Cartesian equals trusted reconstruction, lineage diff is zero."""
    rng = np.random.default_rng(105)
    cart = _random_symmetric_piezo_tensor(rng)
    internal = cartesian_to_voigt(cart, engineering_shear=True)
    from crosspiezo.conventions.voigt import tensor_lineage_metrics
    metrics = tensor_lineage_metrics(internal, cart)
    assert metrics["frobenius_diff_trusted_vs_stored"] < 1e-9
    assert metrics["relative_diff"] < 1e-9
