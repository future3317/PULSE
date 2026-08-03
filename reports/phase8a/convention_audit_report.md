# CrossPiezo tensor-convention audit report

**Date:** 2026-08-03
**Command:** `python run_convention_audit.py`
**Exit code:** 0

## Test result summary

```text
passed=23 failed=0 errors=0 total=23
```

## Details

```text
PASSED:
test_rotation_invariance.py::test_frobenius_norm_is_rotation_invariant
test_rotation_invariance.py::test_kelvin_operator_norm_is_rotation_invariant
test_rotation_invariance.py::test_kelvin_operator_norm_positive_and_bounded
test_rotation_invariance.py::test_longitudinal_matches_brute_force_oracle
test_rotation_invariance.py::test_max_longitudinal_is_rotation_invariant
test_rotation_invariance.py::test_max_longitudinal_modulus_analytic_oracle
test_rotation_invariance.py::test_max_longitudinal_modulus_is_deterministic
test_rotation_invariance.py::test_max_longitudinal_modulus_is_rotation_invariant
test_rotation_invariance.py::test_max_longitudinal_modulus_matches_dense_oracle
test_rotation_invariance.py::test_max_shear_is_withdrawn
test_rotation_invariance.py::test_mp_reported_svd_scalar_is_not_cartesian_rotation_invariant
test_rotation_invariance.py::test_mp_reported_svd_scalar_is_plain_svd
test_piezo_tensor.py::test_pymatgen_voigt_oracle
test_piezo_tensor.py::test_round_trip_does_not_imply_correctness
test_piezo_tensor.py::test_shear_basis_component_value
test_piezo_tensor.py::test_tensor_lineage_detects_broken_project_shear
test_piezo_tensor.py::test_tensor_lineage_detects_broken_stored_shear
test_piezo_tensor.py::test_tensor_lineage_detects_swapped_shear_order
test_piezo_tensor.py::test_tensor_lineage_detects_transposed_voigt
test_piezo_tensor.py::test_tensor_lineage_rejects_unexpected_source_order
test_piezo_tensor.py::test_tensor_lineage_reports_zero_diff_for_consistent_tensors
test_piezo_tensor.py::test_trusted_converter_agrees_with_project_converter
test_piezo_tensor.py::test_work_conjugacy_identity
FAILED:
ERRORS:
```

## Interpretation

A zero exit code means the repository tests confirm:
- F1 (Frobenius norm), F3 (longitudinal maximum), and F4 (Kelvin/Mandel operator norm) are invariant under coordinate rotations;
- Voigt-to-Cartesian conversion and engineering-shear factor-of-two handling are internally consistent;
- F3 spherical optimisation converges to the brute-force dense-grid oracle.

These tests do **not** prove that JARVIS and MP use identical calculation settings; they only rule out low-level convention errors in the CrossPiezo processing pipeline.
