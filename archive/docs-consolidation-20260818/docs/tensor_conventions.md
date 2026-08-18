# CrossPiezo tensor conventions

## Internal representation

- Quantity: piezoelectric stress tensor `e`.
- Shape: full Cartesian `3 x 3 x 3`.
- Minor symmetry: `e_ijk = e_ikj` in the strain indices.
- Unit: `C/m^2`.
- Engineering Voigt order: `xx, yy, zz, yz, xz, xy`.
- Engineering shear components are converted explicitly when expanding to
  Cartesian form.

## Transformation history

Every converted record must retain source Voigt order, internal order, shear
convention, units, stress/strain identity, Cartesian expansion, lattice basis,
atom mapping, rotation parity, point-group action, and source provenance.
Unknown conventions are quarantined rather than inferred.

## Pair transport and metrics

For an accepted structure match, the recovered Cartesian transport is recorded
before comparison. The active cross-source endpoints are coordinate-frame
invariants:

- F1: Cartesian Frobenius norm;
- F3: maximum collinear longitudinal response;
- F4: Kelvin/Mandel operator norm on symmetric strain.

The convention audit and tests validate rotation invariance, Voigt/Cartesian
conversion, engineering-shear handling, and the F3 numerical optimizer. These
tests do not prove that MP and JARVIS used identical electronic-structure
workflows.

## Prohibited transformations

- no silent `e`/`d` conversion;
- no silent sign, unit, shear, or frame change;
- no averaging MP and JARVIS tensors into a physical reference;
- no use of MP's plain Voigt SVD scalar as a cross-source invariant endpoint.
