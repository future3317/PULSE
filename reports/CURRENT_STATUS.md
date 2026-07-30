# CrossPiezo / PULSE — Current Status

> Generated during correctness audit reset.

All Phase 0–5B reports, artifacts, pair manifests, and scientific numbers in this
repository are **provisional** and under active correctness audit.

- Audit branch: `audit/correctness-v1`
- Baseline commit: `ee419516f8641fba18e99266c3288349f96e2785`
- Pre-audit freeze: `artifacts/releases/pre_audit_ee4195/`

Do **not** cite the old numbers in manuscripts or the LaTeX draft until the
`reports/correctness_v1/10_correctness_decision.md` gate has passed.

## What is being audited

- Tensor convention conversions (Voigt ↔ Cartesian, engineering shear, e/d).
- O(3) parity for polar rank-3 tensors.
- Cartesian symmetry operations derived from actual structures.
- Source-native frame reconstruction.
- Structure-match rotation estimation and RMS/max distance handling.
- Crystal-system mapping.
- Orbit discrepancy and exact-frame transport.
- Rotation-invariant ranking functionals.
- e3nn tensor-axis symmetrization and periodic graph handling.
- Prototype / formula split definitions and leakage.
- Source-held-out / counterfactual evaluation labels.
- Hard-coded scientific numbers in reporting scripts.
- PMR definition, bootstrap, and normalization consistency.

## Where new results will appear

- `artifacts/correctness_v1/`
- `reports/correctness_v1/`

## Remaining prohibited without human approval

- New DFT/DFPT calculations.
- MP version shift or third-protocol expansion.
- Full PULSE model development.
- Writing results into the main LaTeX draft.
- Pushing directly to `master`.
