# CrossPiezo source-workflow and tensor-convention comparison

This document records what is actually known about the JARVIS and Materials
Project (MP) piezoelectric data used in the CrossPiezo benchmark, and what is
not documented in the release metadata.  It is intended for the Supplementary
Information convention-comparison table.

## Known from release metadata

| Aspect | JARVIS | MP |
|---|---|---|
| Source database | JARVIS-DFT via pinned GMTNet release | Materials Project |
| Source dataset / snapshot | `YKQ98/GMTNet@7a606a459ee48a320ed38450e391811fb43d5e19` | `materials/piezoelectric` |
| Quantity reported | Piezoelectric stress tensor `e` | Piezoelectric stress tensor `e` |
| Unit | `C/m^2` | `C/m^2` |
| Voigt order | `xx, yy, zz, yz, xz, xy` | `xx, yy, zz, yz, xz, xy` |
| Engineering-shear convention | `True` (off-diagonal Voigt = 2 × tensor shear) | `True` (off-diagonal Voigt = 2 × tensor shear) |
| Structure / tensor frame | JARVIS/GMTNet source-structure Cartesian frame | Materials Project IEEE-oriented structure frame |
| Ionic/electronic decomposition available in release | No (`piezo_voigt_ionic` and `piezo_voigt_electronic` are null) | Yes (both ionic and electronic components present) |
| Source last-updated timestamp | Not present in used release | Present (various timestamps, 2020--2023) |

## Not documented in the release metadata (do not assume identical)

The following items are **not** available in the processed parquet provenance
fields used by CrossPiezo.  They are therefore listed as unknown rather than
guessed.

| Aspect | JARVIS | MP |
|---|---|---|
| Exchange--correlation functional | Not documented | Not documented |
| Pseudopotential family | Not documented | Not documented |
| DFPT vs finite-difference ionic response | Not documented | Not documented |
| Plane-wave cutoff | Not documented | Not documented |
| k-point density / grid | Not documented | Not documented |
| Structure relaxation thresholds (force, stress, energy) | Not documented | Not documented |
| Primitive vs conventional cell default | Not documented | Not documented |
| Sign-convention history | Not documented | Not documented |
| How the IEEE-oriented frame is chosen per structure | Not documented | Operational in MP pipeline, but no per-record rotation matrix is provided |

## What CrossPiezo does to keep the comparison clean

1. **Internal convention**: all tensors are converted to full Cartesian `e` in
   `C/m^2`, with the engineering Voigt order `xx, yy, zz, yz, xz, xy` and
   factor-of-two shear handling.  The conversion is recorded in the
   transformation history of each record.
2. **Structure matching**: the MP relaxed structure is aligned to the JARVIS
   relaxed structure (or vice versa) using a frozen `pymatgen` `StructureMatcher`
   configuration.  The recovered rotation is applied to the Cartesian tensor
   before comparison.  No silent frame reorientation is performed outside this
   per-pair rotation.
3. **Invariants**: all three reported scalars (F1, F3, F4) are coordinate-frame
   invariants, so they are unaffected by the source-specific structure-frame
   choice.
4. **Tests**: rotation invariance, Voigt shear handling, and F3 optimisation
   convergence are covered by the repository test suite and summarised by
   `scripts/run_convention_audit.py`.

## References

- `configs/matching.yaml` -- frozen structure-matcher tolerances.
- `docs/tensor_conventions.md` -- internal tensor-convention definitions.
- `tests/ranking/test_rotation_invariance.py` -- F1/F3/F4 rotation invariance.
- `tests/conventions/test_piezo_tensor.py` -- Voigt/Cartesian and shear tests.
