# CrossPiezo scientific contract

This is the single active reference for data handling, tensor conventions,
matching, statistical estimands, and claim boundaries. Generated CSV/JSON
artifacts and their manifests remain authoritative for numerical values; this
document defines how those values may be produced and interpreted.

## 1. Claim boundary

### Supported claims

- Cross-source disagreement is a source-conditional measurement, not
  experimental truth.
- Global rank correlation and elite-tail overlap describe different parts of a
  screening workflow.
- MP and JARVIS can be compared on the frozen structure-matched panel after
  explicit convention handling.
- A two-source portfolio is a risk-management illustration under a fixed
  budget.

### Claims requiring independent adjudication

Do not claim any of the following without an independent third protocol or
experiment:

- a true tensor, ground-truth consensus, or physically exact ranking;
- experimental validation or physical correctness of either source;
- a protocol uncertainty floor or a general limit on AI screening;
- robust discovery, calibration, or generalization beyond the evaluated panel.

Do not average MP and JARVIS tensors and call the result physical truth. Do not
call a two-source portfolio a validated method comparison.

### Required qualifiers

Exact claims must state the evaluated sources, panel, metric, quantile range,
and uncertainty interval. Undocumented source workflow details are reported as
unknown rather than inferred to be identical. Conventional property controls
are workflow-sensitivity evidence; they do not identify the physical cause of
piezoelectric disagreement.

## 2. Data and provenance contract

`E:/DATA` contains external MP, JARVIS, T2C-Flow, and related releases. It is
read-only: source files are not copied into Git, overwritten, or silently
repaired.

Where available, source records and derived tensors retain:

- source database, release/version, and material identifier;
- formula, structure/CIF, space group, and source structure identifier;
- tensor quantity, contribution, unit, Voigt order, shear convention, and
  frame;
- parser/converter version and transformation history;
- source field, calculation type/version, missingness, and quarantine status.

Frozen derived artifacts are stored in:

- `artifacts/phase6a/panels/` — frozen P0/P2 membership;
- `results/phase7c/` — frozen numerical outputs;
- `results/phase7c/phase7c_manifest.json` — result provenance;
- `results/phase9/` — unified cluster-bootstrap upgrade outputs and manifest.

A report snapshot never replaces its source artifact or manifest. Unknown
metadata remains unknown.

## 3. Structure-matching protocol

Structure matching is the only valid pairing criterion. Chemical or reduced
formula equality may narrow candidates but never creates a paired label.

The authoritative tolerances and tier rules are in `configs/matching.yaml`.
The active panel contains 573 P0 pairs and 207 P2 tight matches; P2 is a
predefined nested sensitivity panel and must not be redefined after viewing
rankings.

For each accepted pair:

1. preserve both source structures and provenance;
2. use explicit cross-references when available, then use formula/species and
   atom count only as candidate filters;
3. apply the frozen `pymatgen` `StructureMatcher` configuration;
4. record match tier, lattice/site distances, space-group relation, recovered
   basis transform, Cartesian rotation, atom mapping, ambiguity, and pass/fail
   reasons;
5. quarantine ambiguous or convention-uncertain records instead of guessing;
6. transport tensors only after acceptance and retain the transformation
   history.

Formula-only pairing, response-driven matching, post-hoc tolerance changes,
silent cell/frame/sign/unit fallbacks, and replacement of the frozen panel are
prohibited.

## 4. Tensor and source-workflow conventions

### Internal representation

- quantity: piezoelectric stress tensor `e`;
- shape: full Cartesian `3 x 3 x 3`;
- minor symmetry: `e_ijk = e_ikj` in the strain indices;
- unit: `C/m^2`;
- engineering Voigt order: `xx, yy, zz, yz, xz, xy`;
- engineering shear components are converted explicitly during Cartesian
  expansion.

Every converted record retains source order, internal order, shear convention,
units, stress/strain identity, Cartesian expansion, lattice basis, atom
mapping, rotation parity, point-group action, and source provenance. Unknown
conventions are quarantined rather than inferred.

For accepted matches, the recovered Cartesian transport is recorded before
comparison. The active scalar endpoints are coordinate-frame invariants:

- F1: Cartesian Frobenius norm;
- F3: maximum collinear longitudinal response;
- F4: Kelvin/Mandel operator norm on symmetric strain.

No silent `e`/`d` conversion, sign/unit/shear/frame change, MP--JARVIS
averaging, or use of MP's plain Voigt SVD scalar as a cross-source invariant is
allowed.

### Known and unknown release metadata

| Aspect | JARVIS | Materials Project |
|---|---|---|
| Quantity | Piezoelectric stress tensor `e` | Piezoelectric stress tensor `e` |
| Unit | `C/m^2` | `C/m^2` |
| Voigt order | `xx, yy, zz, yz, xz, xy` | `xx, yy, zz, yz, xz, xy` |
| Engineering shear | `True` | `True` |
| Tensor frame | JARVIS/GMTNet source-structure frame | MP IEEE-oriented structure frame |
| Ionic/electronic components | Not present in used release | Present in used release |
| Functional, pseudopotential, cutoff, k-points | Not documented | Not documented |
| DFPT vs finite difference | Not documented | Not documented |
| Relaxation and cell-default history | Not documented | Not documented |
| Per-record frame rotation | Not provided | Not provided |

The convention audit and tests validate rotation invariance, Voigt/Cartesian
conversion, engineering-shear handling, and F3 optimization. They do not prove
identical electronic-structure workflows.

## 5. Statistical estimands

The unit of analysis is one frozen structure-matched pair. The primary panel is
P0, P2 is sensitivity, F1 is primary, and F3/F4 are secondary scalars. The
screening grid is `q = 1%` through `50%`.

At each panel, metric, and quantile, report observed top-q overlap, plain
Jaccard, exact chance expectation, chance-adjusted Jaccard, and a simultaneous
95% bootstrap band. nAUCC and banded partial nAUCC are descriptive summaries;
persistent onset is defined by the frozen configuration rather than selected
after inspection.

Phase 9 uses a unified reduced-formula cluster bootstrap with a studentized
sup-norm band. The bootstrap universe size and exact hypergeometric null are
recomputed for each replicate. The frozen Version A paired row bootstrap under
`results/phase7c/` remains a sensitivity layer. Property controls and
portfolio inference resample reduced-formula groups; the full-procedure
portfolio bootstrap re-runs ranking, strategy selection, baseline selection,
and worst-source recall within each replicate. Ties use stable sorting and
seeds, replicate counts, and confidence levels are fixed in configuration.

The primary portfolio estimand is material-level worst-source recall at
`q*=10%` and equal budget. Its paired difference uses the same recall estimand
and full-procedure grouped bootstrap. Earlier group-mean and fixed-selection
quantities are audit diagnostics only.

These are paired benchmark statistics, not estimates of a universal population
parameter and not model-accuracy measurements. Never select metrics or
subgroups after viewing the result to strengthen a claim.

## 6. Independent validation boundary

The pre-registered third-protocol candidate set is retained as a future
independent-validation gate. Its real DFT/DFPT execution is deferred; no
independent tensor result is reported. If resumed, preserve the candidate
manifest, endpoint aggregation, tie policy, code, pseudopotential, numerical
settings, scheduler, and output provenance.
