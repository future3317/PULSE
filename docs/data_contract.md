# CrossPiezo data contract

## Read-only source boundary

`E:/DATA` contains external MP, JARVIS, T2C-Flow, and related releases. It is
read-only. Source files are not copied into Git, overwritten, or silently
repaired.

## Required provenance

Source records and derived tensors should retain, where available:

- source database, dataset, release/version, and material identifier;
- formula, structure/CIF, space group, and source structure identifier;
- tensor quantity, contribution, unit, Voigt order, shear convention, and frame;
- parser/converter version and transformation history;
- source field, calculation type/version, missingness, and quarantine status.

## Frozen derived artifacts

- `artifacts/phase6a/panels/` contains the frozen P0/P2 membership.
- `results/phase7c/` contains the frozen numerical outputs.
- `results/phase7c/phase7c_manifest.json` binds the result files to hashes.

Do not treat a report snapshot as a replacement for the artifact or manifest.
If a source field or convention is not documented, record it as unknown.
