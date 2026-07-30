# CrossPiezo Data Contract

## Source artifacts

- `T2C-Flow/processed/jarvis_piezo.parquet` — 5,000 JARVIS piezoelectric records.
- `T2C-Flow/processed/materials_project_piezo.parquet` — 3,316 MP piezoelectric records.
- `T2C-Flow/processed/jarvis_mp_piezo_overlap.parquet` — 1,266 formula-equal pairs.
- `PiezoJet/processed/jarvis_dfpt_v9_full_public/` — 4,995 strict-factor records.

## Required fields per record

- `source_database`, `source_dataset`, `source_version`
- `material_id`, `external_material_id`
- `formula`, `space_group`, `cif`
- `piezo_voigt_total`, `piezo_cartesian_total`
- `voigt_order`, `engineering_shear`, `unit`

## Invariants

- `E:/DATA` is read-only for this pipeline.
- No large files are copied into the repository.
- All manifests, split hashes and fingerprints are preserved.
- Third-party data is not committed to Git.
