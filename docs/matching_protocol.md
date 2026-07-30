>_# CrossPiezo Strict Matching Protocol

## Tier definitions

- **Tier 0**: shared upstream structure identifier and verified atom mapping.
- **Tier 1**: same composition and atom count, species-aware periodic match
  below frozen tolerances, compatible space group.
- **Tier 2**: same prototype and species mapping but non-negligible relaxed
  geometry difference.
- **Tier 3**: same reduced formula only.

## Frozen tolerances (Phase 0-4)

- `ltol = 0.2` (fractional length tolerance)
- `stol = 0.3` Å (site tolerance)
- `angle_tol = 5.0` degrees
- `primitive_cell = False` for atom mapping; fallback to `True` if first pass
  fails.

## Procedure

1. Parse both CIFs with pymatgen.
2. Reject pairs with different atom counts or reduced formulas (Tier 3).
3. Compare space-group symbols.
4. Run `StructureMatcher` with frozen tolerances.
5. On a match, extract atom mapping and the Cartesian rotation that maps the
   MP lattice into the JARVIS frame.
6. Mark as Tier 1 if space groups are identical or related; otherwise
   quarantine for review.

## Outputs

- `artifacts/pair_manifests/all_matches.parquet`
- `artifacts/pair_manifests/strict_pairs.parquet`
- `artifacts/pair_manifests/quarantined_pairs.parquet`
- `reports/03_pairing_audit.md`
