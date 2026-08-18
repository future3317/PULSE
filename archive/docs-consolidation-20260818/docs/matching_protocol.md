# CrossPiezo structure-matching protocol

Structure matching is the only valid pairing criterion for the active
benchmark. Chemical or reduced-formula equality may narrow candidates but never
creates a paired label.

## Authoritative configuration

The frozen tolerances and tier rules are in `configs/matching.yaml`. Do not
duplicate or silently override them in a report or script.

The active panel contains 573 P0 pairs and 207 P2 tight matches. P2 is a
predefined nested sensitivity panel; it must not be redefined after viewing
rankings.

## Procedure

1. Preserve the original structures and provenance for both sources.
2. Use explicit cross-references when available, then use formula/species and
   atom-count filters only to narrow the candidate set.
3. Apply the frozen `pymatgen` `StructureMatcher` configuration.
4. Record the match tier, lattice/site distances, space-group relation,
   recovered basis transform, Cartesian rotation, atom mapping, ambiguity, and
   pass/fail reasons.
5. Quarantine ambiguous or convention-uncertain records instead of guessing.
6. Transport tensors only after the structure match is accepted and preserve
   the transformation history.

## Prohibited shortcuts

- formula-only pairing;
- choosing a match because its tensor values agree;
- changing tolerances after inspecting disagreement;
- silently using a fallback cell, frame, sign, or unit convention;
- replacing the frozen panel with a newly selected panel.

## Evidence

Panel membership is in `artifacts/phase6a/panels/`; matching summaries are in
`results/phase7c/matching_audit_summary.csv` and the current Phase 8 reports.
