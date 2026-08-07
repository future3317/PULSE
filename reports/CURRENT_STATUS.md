# CrossPiezo current status

This file is a live status snapshot. It intentionally has no manually
maintained date; use the generated audit report and Git history for provenance.

## Active decision

CrossPiezo Version A remains a two-source benchmark and is scientifically
complete after the final technical manuscript revision; it is not yet
administratively ready for submission. The Phase 9 statistical upgrade is
complete. The independent third protocol is candidate-ready and intentionally
deferred; no DFT/DFPT tensor result is included.

## Verified baseline

- `pytest`: 117 passed, 1 skipped.
- Convention audit: 23 passed, 0 failed/errors.
- Phase 7C independent verifier: reconciled, maximum absolute difference
  `1.11e-16`, 0 flags.
- Frozen panel: P0 = 573, P2 = 207.
- Primary portfolio result: balanced-union recall 0.526 with full-procedure
  gain +0.386, 95% CI [0.273, 0.441].
- Unified reduced-formula cluster bootstrap: 2,000 replicates for each of the
  six P0/P2 × F1/F3/F4 combinations. P0 F1 nAUCC = 0.1051, 95% CI
  [0.0565, 0.1530]; P0 F1 persistent onset at delta=0.05 is not reached.
- Tie/cutoff, raw-value and continuous matching-distance sensitivity artifacts
  are written under `results/phase9/`.
- Final manuscript revision: above-chance and practical onset definitions are
  separated, partial nAUCC intervals are identified as percentile-bootstrap
  intervals, the difference-CI wording matches the displayed table, and SI
  Table S7 is split into readable S7a/S7b panels.
- Final paper artifacts: main text 10 pages; supplementary information 11
  pages; both PDFs compile without overfull, underfull or undefined-reference
  warnings and contain no Type 3 fonts.
- Third-protocol candidate set: 48 unique rows with 10/10/10/8/10 strata,
  P2=24/24, maximum crystal-system count 10 and maximum reduced-formula count
  2.

## Submission blockers

1. Add the real `third_protocol_preregistration` BibTeX entry and DOI.
2. Replace anonymous author/institution metadata.
3. Add the permanent archive DOI or private reviewer link.

The final main-text and supplementary builds have no overfull/undefined
reference warnings. The third protocol has not produced any independent tensor
result.

## Deferred scientific extension

The 48-material third-protocol candidate list and preflight record are retained
at `results/phase9/`. If this extension is resumed, it must run on an
independent DFT/DFPT environment with documented pseudopotentials, numerical
settings, scheduler/resource provenance and the recorded candidate-selection
policy. The current paper makes no claim from that future calculation.

## Working rule

Use `docs/PROJECT_GUIDE.md` for current scope and commands. Use frozen CSV/JSON
artifacts and manifests for numbers. Treat older reports and archived plans as
historical evidence, not as active instructions.
