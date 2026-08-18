# Active CrossPiezo decisions and review gates

Only decisions and review gates that can change the current Version A
submission remain here. This is a live file and has no manually maintained date.

## Author-owned blockers

- Which real OSF/Zenodo record should be cited as
  `third_protocol_preregistration`?
- Which author, affiliation, and corresponding-author metadata should replace
  the anonymous manuscript fields?
- Which Zenodo/Figshare archive DOI or private reviewer link should appear in
  Data Availability?

## Completed stricter-venue gates

The Phase 9 upgrade completed the statistical gates below. Their generated
artifacts, rather than prose copied into this file, are the source of truth:

- `results/phase9/cluster_bootstrap_*.csv` — unified reduced-formula cluster
  bootstrap for curve, onset and nAUCC.
- `results/phase9/cutoff_gap_tie_diagnostics.csv` — tie-aware overlap bounds
  and score gaps at every reported cutoff.
- `results/phase9/matching_distance_*.csv` — continuous thresholds and
  distance strata with audit counts and non-identical-space-group counts.
- `results/phase9/raw_*_diagnostics.csv` and `raw_value_summary.csv` — raw and
  relative-difference summaries.

## Third-protocol extension deferred

The 48 candidates are generated and hash-bound, but the project has deferred
real DFT/DFPT execution. No independent tensor result is included, and the
existing MP/JARVIS tensors are not a third protocol. If resumed, use the
candidate manifest and preserve the recorded endpoint aggregation, tie policy,
code, pseudopotential, settings and output provenance.

## Explicitly closed for Version A

- No new panel members or endpoint changes.
- Third-protocol DFT/DFPT and experimental adjudication are deferred.
- No PULSE model development or broad data download.
