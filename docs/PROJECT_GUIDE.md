# CrossPiezo project guide

This is the active operating guide for the repository. It intentionally has no
manually maintained date; provenance comes from Git history, generated reports,
result manifests, and artifact records.

## Active line and scope

CrossPiezo Version A measures screening-resolution agreement between
structure-matched piezoelectric records from Materials Project (MP) and JARVIS.
The central result is that global rank agreement does not determine elite-set
reproducibility. The portfolio analysis is a two-source risk-management
illustration, not evidence of physical truth or independent validation.

The active line includes:

- frozen P0/P2 paired panels;
- F1/F3/F4 screening-resolution curves and simultaneous bands;
- the Phase 9 reduced-formula cluster-bootstrap upgrade;
- tie-aware cutoff, raw-value, matching-distance, property-control, and
  portfolio diagnostics;
- the Version A manuscript and supplementary package.

The third-protocol preflight generated a hash-bound candidate set, but real
DFT/DFPT adjudication is deferred. The active line does not include
experimental adjudication, PULSE model development, new endpoints, panel
expansion, or large-scale data acquisition.

## Frozen evidence

| Item | Current value/source |
|---|---|
| P0 panel | 573 structure-matched pairs |
| P2 panel | 207 tight matches |
| Primary metric | F1 Cartesian Frobenius norm |
| P0 F1 nAUCC | 0.105 over the full 1–50% range |
| Elite partial nAUCC | 0.003 over the 1–10% range |
| Balanced-union recall | 0.526 at P0 F1, q*=10%, equal budget |
| Exact minimax observed-ranking recall | 0.561 (32/57) at P0 F1, q*=10% |
| Exact minimax full-procedure gain | +0.421, 95% CI [0.357, 0.464] |

Frozen point estimates remain authoritative in `results/phase7c/`. Unified
cluster-bootstrap intervals and upgrade diagnostics are in `results/phase9/`
and its manifest. Do not copy numerical values from historical prose into the
manuscript.

## Current decisions and blockers

- Version A remains a two-source benchmark; no independent tensor result is
  reported.
- The third protocol is deferred after preflight. Resuming it requires an
  independent DFT/DFPT environment and complete settings/scheduler/output
  provenance.
- No new panel members, endpoint changes, PULSE model development, or broad
  data download are part of the current line.
- Submission still requires the real preregistration DOI/BibTeX entry, author
  and institution metadata, and a permanent archive DOI or private reviewer
  link.
- The current venue route is a benchmark/materials-informatics submission;
  stronger adjudication claims wait for independent validation.

## Canonical workflow

Run from `E:/CODE/PULSE` with Python 3.11+ and `pyarrow>=23.0`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q
python scripts/run_convention_audit.py
python scripts/verify_phase7c.py
python scripts/run_scientific_upgrade.py
```

The convention audit sets its own subprocess path and BLAS environment. The
Phase 7C verifier must use a PyArrow runtime that reads the frozen panel
Parquet; do not regenerate the panel to work around a local mismatch.

For manuscript artifacts, use the versioned generators in the paper package
and compile the main manuscript and generated supplementary source with TeX
Live. Do not hand-edit generated supplementary TeX.

## Change protocol

For analysis changes:

1. update code/configuration and tests;
2. run the relevant verification scripts;
3. regenerate dependent CSV/JSON/figures/TeX;
4. inspect diffs and update only current explanatory prose;
5. keep artifacts, manifests, and claims synchronized.

For documentation changes, update these canonical files instead of creating a
new kickoff, handoff, versioned plan, or parallel status file. Link to artifacts
and commands rather than duplicating large result tables. Historical material
with recovery or provenance value belongs under `archive/`.

## Documentation map

- [`README.md`](../README.md) — public-facing overview and quick start.
- [`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md) — claim, data, matching,
  tensor, source-workflow, and statistical rules.
- [`CLAUDE.md`](../CLAUDE.md) — agent operating constraints.
- [`reports/README.md`](../reports/README.md) — report/evidence navigation.
- [`archive/docs-consolidation-20260818/`](../archive/docs-consolidation-20260818/)
  — pre-consolidation active documents.

## Key locations

- Manuscript package: `E:/PAPER/CrossPiezo_EliteTail_Instability/`
- Frozen panel: `artifacts/phase6a/panels/`
- Frozen results: `results/phase7c/`
- Phase 9 results: `results/phase9/`
- Current scientific reports: `reports/phase8a/`, `reports/phase8b/`, and
  `reports/phase9/`
- Historical plans and audits: `archive/`
