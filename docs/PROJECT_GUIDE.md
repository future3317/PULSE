# CrossPiezo current project guide

This is the live operating guide for the repository. It deliberately has no
manually maintained date field. Provenance comes from Git history, generated
reports, result manifests, and artifact hashes.

## 1. Active line and scope

CrossPiezo Version A measures screening-resolution agreement between
structure-matched piezoelectric records from Materials Project (MP) and JARVIS.
The central result is that global rank agreement alone does not determine
elite-set reproducibility. The portfolio analysis is a two-source
risk-management illustration, not evidence of physical truth or independent
validation.

The active line includes:

- frozen P0/P2 paired panels;
- F1/F3/F4 screening-resolution curves and simultaneous bands;
- the Phase 9 unified reduced-formula cluster-bootstrap upgrade;
- tie-aware cutoff, raw-value and matching-distance diagnostics;
- conventional ranking diagnostics and property controls;
- the full-procedure grouped-bootstrap portfolio analysis;
- the Version A manuscript and supplementary package.

The Phase 9 third-protocol preflight has generated and hash-bound 48 candidates.
Real DFT/DFPT adjudication is intentionally deferred; the candidate manifest is
ready, but no independent tensor result is part of the active line. A future
run may use the project's server or another documented compute allocation, but
must preserve the pre-registered protocol and record code, pseudopotential,
settings, scheduler and output provenance. The active line does not include
experimental adjudication, PULSE model development, or large-scale data
acquisition.

## 2. Frozen evidence

| Item | Current value/source |
|---|---|
| P0 panel | 573 structure-matched pairs |
| P2 panel | 207 tight matches |
| Primary metric | F1 Cartesian Frobenius norm |
| P0 F1 nAUCC | 0.105 (full 1–50% range) |
| Elite partial nAUCC | 0.003 (1–10%) |
| Balanced-union recall | 0.526 at P0 F1, q*=10%, equal budget |
| Exact minimax observed-ranking recall | 0.561 (32/57) at P0 F1, q*=10%, equal budget |
| Exact minimax full-procedure gain | +0.421, 95% CI [0.357, 0.464] |

The frozen point estimates remain authoritative in `results/phase7c/`; the
unified cluster-bootstrap intervals and upgrade diagnostics are in
`results/phase9/` and its manifest. Do not copy numbers from historical
reports into the manuscript.

## 3. Scientific boundaries

- Pair records by structure matching; chemical formula alone is insufficient.
- Keep MP and JARVIS values source-conditional. Never average them and call the
  result a true tensor, consensus, or physical ground truth.
- Do not claim experimental validation, physical exactness, generalization, or
  a protocol uncertainty floor without the corresponding independent evidence.
- Report the portfolio as a decision/risk-management illustration.
- Treat undocumented source workflow details as unknown rather than inferred.

See `docs/claim_boundary.md`, `docs/tensor_conventions.md`,
`docs/matching_protocol.md`, and `docs/mp_vs_jarvis_conventions.md` for the
technical boundary conditions.

## 4. Canonical workflow

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
Phase 7C verifier must run with a PyArrow runtime that can read the frozen panel
Parquet. Do not regenerate the panel to work around a local PyArrow mismatch.

For manuscript artifacts, run the versioned generators in the paper package,
then compile the main manuscript and supplementary source with TeX Live. The
supplementary `.tex` file is generated and must not be hand-edited.

## 5. Change protocol

When changing analysis behavior:

1. update code/configuration and tests;
2. run the relevant verification scripts;
3. regenerate dependent CSV/JSON/figures/TeX;
4. inspect diffs and update only current explanatory prose;
5. record the result in the current status report or generated audit.

When changing documentation:

- update the canonical document, not a new handoff or kickoff prompt;
- do not add a “last updated” date that someone must maintain manually;
- link to artifacts and commands instead of duplicating large tables;
- move superseded plans to `archive/legacy/` when they retain historical value.

## 6. Current submission blockers

The scientific Version A content is complete. Before venue submission, authors
must still provide:

- the real pre-registration DOI and BibTeX entry;
- author and institution metadata;
- the permanent archive DOI or private reviewer link.

For a stricter venue revision, `reports/open_questions.md` records the remaining
author decisions and the explicitly deferred third-protocol extension. The
statistical reviewer gates are now represented by the Phase 9 artifacts.

## 7. Key locations

- Manuscript package: `E:/PAPER/CrossPiezo_EliteTail_Instability/`
- Frozen panel: `artifacts/phase6a/panels/`
- Frozen results: `results/phase7c/`
- Phase 9 upgrade results: `results/phase9/`
- Current decision reports: `reports/phase8a/` and `reports/phase8b/`
- Phase 9 scientific report: `reports/phase9/01_scientific_upgrade.md`
- Historical plans: `archive/legacy/root_phase_plans/`
