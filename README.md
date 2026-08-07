# CrossPiezo / PULSE

This repository contains the CrossPiezo analysis code, frozen benchmark
artifacts, reproducibility scripts, the Phase 9 scientific upgrade, and
historical phase records.

The active line is **CrossPiezo Version A**: screening-resolution and
two-source portfolio analysis for structure-matched Materials Project and
JARVIS piezoelectric records. The earlier PULSE model proposal and Phase 0–7
development materials remain only as historical context or compatibility
inputs; they are not the current execution target.

## Start here

1. [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) — live scope, rules,
   commands, frozen numbers, and remaining work.
2. [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md) — map of
   active documentation, evidence, and archived history.
3. [`reports/CURRENT_STATUS.md`](reports/CURRENT_STATUS.md) — current
   verification and submission status.
4. `E:/PAPER/CrossPiezo_EliteTail_Instability/` — self-contained manuscript
   and submission package.

## Minimal verification

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q
python scripts/run_convention_audit.py
```

Use a Python 3.11+ environment with `pyarrow>=23.0` for the frozen Parquet
artifacts. Run `python scripts/verify_phase7c.py` from a compatible environment
to reconcile the frozen result tables.

Run `python scripts/run_scientific_upgrade.py` to regenerate the post-Version-A
artifacts under `results/phase9/`. This does not overwrite `results/phase7c/`.
The third-protocol candidate manifest is ready, but real DFT/DFPT adjudication
is intentionally deferred; no independent tensor result is included.

## Documentation policy

Live documents intentionally do not contain manually maintained “last updated”
dates. Use Git history, generated report metadata, result manifests, and
artifact hashes for provenance. Do not hand-copy result numbers into prose;
regenerate reports and manuscript tables from versioned scripts.

Project rules are in [`CLAUDE.md`](CLAUDE.md); the compatibility taskbook entry
is [`CROSSPIEZO_TASKBOOK.md`](CROSSPIEZO_TASKBOOK.md). Stronger-venue review
gates are tracked in [`reports/open_questions.md`](reports/open_questions.md),
not in new kickoff files.
