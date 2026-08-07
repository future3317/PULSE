# CrossPiezo agent instructions

Before changing the project, read:

1. `CROSSPIEZO_TASKBOOK.md`
2. `docs/PROJECT_GUIDE.md`
3. `docs/DOCUMENTATION_INDEX.md`
4. `reports/CURRENT_STATUS.md`

## Active scope

- Work on CrossPiezo Version A: the frozen MP/JARVIS screening-resolution
  benchmark and its submission package.
- Do not start new DFT/DFPT calculations, execute the third protocol, develop
  PULSE models, download data, or expand the panel unless the user explicitly
  authorizes it.
- Treat `E:/DATA` as read-only. Never copy large external datasets into Git.

## Scientific boundaries

- Structure matching, not formula matching, defines paired observations.
- MP and JARVIS tensors are source-conditional observations; never average them
  and call the result a true tensor or ground truth.
- Portfolio analysis is a two-source risk-management illustration, not physical
  validation or a universal method comparison.
- Keep claims within the evidence documented in `docs/claim_boundary.md` and the
  current manuscript Methods section.

## Engineering rules

- Python 3.11+, `pathlib`, type annotations, docstrings, and Pydantic v2 for
  core schemas.
- No silent fallbacks and no `except Exception: pass`.
- Keep randomness seeded and transformations traceable.
- Generate figures, tables, supplementary TeX, and reports only through
  versioned scripts.
- Use `pyarrow>=23.0` for the frozen Parquet artifacts. On Windows, keep BLAS
  threads at one for numerical audits.
- Do not hand-edit generated files. Fix the generator, rerun it, then inspect
  the diff.

## Documentation rules

- Live documentation must not require a manually updated date field.
- Use Git history, generated metadata, manifests, and hashes for provenance.
- Keep current guidance short and actionable. Move superseded plans to
  `archive/legacy/`; delete one-time coordination prompts.
- Never replace an authoritative CSV/JSON value with a number copied from an
  old report or snapshot.

## Verification baseline

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q
python scripts/run_convention_audit.py
python scripts/verify_phase7c.py
```

The primary manuscript package is under
`E:/PAPER/CrossPiezo_EliteTail_Instability/`. Keep its manuscript source,
generated supplementary source, frozen artifacts, and reports synchronized
with the code repository when making a reproducibility change.
