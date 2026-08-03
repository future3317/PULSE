# CrossPiezo Phase 8A submission archive

This archive contains the frozen artifacts needed to reproduce the screening-resolution manuscript and figures.

## Source repository

- Repository: `E:\CODE\PULSE`
- Commit: `90d1ca1b06ecea6cdef82f8d3ae454dafb3ccc7d`
- Commit message: `polish(manuscript): add Nature-style publication figures and cross-references`
- Date: 2026-08-03

## Reconstruction steps

1. Check out the commit above.
2. Install the Python environment used for Phase 7C/8A (Python 3.11+, see repository `pyproject.toml` or `requirements.txt`).
3. Run `python scripts/verify_phase7c.py` to reconcile the frozen CSV artifacts.
4. Run `python scripts/make_screening_resolution_figures.py` to regenerate the five main figures.
5. Run `python scripts/make_screening_resolution_supplementary.py` to regenerate the Supplementary Information LaTeX file.
6. Compile `CrossPiezo_ScreeningResolution_Manuscript.tex` and `CrossPiezo_ScreeningResolution_Supplementary.tex` with `pdflatex -> bibtex -> pdflatex -> pdflatex`.

## Manifest and checksums

- `manifest/phase7c_manifest.json` — hash-bound manifest for Phase 7C artifacts.
- `manifest/phase8a_archive_checksums.json` — SHA-256 checksums for all files in this archive.

## Contents

| Path | Description |
|------|-------------|
| `results/phase7c/*.csv` | Frozen Phase 7C result tables |
| `artifacts/phase6a/panels/panel_membership.parquet` | Frozen P0/P2 panel membership |
| `results/phase8a/manuscript_numbers.json` | Corrected manuscript numbers from CSV reconciliation |
| `configs/third_protocol_phase7c.yaml` | Pre-registered third-protocol configuration (audited, not executed) |
| `scripts/make_screening_resolution_figures.py` | Figure-generation script |
| `scripts/make_screening_resolution_supplementary.py` | Supplementary Information generation script |
| `figures/screening_resolution/*.pdf` | Main-text figure PDFs |

## Notes

- The third protocol is pre-registered but was **not executed** in Phase 8A.
- The final Zenodo/Figshare DOI and private reviewer link will be inserted by the authors before submission.
