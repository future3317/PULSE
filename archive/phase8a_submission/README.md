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
4. Run `python scripts/run_convention_audit.py` to verify tensor conventions and rotation invariance.
5. Run `python scripts/compute_baseline_metrics.py` to generate the conventional ranking diagnostics.
6. Run `python scripts/make_baseline_metric_synthetic_example.py` to generate the synthetic motivating figure.
7. Run `python scripts/make_matching_audit_table.py` and `python scripts/make_matching_distance_discrepancy_figure.py` to generate the matching-audit table and figure.
8. Run `python scripts/make_screening_resolution_figures.py` to regenerate the five main figures.
9. Run `python scripts/make_screening_resolution_supplementary.py` to regenerate the Supplementary Information LaTeX file.
10. Compile `CrossPiezo_ScreeningResolution_Manuscript.tex` and `CrossPiezo_ScreeningResolution_Supplementary.tex` with `pdflatex -> bibtex -> pdflatex -> pdflatex`.

## Manifest and checksums

- `manifest/phase7c_manifest.json` — hash-bound manifest for Phase 7C artifacts.
- `manifest/phase8a_archive_checksums.json` — SHA-256 checksums for all files in this archive.

## Contents

| Path | Description |
|------|-------------|
| `results/phase7c/*.csv` | Frozen Phase 7C result tables |
| `artifacts/phase6a/panels/panel_membership.parquet` | Frozen P0/P2 panel membership |
| `results/phase7c/baseline_metrics_comparison.csv` | Conventional ranking diagnostics on P0 F1 |
| `results/phase7c/matching_audit_summary.csv` | Frozen structure-matching audit summary |
| `results/phase8a/manuscript_numbers.json` | Corrected manuscript numbers from CSV reconciliation |
| `configs/third_protocol_phase7c.yaml` | Pre-registered third-protocol configuration (audited, not executed) |
| `configs/matching.yaml` | Frozen structure-matcher tolerances and tier rules |
| `scripts/make_screening_resolution_figures.py` | Main figure-generation script |
| `scripts/make_screening_resolution_supplementary.py` | Supplementary Information generation script |
| `scripts/compute_baseline_metrics.py` | Conventional ranking diagnostics on frozen P0 F1 |
| `scripts/make_baseline_metric_synthetic_example.py` | Synthetic high-global-$&$#8209;correlation / low elite&#8209;agreement example |
| `scripts/make_matching_audit_table.py` | Structure-matching audit summary table |
| `scripts/make_matching_distance_discrepancy_figure.py` | RMS distance vs F1 rank-discrepancy figure |
| `scripts/run_convention_audit.py` | Tensor convention and rotation-invariance audit runner |
| `scripts/_run_test_module.py` | Helper for subprocess-isolated convention tests |
| `docs/mp_vs_jarvis_conventions.md` | Documented source-workflow and tensor-convention comparison |
| `reports/phase8a/convention_audit_report.md` | Convention audit report |
| `figures/screening_resolution/*.pdf` | Main-text and supplementary figure PDFs |

## Notes

- The third protocol is pre-registered but was **not executed** in Phase 8A.
- The final Zenodo/Figshare DOI and private reviewer link will be inserted by the authors before submission.
