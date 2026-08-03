# Phase 8B Submission Readiness Decision

> Date: 2026-08-03 (updated)
> Decision: **Not Ready for venue submission; hard-stop portfolio estimand resolved**
> Scope: `CrossPiezo_ScreeningResolution_Manuscript.tex`, `CrossPiezo_ScreeningResolution_Supplementary.tex`, frozen Phase 7C artifacts.

## Summary

The hard-stop portfolio estimand mismatch identified in `reports/phase8b/00_frozen_artifact_consistency_audit.md` has been resolved numerically rather than only narratively.
A consistent material-level paired difference with a grouped bootstrap confidence interval was implemented in `src/crosspiezo/analysis/phase7c_stats.py` and wired through `scripts/run_phase7c.py`.
The primary portfolio result is now:

- **Strategy:** `balanced_union` on P0 F1 at $q^*=10\%$, $b=1.0$
- **Worst-source recall:** 0.526
- **Material-level paired difference versus the better single source:** +0.386 (95% CI [0.214, 0.500])

The confidence interval excludes zero, so a restrained portfolio claim has been restored in the main-text abstract and Discussion, and the Supplementary Note~1 / Table~S3 now define and report the material-level estimand consistently.

## Actions completed

- [x] Implemented `material_level_paired_diff_ci` in `src/crosspiezo/analysis/phase7c_stats.py`.
- [x] Wired material-level paired differences into `scripts/run_phase7c.py` Work Package D and E.
- [x] Restricted primary portfolio manuscript numbers to the reference metric `F1_Frobenius`.
- [x] Re-ran `scripts/run_phase7c.py`; regenerated all frozen CSVs, `manuscript_numbers.json`, and `phase7c_manifest.json`.
- [x] Re-ran `scripts/verify_phase7c.py`: reconciled, max diff 1.11e-16, 0 flags.
- [x] Re-ran `pytest -q`: 98 passed, 1 skipped.
- [x] Updated manuscript macros and text to report the corrected portfolio gain.
- [x] Updated Supplementary Note~1 to describe the consistent material-level paired bootstrap.
- [x] Updated Supplementary Table~S3 to use `material_paired_diff_recall` columns.
- [x] Recompiled manuscript (9 pages) and Supplementary (5 pages) without unresolved references or overfull tables.

## Remaining actions before venue submission

1. **Complete Supplementary provenance.** Add tables or text for: exact matching thresholds, manual audit scope/exclusions, full MP/JARVIS field provenance, Voigt/Cartesian and Kelvin/Mandel formulas, F3 optimisation method and validation, bootstrap parameters, tie/duplicate handling, portfolio strategy definitions, and the pre-registered third-protocol candidate list.
2. **Author and institutional metadata.** Replace Anonymous Author(s) / Anonymous Institution.
3. **Archive DOI.** Upload the prepared `archive/phase8a_submission/` bundle to Zenodo/Figshare and insert the private reviewer link or DOI in the Data Availability statement.
4. **Final proof-read.** Check for remaining "accuracy" language, ensure all cross-references resolve, and verify figure fonts/colors.

## Verdict

**Portfolio estimand mismatch: resolved.** The manuscript is internally consistent and the unsupported group-mean paired difference has been replaced by a consistent material-level paired difference with a valid grouped bootstrap CI.

**Venue submission: still Not Ready** because the Supplementary provenance remains incomplete and administrative metadata (authors, archive DOI) are placeholders. These are no longer hard scientific blockers, but a high-impact venue will expect them before review.

## Traceability

All numerical claims in the main text continue to trace to frozen CSV/JSON artifacts via `results/phase7c/phase7c_manifest.json` and `results/phase8a/manuscript_numbers.json`. No hand-entered numbers appear in this report.
