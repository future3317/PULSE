# Phase 8B Submission Readiness Decision

> **Superseded:** this historical readiness snapshot predates the exact
> minimax-selector correction and the current manuscript revision.

> Decision: **Not Ready for venue submission; portfolio estimand resolved via Phase 8C full-procedure grouped bootstrap**
> Scope: `CrossPiezo_ScreeningResolution_Manuscript.tex`, `CrossPiezo_ScreeningResolution_Supplementary.tex`, frozen Phase 7C artifacts.

## Summary

The hard-stop portfolio estimand mismatch identified in `reports/phase8b/00_frozen_artifact_consistency_audit.md` has been resolved numerically rather than only narratively.
A **full-procedure grouped bootstrap** was implemented in `src/crosspiezo/analysis/phase7c_stats.py` and wired through `scripts/run_phase7c.py`.
Each bootstrap replicate now resamples reduced-formula groups with replacement, assigns distinct identities to duplicated occurrences, re-runs the portfolio strategy and both single-source baselines, and recomputes the improvement over the better single source.
The primary portfolio result is now:

- **Strategy:** `balanced_union` on P0 F1 at $q^*=10\%$, $b=1.0$
- **Worst-source recall:** 0.526
- **Full-procedure paired difference versus the better single source:** +0.386 (95% CI [0.273, 0.441])

The confidence interval excludes zero, so a restrained portfolio claim has been restored in the main-text abstract and Discussion, and the Supplementary Note~1 / Table~S3 now define and report the full-procedure material-level estimand consistently.

## Actions completed

- [x] Implemented `full_procedure_portfolio_bootstrap_ci` in `src/crosspiezo/analysis/phase7c_stats.py`.
- [x] Implemented `material_level_paired_diff_ci` (conditional fixed-selection bootstrap) as an auxiliary diagnostic.
- [x] Wired full-procedure portfolio CIs into `scripts/run_phase7c.py` Work Package D and E, producing `portfolio_full_procedure_bootstrap.csv`.
- [x] Restricted primary portfolio manuscript numbers to the reference metric `F1_Frobenius`.
- [x] Re-ran `scripts/run_phase7c.py`; regenerated all frozen CSVs, `manuscript_numbers.json`, and `phase7c_manifest.json`.
- [x] Re-ran `scripts/verify_phase7c.py`: reconciled, max diff 1.11e-16, 0 flags.
- [x] Re-ran `pytest -q`: 16 phase-7C red-team tests passed.
- [x] Added red-team tests for full-procedure seed reproducibility, duplicate-group expansion, zero CI for self-baseline, arithmetic point estimate, and non-negative $\Delta_\text{best}$.
- [x] Updated manuscript macros and text to report the corrected full-procedure portfolio gain.
- [x] Updated Supplementary Note~1 to describe the full-procedure grouped bootstrap.
- [x] Updated Supplementary Table~S3 to use `full_proc_delta_best_recall` columns.

## Remaining actions before venue submission

1. **Complete Supplementary provenance.** Add tables or text for: exact matching thresholds, manual audit scope/exclusions, full MP/JARVIS field provenance, Voigt/Cartesian and Kelvin/Mandel formulas, F3 optimisation method and validation, bootstrap parameters, tie/duplicate handling, portfolio strategy definitions, and the pre-registered third-protocol candidate list.
2. **Author and institutional metadata.** Replace Anonymous Author(s) / Anonymous Institution.
3. **Archive DOI.** Upload the prepared `archive/phase8a_submission/` bundle to Zenodo/Figshare and insert the private reviewer link or DOI in the Data Availability statement.
4. **Final proof-read.** Check for remaining "accuracy" language, ensure all cross-references resolve, and verify figure fonts/colors.

## Verdict

**Portfolio estimand mismatch: resolved via Phase 8C.** The manuscript is internally consistent and the unsupported group-mean paired difference has been replaced by a consistent material-level full-procedure grouped bootstrap CI.

**Venue submission: still Not Ready** because the Supplementary provenance remains incomplete and administrative metadata (authors, archive DOI) are placeholders. These are no longer hard scientific blockers, but a high-impact venue will expect them before review.

## Traceability

All numerical claims in the main text continue to trace to frozen CSV/JSON artifacts via `results/phase7c/phase7c_manifest.json` and `results/phase8a/manuscript_numbers.json`. No hand-entered numbers appear in this report.
