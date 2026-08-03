# Phase 8B Submission Readiness Decision

> Date: 2026-08-03  
> Decision: **Not Ready**  
> Scope: `CrossPiezo_ScreeningResolution_Manuscript.tex`, `CrossPiezo_ScreeningResolution_Supplementary.tex`, frozen Phase 7C artifacts.

## Summary

The hard-stop portfolio estimand mismatch identified in `reports/phase8b/00_frozen_artifact_consistency_audit.md` has been addressed by downgrading the portfolio analysis from the main text to an exploratory Supplementary discussion. The unsupported "global accuracy" narrative has been replaced with "global correlation / aggregate agreement" language throughout. The Supplementary has been expanded with methodological notes on matching, tensor conventions, bootstrap details and portfolio estimands.

However, the manuscript is still **Not Ready** for submission because:

1. **Portfolio analysis remains descriptive only.** The estimand mismatch is documented but not resolved numerically. A reviewer may still ask for a consistent material-level paired difference and confidence interval if the portfolio is to support any causal claim about aggregation. The current treatment is conservative but not a full statistical fix.

2. **Supplementary is improved but not venue-complete.** Notes cover the main methodological gaps, but a high-impact venue will still expect: exact similarity thresholds, the full manual-audit decision log, source field provenance tables, Voigt/Cartesian conversion formulas, F3 optimisation convergence records, bootstrap seed and studentization details, and the complete pre-registered 48-material third-protocol list.

3. **No independent third-protocol or experimental adjudication.** The manuscript correctly limits itself to two-source evidence, but this limits the strength of conclusions about physical correctness.

## Actions completed

- [x] Removed portfolio claim from abstract.
- [x] Replaced "aggregate accuracy" with "aggregate correlation" in abstract and Discussion.
- [x] Removed Section 5 (portfolio) from main text; added a short pointer to Supplementary.
- [x] Moved Figure 4 (illustrative candidates) and Figure 5 (portfolio frontier) to Supplementary as Figures S1 and S2.
- [x] Added Supplementary Note 1 explaining the portfolio estimand mismatch.
- [x] Added Supplementary Notes 2–4 covering structure matching, tensor conventions, and bootstrap/tie handling.
- [x] Updated Table S3 caption to state that paired differences are group-mean differences, not arithmetic differences of Recall.
- [x] Recompiled manuscript (9 pages) and Supplementary (5 pages) without unresolved references or overfull tables.
- [x] Reran `scripts/verify_phase7c.py`: reconciled, max diff 1.11e-16, 0 flags.
- [x] Reran `pytest -q`: 98 passed, 1 skipped.

## Remaining actions before submission

1. **Decide on portfolio statistical treatment.** Either (a) recompute material-level paired differences with a consistent grouped bootstrap CI and restore a restrained main-text claim, or (b) keep the current exploratory Supplementary treatment and add an explicit limitation sentence in the Discussion.

2. **Complete Supplementary provenance.** Add tables or text for: exact matching thresholds, manual audit scope/exclusions, full MP/JARVIS field provenance, Voigt/Cartesian and Kelvin/Mandel formulas, F3 optimisation method and validation, bootstrap parameters, tie/duplicate handling, portfolio strategy definitions, and the pre-registered third-protocol candidate list.

3. **Author and institutional metadata.** Replace Anonymous Author(s) / Anonymous Institution.

4. **Archive DOI.** Upload the prepared `archive/phase8a_submission/` bundle to Zenodo/Figshare and insert the private reviewer link or DOI in the Data Availability statement.

5. **Final proof-read.** Check for remaining "accuracy" language, ensure all cross-references resolve, and verify figure fonts/colors.

## Verdict

**Not Ready.** The manuscript is now internally consistent and the hard-stop contradiction is removed, but the portfolio section is only downgraded, not fixed, and the Supplementary needs more provenance before a high-impact submission.

## Traceability

All numerical claims in the main text continue to trace to frozen CSV/JSON artifacts via `results/phase7c/phase7c_manifest.json` and `results/phase8a/manuscript_numbers.json`. No hand-entered numbers appear in this report.
