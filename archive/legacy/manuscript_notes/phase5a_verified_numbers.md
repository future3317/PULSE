# Phase 5A Verified and Provisional Numbers

> Generated after Phase 5A critical adjudication.
> Do not paste these into the LaTeX Results/Discussion until the researcher
> explicitly approves the Phase 5A decision.

## Verified (frozen after Phase 5A)

| Quantity | Value | Location |
|:---------|------:|:---------|
| Domain/O(3)/symmetry-audited Tier-1 pairs | 538 | `artifacts/phase5a/frozen_summary.json` |
| T1a near-identical relaxed-structure pairs | 439 | `reports/10_structure_mediated_shift.md` |
| Median absolute discrepancy (exact transported) | 0.8051 C/m² | `reports/09_domain_and_o3_transport.md` |
| Median normalized discrepancy (exact transported) | 1.5970 | `reports/09_domain_and_o3_transport.md` |
| Top-50 Jaccard (Frobenius norm) | 0.0753 | `reports/11_ranking_revalidation.md` |
| Kendall τ (Frobenius norm) | 0.2569 | `reports/11_ranking_revalidation.md` |
| PMR (absolute, structural ridge baseline) | 1.031 | `reports/12_in_source_and_pmr.md` |
| PMR 95% CI | [0.947, 1.141] | `reports/12_in_source_and_pmr.md` |
| Polar-domain flips in domain-aware variant | 177 / 538 | `reports/09_domain_and_o3_transport.md` |
| PiezoJet strict-factor stable-optical intersection | 506 | `reports/13_soft_mode_feasibility.md` |

## Provisional

| Quantity | Value | Caveat |
|:---------|------:|:-------|
| Soft-mode factor-only grouped-CV R² | -0.307 ± 0.419 | Does not generalize across prototype groups; in-sample R² = 0.015 |
| Soft-mode combined grouped-CV R² | -0.492 ± 0.544 | Same caveat as factor-only |

## Rejected / not supported

- **Exact S_soft from atom-resolved internal-strain modes**: PiezoJet stores a reduced `(3,3,3)` internal-strain tensor that does not expose the full `3N × 3` matrix, so the LaTeX Eq. (S_soft) cannot be evaluated literally. A scalar proxy was used instead.
- **Causal attribution of JARVIS–MP differences to specific settings**: Not identifiable with two protocols; only correlation with JARVIS-side sensitivity indicators is reported.

## Requires third protocol or model phase

- "True tensor" / "ground truth consensus" claims.
- Experimental validation.
- Calibrated source-conditional prediction regions (PULSE phase).
