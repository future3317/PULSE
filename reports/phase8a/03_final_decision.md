# Phase 8A Final Decision

## Decision

### Submission Ready, Stronger Venue Requires Adjudication

## Rationale

### Why the manuscript is submission-ready

1. **Repository and verification are clean.**
   - Baseline commit `6e4ebf29c18189d115dc29007933edf74bc8c582` verified.
   - `pytest`: 98 passed, 1 skipped.
   - `scripts/verify_phase7c.py`: reconciled, max absolute difference 1.11e-16, 0 flags.
   - Manifest hashes for all 12 Phase 7C artifacts match.
   - v0.7 LaTeX compiles successfully.

2. **Critical number discrepancy is resolved.**
   - `results/phase7c/manuscript_numbers.json` contained stale portfolio primary-best values (0.579 recall / 0.014 paired diff).
   - Authoritative CSV values: `balanced_union` recall 0.526, paired diff +0.012 (95% CI [-0.012, 0.036]).
   - v0.6.tex already had the correct values; v0.7.tex and `results/phase8a/manuscript_numbers.json` now source directly from `portfolio_benchmark.csv`.

3. **Claims are within allowed boundaries.**
   - No PULSE/PMR, componentwise, O(3), soft-mode, or ``true tensor'' claims.
   - Portfolio is framed as two-source robustness, not physical validation.
   - Controls are framed as workflow-sensitivity evidence, not proof of conventional disagreement.
   - Third protocol is pre-registered and explicitly not executed.

4. **Reviewer risks are documented and mitigated.**
   - The reviewer attack matrix addresses materials/computational, statistics/ML, and data/provenance concerns.
   - v0.7 tightens source-pair language, adds limitations, clarifies inferential scope, and promises a permanent archive.

### Why a stronger venue requires adjudication

- The current evidence is limited to **two high-throughput sources** (MP and JARVIS).
- Claims about workflow sensitivity and elite-tail disagreement are **diagnostic**, not adjudicative.
- A top-tier computational-methods venue (e.g., *Nature Computational Science*) or a high-impact materials-informatics venue is likely to expect independent validation beyond two sources.
- The **pre-registered 48-material third protocol** provides the path to that validation. Executing it would upgrade the paper from a two-source benchmark to a three-source adjudication.

## Recommended action

1. Submit `CrossPiezo_ScreeningResolution_Manuscript_v0.7.tex` to **npj Computational Materials** or **Digital Discovery** as the primary route.
2. In parallel, prepare a **Scientific Data** Data Descriptor for the frozen benchmark artifacts.
3. If resources allow, execute the pre-registered 48-material third protocol within 4--8 weeks and revise toward **Nature Computational Science** or a stronger npj Computational Materials adjudication paper.

## Files produced in Phase 8A

| File | Purpose |
|------|---------|
| `reports/phase8a/00_takeover_audit.md` | Verification and number-reconciliation report |
| `reports/phase8a/00b_portfolio_claim_freeze.md` | Frozen equal-budget portfolio claim |
| `reports/phase8a/01_reviewer_attack_matrix.md` | Three-archetype reviewer concerns and fixes |
| `reports/phase8a/02_third_protocol_readiness.md` | Third-protocol audit (not executed) |
| `reports/phase8a/03_final_decision.md` | This decision |
| `submission/phase8a_venue_decision.md` | Venue decision and routes |
| `CrossPiezo_ScreeningResolution_Manuscript_v0.7.tex` | Revised manuscript |
| `results/phase8a/manuscript_numbers.json` | Corrected manuscript numbers from CSV |

## Stop condition

As required, the third protocol is **audited only and not executed** in this Phase 8A session. All Phase 8A deliverables are complete.
