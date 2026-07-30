# Phase 5A Claim Changes

> Status after Phase 5A critical adjudication.  Benchmark Go achieved;
> Strong Go blocked by soft-mode grouped-CV instability.

## Claims that can be strengthened (with verified numbers)

1. **Cross-protocol tensor disagreement is large and comparable to in-source model error.**
   - Median absolute discrepancy = 0.805 C/m².
   - PMR = 1.03 [0.95, 1.14] against a reproducible structural ridge baseline.
   - Rankings are unstable: top-50 Jaccard = 0.075, Kendall τ = 0.257.

2. **Disagreement survives rigorous O(3) transport, polar-domain equivalence, and common point-group projection.**
   - Median normalized discrepancy remains ~1.47–1.88 across all variants.

## Claims that must be narrowed

1. **"Mechanistic amplification by soft optical modes"** → narrow to:
   - "JARVIS-side soft-mode and ionic-response indicators correlate with cross-protocol discrepancy in-sample, but the relationship does not generalize across prototype groups in grouped CV."

2. **"PULSE model"** → defer to a later phase.  Phase 5A ends at Benchmark Go; full PULSE development requires Strong Go or explicit researcher approval.

## Claims that must be removed or cannot be made

- Any statement that one database is experimentally true or universally more accurate.
- Causal attribution of a specific JARVIS–MP setting to observed differences.
- "True tensor" / "ground truth consensus" without a third protocol or experiment.

## LaTeX placeholders that can now be filled (after researcher approval)

- `\TBD{strict matched count}` → 538 Tier-1 pairs (439 T1a).
- `\TBD{Compare paired-source discrepancy with in-source errors...}` → PMR = 1.03 [0.95, 1.14].
- `\TBD{Report model rankings across all frozen splits...}` → top-50 Jaccard = 0.075, Kendall τ = 0.257.

## LaTeX placeholders that must remain TBD

- `\TBD{Report mode-resolved analysis on the strict-factor subset...}` → relationship is exploratory only; grouped CV does not support a strong mechanistic claim.
- Third-protocol adjudication set remains optional / not executed.
