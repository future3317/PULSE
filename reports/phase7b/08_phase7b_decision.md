# Phase 7B Final Decision

**Decision: Material Go (proceed with screening-resolution manuscript v0.5)**

This decision follows the corrected Phase 7B statistical audit and robust-portfolio benchmark. All gate items below are satisfied on the frozen P0/P2 panels with coordinate-invariant response scalars F1/F3/F4. No panel membership, source data, or DFT was added.

## Gate checklist

| # | Criterion | Required evidence | Status |
|---|-----------|-------------------|--------|
| 1 | nAUCC and persistent-onset estimates are reproducible in an independent implementation | `results/phase7b/verification_summary.json`: max absolute difference between main and independent scripts = 1.11e-16, 0 flags | Pass |
| 2 | Elite-tail gap remains under P2 and high-response sensitivities | P2 F1 nAUCC = 0.060 (P0 = 0.105); P0/P2 high-response nAUCC negative (-0.048 to -0.115) | Pass |
| 3 | At least two control properties show positive Δτ or ΔnAUCC vs F1 | volume Δτ = 0.722 [0.651, 0.798], ΔnAUCC = 0.813; band_gap Δτ = 0.663 [0.591, 0.736], ΔnAUCC = 0.792; energy_above_hull and dielectric_trace also positive | Pass |
| 4 | Robust portfolio beats single-source strategies on frozen holdout, with correct metrics | balanced_union/average_percentile/intersection_first consistently exceed jarvis_only/mp_only in worst-source recall and NDCG; metrics use full-universe IDCG, minimax regret = 1 − worst-source recall, deterministic selection | Pass |
| 5 | Chemistry classification, FDR, and missingness audit pass | Anion groups parsed with `pymatgen.Composition` (no substring); subgroup min-N = 30; FDR via Benjamini–Hochberg; elastic bulk modulus marked `insufficient_N` because only MP reports it | Pass |
| 6 | v0.5 manuscript compiles and all numbers trace to the manifest | `CrossPiezo_ScreeningResolution_Manuscript_v0.5.pdf` generated; `results/phase7b/phase7b_manifest.json` records file hashes and key numbers | Pass |

## Key numbers

- P0 F1 nAUCC = 0.105; pointwise LCB > 0 first at q = 33%; no five-quantile persistent onset at δ = 0.05.
- P2 F1 nAUCC = 0.060; no persistent onset at δ = 0.05.
- Strongest control contrast (P0): volume τ = 0.967, nAUCC = 0.918 vs F1 τ = 0.245, nAUCC = 0.105.
- Best holdout worst-source recall (P0 F1, budget factor 2.0): balanced_union = 1.000.

## Caveats

- The electronic/ionic tensor decomposition is available for fewer than 100 matched pairs; it is marked `insufficient_N` and no conclusion is drawn.
- Portfolio results are evaluated on a deterministic holdout fold (hash of `pair_id` modulo 5). They demonstrate two-source decision robustness, not independent physical validation.
- The elastic bulk modulus control is reported by MP only in this processed release, so a cross-source comparison is not possible.

## Next step

Proceed with the v0.5 manuscript, figure refinement, and any journal-specific formatting. Do not execute the third protocol or revive PULSE/PMR/componentwise/O(3)/soft-mode causal claims without a new taskbook.
