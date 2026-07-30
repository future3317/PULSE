# Phase 5A Corrections

> This report formally corrects conclusions from Phase 5A that do not survive
> the Benchmark Consolidation gate.  It is a prerequisite for Phase 5B.

## 1. Phase 5A PMR is not valid evidence

Phase 5A reported:

- PMR (absolute, structural ridge) = 1.031, 95% CI [0.947, 1.141].

This used a `structural_ridge` baseline with O(3)-invariant composition +
distance-histogram features.  The baseline failed the minimum skill gate:

| train → eval | zero MAE | structural_ridge MAE |
|:-------------|---------:|---------------------:|
| JARVIS → JARVIS | 0.494 | **0.516** |
| MP → MP | 0.914 | **1.046** |

In both in-source tests the ridge model is **worse than the zero predictor**.
Therefore the denominator of the Phase 5A PMR is not a valid estimate of
in-source model error.  The ratio 1.031 cannot be interpreted as "protocol gap
is comparable to competent model error".

## 2. PMR statistic mixed median and mean

Phase 5A computed:

```text
PMR = median_paired_discrepancy / mean_in_source_MAE
```

This mixes a median numerator with a mean denominator.  Phase 5B will report
mean/mean and median/median variants separately, on the same test universe,
with paired bootstrap confidence intervals.

## 3. Normalized symmetry residual is extremely high

Phase 5A reported normalized symmetry residuals using the common matched
structure point group on tensors expressed in the unified CIF frame:

| source | median normalized residual |
|:-------|---------------------------:|
| JARVIS | 0.92 |
| MP | 0.43 |

A median normalized residual of 0.92 for JARVIS indicates that the tensor is
almost entirely inconsistent with the CIF-setting point group.  This is a red
flag that the tensor may be reported in a source-native calculation frame that
is not the same as the T2C-Flow CIF frame.  Phase 5B must restore the
source-native tensor frame before any componentwise, cosine, or directional
comparison.

## 4. Soft-mode mechanism is not established

Phase 5A reported factor-only grouped-CV R² = -0.307 ± 0.419 and combined
R² = -0.492 ± 0.544.  Negative grouped-CV R² means the JARVIS-side soft-mode
indicators do **not** generalize across prototype groups.  The mechanism claim
must be withdrawn from the title/abstract/main contributions until either:

- the full atom-resolved internal-strain matrix Λ is recovered and re-validated,
  or
- an independent third-protocol adjudication set supports the mechanism.

## Consequences for Phase 5B

1. Do not use Phase 5A PMR = 1.031 in any paper claim.
2. Build source-native frame residuals before componentwise comparisons.
3. Train at least one O(3)-equivariant model that beats zero/mean in-source.
4. Recompute PMR only from models that pass the valid-model gate.
5. Remove or downgrade soft-mode mechanism claims pending full Λ recovery.
