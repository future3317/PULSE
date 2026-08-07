# CrossPiezo statistical plan

This is the current analysis contract. Frozen parameter values live in
`configs/phase7c.yaml`; Phase 9 upgrade parameters are implemented in
`scripts/run_scientific_upgrade.py`. Generated CSV/JSON artifacts and their
manifests are the authoritative numerical record.

## Primary universe and endpoints

- Unit of analysis: one frozen structure-matched pair.
- Primary panel: P0; sensitivity panel: P2.
- Primary scalar: F1 Cartesian Frobenius norm.
- Secondary scalars: F3 longitudinal maximum and F4 Kelvin/Mandel operator
  norm.
- Screening grid: q = 1% through 50%, with elite, intermediate, and broad
  bands.

## Concordance

For each panel, metric, and quantile, report observed top-q overlap, plain
Jaccard, exact chance expectation, chance-adjusted Jaccard, and a simultaneous
95% bootstrap band. Report nAUCC and banded partial nAUCC as descriptive
summaries. Persistent onset is defined by the frozen configuration, not chosen
after inspecting the results.

## Uncertainty and dependence

- Phase 9 screening-resolution curve, onset and nAUCC intervals use a unified
  reduced-formula cluster bootstrap with a studentized sup-norm band over the
  quantile grid. The bootstrap universe size and exact hypergeometric null are
  recomputed for every replicate.
- The frozen Version A paired row-bootstrap curve remains a sensitivity layer
  under `results/phase7c/`.
- Property controls and portfolio inference resample reduced-formula groups.
- Portfolio full-procedure bootstrap re-runs source ranking, strategy
  selection, baseline selection, and worst-source recall within each replicate.
- Ties use stable sorting; duplicate reduced formulas remain in the panel and
  are represented by distinct bootstrap identities when required.
- Seeds, replicate counts, and confidence levels are frozen in the config.

The Phase 9 upgrade also reports tie-induced overlap bounds, score gaps,
continuous matching-distance thresholds and strata, and raw-value relative
difference summaries under `results/phase9/`. The pre-registered third protocol
is a separate independent-validation gate: its 48 candidates are hash-bound,
but real DFT/DFPT execution is intentionally deferred and no independent tensor
result is reported.

## Portfolio estimand

The primary portfolio metric is material-level worst-source recall at q*=10%
and equal budget. The reported paired difference uses the same recall estimand
and the full-procedure grouped bootstrap. Earlier group-mean and fixed-selection
quantities remain audit diagnostics only.

## Interpretation

These are paired benchmark statistics, not estimates of a universal population
parameter and not model-accuracy measurements. Never select metrics or
subgroups after viewing the result to strengthen a claim.
