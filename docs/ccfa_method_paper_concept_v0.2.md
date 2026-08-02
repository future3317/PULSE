# CCF-A Method Paper Concept: Source-Robust Ranking under Conflicting Scientific Labels (v0.2)

## 1. General problem

Multiple independent computational protocols, database versions, or experimental assays can assign conflicting labels or scores to the same material.  Existing ranking methods optimise agreement with a single source and are not guaranteed to produce candidates that are robust across sources.  The generic task is:

> Given noisy, source-conditional score vectors for a shared set of items, learn a ranking or selection rule whose top-k performance is stable under the worst source.

This is a methodological, benchmark-driven problem: we do not assume access to a single physical ground truth.

## 2. Core research questions

1. When does cross-source disagreement reflect irreducible convention differences versus reducible noise?
2. Can a source-robust ranking objective guarantee a target worst-source recall without oracle labels?
3. Which protocols (abstention, distributionally robust optimisation, rank aggregation, held-out-source validation) yield measurable gains on real multi-source benchmarks?

## 3. Candidate methods

- **Distributionally robust optimisation (DRO) ranking**: minimise worst-source ranking loss over a source uncertainty set.
- **Set-valued labels**: treat each item's label as an interval or set across sources and define conservative dominance relations.
- **Abstention**: allow the model to decline ranking items with high cross-source disagreement.
- **Rank aggregation with source reliability**: learn source weights from held-out-source validation.
- **Deterministic robust portfolios**: explicit two-source selection rules (maximin, balanced union, disagreement abstention) with finite-sample regret bounds.

## 4. Theoretical propositions

1. **Rank-correlation upper bound**: for any deterministic ranking, worst-source NDCG@k is bounded above by a monotone function of source-source Kendall tau and the elite fraction q*.
2. **Abstention guarantee**: under a bounded-disagreement model, there exists an abstention rule that achieves a target worst-source recall at a quantifiable coverage cost.
3. **DRO generalisation**: DRO ranking generalises across sources if the source distribution shift is bounded in Wasserstein or Jensen-Shannon divergence.
4. **Portfolio minimax regret**: for two sources, the greedy maximin-oracle portfolio has minimax recall regret no larger than the single-source oracle that ignores the other source.

## 5. Multi-source benchmarks (at least three)

1. **CrossPiezo** (piezoelectric response, JARVIS vs Materials Project) — current project, frozen matched-pair panels P0/P2.
2. **MatBench-Dielectric / MP + AFLOW + OQMD elastic moduli** — scalar property, three independent high-throughput sources.
3. **Perovskite band gaps** from different DFT functionals and high-throughput experiments — mixed computational/experimental source labels.

## 6. Baselines

- Single-source ranker trained on pooled data.
- Borda / Copeland rank aggregation.
- Domain-adversarial or source-conditioned neural ranker.
- Conservative maximin selection.
- Oracle single-source upper bound (for regret calibration only).

## 7. Held-out-source protocol

- Train on a subset of sources; validate on a held-out source.
- Report worst-source and average-source Recall@k, NDCG@k, and minimax regret.
- Require that improvements are not due to test-set leakage from shared upstream data.
- CrossPiezo-specific: deterministic grouped split by hashed pair_id; disagreement-abstention lambda tuned on dev and evaluated on a frozen holdout.

## 8. Computational budget

- Data scale: 1k-10k matched items per benchmark.
- Model scale: light-weight neural ranker or gradient-boosted ranker; no large language models required.
- Compute: CPU/GPU training within a few GPU-days per benchmark.

## 9. Method Go / No-Go risks

**Go**: Theoretical bounds are non-vacuous, worst-source gains exceed 10% over single-source baselines on at least two benchmarks, and abstention meaningfully reduces regret while preserving coverage.

**No-Go**: Disagreement is dominated by irreducible convention errors, worst-source performance cannot be improved without oracle source labels, or benchmarks lack held-out-source independence.
