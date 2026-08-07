# CCF-A Method Paper Concept: Source-Robust Ranking under Conflicting Scientific Labels

## 1. General problem

Multiple independent computational protocols, database versions, or experimental assays can assign conflicting labels or scores to the same material.  Existing ranking methods optimize agreement with a single source and are not guaranteed to produce candidates that are robust across sources.  The generic task is:

> Given noisy, source-conditional score vectors for a shared set of items, learn a ranking or selection rule whose top-k performance is stable under the worst source.

## 2. Candidate methods

- **Distributionally robust optimization (DRO) ranking**: minimize worst-source ranking loss over a source uncertainty set.
- **Set-valued labels**: treat each item's label as an interval or set across sources and define conservative dominance relations.
- **Abstention**: allow the model to decline ranking items with high cross-source disagreement.
- **Rank aggregation with source reliability**: learn source weights from held-out-source validation.

## 3. Provable propositions

1. Worst-source NDCG of any deterministic ranking is bounded above by a function of source-source Kendall tau.
2. Under a bounded-disagreement model, there exists an abstention rule that guarantees a target worst-source recall.
3. DRO ranking generalizes across sources if the source distribution shift is bounded in Wasserstein/JS divergence.

## 4. At least three multi-source benchmarks

1. **CrossPiezo** (piezoelectric response, JARVIS vs MP) — current project.
2. **MatBench-Dielectric** or **MP + AFLOW + OQMD** elastic moduli — scalar property, three sources.
3. **Perovskite band gaps** from different DFT functionals or high-throughput experiments — small-molecule / inorganic mixed source.

## 5. Baselines

- Single-source ranker trained on pooled data.
- Borda / Copeland rank aggregation.
- Domain-adversarial or source-conditioned neural ranker.
- Conservative maximin selection.

## 6. Held-out-source protocol

- Train on a subset of sources; validate on a held-out source.
- Report worst-source and average-source Recall@k, NDCG@k, and rank regret.
- Require that improvements are not due to test-set leakage from shared upstream data.

## 7. Computational budget

- Data scale: 1k-10k matched items per benchmark.
- Model scale: light-weight neural ranker or gradient-boosted ranker; no large language models required.
- Compute: CPU/GPU training within a few GPU-days per benchmark.

## 8. Go/No-Go risks

- **Go**: Theoretical bounds are non-vacuous, worst-source gains exceed 10\% over single-source baselines on at least two benchmarks, and abstention meaningfully reduces regret.
- **No-Go**: Disagreement is dominated by irreducible convention errors, worst-source performance cannot be improved without oracle source labels, or benchmarks lack held-out-source independence.
