# Phase 8A Portfolio Claim Freeze

> **Superseded:** this historical freeze predates the exact minimax-selector
> correction. Use results/phase7c/portfolio_benchmark.csv and
> reports/phase7c/04_portfolio.md for current values.

> Source: `results/phase7c/portfolio_benchmark.csv` and `results/phase7c/portfolio_paired_differences.csv`  
> Scope: P0/P2 × F1/F3/F4, equal budget (`budget_factor = 1.0`)

## 1. Decision

The equal-budget primary claim is frozen as:

> **The deterministic `balanced_union` strategy achieves the highest or tied-highest worst-source recall across all panel/metric combinations, and in grouped 5-fold cross-evaluation it produces large, statistically significant improvements in worst-source recall relative to the better single-source baseline.**

This is a **two-source robustness** claim, not physical validation.

## 2. Full-panel worst-source recall at budget = 1.0

| Panel | Metric | Best deterministic strategy | Worst-source recall | Notes |
|-------|--------|----------------------------|---------------------|-------|
| P0 | F1_Frobenius | `balanced_union` | **0.526** | Next best: `minimax_oracle` 0.474 (unachievable) |
| P0 | F3_Longitudinal | `balanced_union` | **0.509** | Next best: consensus strategies 0.439 |
| P0 | F4_KelvinOp | `balanced_union` | **0.579** | Next best: consensus strategies 0.456 |
| P2 | F1_Frobenius | `balanced_union` (tied) | **0.450** | Tied with `minimax_oracle` 0.450 |
| P2 | F3_Longitudinal | `balanced_union` | **0.500** | Next best: consensus strategies 0.400 |
| P2 | F4_KelvinOp | `balanced_union` (tied) | **0.450** | Tied with `minimax_oracle` 0.450 |

`balanced_union` is the practical best strategy in every case.

## 3. Full-panel paired difference vs better single-source baseline

Full-panel differences are modest and their 95% CIs include zero. They should be read as decision examples, not as proof of large method gains.

| Panel | Metric | Strategy | Paired diff recall | 95% CI |
|-------|--------|----------|--------------------|--------|
| P0 | F1_Frobenius | `balanced_union` | +0.012 | [-0.012, 0.036] |
| P0 | F3_Longitudinal | `balanced_union` | +0.014 | [-0.012, 0.038] |
| P0 | F4_KelvinOp | `balanced_union` | +0.014 | [-0.008, 0.038] |
| P2 | F1_Frobenius | `balanced_union` | +0.021 | [-0.016, 0.059] |
| P2 | F3_Longitudinal | `balanced_union` | +0.021 | [-0.016, 0.059] |
| P2 | F4_KelvinOp | `balanced_union` | +0.021 | [-0.021, 0.064] |

## 4. Grouped 5-fold CV paired difference vs better single-source baseline

CV differences are larger and their 95% CIs exclude zero, reflecting variance reduction from grouping by composition.

| Panel | Metric | Strategy | Paired diff recall | 95% CI |
|-------|--------|----------|--------------------|--------|
| P0 | F1_Frobenius | `balanced_union` | +0.342 | [0.284, 0.389] |
| P0 | F3_Longitudinal | `balanced_union` | +0.364 | [0.295, 0.422] |
| P0 | F4_KelvinOp | `balanced_union` | +0.357 | [0.329, 0.389] |
| P2 | F1_Frobenius | `balanced_union` | +0.307 | [0.230, 0.407] |
| P2 | F3_Longitudinal | `balanced_union` | +0.347 | [0.267, 0.430] |
| P2 | F4_KelvinOp | `balanced_union` (tied) | +0.397 | [0.313, 0.480] |

## 5. Budget = 2.0 coverage upper bound

`balanced_union` at `budget_factor = 2.0` achieves worst-source recall **1.000** by construction (it contains the union of the two source top-$q^*$ sets). This is a coverage upper bound, not an algorithmic gain, and must be labelled as such.

## 6. Not-a-universal-winner check

A single strategy is not universally best on every metric:

- `minimax_oracle` ties or beats `balanced_union` in P2 F1 and P2 F4, but it is an oracle and not implementable.
- In full-panel paired differences, some single-source or consensus strategies have marginally higher point estimates than `balanced_union` in P2, but differences are tiny and confidence intervals overlap heavily.
- No full-panel paired difference excludes zero.

Nevertheless, `balanced_union` is the strongest deterministic strategy on the primary metric (worst-source recall) across the entire P0/P2 × F1/F3/F4 grid. Therefore the frozen claim uses `balanced_union` as the primary example, while noting that gains are modest in full-panel evaluation and larger in grouped CV.

## 7. Allowed wording

- "Equal-budget balanced-union improves worst-source recall relative to JARVIS-only or MP-only selection."
- "In grouped 5-fold CV, balanced-union improves worst-source recall by +0.30 to +0.40 with 95% CIs excluding zero."
- "Balanced-union at twice the elite budget is a coverage upper bound with recall 1.0 by construction."

## 8. Forbidden wording

- "balanced-union validates physical truth"
- "two-source consensus is experimentally validated"
- "portfolio proves MP/JARVIS correctness"
- any claim that the full-panel point estimates are statistically significant
