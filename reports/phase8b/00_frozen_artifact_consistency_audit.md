# Phase 8B Frozen Artifact Consistency Audit

> **Historical audit:** this report records the pre-correction artifact state.
> Current results are in reports/phase7c/04_portfolio.md.

> Auditor: Claude Code  
> Scope: `CrossPiezo_ScreeningResolution_Manuscript.tex`, `CrossPiezo_ScreeningResolution_Supplementary.tex`, frozen Phase 7C artifacts in `results/phase7c/`, and the `results/phase8a/manuscript_numbers.json` reconciliation.  
> Baseline commit: `90d1ca1b06ecea6cdef82f8d3ae454dafb3ccc7d`  

## 1. Audit method

For each numerical claim in the main text and Supplementary Table S1–S4 we trace the value to a frozen CSV/JSON artifact and, where possible, recompute it from rawer columns.  
Evidence priority: **CSV/JSON artifact > manifest hash > code > report prose > PDF snapshot**.

Commands run:

```bash
python scripts/verify_phase7c.py          # reconciled, max diff 1.11e-16, 0 flags
pytest -q                                 # 98 passed, 1 skipped
pdflatex -> bibtex -> pdflatex -> pdflatex # main manuscript compiled, no undefined citations
```

## 2. Main-text claim traceability

| Claim in manuscript | Artifact | Row/field | Value | Status |
|---|---|---|---|---|
| P0 $n=573$, P2 $n=207$ | `artifacts/phase6a/panels/panel_membership.parquet` | panel counts | 573 / 207 | OK |
| P0 F1 full nAUCC 0.105 | `results/phase7c/concordance_summary.csv` | P0, F1_Frobenius, `nAUCC` | 0.10507 | OK |
| P0 F3 full nAUCC 0.110 | `results/phase7c/concordance_summary.csv` | P0, F3_Longitudinal, `nAUCC` | 0.11016 | OK |
| P0 F4 full nAUCC 0.106 | `results/phase7c/concordance_summary.csv` | P0, F4_KelvinOp, `nAUCC` | 0.10572 | OK |
| P2 F1 full nAUCC 0.060 | `results/phase7c/concordance_summary.csv` | P2, F1_Frobenius, `nAUCC` | 0.06023 | OK |
| P0 F1 elite partial nAUCC 0.003 | `results/phase7c/concordance_bands.csv` | P0, F1_Frobenius, elite, `partial_nAUCC` | 0.00301 | OK |
| P0 F1 intermediate partial nAUCC 0.076 | `results/phase7c/concordance_bands.csv` | P0, F1_Frobenius, intermediate, `partial_nAUCC` | 0.07586 | OK |
| P0 F1 broad partial nAUCC 0.145 | `results/phase7c/concordance_bands.csv` | P0, F1_Frobenius, broad, `partial_nAUCC` | 0.14543 | OK |
| P2 elite/intermediate/broad values | `results/phase7c/concordance_bands.csv` | as above | match | OK |
| P0 F1 dual-high $f=0.10$, $N=58$, nAUCC $=-0.046$ | `results/phase7c/high_response_sensitivity.csv` | P0, F1_Frobenius, 0.1, dual_high | 58 / -0.04580 | OK |
| P0 volume $\Delta\tau=0.722$, $\Delta$nAUCC $=0.813$ | `results/phase7c/property_controls.csv` | P0, volume | 0.72197 / 0.81292 | OK |
| P0 band gap $\Delta\tau=0.663$, $\Delta$nAUCC $=0.792$ | `results/phase7c/property_controls.csv` | P0, band_gap | 0.66282 / 0.79198 | OK |
| Primary portfolio best = balanced union | `results/phase7c/portfolio_benchmark.csv` | P0, F1, full_panel, $b=1.0$ | recall 0.526 | **MISMATCH with paired diff** |
| Balanced-union paired diff $+0.012$ | `results/phase7c/portfolio_benchmark.csv` | P0, F1, full_panel, $b=1.0$ | 0.01210 | **INCONSISTENT** |

All screening-resolution and control claims reconcile to machine precision.  The **portfolio section fails consistency** (see §3).

## 3. Critical finding: portfolio paired-diff / recall mismatch

### 3.1 What the table shows

From `results/phase7c/portfolio_benchmark.csv`, P0 / F1_Frobenius / full_panel / $b=1.0$:

| Strategy | `worst_source_recall` | `paired_diff_recall` | 95% CI |
|---|---:|---:|---|
| jarvis_only | 0.14035 | 0.0 | — |
| mp_only | 0.14035 | 0.01411 | [-0.022, 0.050] |
| average_percentile | 0.45614 | 0.01411 | [-0.012, 0.042] |
| balanced_union | **0.52632** | **0.01210** | [-0.012, 0.036] |

### 3.2 Why the numbers do not match

`worst_source_recall` is a **material-level (micro) metric**: it counts how many of the top-$q$ materials are captured by the portfolio, divided by $q^* n$, with no group weighting.

`paired_diff_recall` is computed in `scripts/run_phase7c.py:_paired_diff_ci_vs_baseline` by first computing **group-level worst-source recalls** (`src/crosspiezo/analysis/phase7c_stats.py:_group_level_worst_recall`), i.e. one recall value per reduced formula, and then taking the mean difference across groups.  Because each group contains only a handful of materials, the group-mean differences are small and the baseline is often already competitive within each group.  The result is a paired difference of ~0.01 even though the material-level recall improves by ~0.39.

This is an **estimand mismatch**, not a coding bug in the sense of an inverted sign or wrong column.  But it makes the table internally contradictory: a reader naturally expects `paired_diff_recall` to equal `worst_source_recall(strategy) − worst_source_recall(baseline)`.

### 3.3 Corrected material-level differences

Using the same `worst_source_recall` estimand for both columns gives:

| Strategy | Material-level $\Delta$Recall vs jarvis_only |
|---|---:|
| average_percentile | 0.31579 |
| balanced_union | **0.38596** |

These are not tiny exploratory gains; they reflect that rank aggregation genuinely covers more of the elite set in absolute terms.  The confidence intervals must be recomputed for this estimand before any claim is made.

### 3.4 Recommendation

Until the paired difference is recomputed with the same estimand as the reported recall, the portfolio section should not be presented as a validated decision result.  Two safe options:

1. **Recompute** `paired_diff_recall` and its grouped paired-bootstrap CI using the material-level `worst_source_recall` difference, and update the manuscript/Supplementary accordingly.  This will give a positive, likely significant point estimate.
2. **Downgrade** the portfolio analysis to an exploratory Supplementary discussion, remove it from the abstract and main-text Results, and explicitly state that the full-panel comparison is descriptive and not a paired statistical test.

Option 2 is the conservative path if a quick fix is not desired; Option 1 is required if the main text continues to claim that aggregation "changes the coverage–quality trade-off".

## 4. Global accuracy narrative mismatch

### 4.1 What the data actually show

Supplementary Table S2 allows the F1 Kendall $\tau$ to be backed out:

- Volume $\tau \approx 0.967$, $\Delta\tau = 0.722$  →  F1 $\tau \approx 0.245$.
- Band gap $\tau \approx 0.908$, $\Delta\tau = 0.663$  →  F1 $\tau \approx 0.245$.

The F1 global cross-source rank correlation is therefore already low (~0.25).  The elite-tail gap is a further degradation of an already modest global agreement, not a hidden failure inside an otherwise high-accuracy benchmark.

### 4.2 Unsupported claims in current text

The manuscript uses phrases such as:

- "aggregate accuracy"
- "global model accuracy"
- "a model can achieve strong aggregate accuracy"
- "aggregate accuracy can overstate the reproducible resolution"

No ML model is trained or tested in this work.  The evidence supports only statements about **global rank correlation / aggregate agreement** between two independently generated high-throughput databases.  Claims about "model accuracy" are out of scope and should be removed or replaced with "global correlation", "rank agreement", or "aggregate concordance".

## 5. Supplementary gaps (blocking for high-impact venues)

The current Supplementary is 3 pages and lacks the information needed to reproduce or adjudicate the findings:

- Exact structure-matching thresholds and all filtering rules for P0/P2.
- Manual audit scope, excluded cases, and ambiguous matches.
- Full provenance of MP/JARVIS fields: units, proper/improper tensor, clamped/relaxed-ion contributions, code versions.
- Voigt → full Cartesian tensor conversion formula and Kelvin/Mandel convention.
- F3 numerical optimization method, convergence tolerances, and independent validation.
- Handling of ties, rounding of $k$, and duplicate/reduced-formula sets in top-$q$ selection.
- Bootstrap replicates, seed, studentization details, and coverage simulations for small panels.
- Sensitivity of persistent-onset to $\delta$ and consecutive-quantile length.
- Formal definitions of all portfolio strategies.
- Complete candidate lists, match scores, and audit decisions.

Without these, a reviewer can reasonably argue that the observed disagreement is an artifact of field misalignment rather than meaningful workflow sensitivity.

## 6. Stop/go decision

**Current status: Not Ready.**

The hard stop is the portfolio estimand mismatch.  The narrative overreach on "accuracy" and the thin Supplementary are also high risks for a journal like *npj Computational Materials* or *Digital Discovery*.

Minimum actions before any submission:

1. Resolve the portfolio paired-diff/recall inconsistency (Option 1 or 2 above).
2. Replace "accuracy" language with "correlation/agreement/concordance" throughout.
3. Expand Supplementary with the missing provenance and methodological details.
4. Recompile and re-run `scripts/verify_phase7c.py` + `pytest`.
5. Produce a new `reports/phase8b/01_submission_readiness_decision.md` with a clear `Submission Ready` / `Not Ready` verdict.

## 7. Traceability checksum

All values above were extracted from the frozen files whose SHA-256 hashes are recorded in `results/phase7c/phase7c_manifest.json`.  No hand-entered numbers appear in this report.
