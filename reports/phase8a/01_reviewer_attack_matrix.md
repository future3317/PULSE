# Phase 8A Reviewer Attack Matrix

This matrix simulates three reviewer archetypes, lists major and minor concerns, assesses whether the current evidence resolves each concern, and gives the precise location in the v0.7 manuscript where a fix or clarification is needed.

## A. Materials / computational reviewer

### Major concerns

| # | Concern | Resolved by current evidence? | Exact fix location in v0.7 |
|---|---------|------------------------------|----------------------------|
| 1 | **DFPT convention alignment.** MP and JARVIS may use different Voigt orderings, unit conventions, or stress-vs-strain tensor definitions. The observed disagreement could partly reflect unaligned conventions rather than genuine workflow sensitivity. | Partially. Coordinate-invariant scalars (F1/F3/F4) reduce but do not eliminate convention dependence. MP plain Voigt SVD is only used for source-field audit. | Methods §2.1: add explicit statement that invariant scalars are necessary but not sufficient for convention alignment; Discussion §4: note convention alignment as a limitation and future work. |
| 2 | **Only two sources.** The paper generalises about "cross-database" agreement, but only MP and JARVIS are compared. The conclusions may be specific to these two databases. | Not resolved. The manuscript already avoids generalisation claims, but the framing could be tightened. | Title / Abstract / Introduction: replace "cross-database" with "MP and JARVIS" or "two high-throughput sources"; Discussion §4: state explicitly that results are source-pair specific. |
| 3 | **Small elite-tail samples.** P0 n=573 and P2 n=207; at q=1% there are only ~6 (P0) or ~2 (P2) materials. Power to detect elite-tail agreement is low. | Partially. The paper reports this as evidence of weak elite-tail resolution, but the sample-size limitation should be more prominent. | Results §3.1: add effective sample sizes per quantile; Discussion §4: state that elite-tail conclusions are lower-bound statements limited by sample size. |
| 4 | **Physical relevance of F3.** F3 is the true collinear longitudinal maximum. Its relevance to polycrystalline ceramic or thin-film screening is not justified; F1/F4 may be more appropriate for different applications. | Not resolved. The manuscript motivates F1/F4 but says little about F3 use-case. | Methods §2.1: add one sentence on when F3 is physically meaningful (single-crystal longitudinal actuation) and when F1/F4 are preferred. |
| 5 | **No physical adjudication.** Claims about "workflow sensitivity" are inferred from cross-source disagreement, but without a third protocol or experiment the source of disagreement (convention vs physics vs numerical error) cannot be assigned. | Not resolved. The manuscript already acknowledges this, but the limitation wording can be strengthened. | Discussion §4: lead with the statement that cross-source disagreement identifies sensitivity but does not identify the cause; Conclusion §6: repeat that independent validation is required. |

### Minor concerns

| # | Concern | Resolved? | Fix location |
|---|---------|-----------|--------------|
| 1 | **Calculation version details.** Main text does not state MP/JARVIS calculation versions; only Table 5 gives a high-level provenance audit. | Partially. Table 5 exists but is sparse. | Methods §2.4: add MP and JARVIS database versions and calculation schemas; Table 5: expand with version strings if available. |
| 2 | **Energy above hull as control.** It is flagged as same-field copy (>5% identical values); including it as a control may confuse readers. | Partially. The flag is disclosed, but the rationale for inclusion could be clearer. | Methods §2.4: state that energy_above_hull is retained as a well-understood negative control despite the copy flag because zero hull values are physically expected; Table 5: keep flag. |
| 3 | **Dielectric trace consistency.** Dielectric trace is also a DFPT tensor scalar but is far more cross-source consistent than piezoelectric response. Why? | Partially. The manuscript notes higher workflow sensitivity for piezo but does not explain dielectric specifically. | Discussion §4: add a sentence that dielectric is a second-order property with typically smoother k-space convergence and fewer convention ambiguities than piezoelectric stress tensors. |
| 4 | **Crystal-system coverage.** P0/P2 composition/crystal-system balance is not described; elite-tail materials may be dominated by a few systems. | Not resolved. | Results §3.1 or Methods §2.1: add a supplementary table or sentence on crystal-system and composition distribution of P0/P2. |
| 5 | **Third protocol timeline.** The 48-material protocol is pre-registered but not executed; reviewers may ask when it will be done. | Not resolved. | Discussion §5: add a sentence on pre-registered status and estimated timeline, or state that the current paper does not depend on it. |

---

## B. Statistics / machine-learning reviewer

### Major concerns

| # | Concern | Resolved by current evidence? | Exact fix location in v0.7 |
|---|---------|------------------------------|----------------------------|
| 1 | **Multiple comparisons.** The paper tests many metrics, bands, panels, and strategies without a familywise error correction. Some "significant" bands may be false positives. | Partially. Controls use BH FDR; screening-resolution bands and portfolio CIs do not. The claims are mostly descriptive, not hypothesis-driven. | Methods §2.3 and §2.5: explicitly state that CIs are simultaneous for the screening curve but that exploratory band/portfolio comparisons are not corrected for multiple comparisons; Results: frame banded nAUCC as descriptive, not inferential. |
| 2 | **Hypergeometric null independence assumption.** The chance-adjusted overlap uses a hypergeometric null that assumes independent rankings. Because MP and JARVIS compute the same physical materials, rankings may be correlated under any null. | Partially. The null tests whether top-q sets are independent given the shared universe, which is the standard chance baseline. | Methods §2.3: add a sentence defending the hypergeometric null as a chance baseline conditional on the shared universe; Discussion §4: note that any positive correlation under the null would make observed agreement look stronger, so weak agreement is a conservative finding. |
| 3 | **Grouped bootstrap justification.** Portfolio paired differences use grouping by `reduced_formula`. The choice of grouping variable is not validated; other groupings (crystal system, prototype) could change CI width. | Not resolved. | Methods §2.5: justify `reduced_formula` grouping as the strongest known structure-composition cluster; Supplementary: include a sensitivity table with alternative groupings if feasible. |
| 4 | **nAUCC sampling distribution.** nAUCC integrates autocorrelated quantile estimates. The manuscript reports point estimates and simultaneous bands but does not characterise the sampling distribution of the area itself. | Partially. The bands are simultaneous, but the area is not given a CI. | Methods §2.3: state that nAUCC is a summary index and that uncertainty is conveyed via simultaneous bands on the underlying curve, not via an area CI. |
| 5 | **Composition-blocked CV leakage.** Grouped 5-fold CV splits by `reduced_formula`, but different structure entries or DFT relaxations of the same formula may still leak information across folds, inflating CV gains. | Partially. The grouping reduces but may not eliminate leakage. | Methods §2.5: explicitly define the fold construction and state that leakage could remain across different structural polymorphs of the same formula; Results §3.4: report CV gains as an optimistic upper bound relative to full-panel evaluation. |

### Minor concerns

| # | Concern | Resolved? | Fix location |
|---|---------|-----------|--------------|
| 1 | **Bootstrap replicate count.** 2000 replicates is standard but no sensitivity to replicate count is shown. | Resolved by convention. | Methods §2.3 / §2.5: keep 2000 and cite Efron & Tibshirani; optional supplementary note. |
| 2 | **Random seed sensitivity.** All random processes are fixed to seeds, but no seed-sensitivity analysis is reported. | Resolved by reproducibility requirement. | Methods §2: state seeds are fixed for reproducibility; no seed-sensitivity sweep is claimed. |
| 3 | **Portfolio regret distribution.** Table 4 reports only worst-source recall and NDCG; the distribution of regret across sources/materials is hidden. | Not resolved. | Results §3.4 / Supplementary: add a regret distribution figure or table (e.g., histogram of per-material regret). |
| 4 | **Confidence band calibration.** No simulation confirms that the simultaneous 95% bands have nominal coverage. | Not resolved. | Methods §2.3: state bands are constructed by the studentized sup-norm bootstrap; Discussion §4: note coverage is asymptotic and finite-sample calibration is a limitation. |
| 5 | **Missing data exclusion.** The number of materials excluded because of missing tensors or failed matches is not stated in the main text. | Partially. Handoff mentions processed records and overlap, but main text does not. | Methods §2.1: add counts of processed records, preliminary overlap, P0, and P2. |

---

## C. Data and provenance reviewer

### Major concerns

| # | Concern | Resolved by current evidence? | Exact fix location in v0.7 |
|---|---------|------------------------------|----------------------------|
| 1 | **Structure-matching reproducibility.** P0/P2 panel membership is frozen, but the exact structure-matching protocol (tolerances, symmetrization, space-group handling) is not described in enough detail to reproduce. | Partially. Phase 6A/6B reports exist but are not referenced. | Methods §2.1: add a paragraph summarising the structure-matching protocol and cite the frozen Phase 6A report; Data Availability: provide panel membership file path and checksum. |
| 2 | **Per-entry provenance.** Table 5 gives aggregate provenance flags but not per-entry source fields, calculation IDs, or relaxation parameters. | Not resolved. | Data Availability §5: state that per-entry provenance records are included in the repository as `results/phase7c/control_provenance.csv`; Table 5: refer readers to the CSV for full per-entry records. |
| 3 | **Correlated errors beyond copy flags.** The same-field-copy flag only catches identical values >5%. MP and JARVIS may share methodology, pseudopotentials, or upstream data, producing correlated non-identical errors. | Not resolved. | Discussion §4: acknowledge that the copy flag is a minimal check and cannot rule out methodological correlation; Conclusion §6: call for independent third-protocol validation. |
| 4 | **Third-protocol selection bias.** The 48-material strata select consensus elite, JARVIS-only elite, MP-only elite, etc. Because the strata are defined using the same two sources being evaluated, the adjudication may be circular unless the third protocol is fully independent. | Partially. The strata are designed to test disagreement, but the pre-registration must be clear about independence. | Methods / Discussion: explicitly state that the third protocol uses independently generated inputs (relaxed structures, k-grids, pseudopotentials) and that strata are chosen to maximise diagnostic coverage, not to favour either source. |
| 5 | **Long-term artifact availability.** Hash-bound artifacts are in the repository, but no DOI or permanent archive is referenced. Git history can be rewritten and binary artifacts may not be permanent. | Not resolved. | Data Availability §5: add a statement that a Zenodo/Figshare archive with the frozen manifest will be created upon acceptance; include the manifest hash in the paper. |

### Minor concerns

| # | Concern | Resolved? | Fix location |
|---|---------|-----------|--------------|
| 1 | **Manifest timestamp only in UTC.** Local execution provenance (machine, timezone, user) is not recorded. | Resolved by commit hash. | No fix needed; commit hash provides reproducibility anchor. |
| 2 | **No code version tag.** Manifest records commit hash but not a human-readable release tag. | Resolved by commit hash. | Data Availability §5: mention that the commit hash in the manifest uniquely identifies the code state. |
| 3 | **Repository may be private at submission.** "Released in the CrossPiezo repository" is vague if the repo is not yet public. | Not resolved. | Data Availability §5: add a link/DOI placeholder or state that the repository will be made public on submission/preprint. |
| 4 | **Panel membership checksum absent from Phase 7C manifest.** The manifest hashes Phase 7C outputs but does not hash the frozen `artifacts/phase6a/panels/panel_membership.parquet`. | Not resolved. | Methods §2.1: state the panel membership path and its SHA-256; ideally add it to the Phase 7C manifest in v0.7 (without invalidating v0.6). |
| 5 | **Third-protocol fail policy is untested.** Replacement rules for failed candidates are described but not exercised. | Resolved by pre-registration. | `configs/third_protocol_phase7c.yaml`: keep policy; Discussion §5: note that failures will be documented and quarantined per the pre-registered policy. |

---

## Cross-cutting recommendations for v0.7

1. **Tighten source-pair language.** Avoid "cross-database" as a general claim; use "MP and JARVIS" or "these two high-throughput sources."
2. **Lead with limitations.** The Discussion should open by stating that the study identifies workflow sensitivity but cannot assign its cause without a third protocol or experiment.
3. **Clarify inferential scope.** Banded nAUCC and full-panel portfolio differences are descriptive; only CV paired differences with CIs excluding zero are reported as statistically robust.
4. **Add sample-size context.** Report effective sample sizes per quantile and panel.
5. **Archive promise.** State that frozen artifacts will be archived with a DOI and that the manifest hash is included in the paper.
