# Phase 8A Venue Decision

## Executive summary

The current v0.7 manuscript is suitable for submission to **benchmark and materials-informatics journals**. For a stronger computational-methods venue, the pre-registered 48-material third-protocol adjudication should be executed first. A separate **data descriptor** route is also viable.

## Route 1: Submit the current manuscript (benchmark / materials informatics)

### Recommended venues

| Venue | Fit | Notes |
|-------|-----|-------|
| **npj Computational Materials** | Strong | Explicitly covers high-throughput methods, materials data mining, and ML/AI for materials design. The screening-resolution benchmark and portfolio analysis fit the scope well. | [Aims & Scope](https://www.nature.com/npjcompumats/aims) |
| **Digital Discovery** (RSC) | Strong | Publishes AI/ML and high-throughput computational methods for materials/molecular discovery, with strong data-availability requirements. | [Journal homepage](https://www.rsc.org/publishing/journals/digital-discovery) |
| **Computational Materials Science** | Good | Established venue for computational materials methodology and benchmark studies. Slightly lower impact but appropriate for a focused benchmark paper. | (Scope: computational materials science methods and benchmarks) |
| **Journal of Chemical Information and Modeling** | Good | Accepts benchmark and data-quality studies in chemical/materials informatics. | (Scope: chemical informatics, computational chemistry, data) |

### Why this route works now

- Claims are strictly two-source robustness: no physical-validation claims.
- All numbers are hash-bound and independently verified.
- The third protocol is pre-registered, satisfying reviewers who ask for an adjudication plan.
- The reviewer attack matrix (Phase 8A) has been used to tighten language and add limitations.

## Route 2: Stronger venue after executing the 48-material third protocol

### Recommended venues

| Venue | Why stronger | What the third protocol adds |
|-------|--------------|------------------------------|
| **npj Computational Materials** (with third-protocol results) | Same venue, stronger story | Independent computational validation transforms the robustness claim into a three-source adjudication, strengthening the ``limits of screening'' narrative. |
| **Nature Computational Science** | Higher tier | A clean computational-adjudication result (48 materials × independent DFT) is exactly the kind of benchmark that could be competitive here. | (Scope: computational science, methods, and benchmarks) |
| **Digital Discovery** (with third-protocol results) | Stronger within RSC portfolio | Independent protocol provides the ``experimental or independent computational validation'' that elevates the paper from a benchmark to an adjudication study. |

### Caution

- A top-tier materials-science journal (e.g., *Nature Materials*, *Advanced Materials*) is unlikely without **experimental** adjudication, because the manuscript does not claim discovery of new materials.
- The third protocol must be executed exactly as pre-registered; any post-hoc selection of materials would invalidate the adjudication.

## Route 3: Data-paper route

### Recommended venues

| Venue | Fit | Notes |
|-------|-----|-------|
| **Scientific Data** | Strong | Publishes Data Descriptors for scientifically valuable datasets. The frozen CrossPiezo panel, hash-bound CSV artifacts, and third-protocol pre-registration form a reusable dataset. | [Aims & scope](https://www.nature.com/sdata/aims-and-scope) |
| **Data in Brief** | Good | Shorter data descriptors, lower barrier, suitable for releasing the panel and benchmark outputs. | (Scope: datasets and data articles) |

### What would be required

- Deposit the frozen panel, manifest, and all CSV artifacts in a public repository (e.g., Zenodo/Figshare) before submission.
- Reformat the manuscript to the Data Descriptor structure: Background & Summary, Methods, Data Records, Technical Validation, Usage Notes, Code Availability.
- Avoid hypothesis-testing language; focus on dataset description and reuse.

## Recommendation

**Primary:** Submit v0.7 to **npj Computational Materials** or **Digital Discovery** as a benchmark/material-informatics article.

**Conditional:** If the 48-material third protocol is executed within the next 4--8 weeks, revise to a stronger adjudication narrative and target **Nature Computational Science** or **npj Computational Materials** with the additional results.

**Parallel:** Prepare a **Scientific Data** Data Descriptor for the frozen benchmark artifacts, to be submitted either before or alongside the primary article.

## Sources

- npj Computational Materials aims & scope: [https://www.nature.com/npjcompumats/aims](https://www.nature.com/npjcompumats/aims)
- npj Computational Materials submission guidelines: [https://www.nature.com/npjcompumats/for-authors-and-referees/submission-guidelines](https://www.nature.com/npjcompumats/for-authors-and-referees/submission-guidelines)
- Digital Discovery (RSC): [https://www.rsc.org/publishing/journals/digital-discovery](https://www.rsc.org/publishing/journals/digital-discovery)
- Digital Discovery author guidelines: [https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery)
- Scientific Data aims & scope: [https://www.nature.com/sdata/aims-and-scope](https://www.nature.com/sdata/aims-and-scope)
- Scientific Data author instructions: [https://www.nature.com/sdata/author-instructions](https://www.nature.com/sdata/author-instructions)
