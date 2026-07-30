# Phase 6A Manuscript Outline

## Title
CrossPiezo: A Provenance-Audited Benchmark Reveals Unstable Piezoelectric Screening Across Materials Databases

## Abstract
We audit 8,316 processed piezoelectric tensor records from JARVIS and the Materials Project (MP), identify 573 strict structure-matched pairs, and show that coordinate-invariant response rankings are only weakly consistent across sources. The instability is not driven by near-zero noise and persists in high-response subsets. We release the CrossPiezo-Invariant-v1 benchmark with explicit metric definitions, nested panels, and observed cross-database intervals.

## Main sections
1. Data lineage and conversion audit (8,316 records).
2. Metric definitions: F1, F_MP_SVD, F3, F4.
3. Strict structure-matched invariant panel (P0-P3).
4. Cross-source ranking and threshold stability.
5. Robust screening and observed intervals.
6. Limitations and next steps (MP version shift).

## Allowed claims
- 8,316 processed tensor records have consistent trusted/project/stored conversion.
- On 573 structure-matched materials, JARVIS and MP coordinate-invariant rankings are weakly consistent.
- Database definitions and label provenance need explicit versioning.
- Single-database high-response candidates are not automatically cross-database robust.

## Forbidden claims
- Protocol uncertainty floor; real tensors; componentwise disagreement; soft-mode mechanism; PULSE calibration; PMR > 1; one database is more accurate.

## Commit
44328ef610190bbd6d84e1d1873cadd4b99e054d
