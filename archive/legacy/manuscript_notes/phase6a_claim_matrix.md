# Phase 6A Claim Matrix

| Claim | Evidence | Location | Allowed |
|-------|----------|----------|---------|
| 8,316 tensor conversions verified | all-record conversion | reports/phase6a/* | Yes |
| F1/F3/F4 are rotation invariant | tests/ranking/test_rotation_invariance.py | pytest | Yes |
| F_MP_SVD is source-field only | metric definition audit + rotation test | reports/phase6a/01* | Yes |
| 573 strict matched pairs | panel membership parquet | artifacts/phase6a/panels/* | Yes |
| Cross-source ranking weakly consistent | tau, top-fraction overlap, chance-adjusted Jaccard | reports/phase6a/04* | Yes |
| Instability not near-zero driven | robustness checks | artifacts/phase6a/ranking/robustness_checks.json | Yes |
| Consensus/disputed candidates | robust screening | reports/phase6a/05* | Yes |
| MP version-shift not runnable locally | feasibility report | reports/phase6a/06* | Yes |
