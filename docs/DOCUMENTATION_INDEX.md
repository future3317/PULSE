# CrossPiezo documentation index

This index separates live instructions from evidence and historical material.
It is the preferred navigation page for future agents and collaborators.

## Live entry points

| Document | Purpose |
|---|---|
| `README.md` | Repository entry point and minimal commands |
| `CLAUDE.md` | Agent rules and safe operating constraints |
| `CROSSPIEZO_TASKBOOK.md` | Stable compatibility pointer to the live guide |
| `docs/PROJECT_GUIDE.md` | Current scope, source of truth, workflow, and blockers |
| `reports/CURRENT_STATUS.md` | Current verification and readiness snapshot |
| `reports/open_questions.md` | Only unresolved decisions that still affect the active line |
| `reports/README.md` | Report directory map |

## Scientific and reproducibility references

| Document | Purpose |
|---|---|
| `docs/claim_boundary.md` | Allowed and forbidden scientific claims |
| `docs/data_contract.md` | Source-artifact and read-only data rules |
| `docs/matching_protocol.md` | Structure matching and tier definitions |
| `docs/tensor_conventions.md` | Internal tensor representation and transformations |
| `docs/mp_vs_jarvis_conventions.md` | Known and unknown source-workflow conventions |
| `docs/statistical_plan.md` | Current screening-resolution and bootstrap estimands |

## Current evidence

- `reports/phase7c/`: frozen Phase 7C work-package reports.
- `reports/phase8a/`: takeover, reviewer, portfolio, and third-protocol audits.
- `reports/phase8b/`: artifact consistency and submission-readiness decisions.
- `reports/phase9/`: generated statistical-upgrade report and deferred
  third-protocol preflight result.
- `submission/phase8a_venue_decision.md`: venue route recommendation.
- `archive/phase8a_submission/`: frozen reproducibility bundle.

## Historical material

- `archive/legacy/root_phase_plans/`: superseded Phase 5–8 plans and audit
  taskbooks.
- `archive/legacy/manuscript_notes/`: superseded manuscript planning notes.
- `archive/legacy/live_status/`: obsolete status and question snapshots.
- `archive/legacy/future_work/`: superseded general method concepts retained
  as future-work provenance.
- `reports/` also contains older generated phase reports because their paths
  are used by historical runners and they preserve scientific provenance.

The current numeric split is intentional: frozen Phase 7C point estimates live
under `results/phase7c/`, while the post-Version-A statistical upgrade lives
under `results/phase9/`. The latter does not overwrite the frozen layer.

## Documentation maintenance

Live documents do not need manual dates. Generated reports may include the
runtime date and commit that produced them; those fields are provenance, not a
manual checklist. A new kickoff prompt or parallel handoff should not be added
for routine iterations.
