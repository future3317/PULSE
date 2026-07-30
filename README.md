# CrossPiezo / PULSE — Phase 0-4 Audit

This repository contains the first-round feasibility audit for the CrossPiezo
cross-protocol piezoelectric benchmark and the PULSE (Protocol-Uncertainty
Learning for Symmetry-Equivariant tensors) proposal.

## Scope

The audit covers Phase 0–4 of `CROSSPIEZO_TASKBOOK.md`:

1. Environment, manuscript contract, and claim-boundary extraction.
2. Read-only inventory of `E:/DATA` piezoelectric assets.
3. Tensor convention validation (Voigt/Cartesian, symmetry projection, O(3) covariance).
4. Strict JARVIS–MP structure matching and discrepancy atlas.
5. Feasibility metrics and a frozen Go / Narrow / No-Go decision.

It explicitly does **not** train PULSE, run large-scale SOTA reproductions,
download data, or modify LaTeX result placeholders.

## Quick start

```bash
# Using the local EGNN conda environment (Python 3.11, dependencies already present)
conda activate EGNN
pip install -e .
python scripts/run_phase0_4.py
```

After running, inspect:

- `reports/00_environment_and_scope.md`
- `reports/01_data_inventory.md`
- `reports/02_convention_audit.md`
- `reports/03_pairing_audit.md`
- `reports/04_feasibility_results.md`
- `reports/05_go_no_go.md`

## Tests

```bash
pytest tests/
```

## Project rules

See `CLAUDE.md` and `CROSSPIEZO_TASKBOOK.md`.
