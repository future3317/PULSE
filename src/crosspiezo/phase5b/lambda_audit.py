"""Audit PiezoJet strict-factor cache for full atom-resolved internal strain Λ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def audit_piezojet_lambda(
    factor_root: Path,
    jarvis_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Inspect strict-factor records for atom-resolved Λ and related fields."""
    factor_root = Path(factor_root)
    if not factor_root.exists():
        return {"status": "missing_root", "n_files": 0}

    pt_files = sorted(factor_root.glob("*.pt"))
    n_total_files = len(pt_files)

    inspected: list[dict[str, Any]] = []
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        jid = data.get("jid")
        if jarvis_ids is not None and jid not in jarvis_ids:
            continue

        ist = data.get("internal_strain_tensors")
        born = data.get("born_charges")
        eigen = data.get("dynamical_eigenvalues")
        eigenvec = data.get("dynamical_eigenvectors")
        fc = data.get("force_constants")
        epsilon = data.get("epsilon")
        total = data.get("total_piezo_source")

        def _shape(obj: Any) -> tuple[int, ...] | None:
            if hasattr(obj, "shape"):
                return tuple(obj.shape)
            return None

        n_atoms = None
        if _shape(born) and len(_shape(born)) >= 1:
            n_atoms = _shape(born)[0]

        # A full atom-resolved Λ would be (n_atoms, 3, 3, 3) or flattened (3*n_atoms, 6)
        # for the internal-strain contribution to piezo.
        ist_shape = _shape(ist)
        full_lambda_candidate = False
        if ist_shape in [(n_atoms, 3, 3, 3), (3 * n_atoms, 6), (n_atoms, 3, 6)] if n_atoms else False:
            full_lambda_candidate = True

        inspected.append({
            "jid": jid,
            "file": pt_file.name,
            "n_atoms": n_atoms,
            "born_shape": _shape(born),
            "internal_strain_shape": ist_shape,
            "force_constant_shape": _shape(fc),
            "dynamical_eigenvalue_shape": _shape(eigen),
            "dynamical_eigenvector_shape": _shape(eigenvec),
            "epsilon_type": type(epsilon).__name__ if epsilon is not None else None,
            "total_piezo_shape": _shape(total),
            "full_lambda_candidate": full_lambda_candidate,
        })

    df = pd.DataFrame(inspected)
    if df.empty:
        return {"status": "no_records", "n_total_files": n_total_files, "inspected": 0}

    shape_counts = df["internal_strain_shape"].value_counts().to_dict()
    full_count = int(df["full_lambda_candidate"].sum())

    return {
        "status": "audited",
        "n_total_files": n_total_files,
        "inspected": len(df),
        "full_lambda_candidate_count": full_count,
        "internal_strain_shape_counts": shape_counts,
        "records": df,
    }
