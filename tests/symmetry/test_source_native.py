"""Red tests for C-04: source-native residuals must use each source's own frame.

The old implementation used a single shared space-group symbol taken from the
JARVIS row, so the ``native_residual`` for MP was computed in the wrong frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crosspiezo.phase5b.panels import compute_source_native_residuals


def _tensor_with_single_component(i: int, j: int, k: int, value: float) -> np.ndarray:
    t = np.zeros((3, 3, 3), dtype=np.float64)
    t[i, j, k] = value
    t[i, k, j] = value
    return t


def _minimal_df(cif: str | None = None) -> pd.DataFrame:
    tensor = _tensor_with_single_component(0, 0, 0, 1.0)
    return pd.DataFrame({
        "jarvis_id": ["J-1"],
        "mp_id": ["MP-1"],
        "space_group": [62],
        "crystal_system": ["orthorhombic"],
        "jarvis_tensor": [tensor],
        "mp_tensor_raw": [tensor],
        "mp_tensor_aligned": [tensor],
        "jarvis_cif": [cif],
        "mp_cif": [cif],
        "rotation": [None],
    })


def test_missing_structures_yield_unresolved_native_residual():
    """Computing native residuals without the source CIFs must not fall back to a
    shared abstract space group; native residual must be NaN."""
    df = _minimal_df(cif=None)
    result = compute_source_native_residuals(df)
    assert not result.empty
    assert "native_frame_status" in result.columns
    assert all(status == "native_frame_unresolved" for status in result["native_frame_status"])
    for col in result.columns:
        if "native_residual" in col:
            assert all(np.isnan(result[col]))


def test_source_native_uses_source_specific_structures():
    """With valid CIFs, native residuals are computed from each source's own point
    group and the status is verified."""
    import numpy as np
    from pymatgen.core.structure import Structure

    lattice = np.eye(3) * 3.0
    s = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    cif = s.to(fmt="cif")
    df = _minimal_df(cif=cif)
    result = compute_source_native_residuals(df)
    assert not result.empty
    assert "native_frame_status" in result.columns
    assert all(status == "native_frame_verified" for status in result["native_frame_status"])
