"""Red tests for C-04: source-native residuals must use each source's own frame.

The current implementation uses a single shared space-group symbol taken from the
JARVIS row, so the ``native_residual`` for MP is computed in the wrong frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosspiezo.phase5b.panels import compute_source_native_residuals


def _tensor_with_single_component(i: int, j: int, k: int, value: float) -> np.ndarray:
    t = np.zeros((3, 3, 3), dtype=np.float64)
    t[i, j, k] = value
    t[i, k, j] = value
    return t


def test_requires_source_structures():
    """Computing native residuals without the source CIFs must fail closed."""
    df = pd.DataFrame({
        "jarvis_id": ["J-1"],
        "mp_id": ["MP-1"],
        "space_group": [62],
        "jarvis_tensor": [_tensor_with_single_component(0, 0, 0, 1.0)],
        "mp_tensor_raw": [_tensor_with_single_component(0, 0, 0, 1.0)],
        "rotation": [None],
    })
    with pytest.raises((KeyError, ValueError)):
        compute_source_native_residuals(df)


def test_no_structures_yields_unresolved_native_residual():
    """If source-native structures cannot be recovered, native residual is NaN."""
    df = pd.DataFrame({
        "jarvis_id": ["J-1"],
        "mp_id": ["MP-1"],
        "space_group": [62],
        "crystal_system": ["orthorhombic"],
        "jarvis_tensor": [_tensor_with_single_component(0, 0, 0, 1.0)],
        "mp_tensor_raw": [_tensor_with_single_component(0, 0, 0, 1.0)],
        "jarvis_cif": [None],
        "mp_cif": [None],
        "rotation": [None],
    })
    result = compute_source_native_residuals(df)
    assert result.empty or all(
        np.isnan(result[col].values[0]) for col in result.columns if "native_residual" in col
    )


def test_native_frame_verified_status_recorded():
    """The result must report whether the native frame was verified or reconstructed."""
    result = compute_source_native_residuals(pd.DataFrame())
    assert "native_frame_status" in result.columns
