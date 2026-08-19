from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_resolution_extensions.py"
SPEC = importlib.util.spec_from_file_location("resolution_extensions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_global_rank_blindness_construction_is_exact() -> None:
    result = MODULE.global_rank_blindness(100)
    row = result.loc[result["block_size"] == 5].iloc[0]
    assert row["top_k_overlap"] == 0
    assert np.isclose(row["kendall_tau"], 1 - 4 * 25 / (100 * 99))
    assert np.isclose(row["spearman_rho"], 1 - 12 * 125 / (100 * (100**2 - 1)))


def test_retention_uses_source_top_set_as_denominator() -> None:
    panel = pd.DataFrame({"left": [5, 4, 3, 2, 1], "right": [5, 4, 1, 3, 2]})
    result = MODULE.retention_map(panel, "left", "right")
    row = result[(result["q_source_percent"] == 20) & (result["q_target_percent"] == 20)].iloc[0]
    assert row["source_k"] == 1
    assert row["target_k"] == 1
    assert row["overlap"] == 1
    assert row["retention"] == 1.0
