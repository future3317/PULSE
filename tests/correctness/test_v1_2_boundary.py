"""Execution-boundary tests for Correctness v1.2.

The v1.2 pipeline must not import or call Phase 5A/5B, baselines, PMR,
soft-mode, e3nn, or O(3) transport analysis.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_correctness_v1_2.py"

_BANNED_MODULES = {
    "run_phase5a",
    "run_phase5b",
    "baseline",
    "PMR",
    "soft_mode",
    "e3nn",
    "o3_transport",
    "O3 transport",
}


def _collect_imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                modules.add(node.module.split(".")[0])
    return modules


def test_v1_2_script_does_not_import_banned_modules():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    imported = _collect_imported_modules(source)
    banned = imported & _BANNED_MODULES
    assert not banned, f"v1.2 script imports banned modules: {banned}"


def test_v1_2_script_import_does_not_load_banned_modules():
    """Importing the v1.2 script must not load any banned module namespace."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
        "import scripts.run_correctness_v1_2 as v12\n"
        "loaded = set(sys.modules.keys())\n"
        f"banned = {sorted(_BANNED_MODULES)}\n"
        "found = [m for m in banned if m in loaded]\n"
        "print('FOUND:', found)\n"
        "assert not found, f'banned modules loaded: {found}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Boundary test failed:\n{result.stdout}\n{result.stderr}")
