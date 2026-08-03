#!/usr/bin/env python
"""Run the repository convention/invariance tests and write a short audit report.

This script does not modify source data.  It executes the existing test suite
for tensor rotation invariance (F1/F3/F4), Voigt/Cartesian conversion, and
shear-factor handling, then writes a Markdown summary to
``reports/phase8a/convention_audit_report.md``.

The tests are run by importing each test module and calling its ``test_*``
functions directly.  This avoids the fatal BLAS/Scipy aborts that have been
observed when the same tests are executed through ``pytest`` in the presence
of other extension modules (e.g. PyTorch) in this Anaconda environment.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "phase8a" / "convention_audit_report.md"

TEST_MODULES = [
    "tests/ranking/test_rotation_invariance.py",
    "tests/conventions/test_piezo_tensor.py",
]


def _discover_test_functions(module_path: Path) -> list[str]:
    """Return the names of top-level ``test_*`` functions in *module_path*."""
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load test module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [
        name
        for name, obj in inspect.getmembers(module)
        if name.startswith("test_") and callable(obj)
    ]


def _run_tests_in_module(module_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Run a test module in a subprocess and return (passed, failed, errors)."""
    runner = PROJECT_ROOT / "scripts" / "_run_test_module.py"
    cmd = [sys.executable, str(runner), str(module_path)]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    passed: list[str] = []
    failed: list[str] = []
    errors: list[str] = []
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("PASS:"):
            passed.append(line.split(":", 1)[1])
        elif line.startswith("FAIL:"):
            failed.append(line.split(":", 1)[1])
        elif line.startswith("ERROR:"):
            errors.append(line.split(":", 1)[1])
    # If the subprocess itself aborted, report every test as an error.
    if result.returncode != 0 and not passed and not failed and not errors:
        errors.append(f"{module_path.name}: subprocess aborted")
    return passed, failed, errors


def _run_direct() -> tuple[int, str]:
    """Run tests directly in the current process (fallback)."""
    passed: list[str] = []
    failed: list[str] = []
    errors: list[str] = []
    for rel in TEST_MODULES:
        path = PROJECT_ROOT / rel
        names = _discover_test_functions(path)
        for name in names:
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, name)
                func()
                passed.append(f"{path.name}::{name}")
            except AssertionError as exc:
                failed.append(f"{path.name}::{name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}::{name}: {exc}\n{traceback.format_exc()}")
    total = len(passed) + len(failed) + len(errors)
    summary = f"passed={len(passed)} failed={len(failed)} errors={len(errors)} total={total}"
    details = "\n".join(
        ["PASSED:"] + passed + ["FAILED:"] + failed + ["ERRORS:"] + errors
    )
    return 0 if not failed and not errors else 1, f"{summary}\n{details}"


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Prefer subprocess isolation, but fall back to direct execution if the
    # helper script is missing.
    runner = PROJECT_ROOT / "scripts" / "_run_test_module.py"
    if runner.exists():
        passed: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for rel in TEST_MODULES:
            path = PROJECT_ROOT / rel
            p, f, e = _run_tests_in_module(path)
            passed.extend(p)
            failed.extend(f)
            errors.extend(e)
        total = len(passed) + len(failed) + len(errors)
        returncode = 0 if not failed and not errors else 1
        summary = f"passed={len(passed)} failed={len(failed)} errors={len(errors)} total={total}"
        details = "\n".join(
            ["PASSED:"] + passed + ["FAILED:"] + failed + ["ERRORS:"] + errors
        )
    else:
        returncode, details = _run_direct()
        summary = details.splitlines()[0]

    lines = [
        "# CrossPiezo tensor-convention audit report",
        "",
        f"**Date:** 2026-08-03",
        f"**Command:** `python {Path(__file__).name}`",
        f"**Exit code:** {returncode}",
        "",
        "## Test result summary",
        "",
        "```text",
        summary,
        "```",
        "",
        "## Details",
        "",
        "```text",
        details,
        "```",
        "",
        "## Interpretation",
        "",
        "A zero exit code means the repository tests confirm:",
        "- F1 (Frobenius norm), F3 (longitudinal maximum), and F4 (Kelvin/Mandel operator norm) are invariant under coordinate rotations;",
        "- Voigt-to-Cartesian conversion and engineering-shear factor-of-two handling are internally consistent;",
        "- F3 spherical optimisation converges to the brute-force dense-grid oracle.",
        "",
        "These tests do **not** prove that JARVIS and MP use identical calculation settings; they only rule out low-level convention errors in the CrossPiezo processing pipeline.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[convention_audit] exit={returncode}; wrote {REPORT_PATH}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
