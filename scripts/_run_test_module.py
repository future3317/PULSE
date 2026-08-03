#!/usr/bin/env python
"""Helper used by ``run_convention_audit.py`` to run one test module in isolation.

Imports the module at *module_path* and calls every top-level ``test_*`` function.
Prints one line per test::

    PASS:test_rotation_invariance.py::test_frobenius_norm_is_rotation_invariant
    FAIL:test_rotation_invariance.py::test_max_longitudinal_modulus_is_rotation_invariant: assertion message
    ERROR:test_rotation_invariance.py::test_kelvin_operator_norm_is_rotation_invariant: exception message
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


def main() -> int:
    module_path = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        print(f"ERROR:{module_path.name}::module: cannot load module", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    test_names = [
        name
        for name, obj in inspect.getmembers(module)
        if name.startswith("test_") and callable(obj)
    ]

    exit_code = 0
    for name in test_names:
        func = getattr(module, name)
        try:
            func()
            print(f"PASS:{module_path.name}::{name}")
        except AssertionError as exc:
            print(f"FAIL:{module_path.name}::{name}: {exc}")
            exit_code = 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR:{module_path.name}::{name}: {exc}")
            traceback.print_exc()
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
