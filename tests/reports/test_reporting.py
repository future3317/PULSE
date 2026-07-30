"""Red tests for C-14: reporting scripts must not hard-code scientific numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = PROJECT_ROOT / "scripts" / "compile_phase5b_reports.py"


def _script_text() -> str:
    return COMPILE_SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "literal",
    [
        "0.07526881720430108",
        "0.2568831475074093",
        "538",
        "15",
    ],
)
def test_no_hardcoded_phase5b_numbers_in_compile_script(literal: str) -> None:
    """Old summary numbers must be read from artifacts, not embedded as literals."""
    text = _script_text()
    assert literal not in text, (
        f"compile script contains hardcoded scientific number: {literal}"
    )


def test_report_decision_reads_ranking_from_artifact() -> None:
    """report_decision must derive top-50 Jaccard and Kendall tau from artifacts."""
    text = _script_text()
    # The function signature or body should accept ranking results, not assign them.
    pattern = re.compile(r"top50_jaccard\s*=\s*[0-9.]")
    assert not pattern.search(text), "top50_jaccard is hardcoded"
    pattern = re.compile(r"kendall_tau\s*=\s*[0-9.]")
    assert not pattern.search(text), "kendall_tau is hardcoded"
