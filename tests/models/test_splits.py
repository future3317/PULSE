"""Red tests for C-12 and C-13: prototype/formula split leakage and evaluation
label accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pandas as pd

from crosspiezo.analysis.soft_mode import _formula_to_prototype as soft_mode_prototype
from crosspiezo.models.trainer import (
    _formula_to_prototype as trainer_prototype,
    build_paired_counterfactual_eval,
    paired_counterfactual_is_disjoint,
)
from scripts.train_e3nn import _baseline_splits


def test_formula_to_prototype_uses_reduced_stoichiometry():
    """Na2Cl2 and NaCl reduce to the same anonymous formula AB."""
    a = trainer_prototype("Na2Cl2")
    b = trainer_prototype("NaCl")
    assert a == b == "AB", f"expected AB, got {a} and {b}"


def test_prototype_helper_consistent_across_modules():
    """Both modules should agree on a real prototype label (or both renamed)."""
    formula = "CaTiO3"
    assert soft_mode_prototype(formula) == trainer_prototype(formula), (
        "prototype definitions diverge between modules"
    )


def test_source_held_out_requires_exclusion_of_eval_source_from_training():
    """A split named 'source_held_out_X' must be evaluated on a model that never
    saw source X during training.  With only two sources, pooled models are not
    source-held-out."""
    splits = _baseline_splits([], [], [], [])
    for name, _tr, _ev, train, eval_source, split in splits:
        if not split.startswith("source_held_out"):
            continue
        held_source = split.split("_")[-1]
        assert train != "pooled", (
            f"{name} {split} uses a pooled train_source; pooled models cannot be "
            "source-held-out with only two sources"
        )
        assert held_source not in train.split("+"), (
            f"{name} {split} evaluated on model trained with {train}"
        )


def test_train_pool_excludes_test_formula_and_prototype():
    """Training pool must have zero intersection with test on ID, formula, and prototype."""
    test_panel = pd.DataFrame({
        "jarvis_id": ["J-1"],
        "mp_id": ["MP-1"],
        "formula": ["NaCl"],
        "prototype": ["AB"],
    })
    test_jids = set(test_panel["jarvis_id"])
    test_mids = set(test_panel["mp_id"])
    test_formulas = set(test_panel["formula"])
    test_prototypes = set(test_panel["prototype"])

    records = [
        {"id": "J-1", "formula": "NaCl", "prototype": "AB", "source": "jarvis"},
        {"id": "J-2", "formula": "NaCl", "prototype": "AB", "source": "jarvis"},
        {"id": "J-3", "formula": "Na2Cl2", "prototype": "AB", "source": "jarvis"},
        {"id": "J-4", "formula": "KCl", "prototype": "AB", "source": "jarvis"},
        {"id": "J-5", "formula": "SiO2", "prototype": "AB2", "source": "jarvis"},
    ]

    pool = [
        r for r in records
        if r["id"] not in test_jids and r["id"] not in test_mids
        and r["formula"] not in test_formulas
        and r.get("prototype") not in test_prototypes
    ]
    assert len(pool) == 1
    assert pool[0]["id"] == "J-5"


def test_counterfactual_eval_uses_paired_counterpart():
    """Paired-counterfactual predictions must be made on the mate from the other
    source, not on the same-source test set."""
    test_panel = pd.DataFrame({
        "jarvis_id": ["J-1", "J-2"],
        "mp_id": ["MP-1", "MP-2"],
    })
    jarvis_records = [
        {"id": "J-1", "source": "jarvis", "formula": "A"},
        {"id": "J-2", "source": "jarvis", "formula": "B"},
    ]
    mp_records = [
        {"id": "MP-1", "source": "mp", "formula": "A"},
        {"id": "MP-2", "source": "mp", "formula": "B"},
    ]
    paired = build_paired_counterfactual_eval(test_panel, jarvis_records, mp_records)

    assert all(r["source"] == "mp" for r in paired["paired_counterfactual_jarvis"])
    assert all(r["source"] == "jarvis" for r in paired["paired_counterfactual_mp"])
    jarvis_eval_ids = {r["id"] for r in paired["paired_counterfactual_jarvis"]}
    mp_eval_ids = {r["id"] for r in paired["paired_counterfactual_mp"]}
    assert jarvis_eval_ids.isdisjoint({"J-1", "J-2"})
    assert mp_eval_ids.isdisjoint({"MP-1", "MP-2"})
    assert paired_counterfactual_is_disjoint(paired, {"J-3", "MP-3"})
