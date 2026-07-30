"""Red tests for C-12 and C-13: prototype/formula split leakage and evaluation
label accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosspiezo.analysis.soft_mode import _formula_to_prototype as soft_mode_prototype
from crosspiezo.models.trainer import _formula_to_prototype as trainer_prototype


def test_formula_to_prototype_is_not_element_set():
    """Na2Cl2 and NaCl are the same chemical system but different prototypes."""
    a = trainer_prototype("Na2Cl2")
    b = trainer_prototype("NaCl")
    assert a != b, f"prototype collapsed stoichiometries: {a} == {b}"


def test_prototype_helper_consistent_across_modules():
    """Both modules should agree on a real prototype label (or both renamed)."""
    formula = "CaTiO3"
    assert soft_mode_prototype(formula) == trainer_prototype(formula), (
        "prototype definitions diverge between modules"
    )


def test_source_held_out_requires_exclusion_of_eval_source_from_training():
    """A split named 'source_held_out_X' must be evaluated on a model that never
    saw source X during training."""
    rows = [
        {"model_name": "mlp", "train_source": "pooled", "eval_source": "jarvis", "split_type": "source_held_out_jarvis"},
        {"model_name": "ridge", "train_source": "jarvis", "eval_source": "mp", "split_type": "source_held_out_mp"},
    ]
    for row in rows:
        split = row["split_type"]
        held_source = split.split("_")[-1]
        train = row["train_source"]
        assert train != "pooled", "pooled model cannot be source-held-out with only two sources"
        assert held_source not in train.split("+"), (
            f"{split} evaluated on model trained with {train}"
        )


def test_counterfactual_eval_uses_paired_counterpart():
    """Paired-counterfactual predictions must be made on the mate from the other
    source, not on the same-source test set."""
    # This is a placeholder for a leakage test on the actual prediction manifest.
    # After the fix, a function must return zero intersection between
    # paired_counterfactual eval IDs and in-source eval IDs.
    assert False, "paired-counterfactual leakage test not implemented in current code"
