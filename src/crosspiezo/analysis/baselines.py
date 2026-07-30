"""Lightweight reproducible in-source baselines for Phase 5A PMR.

These baselines are intentionally simple.  Their purpose is to provide a
comparable in-source error scale for the protocol-to-model ratio, not to
reproduce published SOTA numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class BaselineResult:
    """Test-set performance of one baseline under one train/eval combination."""

    baseline_name: str
    train_source: str
    eval_source: str
    seed: int
    n_train: int
    n_test: int
    absolute_frobenius_mae: float
    normalized_frobenius_mae: float
    component_mae: float
    cosine_similarity: float
    amplitude_ratio: float


def _reduced_anonymous_prototype(formula: str) -> str:
    """Reduced anonymous-formula prototype key (e.g. Na2Cl2 and NaCl -> AB)."""
    import math

    from pymatgen.core.composition import Composition

    comp = Composition(formula)
    items = sorted(comp.items(), key=lambda item: (-item[1], str(item[0])))
    amounts = [int(amount) for _, amount in items]
    gcd = math.gcd(*amounts) if len(amounts) > 1 else amounts[0]
    reduced = [a // gcd for a in amounts]
    return "".join(f"{chr(ord('A') + i)}{(a if a > 1 else '')}" for i, a in enumerate(reduced))


def _composition_features(formula: str) -> np.ndarray:
    """A simple deterministic composition vector (first 94 elements)."""
    from pymatgen.core.composition import Composition

    comp = Composition(formula)
    vec = np.zeros(94, dtype=np.float64)
    for el, amt in comp.items():
        z = el.Z
        if 1 <= z <= 94:
            vec[z - 1] = float(amt)
    return vec / (np.sum(vec) + 1e-12)


def _distance_histogram_features(cif_string: str, n_bins: int = 20) -> np.ndarray:
    """O(3)-invariant pairwise-distance histogram for a CIF structure."""
    from pymatgen.core.structure import Structure

    try:
        struct = Structure.from_str(cif_string, fmt="cif")
    except Exception:  # noqa: BLE001
        return np.zeros(n_bins, dtype=np.float64)
    if len(struct) < 2:
        return np.zeros(n_bins, dtype=np.float64)
    coords = struct.cart_coords
    dists = cdist(coords, coords)
    # Exclude self-distances and take the upper triangle.
    triu_idx = np.triu_indices_from(dists, k=1)
    values = dists[triu_idx]
    if len(values) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 10.0))
    return np.asarray(hist, dtype=np.float64) / (len(values) + 1e-12)


def _flatten_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flatten a 3x3x3 Cartesian tensor into 27 components."""
    return np.asarray(tensor, dtype=np.float64).ravel()


def _unflatten_tensor(vec: np.ndarray) -> np.ndarray:
    """Reshape a 27-vector into a 3x3x3 Cartesian tensor."""
    t = np.asarray(vec, dtype=np.float64).reshape(3, 3, 3)
    # Enforce last-index symmetry.
    return 0.5 * (t + t.transpose(0, 2, 1))


def _build_feature_matrix(
    records: list[dict[str, Any]],
    feature_mode: str,
) -> np.ndarray:
    """Build a feature matrix from records."""
    feats: list[np.ndarray] = []
    for rec in records:
        if feature_mode == "composition":
            feats.append(_composition_features(rec["formula"]))
        elif feature_mode == "structure":
            comp = _composition_features(rec["formula"])
            dist = _distance_histogram_features(rec["cif"])
            feats.append(np.concatenate([comp, dist]))
        elif feature_mode == "source_token":
            comp = _composition_features(rec["formula"])
            token = np.zeros(2, dtype=np.float64)
            if rec["source"] == "jarvis":
                token[0] = 1.0
            elif rec["source"] == "mp":
                token[1] = 1.0
            feats.append(np.concatenate([comp, token]))
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
    return np.asarray(feats, dtype=np.float64)


def _prepare_records(
    jarvis_df: Any,
    mp_df: Any,
    tensor_key: str = "piezo_cartesian_total",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert dataframes to unified record lists with tensors and CIFs."""
    import ast

    def _to_array(value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            return np.asarray(parsed, dtype=np.float64)
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=np.float64)
        return None

    jarvis_records: list[dict[str, Any]] = []
    for _, row in jarvis_df.iterrows():
        tensor = _to_array(row.get(tensor_key))
        if tensor is None or tensor.shape != (3, 3, 3):
            continue
        jarvis_records.append({
            "id": row["material_id"],
            "formula": row["formula"],
            "prototype": _reduced_anonymous_prototype(row["formula"]),
            "cif": row["cif"],
            "source": "jarvis",
            "tensor": tensor,
        })

    mp_records: list[dict[str, Any]] = []
    for _, row in mp_df.iterrows():
        tensor = _to_array(row.get(tensor_key))
        if tensor is None or tensor.shape != (3, 3, 3):
            continue
        mp_records.append({
            "id": row["material_id"],
            "formula": row["formula"],
            "prototype": _reduced_anonymous_prototype(row["formula"]),
            "cif": row["cif"],
            "source": "mp",
            "tensor": tensor,
        })

    return jarvis_records, mp_records


def zero_baseline(
    test_records: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    seed: int = 42,
) -> BaselineResult:
    """Predict the zero tensor for every test record."""
    preds = [np.zeros((3, 3, 3), dtype=np.float64) for _ in test_records]
    return _collect_result("zero", train_source, eval_source, seed, [], test_records, preds)


def composition_mean_baseline(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    seed: int = 42,
    source_specific: bool = False,
) -> BaselineResult:
    """Predict the mean tensor per composition (global or source-specific)."""
    from collections import defaultdict

    means: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 3, 3), dtype=np.float64))
    counts: dict[str, int] = defaultdict(int)
    for rec in train_records:
        key = rec["source"] if source_specific else "global"
        means[rec["formula"] + "_" + key] += rec["tensor"]
        counts[rec["formula"] + "_" + key] += 1
    for k in means:
        means[k] /= max(counts[k], 1)

    preds: list[np.ndarray] = []
    for rec in test_records:
        key = rec["source"] if source_specific else "global"
        pred = means.get(rec["formula"] + "_" + key, np.zeros((3, 3, 3), dtype=np.float64))
        preds.append(pred)
    name = "source_specific_composition" if source_specific else "composition_mean"
    return _collect_result(name, train_source, eval_source, seed, train_records, test_records, preds)


def structural_ridge_baseline(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    feature_mode: str = "structure",
    seed: int = 42,
    alpha: float = 1.0,
) -> BaselineResult:
    """Ridge regression on composition (+ distance histogram) features.

    This is an O(3)-invariant input baseline, not a full equivariant model.
    It serves as a reproducible lower-complexity reference for PMR.
    """
    rng = np.random.default_rng(seed)
    x_train = _build_feature_matrix(train_records, feature_mode)
    y_train = np.asarray([_flatten_tensor(r["tensor"]) for r in train_records], dtype=np.float64)
    x_test = _build_feature_matrix(test_records, feature_mode)

    # Add random Fourier features for non-linearity.
    rbf = RBFSampler(gamma=0.05, n_components=256, random_state=int(rng.integers(2**31)))
    x_train_rbf = rbf.fit_transform(x_train)
    x_test_rbf = rbf.transform(x_test)

    model = Ridge(alpha=alpha, random_state=int(rng.integers(2**31)))
    model.fit(x_train_rbf, y_train)
    y_pred = model.predict(x_test_rbf)
    preds = [_unflatten_tensor(v) for v in y_pred]
    name = "structural_ridge" if feature_mode == "structure" else "composition_ridge"
    return _collect_result(name, train_source, eval_source, seed, train_records, test_records, preds)


def mlp_invariant_baseline(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    seed: int = 42,
    feature_mode: str = "structure",
) -> BaselineResult:
    """Small MLP on O(3)-invariant composition+structure features.

    This is a fast, reproducible non-linear baseline.  It is O(3)-invariant,
    not a full equivariant tensor model, and is reported separately from the
    e3nn baseline.
    """
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(seed)
    x_train = _build_feature_matrix(train_records, feature_mode)
    y_train = np.asarray([_flatten_tensor(r["tensor"]) for r in train_records], dtype=np.float64)
    x_test = _build_feature_matrix(test_records, feature_mode)

    if len(x_train) == 0 or len(x_test) == 0 or x_train.shape[1] == 0:
        preds = [np.zeros((3, 3, 3), dtype=np.float64) for _ in test_records]
        return _collect_result("mlp_invariant", train_source, eval_source, seed, train_records, test_records, preds)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=int(rng.integers(2**31)),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    preds = [_unflatten_tensor(v) for v in y_pred]
    return _collect_result("mlp_invariant", train_source, eval_source, seed, train_records, test_records, preds)


def source_token_baseline(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    train_source: str,
    eval_source: str,
    seed: int = 42,
) -> BaselineResult:
    """Composition baseline with a source token appended to features."""
    return structural_ridge_baseline(
        train_records,
        test_records,
        train_source,
        eval_source,
        feature_mode="source_token",
        seed=seed,
        alpha=1.0,
    )


def _collect_result(
    baseline_name: str,
    train_source: str,
    eval_source: str,
    seed: int,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    predictions: list[np.ndarray],
) -> BaselineResult:
    """Aggregate metrics for a baseline."""
    abs_errors: list[float] = []
    norm_errors: list[float] = []
    comp_errors: list[float] = []
    cosines: list[float] = []
    amp_ratios: list[float] = []
    for rec, pred in zip(test_records, predictions, strict=True):
        true = rec["tensor"]
        diff = true - pred
        abs_err = float(np.linalg.norm(diff))
        norm = float(0.5 * (np.linalg.norm(true) + np.linalg.norm(pred)) + 1e-6)
        abs_errors.append(abs_err)
        norm_errors.append(abs_err / norm)
        comp_errors.append(float(np.mean(np.abs(diff))))
        cosines.append(_cosine(true, pred))
        amp_ratios.append(_amplitude_ratio(true, pred))

    return BaselineResult(
        baseline_name=baseline_name,
        train_source=train_source,
        eval_source=eval_source,
        seed=seed,
        n_train=len(train_records),
        n_test=len(test_records),
        absolute_frobenius_mae=float(np.mean(abs_errors)) if abs_errors else float("nan"),
        normalized_frobenius_mae=float(np.mean(norm_errors)) if norm_errors else float("nan"),
        component_mae=float(np.mean(comp_errors)) if comp_errors else float("nan"),
        cosine_similarity=float(np.mean(cosines)) if cosines else float("nan"),
        amplitude_ratio=float(np.mean(amp_ratios)) if amp_ratios else float("nan"),
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    lf = left.ravel()
    rf = right.ravel()
    denom = np.linalg.norm(lf) * np.linalg.norm(rf) + 1e-12
    return float(np.dot(lf, rf) / denom)


def _amplitude_ratio(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) + 1e-12
    return float(np.linalg.norm(right) / denom)


def compute_pmr(
    paired_absolute_discrepancies: np.ndarray,
    in_source_absolute_maes: list[float],
) -> dict[str, float]:
    """Protocol-to-model ratio with bootstrap CI.

    PMR = median paired discrepancy / mean in-source MAE.
    """
    from crosspiezo.analysis.discrepancy import bootstrap_ci

    discs = np.asarray(paired_absolute_discrepancies, dtype=np.float64)
    median_disc = float(np.median(discs))
    mean_mae = float(np.mean(in_source_absolute_maes))
    pmr = median_disc / (mean_mae + 1e-12)
    _, lo, hi = bootstrap_ci(discs, statistic="median")
    pmr_lo = lo / (mean_mae + 1e-12)
    pmr_hi = hi / (mean_mae + 1e-12)
    return {
        "pmr": pmr,
        "pmr_ci95_low": pmr_lo,
        "pmr_ci95_high": pmr_hi,
        "median_paired_discrepancy": median_disc,
        "mean_in_source_mae": mean_mae,
    }
