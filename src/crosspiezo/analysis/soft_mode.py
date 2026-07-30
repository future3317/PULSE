"""Soft-mode / microscopic-factor feasibility analysis for Phase 5A.

All factor-based claims are JARVIS-side sensitivity indicators; they are not
attributed to a specific JARVIS--MP protocol setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SoftModeResult:
    """Result of one regression model predicting cross-protocol discrepancy."""

    model_name: str
    n_pairs: int
    r2: float
    rmse: float
    pearson_r: float
    pearson_pvalue: float
    grouped_cv_r2_mean: float
    grouped_cv_r2_std: float


def load_piezojet_records(
    factor_root: Any,
    material_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load PiezoJet strict-factor records indexed by JARVIS material id.

    Parameters
    ----------
    factor_root: pathlib.Path or str
        Directory containing the ``*.pt`` files.
    material_ids: set[str] | None
        If provided, only load records whose JID is in this set.
    """
    import torch

    factor_root = factor_root if hasattr(factor_root, "glob") else __import__("pathlib").Path(factor_root)
    records: dict[str, dict[str, Any]] = {}
    for pt_file in sorted(factor_root.glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        jid = data.get("jid")
        if jid is None:
            continue
        if material_ids is not None and jid not in material_ids:
            continue
        records[jid] = data
    return records


def _to_numpy(obj: Any) -> np.ndarray | None:
    """Convert a torch.Tensor or list to a NumPy array."""
    if hasattr(obj, "detach"):
        return np.asarray(obj.detach().cpu().numpy(), dtype=np.float64)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj, dtype=np.float64)
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=np.float64)
    return None


def compute_soft_mode_features(data: dict[str, Any]) -> dict[str, float] | None:
    """Compute JARVIS-side physical sensitivity indicators from a PiezoJet record.

    Returns None if the record is missing required fields or is dynamically
    unstable (non-positive optical eigenvalues after removing translations).
    """
    eigen = _to_numpy(data.get("dynamical_eigenvalues"))
    if eigen is None:
        return None
    # PiezoJet stores eigenvalues in a convention where stable optical modes are
    # negative and the three acoustic (zero-frequency) modes are closest to zero.
    sorted_idx = np.argsort(eigen)
    optical = eigen[sorted_idx[:-3]]
    if len(optical) == 0 or np.any(optical >= 0):
        return None
    min_eigen = float(np.min(optical))  # most negative = softest stable mode
    max_eigen = float(np.max(optical))  # least negative = hardest mode
    condition_number = max_eigen / (min_eigen + 1e-12)

    born = _to_numpy(data.get("born_charges"))
    born_norm = float(np.linalg.norm(born)) if born is not None else float("nan")

    ist = _to_numpy(data.get("internal_strain_tensors"))
    ist_norm = float(np.linalg.norm(ist)) if ist is not None else float("nan")

    fc = _to_numpy(data.get("force_constants"))
    fc_norm = float(np.linalg.norm(fc)) if fc is not None else float("nan")

    epsilon = data.get("epsilon")
    eps_ion = None
    if isinstance(epsilon, dict):
        eps_ion = _to_numpy(epsilon.get("ionic"))
    eps_ion_norm = float(np.linalg.norm(eps_ion)) if eps_ion is not None else float("nan")

    total = _to_numpy(data.get("total_piezo_source"))
    ionic = _to_numpy(data.get("ionic_piezo_source"))
    if total is not None and ionic is not None:
        total_norm = float(np.linalg.norm(total))
        ionic_norm = float(np.linalg.norm(ionic))
        ionic_fraction = ionic_norm / (total_norm + 1e-12)
    else:
        ionic_fraction = float("nan")

    # S_soft proxy: sum_m |z_m| |l_m| / (|lambda_m| + delta)^2 over optical modes.
    # The PiezoJet internal_strain_tensors are stored in a reduced (3,3,3) shape
    # that does not expose the full 3N atom-resolved internal-strain matrix, so
    # the exact l_m contraction is not always possible.  We compute the mode-
    # resolved Born-effective-charge vector z_m and fall back to a scalar proxy
    # when the atom-resolved internal strain is unavailable.
    s_soft = float("nan")
    eigenvectors = _to_numpy(data.get("dynamical_eigenvectors"))
    if born is not None and eigenvectors is not None:
        try:
            optical_idx = sorted_idx[:-3]
            opt_eigen = eigen[optical_idx]
            opt_eigenvectors = eigenvectors[optical_idx]
            # born shape: (n_atoms, 3, 3) = Z*_{i, alpha, beta}
            # eigenvectors shape: (n_modes, n_atoms, 3) = v_{m, i, alpha}
            # z_m[beta] = sum_{i,alpha} Z*_{i,alpha,beta} * v_{m,i,alpha}
            z_m = np.einsum("iax,mia->mx", born, opt_eigenvectors)  # (n_optical, 3)
            z_norm = np.linalg.norm(z_m, axis=1)
            delta = 1e-4
            if ist is not None and ist.shape == (3, 3, 3):
                # Scalar proxy using available reduced internal-strain norm.
                ist_norm_scalar = float(np.linalg.norm(ist))
                s_soft = float(np.sum(z_norm * ist_norm_scalar / ((np.abs(opt_eigen) + delta) ** 2)))
            else:
                s_soft = float(np.sum(z_norm / ((np.abs(opt_eigen) + delta) ** 2)))
        except Exception:  # noqa: BLE001
            s_soft = float("nan")

    return {
        "min_optical_eigenvalue": min_eigen,
        "max_optical_eigenvalue": max_eigen,
        "optical_condition_number": condition_number,
        "born_charge_norm": born_norm,
        "internal_strain_norm": ist_norm,
        "force_constant_norm": fc_norm,
        "epsilon_ionic_norm": eps_ion_norm,
        "ionic_fraction": ionic_fraction,
        "S_soft": s_soft,
    }


def _formula_to_prototype(formula: str) -> str:
    """Reduced anonymous-formula prototype key (e.g. AB2).

    The stoichiometric coefficients are divided by their greatest common
    divisor so that scaled formulas such as Na2Cl2 and NaCl both map to AB.
    This is still an anonymous formula, not a full structure prototype.
    """
    import math

    from pymatgen.core.composition import Composition

    comp = Composition(formula)
    items = sorted(comp.items(), key=lambda item: (-item[1], str(item[0])))
    amounts = [int(amount) for _, amount in items]
    gcd = math.gcd(*amounts) if len(amounts) > 1 else amounts[0]
    reduced = [a // gcd for a in amounts]
    return "".join(f"{chr(ord('A') + i)}{a}" for i, a in enumerate(reduced))


def nested_regression_analysis(
    df: pd.DataFrame,
    target_col: str = "absolute_discrepancy",
    group_col: str = "prototype",
    seed: int = 42,
) -> list[SoftModeResult]:
    """Fit nested models predicting discrepancy from chemistry/structure/factors.

    Models:
      * chemistry_only: composition-derived numeric features.
      * structure_only: structure-match distances + volume ratio.
      * factor_only: microscopic JARVIS-side factors.
      * combined: all of the above.

    Grouped CV is performed by ``group_col`` (prototype by default).
    """
    rng = np.random.default_rng(seed)

    feature_groups: dict[str, list[str]] = {
        "chemistry_only": ["n_elements", "max_z"],
        "structure_only": ["volume_ratio", "lattice_distance", "site_distance"],
        "factor_only": [
            "min_optical_eigenvalue",
            "optical_condition_number",
            "born_charge_norm",
            "internal_strain_norm",
            "force_constant_norm",
            "epsilon_ionic_norm",
            "ionic_fraction",
            "S_soft",
        ],
        "combined": [],
    }
    # Combined gets all available feature columns.
    all_features = []
    for v in feature_groups.values():
        all_features.extend(v)
    feature_groups["combined"] = sorted(set(all_features))

    results: list[SoftModeResult] = []
    groups = df[group_col].values
    y = df[target_col].values
    for model_name, cols in feature_groups.items():
        available = [c for c in cols if c in df.columns and df[c].notna().any()]
        if not available:
            continue
        x = df[available].values
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        xv = x[valid]
        yv = y[valid]
        gv = groups[valid]
        if len(yv) < 10:
            continue

        scaler = StandardScaler()
        xs = scaler.fit_transform(xv)
        model = Ridge(alpha=1.0, random_state=int(rng.integers(2**31)))
        model.fit(xs, yv)
        y_pred = model.predict(xs)
        r2 = float(model.score(xs, yv))
        rmse = float(np.sqrt(np.mean((yv - y_pred) ** 2)))
        r, p = stats.pearsonr(yv, y_pred)

        # Grouped CV by prototype.
        cv_scores: list[float] = []
        gkf = GroupKFold(n_splits=min(5, len(np.unique(gv))))
        for train_idx, test_idx in gkf.split(xs, yv, groups=gv):
            model_cv = Ridge(alpha=1.0, random_state=int(rng.integers(2**31)))
            model_cv.fit(xs[train_idx], yv[train_idx])
            score = model_cv.score(xs[test_idx], yv[test_idx])
            cv_scores.append(float(score))
        cv_mean = float(np.mean(cv_scores)) if cv_scores else float("nan")
        cv_std = float(np.std(cv_scores)) if cv_scores else float("nan")

        results.append(SoftModeResult(
            model_name=model_name,
            n_pairs=len(yv),
            r2=r2,
            rmse=rmse,
            pearson_r=float(r) if r is not None else float("nan"),
            pearson_pvalue=float(p) if p is not None else float("nan"),
            grouped_cv_r2_mean=cv_mean,
            grouped_cv_r2_std=cv_std,
        ))

    return results
