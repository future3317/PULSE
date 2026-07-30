"""Strict structure matching for CrossPiezo cross-source pairs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher as PMGStructureMatcher
from pymatgen.core.structure import Structure
from scipy.linalg import svd

from crosspiezo.schemas import MatchRecord, MatchTier

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a single structure-match attempt."""

    match_key: str
    tier: MatchTier
    fit: bool
    rms_distance: float | None = None
    max_distance: float | None = None
    atom_mapping: list[int] | None = None
    lattice_distance: float | None = None
    cartesian_rotation: list[list[float]] | None = None
    fractional_translation: list[float] | None = None
    unimodular_cell_transform: list[list[float]] | None = None
    space_group_relation: str | None = None
    ambiguity: str | None = None
    reasons: list[str] | None = None
    rotation_class: str | None = None
    kabsch_rms: float | None = None


def parse_cif(cif_string: str) -> Structure:
    """Parse a CIF string into a pymatgen Structure."""
    return Structure.from_str(cif_string, fmt="cif")


def _as_structure(value: str | Structure) -> Structure:
    """Accept either a serialized CIF string or an existing pymatgen Structure."""
    if isinstance(value, Structure):
        return value
    return parse_cif(value)


def _lattice_distance(s1: Structure, s2: Structure) -> float:
    """Fractional lattice-parameter distance after volume scaling."""
    l1 = np.array(s1.lattice.abc)
    l2 = np.array(s2.lattice.abc)
    # Scale s2 to s1 volume
    vol_ratio = s1.volume / (s2.volume + 1e-12)
    l2_scaled = l2 * (vol_ratio ** (1.0 / 3.0))
    return float(np.max(np.abs(l1 - l2_scaled) / (np.abs(l1) + 1e-6)))


def _space_group_relation(s1: Structure, s2: Structure) -> str:
    """Compare space-group symbols."""
    try:
        sg1 = s1.get_space_group_info()[0]
        sg2 = s2.get_space_group_info()[0]
    except Exception:  # noqa: BLE001
        return "unknown"
    if sg1 == sg2:
        return "identical"
    # Accept if one symbol is a substring (e.g., different settings)
    if sg1 in sg2 or sg2 in sg1:
        return "related_setting"
    return "different"


def _rotation_from_lattices(
    s_left: Structure,
    s_right: Structure,
    unimodular_transform: np.ndarray,
) -> np.ndarray | None:
    """Recover the orthogonal Cartesian rotation from the matched lattices.

    The structure matcher supplies an integer unimodular matrix ``M`` such that
    the right fractional coordinates map onto the left cell.  The deformation
    gradient ``F = L_right @ M.T @ inv(L_left)`` encodes both the basis change
    and the physical rotation/strain.  Its polar factor is the orthogonal
    rotation that transports tensors from the right Cartesian frame to the left
    Cartesian frame.  Both proper and improper rotations are preserved.
    """
    left_lattice = np.asarray(s_left.lattice.matrix, dtype=np.float64)
    right_lattice = np.asarray(s_right.lattice.matrix, dtype=np.float64)
    m = np.asarray(unimodular_transform, dtype=np.float64)
    try:
        f = right_lattice @ m.T @ np.linalg.inv(left_lattice)
    except np.linalg.LinAlgError:
        return None
    u, _, vh = svd(f)
    rotation = u @ vh
    # Preserve determinant sign when the raw deformation is improper.
    if np.linalg.det(rotation) > 0.0 and np.linalg.det(f) < 0.0:
        rotation = -rotation
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        return None
    return rotation


def _periodic_image_aligned_coords(
    coords_left: np.ndarray,
    coords_right: np.ndarray,
    lattice_left: np.ndarray,
) -> np.ndarray:
    """Shift right Cartesian coordinates by lattice images to align with left.

    For each atom pair, choose the lattice-image shift that minimizes the
    Cartesian distance to the left coordinate.
    """
    inv_lattice = np.linalg.inv(lattice_left)
    frac_diff = (coords_right @ inv_lattice) - (coords_left @ inv_lattice)
    # Nearest integer image shift in the left fractional basis.
    image_shift = np.rint(frac_diff)
    return coords_right - image_shift @ lattice_left


def _kabsch_rotation(
    coords_left: np.ndarray,
    coords_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the best proper and improper orthogonal transforms aligning right to left.

    Returns
    -------
    rotation_proper:
        Best proper rotation (det = +1).
    rotation_improper:
        Best improper rotation (det = -1), if distinct; otherwise None.
    rms_proper:
        RMS reconstruction error for the proper rotation.
    rms_improper:
        RMS reconstruction error for the improper rotation.
    """
    left = np.asarray(coords_left, dtype=np.float64)
    right = np.asarray(coords_right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Coordinate arrays must have the same shape")

    centroid_left = np.mean(left, axis=0)
    centroid_right = np.mean(right, axis=0)
    p = left - centroid_left
    q = right - centroid_right

    h = q.T @ p
    u, _, vh = svd(h)
    rot_proper = vh.T @ u.T
    if np.linalg.det(rot_proper) < 0:
        # Force proper by reflecting the last singular vector.
        vh[-1, :] *= -1
        rot_proper = vh.T @ u.T

    # Improper candidate: reflect one axis after SVD.
    vh_imp = vh.copy()
    vh_imp[-1, :] *= -1
    rot_improper = vh_imp.T @ u.T
    if np.linalg.det(rot_improper) > 0:
        rot_improper = -rot_improper

    def _rms(r: np.ndarray) -> float:
        pred = q @ r.T + centroid_left
        return float(np.sqrt(np.mean((pred - left) ** 2)))

    return rot_proper, rot_improper, _rms(rot_proper), _rms(rot_improper)


def _rotation_between_matched_structures(
    matcher: PMGStructureMatcher,
    s_left: Structure,
    s_right: Structure,
) -> tuple[np.ndarray | None, np.ndarray | None, list[float] | None, list[int] | None, str | None, float | None]:
    """Estimate the Cartesian rotation and lattice transform for a matched pair.

    Returns
    -------
    rotation:
        3x3 orthogonal Cartesian rotation (det may be -1).
    unimodular_transform:
        Integer matrix mapping s_right fractional coordinates to the s_left cell.
    fractional_translation:
        Translation vector in the s_left fractional basis.
    atom_mapping:
        ``mapping[i]`` is the index in s_right corresponding to site ``i`` in s_left.
    rotation_class:
        One of ``basis_relabel``, ``proper_rotation``, ``improper_relation``,
        ``deformation``, or ``unresolved``.
    """
    try:
        transform = matcher.get_transformation(s_left, s_right)
        unimodular_transform = np.asarray(transform[0], dtype=np.float64)
        fractional_translation = [float(x) for x in transform[1]]
        atom_mapping = [int(x) for x in transform[2]]
    except Exception:  # noqa: BLE001
        return None, None, None, None, "unresolved", None

    mapping_array = np.asarray(atom_mapping, dtype=np.int64)
    coords_left = np.asarray(s_left.cart_coords, dtype=np.float64)
    coords_right_raw = np.asarray(s_right.cart_coords, dtype=np.float64)[mapping_array]
    lattice_left = np.asarray(s_left.lattice.matrix, dtype=np.float64)

    # Apply periodic image correction so that mapped atoms are in the same cell.
    coords_right = _periodic_image_aligned_coords(coords_left, coords_right_raw, lattice_left)

    # If image-corrected coordinates already coincide, this is a pure basis relabel.
    if np.allclose(coords_left, coords_right, atol=1e-5):
        return (
            np.eye(3, dtype=np.float64),
            unimodular_transform,
            fractional_translation,
            atom_mapping,
            "basis_relabel",
            0.0,
        )

    try:
        rot_proper, rot_improper, rms_proper, rms_improper = _kabsch_rotation(coords_left, coords_right)
    except Exception:  # noqa: BLE001
        return None, unimodular_transform, fractional_translation, atom_mapping, "unresolved", None

    # Choose the solution with lower RMS; improper is allowed if it reconstructs better.
    if rms_improper < rms_proper - 1e-6:
        rotation = rot_improper
        rotation_class = "improper_relation"
    else:
        rotation = rot_proper
        rotation_class = "proper_rotation"

    # If even the best Kabsch solution does not reproduce coordinates well, the
    # match contains real deformation/relaxation and the rotation is unresolved.
    best_rms = min(rms_proper, rms_improper)
    if best_rms > 0.5:
        return None, unimodular_transform, fractional_translation, atom_mapping, "deformation", best_rms

    return rotation, unimodular_transform, fractional_translation, atom_mapping, rotation_class, best_rms


def match_structures(
    left_key: str,
    right_key: str,
    left_cif: str | Structure,
    right_cif: str | Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    primitive_cell: bool = False,
    attempt_supercell: bool = False,
) -> MatchResult:
    """Run the frozen strict-match protocol on a candidate pair.

    Tier assignment:
      - Tier 0 is reserved for explicit upstream provenance matches; this
        routine assigns Tier 1 for a successful structure match and Tier 3
        for formula-only input.
      - Tier 2 is not assigned automatically; it requires a prototype-aware
        analysis that this minimal Phase-3 routine does not perform.
    """
    match_key = f"{left_key}__{right_key}"
    reasons: list[str] = []
    try:
        s_left = _as_structure(left_cif)
        s_right = _as_structure(right_cif)
    except Exception as exc:  # noqa: BLE001
        return MatchResult(
            match_key=match_key,
            tier=MatchTier.QUARANTINE,
            fit=False,
            reasons=[f"cif_parse_error: {exc}"],
        )

    if len(s_left) != len(s_right):
        return MatchResult(
            match_key=match_key,
            tier=MatchTier.TIER_3,
            fit=False,
            reasons=["different_atom_counts"],
        )

    if s_left.composition.reduced_formula != s_right.composition.reduced_formula:
        return MatchResult(
            match_key=match_key,
            tier=MatchTier.TIER_3,
            fit=False,
            reasons=["different_reduced_formula"],
        )

    sg_rel = _space_group_relation(s_left, s_right)
    if sg_rel == "different":
        reasons.append("space_group_different")

    matcher = PMGStructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=primitive_cell,
        scale=True,
        attempt_supercell=attempt_supercell,
    )

    fit = matcher.fit(s_left, s_right)
    if not fit:
        reasons.extend(["structure_match_failed", f"space_group_relation:{sg_rel}"])
        return MatchResult(
            match_key=match_key,
            tier=MatchTier.QUARANTINE,
            fit=False,
            lattice_distance=_lattice_distance(s_left, s_right),
            space_group_relation=sg_rel,
            reasons=reasons,
        )

    rms = matcher.get_rms_dist(s_left, s_right)  # type: ignore[no-untyped-call]
    # get_rms_dist returns (rms displacement, maximum distance).
    rms_dist, max_dist = (float(rms[0]), float(rms[1])) if rms is not None else (None, None)

    lattice_dist = _lattice_distance(s_left, s_right)

    rotation, unimodular_transform, fractional_translation, atom_mapping, rotation_class, kabsch_rms = (
        _rotation_between_matched_structures(matcher, s_left, s_right)
    )

    if sg_rel == "different":
        # Even if pymatgen fit succeeded, differing space groups are suspicious.
        reasons.append("space_group_mismatch_kept_for_review")
        tier = MatchTier.QUARANTINE
    else:
        tier = MatchTier.TIER_1

    # If the rotation reconstruction is unresolved/deformation, downgrade to quarantine.
    if rotation_class in ("deformation", "unresolved"):
        reasons.append(f"rotation_unresolved:{rotation_class}")
        tier = MatchTier.QUARANTINE

    return MatchResult(
        match_key=match_key,
        tier=tier,
        fit=True,
        rms_distance=rms_dist,
        max_distance=max_dist,
        atom_mapping=atom_mapping,
        lattice_distance=lattice_dist,
        cartesian_rotation=rotation.tolist() if rotation is not None else None,
        fractional_translation=fractional_translation,
        unimodular_cell_transform=unimodular_transform.tolist() if unimodular_transform is not None else None,
        space_group_relation=sg_rel,
        reasons=reasons or None,
        rotation_class=rotation_class,
        kabsch_rms=kabsch_rms,
    )


def to_match_record(result: MatchResult) -> MatchRecord:
    """Convert a MatchResult to a Pydantic MatchRecord."""
    return MatchRecord(
        match_key=result.match_key,
        left_structure_key=result.match_key.split("__")[0],
        right_structure_key=result.match_key.split("__")[1],
        match_tier=result.tier,
        atom_permutation=result.atom_mapping,
        lattice_distance=result.lattice_distance,
        cartesian_rotation=result.cartesian_rotation,
        fractional_translation=result.fractional_translation,
        unimodular_cell_transform=result.unimodular_cell_transform,
        site_distance=result.max_distance,
        space_group_relation=result.space_group_relation,
        ambiguity=result.ambiguity,
        rotation_class=result.rotation_class,
        kabsch_rms=result.kabsch_rms,
        pass_fail_reasons=result.reasons or [],
    )
