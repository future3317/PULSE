"""Build CrossPiezo Core/Extended benchmark panels for Phase 5B."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from crosspiezo.conventions.symmetry import (
    point_group_rotations,
    symmetry_residual,
)
from crosspiezo.conventions.voigt import voigt_to_cartesian


def _crystal_system(space_group_number: int) -> str:
    """Standard International Tables crystal-system ranges."""
    if 1 <= space_group_number <= 2:
        return "triclinic"
    if 3 <= space_group_number <= 15:
        return "monoclinic"
    if 16 <= space_group_number <= 74:
        return "orthorhombic"
    if 75 <= space_group_number <= 142:
        return "tetragonal"
    if 143 <= space_group_number <= 167:
        return "trigonal"
    if 168 <= space_group_number <= 194:
        return "hexagonal"
    if 195 <= space_group_number <= 230:
        return "cubic"
    return "unknown"


def _to_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    return None


def _tensor_from_row(row: pd.Series) -> np.ndarray | None:
    cart = _to_array(row.get("piezo_cartesian_total"))
    if cart is not None and cart.shape == (3, 3, 3):
        return cart
    voigt = _to_array(row.get("piezo_voigt_total"))
    if voigt is not None and voigt.shape == (3, 6):
        return voigt_to_cartesian(voigt, engineering_shear=True)
    return None


def _space_group_symbol(space_group: Any) -> str | None:
    try:
        return f"{int(float(space_group))}"
    except Exception:  # noqa: BLE001
        return None





def _transport_rotations(rotations: list[np.ndarray], frame_rotation: np.ndarray) -> list[np.ndarray]:
    """Conjugate source-frame point-group rotations into the common frame."""
    rot = np.asarray(frame_rotation, dtype=np.float64)
    rot_inv = np.linalg.inv(rot)
    return [np.asarray(rot @ g @ rot_inv, dtype=np.float64) for g in rotations]


def _normalized_residual(tensor: np.ndarray, rotations: list[np.ndarray] | None) -> tuple[float, float]:
    if rotations is None or len(rotations) == 0:
        return float("nan"), float("nan")
    raw = symmetry_residual(tensor, rotations)
    norm = np.linalg.norm(tensor) + 1e-12
    return raw, raw / norm


def build_enriched_pairs(data_root: Path) -> pd.DataFrame:
    """Load JARVIS/MP records and strict pairs with full tensors and transforms."""
    import yaml

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "data_sources.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sources = cfg["sources"]["t2c_flow"]
    root = Path(data_root) / "T2C-Flow" if data_root is not None else Path(sources["root"])
    jarvis = pd.read_parquet(root / sources["records"]["jarvis_piezo"])
    mp = pd.read_parquet(root / sources["records"]["mp_piezo"])
    pairs = pd.read_parquet(Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "pair_manifests" / "strict_pairs.parquet")
    matches = pd.read_parquet(Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "pair_manifests" / "all_matches.parquet")
    match_by_key = {row["match_key"]: row for _, row in matches.iterrows()}

    jarvis_by_id = {row["material_id"]: row for _, row in jarvis.iterrows()}
    mp_by_id = {row["material_id"]: row for _, row in mp.iterrows()}

    records: list[dict[str, Any]] = []
    for _, prow in pairs.iterrows():
        jid = prow["jarvis_id"]
        mid = prow["mp_id"]
        jrow = jarvis_by_id.get(jid)
        mrow = mp_by_id.get(mid)
        if jrow is None or mrow is None:
            continue
        jtensor = _tensor_from_row(jrow)
        mtensor = _tensor_from_row(mrow)
        if jtensor is None or mtensor is None:
            continue

        match_key = f"jarvis:{jid}__mp:{mid}"
        mrec = match_by_key.get(match_key)
        rotation = None
        atom_perm = None
        if mrec is not None and mrec["cartesian_rotation"] is not None:
            rot_val = mrec["cartesian_rotation"]
            if isinstance(rot_val, np.ndarray) and rot_val.dtype == object:
                rotation = np.stack([np.asarray(v, dtype=np.float64) for v in rot_val])
            else:
                rotation = np.asarray(rot_val, dtype=np.float64)
            atom_perm = mrec["atom_permutation"]

        sg_num = int(float(jrow["space_group"])) if pd.notna(jrow["space_group"]) else None
        mtensor_aligned = mtensor
        if rotation is not None:
            mtensor_aligned = np.einsum("il,jm,kn,lmn->ijk", rotation, rotation, rotation, mtensor)

        records.append({
            "jarvis_id": jid,
            "mp_id": mid,
            "formula": jrow["formula"],
            "space_group": sg_num,
            "crystal_system": _crystal_system(sg_num) if sg_num else "unknown",
            "jarvis_tensor": jtensor,
            "mp_tensor_raw": mtensor,
            "mp_tensor_aligned": mtensor_aligned,
            "rotation": rotation,
            "atom_permutation": atom_perm,
            "rms_distance": prow["rms_distance"],
            "max_distance": prow["max_distance"],
            "lattice_distance": prow["lattice_distance"],
            "space_group_relation": prow["space_group_relation"],
            "jarvis_norm": float(np.linalg.norm(jtensor)),
            "mp_norm": float(np.linalg.norm(mtensor_aligned)),
            "jarvis_cif": jrow["cif"],
            "mp_cif": mrow["cif"],
        })

    return pd.DataFrame(records)


def _structure_from_cif(cif_string: str | None) -> Structure | None:
    """Parse a CIF string into a pymatgen Structure, or None if unavailable."""
    if not isinstance(cif_string, str):
        return None
    try:
        return Structure.from_str(cif_string, fmt="cif")
    except Exception:  # noqa: BLE001
        return None


def compute_source_native_residuals(enriched: pd.DataFrame) -> pd.DataFrame:
    """Compute native, transported, and common symmetry residuals per source.

    Each source's native residual is computed with the point group of that
    source's actual CIF structure, not with a shared abstract space-group symbol.
    """
    from pymatgen.core.structure import Structure

    rows: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        rotation = row["rotation"]

        jarvis_struct = _structure_from_cif(row.get("jarvis_cif"))
        mp_struct = _structure_from_cif(row.get("mp_cif"))

        source_structures: dict[str, Structure | None] = {
            "jarvis": jarvis_struct,
            "mp": mp_struct,
        }

        # Determine a common structure for the common-frame residual.  Prefer the
        # JARVIS structure when available; otherwise use MP.
        common_struct = jarvis_struct if jarvis_struct is not None else mp_struct

        for source, tensor_key in [("jarvis", "jarvis_tensor"), ("mp", "mp_tensor_raw")]:
            tensor = row[tensor_key]
            source_struct = source_structures[source]

            if source_struct is None:
                rows.append({
                    "jarvis_id": row["jarvis_id"],
                    "mp_id": row["mp_id"],
                    "source": source,
                    "space_group": row["space_group"],
                    "crystal_system": row["crystal_system"],
                    "norm": float(np.linalg.norm(tensor)),
                    "native_residual_raw": float("nan"),
                    "native_residual_normalized": float("nan"),
                    "transport_residual_raw": float("nan"),
                    "transport_residual_normalized": float("nan"),
                    "common_residual_raw": float("nan"),
                    "common_residual_normalized": float("nan"),
                    "native_frame_status": "native_frame_unresolved",
                })
                continue

            try:
                source_rots = point_group_rotations(source_struct)
            except Exception:  # noqa: BLE001
                source_rots = []

            # Native frame: source CIF point group.
            native_raw, native_norm = _normalized_residual(tensor, source_rots)
            native_status = "native_frame_verified" if source_rots else "native_frame_unresolved"

            # Transported frame: conjugate source point group to common frame.
            if rotation is not None and source_rots:
                trans_rots = _transport_rotations(source_rots, rotation)
                # For MP, the tensor in common frame is mp_tensor_aligned.
                t_tensor = row["jarvis_tensor"] if source == "jarvis" else row["mp_tensor_aligned"]
                transport_raw, transport_norm = _normalized_residual(t_tensor, trans_rots)
            else:
                transport_raw, transport_norm = float("nan"), float("nan")

            # Common frame: common point group.
            if common_struct is not None:
                try:
                    common_rots = point_group_rotations(common_struct)
                except Exception:  # noqa: BLE001
                    common_rots = []
            else:
                common_rots = []
            c_tensor = row["jarvis_tensor"] if source == "jarvis" else row["mp_tensor_aligned"]
            common_raw, common_norm = _normalized_residual(c_tensor, common_rots)

            rows.append({
                "jarvis_id": row["jarvis_id"],
                "mp_id": row["mp_id"],
                "source": source,
                "space_group": row["space_group"],
                "crystal_system": row["crystal_system"],
                "norm": float(np.linalg.norm(tensor)),
                "native_residual_raw": native_raw,
                "native_residual_normalized": native_norm,
                "transport_residual_raw": transport_raw,
                "transport_residual_normalized": transport_norm,
                "common_residual_raw": common_raw,
                "common_residual_normalized": common_norm,
                "native_frame_status": native_status,
            })
    return pd.DataFrame(rows)


def assign_sublayers(enriched: pd.DataFrame) -> pd.DataFrame:
    """Assign T1a/T1b/T1c sublayers from structure-mediated shift metrics."""
    from pymatgen.core.structure import Structure

    sublayers: list[str] = []
    for _, row in enriched.iterrows():
        try:
            jstruct = Structure.from_str(row["jarvis_cif"], fmt="cif")
            mstruct = Structure.from_str(row["mp_cif"], fmt="cif")
        except Exception:  # noqa: BLE001
            sublayers.append("unparsed")
            continue

        lattice_strain = float(np.max(np.abs(np.array(jstruct.lattice.abc) / np.array(mstruct.lattice.abc) - 1.0)))
        site_max = row["max_distance"] if pd.notna(row["max_distance"]) else float("inf")
        try:
            sg_equal = jstruct.get_space_group_info()[1] == mstruct.get_space_group_info()[1]
        except Exception:  # noqa: BLE001
            sg_equal = False

        if site_max < 0.05 and lattice_strain < 0.05 and sg_equal:
            sublayers.append("T1a")
        elif site_max < 0.2 and lattice_strain < 0.1:
            sublayers.append("T1b")
        else:
            sublayers.append("T1c")
    enriched = enriched.copy()
    enriched["sublayer"] = sublayers
    return enriched


def build_core_extended_panels(
    enriched: pd.DataFrame,
    residual_df: pd.DataFrame,
    native_threshold: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (extended, core, exclusion log) dataframes."""
    enriched = assign_sublayers(enriched)

    # Map residual metrics to pairs.
    residual_pivot = residual_df.pivot_table(
        index=["jarvis_id", "mp_id"],
        columns="source",
        values=["native_residual_normalized", "transport_residual_normalized", "common_residual_normalized"],
    )
    residual_pivot.columns = [f"{metric}_{source}" for metric, source in residual_pivot.columns]
    residual_pivot = residual_pivot.reset_index()

    merged = enriched.merge(residual_pivot, on=["jarvis_id", "mp_id"], how="left")

    # Extended: all Tier-1 pairs with required tensors.
    extended = merged.copy()

    # Core: T1a plus residual audit pass or explained by transport.
    def _core_ok(r: pd.Series) -> bool:
        if r["sublayer"] != "T1a":
            return False
        for source in ["jarvis", "mp"]:
            native = r.get(f"native_residual_normalized_{source}")
            transport = r.get(f"transport_residual_normalized_{source}")
            common = r.get(f"common_residual_normalized_{source}")
            if pd.isna(native):
                return False
            if native <= native_threshold:
                continue
            # If native is high but transport is low, the frame mismatch explains it.
            if pd.notna(transport) and transport <= native_threshold and pd.notna(common) and common <= native_threshold:
                continue
            return False
        return True

    core_mask = merged.apply(_core_ok, axis=1)
    core = merged[core_mask].copy()

    exclusions = merged[~core_mask].copy()
    exclusions["reason"] = exclusions.apply(lambda r: _exclusion_reason(r, native_threshold), axis=1)

    return extended, core, exclusions


def _exclusion_reason(r: pd.Series, threshold: float) -> str:
    if r["sublayer"] != "T1a":
        return f"not_T1a ({r['sublayer']})"
    reasons: list[str] = []
    for source in ["jarvis", "mp"]:
        native = r.get(f"native_residual_normalized_{source}")
        transport = r.get(f"transport_residual_normalized_{source}")
        common = r.get(f"common_residual_normalized_{source}")
        if pd.isna(native):
            reasons.append(f"{source}_missing_residual")
        elif native > threshold:
            if pd.notna(transport) and transport <= threshold and pd.notna(common) and common <= threshold:
                continue
            reasons.append(f"{source}_native_residual_{native:.3f}")
    return ";".join(reasons) if reasons else "unknown"
