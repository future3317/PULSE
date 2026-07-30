"""Immutable data contracts for CrossPiezo / PULSE Phase 0-4."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MatchTier(StrEnum):
    """Structure-match tier for a cross-source pair."""

    TIER_0 = "tier_0"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    UNMATCHED = "unmatched"
    QUARANTINE = "quarantine"


class SourceArtifact(BaseModel):
    """A frozen upstream data artifact."""

    model_config = ConfigDict(frozen=True)

    source_name: str
    source_version: str
    path: Path
    sha256_or_fingerprint: str | None = None
    size_bytes: int | None = None
    license: str | None = None
    parser_version: str = "0.1.0"
    frozen_status: str = "frozen"
    role: str | None = None


class StructureRecord(BaseModel):
    """A structure with provenance and a unique internal key."""

    model_config = ConfigDict(frozen=True)

    structure_key: str
    source_name: str
    source_version: str
    material_id: str
    source_structure_id: str | None = None
    formula: str
    atomic_numbers: list[int]
    lattice: list[list[float]]  # 3x3 row vectors (Angstrom)
    fractional_coordinates: list[list[float]]  # nsites x 3
    space_group: int | None = None
    primitive_or_conventional: str = "unknown"
    structure_hash: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class TensorRecord(BaseModel):
    """A piezoelectric tensor with full convention history."""

    model_config = ConfigDict(frozen=True)

    tensor_key: str
    structure_key: str
    tensor_type: str = "piezo_stress"  # e_total, e_ionic, e_electronic, d_total, ...
    contribution: str = "total"  # total / electronic / ionic
    raw_shape: tuple[int, ...] = (3, 6)
    raw_voigt_order: str = "unknown"
    internal_voigt_order: str = "xx,yy,zz,yz,xz,xy"
    shear_convention: str = "engineering"
    unit: str = "C/m^2"
    cartesian_tensor: list[list[list[float]]]  # 3x3x3
    point_group: str | None = None
    symmetry_residual: float | None = None
    source_functional: str | None = None
    source_code: str | None = None
    transformation_history: list[dict[str, Any]] = Field(default_factory=list)


class MatchRecord(BaseModel):
    """A cross-source structure match."""

    model_config = ConfigDict(frozen=True)

    match_key: str
    left_structure_key: str
    right_structure_key: str
    match_tier: MatchTier
    unimodular_cell_transform: list[list[float]] | None = None
    fractional_translation: list[float] | None = None
    cartesian_rotation: list[list[float]] | None = None
    atom_permutation: list[int] | None = None
    lattice_distance: float | None = None
    site_distance: float | None = None
    space_group_relation: str | None = None
    ambiguity: str | None = None
    rotation_class: str | None = None
    kabsch_rms: float | None = None
    pass_fail_reasons: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    """A single reproducibility event."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    artifact: str
    check_name: str
    status: str
    message: str
    code_commit: str | None = None
    config_hash: str | None = None


def array_to_nested(arr: np.ndarray) -> list[Any]:
    """Convert a NumPy array to nested Python lists."""
    return list(arr.tolist())


def nested_to_array(nested: list[Any]) -> np.ndarray:
    """Convert nested Python lists to a NumPy array."""
    return np.asarray(nested, dtype=np.float64)
