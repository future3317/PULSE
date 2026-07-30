"""Read-only data inventory scanner for CrossPiezo Phase 1."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crosspiezo.schemas import SourceArtifact


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _head_tail_hash(path: Path, n_bytes: int = 4096) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
        if size > n_bytes:
            f.seek(max(n_bytes, size - n_bytes))
            h.update(f.read(n_bytes))
    h.update(str(size).encode())
    return h.hexdigest()


def _classify_asset(path: Path, role: str | None) -> dict[str, Any]:
    stat = path.stat()
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    if size <= 10 * 1024 * 1024 or path.suffix in {".json", ".jsonl", ".yaml", ".yml", ".md", ".csv"}:
        fingerprint = _sha256_file(path)
        scan_strategy = "full_sha256"
    else:
        fingerprint = _head_tail_hash(path)
        scan_strategy = "head_tail_hash"
    return {
        "path": str(path),
        "size_bytes": size,
        "mtime_utc": mtime,
        "extension": path.suffix,
        "fingerprint": fingerprint,
        "scan_strategy": scan_strategy,
        "role": role,
        "exists": True,
    }


def scan_t2c_flow(root: Path, config: dict[str, Any]) -> list[SourceArtifact]:
    """Inventory the T2C-Flow piezoelectric assets."""
    records: list[SourceArtifact] = []
    root = Path(config["root"])
    records.append(SourceArtifact(
        source_name="T2C-Flow",
        source_version="2026-07-16",
        path=root / "MANIFEST.json",
        sha256_or_fingerprint=_sha256_file(root / "MANIFEST.json"),
        license="ODC-BY / original database terms",
        role="manifest",
    ))
    records.append(SourceArtifact(
        source_name="T2C-Flow",
        source_version="2026-07-16",
        path=root / "README.md",
        role="documentation",
    ))
    for key, rel in config["records"].items():
        path = Path(config["root"]) / rel
        if path.exists():
            rec = _classify_asset(path, role=f"t2c_{key}")
            records.append(SourceArtifact(
                source_name="T2C-Flow",
                source_version="2026-07-16",
                path=path,
                sha256_or_fingerprint=rec["fingerprint"],
                license="ODC-BY / original database terms",
                role=rec["role"],
            ))
    return records


def scan_piezojet(config: dict[str, Any]) -> list[SourceArtifact]:
    """Inventory PiezoJet strict-factor assets."""
    records: list[SourceArtifact] = []
    root = Path(config["root"])
    manifest = root / config["manifest"]
    if manifest.exists():
        records.append(SourceArtifact(
            source_name="PiezoJet",
            source_version="jarvis_dfpt_v9_full_public",
            path=manifest,
            sha256_or_fingerprint=_sha256_file(manifest),
            role="manifest",
        ))
    factor_root = Path(config["strict_factors"]["root"])
    files = sorted(factor_root.glob("*.pt"))
    # Represent the cohort by its manifest; do not hash 5k binary files individually.
    records.append(SourceArtifact(
        source_name="PiezoJet",
        source_version="jarvis_dfpt_v9_full_public",
        path=factor_root,
        sha256_or_fingerprint=f"count:{len(files)}",
        role="strict_factor_files",
    ))
    return records


def build_inventory(config_path: Path) -> pd.DataFrame:
    """Build the read-only data inventory from the data-sources config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    artifacts: list[SourceArtifact] = []
    sources = cfg.get("sources", {})
    if "t2c_flow" in sources:
        artifacts.extend(scan_t2c_flow(Path(sources["t2c_flow"]["root"]), sources["t2c_flow"]))
    if "piezojet" in sources:
        artifacts.extend(scan_piezojet(sources["piezojet"]))
    rows = [a.model_dump() for a in artifacts]
    df = pd.DataFrame(rows)
    df["path"] = df["path"].astype(str)
    return df


def load_data_sources(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return dict(yaml.safe_load(f))
