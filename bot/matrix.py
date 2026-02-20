from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .constants import DIMENSIONS

logger = logging.getLogger(__name__)


class MatrixError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise MatrixError(f"Matrix JSON must be an object: {path}")
    return payload


def _load_supplemental_matrices(primary_matrix_path: Path) -> list[dict[str, Any]]:
    supplemental_dir = primary_matrix_path.parent / "matrices"
    if not supplemental_dir.exists() or not supplemental_dir.is_dir():
        return []

    matrices: list[dict[str, Any]] = []
    for file_path in sorted(supplemental_dir.glob("*.json")):
        try:
            payload = _load_json(file_path)
        except Exception as exc:
            logger.warning("Skip invalid supplemental matrix '%s': %s", file_path, exc)
            continue

        payload.setdefault("name", file_path.stem)
        payload.setdefault("source", str(file_path))
        matrices.append(payload)

    return matrices


def load_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MatrixError(f"Matrix file not found: {path}")

    matrix = _load_json(path)

    missing_dimensions = [d for d in DIMENSIONS if d not in matrix.get("dimensions", {})]
    if missing_dimensions:
        raise MatrixError(f"Matrix missing dimensions: {', '.join(missing_dimensions)}")

    matrix["supplemental_matrices"] = _load_supplemental_matrices(path)
    return matrix


def required_dimensions_for_track(track: str | None) -> list[str]:
    normalized = (track or "").upper()
    include_management = normalized == "M"

    dimensions = []
    for dim in DIMENSIONS:
        if dim == "management" and not include_management:
            continue
        dimensions.append(dim)
    return dimensions
