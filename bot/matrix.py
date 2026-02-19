from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DIMENSIONS


class MatrixError(RuntimeError):
    pass


def load_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MatrixError(f"Matrix file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        matrix = json.load(f)

    missing_dimensions = [d for d in DIMENSIONS if d not in matrix.get("dimensions", {})]
    if missing_dimensions:
        raise MatrixError(f"Matrix missing dimensions: {', '.join(missing_dimensions)}")

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
