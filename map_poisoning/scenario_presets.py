"""Validated, reproducible scenario geometry for converted experiment maps."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

from .planning import astar

Cell = tuple[int, int]


@dataclass(frozen=True)
class FixedScenarioPreset:
    preset_id: str
    expected_shape: tuple[int, int]
    robot_starts: Mapping[int, Cell]
    delivery_points: tuple[Cell, ...]
    expected_map_hash: str | None = None
    description: str = ""


# Coordinates were selected from the largest 4-connected free-space component
# of each converted map.  Every point has a free 3x3 neighbourhood; the
# geographically separated points deliberately create crossing warehouse
# routes instead of four nearly coincident delivery actions.
PRESETS: dict[str, FixedScenarioPreset] = {
    "warehouse_002": FixedScenarioPreset(
        "warehouse_002", (188, 192),
        {0: (101, 105), 1: (37, 15), 2: (119, 28)},
        ((42, 139), (66, 63), (40, 89), (123, 133)),
        "b8f295ab98f5ebef3970dafa537a63468ed8c0f417ec5a47223328a90e447b3b",
        "Separated clear cells spanning the converted warehouse-002 zones.",
    ),
    "warehouse_005": FixedScenarioPreset(
        "warehouse_005", (48, 80),
        {0: (36, 35), 1: (24, 6), 2: (16, 52)},
        ((14, 25), (44, 15), (45, 51), (29, 21)),
        "784921874e5fbaad904527cc019bf0d960b600820de1fb1800f075905804e523",
        "Separated clear cells spanning the converted warehouse-005 zones.",
    ),
    "warehouse_005_rotated": FixedScenarioPreset(
        "warehouse_005_rotated", (35, 52),
        {0: (26, 5), 1: (4, 48), 2: (32, 37)},
        ((9, 21), (14, 39), (25, 25), (25, 49)),
        "6a10880abafdd65e8b0a38d3d4463164b87cd4fbbb873742cbd110df4ee5fe62",
        "Rotated-map geometry is explicit and does not alter default warehouse authoring.",
    ),
}


PRESET_MAP_RELATIVE_PATHS: dict[str, str] = {
    "warehouse_002": "converted_maps/maps_002_map/static_grid.npy",
    "warehouse_005": "converted_maps/maps_005_map/static_grid.npy",
    "warehouse_005_rotated": "converted_maps/maps_005_map_rotated/static_grid.npy",
}


def map_path_for_preset(preset_id: str) -> str:
    """Return the packaged converted NPY map for a fixed preset."""
    preset_for_id(preset_id)  # validate identifier first
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    path = root / PRESET_MAP_RELATIVE_PATHS[preset_id]
    if not path.exists():
        raise FileNotFoundError(f"converted map for scenario preset {preset_id} is missing: {path}")
    return str(path)


def validate_fixed_preset(grid: np.ndarray, preset: FixedScenarioPreset) -> None:
    """Reject changed, blocked, duplicated, or disconnected experiment geometry."""
    prefix = f"scenario preset {preset.preset_id} is invalid"
    if tuple(grid.shape) != preset.expected_shape:
        raise ValueError(f"{prefix}: expected shape {preset.expected_shape}, got {tuple(grid.shape)}")
    if preset.expected_map_hash and hashlib.sha256(grid.tobytes()).hexdigest() != preset.expected_map_hash:
        raise ValueError(f"{prefix}: map hash does not match the validated conversion")
    points = list(preset.robot_starts.items()) + list(enumerate(preset.delivery_points))
    for label, cell in points:
        if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
            raise ValueError(f"{prefix}: {label} point {cell} is out of bounds")
        if grid[cell] != 0:
            raise ValueError(f"{prefix}: point {cell} is blocked")
    starts = list(preset.robot_starts.values())
    if len(starts) != len(set(starts)):
        raise ValueError(f"{prefix}: robot starts must be distinct")
    if len(preset.delivery_points) != len(set(preset.delivery_points)):
        raise ValueError(f"{prefix}: delivery points must be distinct")
    if set(starts) & set(preset.delivery_points):
        raise ValueError(f"{prefix}: a robot start equals a delivery point")

    rows, cols = grid.shape
    traversable = lambda cell: (
        1.0 if 0 <= cell[0] < rows and 0 <= cell[1] < cols and grid[cell] == 0
        else float("inf")
    )
    for robot_id, start in preset.robot_starts.items():
        for index, target in enumerate(preset.delivery_points):
            if astar(start, target, traversable) is None:
                raise ValueError(f"{prefix}: robot {robot_id} start cannot reach delivery point {index}")


def preset_for_id(preset_id: str) -> FixedScenarioPreset:
    try:
        return PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"unknown scenario preset: {preset_id}") from exc


def preset_for_hash(map_hash: str) -> FixedScenarioPreset | None:
    return next((preset for preset in PRESETS.values() if preset.expected_map_hash == map_hash), None)
