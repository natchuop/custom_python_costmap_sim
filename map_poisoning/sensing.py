"""Deterministic 360-degree line-of-sight sensing for the grid simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .models import ClaimType

LIDAR_RANGE_CELLS = 5
LIDAR_MIN_CONFIDENCE = 0.60


@dataclass(frozen=True)
class LidarReading:
    claim: ClaimType
    sensor_confidence: float
    distance: float


def sensor_confidence_for_distance(distance: float, radius: int = LIDAR_RANGE_CELLS) -> float:
    """Return the configured confidence for a Euclidean range measurement.

    Confidence is 1.0 at one-cell range and falls linearly to 0.60 at the
    five-cell maximum. The robot's own cell is treated as confidence 1.0.
    """
    if distance <= 1.0:
        return 1.0
    if distance >= float(radius):
        return LIDAR_MIN_CONFIDENCE
    span = max(1.0, float(radius - 1))
    fraction = (distance - 1.0) / span
    return 1.0 - (1.0 - LIDAR_MIN_CONFIDENCE) * fraction


def _supercover_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Grid cells touched by the segment between two cell centers.

    This integer supercover-style walk is deterministic and conservative at
    corners, which is useful for occupancy occlusion on a grid.
    """
    r0, c0 = start
    r1, c1 = end
    dr = r1 - r0
    dc = c1 - c0
    nr = abs(dr)
    nc = abs(dc)
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    r, c = r0, c0
    ir = ic = 0
    cells = [(r, c)]
    while ir < nr or ic < nc:
        # Compare crossings of the next horizontal/vertical grid boundaries.
        left = (1 + 2 * ir) * nc
        right = (1 + 2 * ic) * nr
        if left == right:
            r += sr
            c += sc
            ir += 1
            ic += 1
        elif left < right:
            r += sr
            ir += 1
        else:
            c += sc
            ic += 1
        cells.append((r, c))
    return cells


def _visible(truth_grid, start: tuple[int, int], target: tuple[int, int]) -> bool:
    """The target is visible if no earlier cell on its ray is blocked."""
    ray = _supercover_line(start, target)
    for cell in ray[1:-1]:
        if truth_grid[cell]:
            return False
    return True


def lidar_observations(
    truth_grid,
    position: tuple[int, int],
    other_positions: Iterable[tuple[int, int]] | None = None,
    *,
    radius: int = LIDAR_RANGE_CELLS,
) -> dict[tuple[int, int], LidarReading]:
    """Return a circular, 360-degree, line-of-sight occupancy scan.

    Other robots remain traffic participants rather than map obstacles. The
    first physical obstacle on a ray is visible; cells behind it are not.
    """
    rows, cols = truth_grid.shape
    seen: dict[tuple[int, int], LidarReading] = {}
    r0, c0 = position
    radius_sq = radius * radius
    for row in range(max(0, r0 - radius), min(rows, r0 + radius + 1)):
        for col in range(max(0, c0 - radius), min(cols, c0 + radius + 1)):
            dr = row - r0
            dc = col - c0
            squared = dr * dr + dc * dc
            if squared > radius_sq:
                continue
            cell = (row, col)
            if not _visible(truth_grid, position, cell):
                continue
            distance = math.sqrt(squared)
            seen[cell] = LidarReading(
                ClaimType.BLOCKED if truth_grid[cell] else ClaimType.FREE,
                sensor_confidence_for_distance(distance, radius),
                distance,
            )
    return seen
