"""Deterministic local sensing for the modular grid simulator."""
from __future__ import annotations

from typing import Iterable

from .models import ClaimType


def lidar_observations(
    truth_grid,
    position: tuple[int, int],
    other_positions: Iterable[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], ClaimType]:
  """Return a radius-five local occupancy observation.

  Other robots are not environmental obstacles.  This intentionally simple
  sensor makes the information available to trust verification explicit and
  reproducible on every supported map.
  """
  rows, cols = truth_grid.shape
  seen: dict[tuple[int, int], ClaimType] = {}
  radius = 5
  occupied_by_robot = set(other_positions or ())
  for row in range(max(0, position[0] - radius), min(rows, position[0] + radius + 1)):
    for col in range(max(0, position[1] - radius), min(cols, position[1] + radius + 1)):
      cell = (row, col)
      if abs(row - position[0]) + abs(col - position[1]) > radius or cell in occupied_by_robot:
        continue
      seen[cell] = ClaimType.BLOCKED if truth_grid[cell] else ClaimType.FREE
  return seen
