"""Sensor models aligned with the validated warehouse LiDAR model."""
from __future__ import annotations

from typing import Iterable

from .models import ClaimType


def _claim_from_truth_value(value: int) -> ClaimType:
  import sim2

  state = sim2.CellState(int(value))
  if state in (
      sim2.CellState.OCCUPIED_STATIC,
      sim2.CellState.OCCUPIED_DYNAMIC,
      sim2.CellState.TEMPORARILY_BLOCKED,
  ):
    return ClaimType.BLOCKED
  return ClaimType.FREE


def lidar_observations(
    truth_grid,
    position: tuple[int, int],
    other_positions: Iterable[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], ClaimType]:
  """Ray-cast lidar from a grid cell center, matching legacy range and rays."""
  import sim2

  world = sim2.GridWorld(truth_grid)
  origin_xy = sim2.cell_to_xy(position)
  observations, _ = world.observe_cells_lidar(
      origin_xy,
      max_range_cells=sim2.LIDAR_RANGE_CELLS,
      num_rays=sim2.LIDAR_NUM_RAYS,
      step_cells=sim2.LIDAR_STEP_CELLS,
      robot_positions=set(other_positions or ()),
  )
  return {cell: _claim_from_truth_value(int(state)) for cell, state in observations.items()}
