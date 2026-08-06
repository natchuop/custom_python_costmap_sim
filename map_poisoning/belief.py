"""Per-robot direct belief state, intentionally separate from peer fusion."""
from __future__ import annotations

import math
import numpy as np

from .models import Cell, ClaimType, DirectObservation


class RobotBeliefMap:
    """A robot's local, directly observed view of the world.

    Peer claims are never written into this map.  They are retained by that
    robot's fusion engine so changing source trust can safely reweight them.
    """
    def __init__(self, static_grid: np.ndarray):
        self.static_grid = np.asarray(static_grid, dtype=np.uint8)
        self.rows, self.cols = self.static_grid.shape
        self.direct: dict[Cell, DirectObservation] = {}

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def observe(self, observation: DirectObservation) -> bool:
        if not self.in_bounds(observation.cell) or self.static_grid[observation.cell]:
            return False
        previous = self.direct.get(observation.cell)
        self.direct[observation.cell] = observation
        return previous is None or previous.claim != observation.claim

    def direct_state(self, cell: Cell) -> ClaimType | None:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return ClaimType.BLOCKED
        item = self.direct.get(cell)
        return item.claim if item else None

    def traversal_cost(self, cell: Cell, step: int, fusion) -> float:
        state = self.direct_state(cell)
        if state == ClaimType.BLOCKED:
            return math.inf
        if state == ClaimType.FREE:
            # A fresh local sensor observation takes precedence over peer
            # blockage evidence until this recipient observes the cell again.
            return 1.0
        return fusion.routing_cost(cell, step)
