"""Per-robot direct belief state, intentionally separate from peer fusion."""
from __future__ import annotations

import math
import numpy as np

from .models import Cell, ClaimType, DirectObservation


class RobotBeliefMap:
    """A robot's local, directly observed view of the world."""

    UNKNOWN_TRAVERSAL_COST = 3.0

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

    def has_direct_free(self, cell: Cell) -> bool:
        return self.direct_state(cell) == ClaimType.FREE

    def is_blocked_for_planning(self, cell: Cell, fusion, step: int) -> bool:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return True
        direct = self.direct_state(cell)
        if direct == ClaimType.BLOCKED:
            return True
        if direct == ClaimType.FREE:
            return fusion.footprint_hard_blocked([cell], step)
        return fusion.footprint_hard_blocked([cell], step)

    def traversal_cost(self, cell: Cell, step: int, fusion) -> float:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return math.inf
        if self.is_blocked_for_planning(cell, fusion, step):
            return math.inf
        direct = self.direct_state(cell)
        if direct == ClaimType.FREE:
            return 1.0
        max_cost = self.UNKNOWN_TRAVERSAL_COST if direct is None else 1.0
        peer_cost = fusion.routing_cost(cell, step)
        if math.isinf(peer_cost):
            return math.inf
        return max(max_cost, peer_cost)
