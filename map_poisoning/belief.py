"""Per-robot direct belief state, intentionally separate from peer fusion."""
from __future__ import annotations

import math
import numpy as np

from .models import Cell, ClaimType, DirectObservation


class RobotBeliefMap:
    """A robot's local, directly observed view of the world."""

    UNKNOWN_TRAVERSAL_COST = 3.0
    DEFAULT_MEMORY_STEPS = 12

    def __init__(self, static_grid: np.ndarray, memory_steps: int = DEFAULT_MEMORY_STEPS):
        self.static_grid = np.asarray(static_grid, dtype=np.uint8)
        self.rows, self.cols = self.static_grid.shape
        self.memory_steps = max(0, int(memory_steps))
        self.direct: dict[Cell, DirectObservation] = {}

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def observe(self, observation: DirectObservation) -> bool:
        if not self.in_bounds(observation.cell) or self.static_grid[observation.cell]:
            return False
        previous = self.direct.get(observation.cell)
        self.direct[observation.cell] = observation
        return previous is None or previous.claim != observation.claim

    def observation_status(self, cell: Cell, step: int | None = None) -> tuple[ClaimType | None, str]:
        """Return ``(claim, freshness)`` with freshness in {fresh, stale, unknown}."""
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return ClaimType.BLOCKED, "fresh"
        item = self.direct.get(cell)
        if item is None:
            return None, "unknown"
        if step is not None and item.step > step:
            return None, "unknown"
        if step is None or step - item.step <= self.memory_steps:
            return item.claim, "fresh"
        return item.claim, "stale"

    def direct_state(self, cell: Cell, step: int | None = None) -> ClaimType | None:
        claim, freshness = self.observation_status(cell, step)
        return claim if freshness == "fresh" else None

    def has_direct_free(self, cell: Cell, step: int | None = None) -> bool:
        return self.direct_state(cell, step) == ClaimType.FREE

    def is_blocked_for_planning(self, cell: Cell, fusion, step: int, *, hard_blocked_fn=None) -> bool:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return True
        claim, freshness = self.observation_status(cell, step)
        if freshness == "fresh" and claim == ClaimType.BLOCKED:
            return True
        if freshness == "fresh" and claim == ClaimType.FREE:
            # A robot currently looking at a free cell cannot be hard-blocked by peers.
            return False
        check = hard_blocked_fn or (lambda: fusion.footprint_hard_blocked([cell], step))
        return bool(check())

    def traversal_cost(self, cell: Cell, step: int, fusion, *, routing_cost_fn=None, hard_blocked_fn=None) -> float:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return math.inf
        if self.is_blocked_for_planning(cell, fusion, step, hard_blocked_fn=hard_blocked_fn):
            return math.inf
        claim, freshness = self.observation_status(cell, step)
        if freshness == "fresh" and claim == ClaimType.FREE:
            return 1.0
        peer_cost = (
            routing_cost_fn(cell, step)
            if routing_cost_fn is not None
            else fusion.routing_cost(cell, step)
        )
        if math.isinf(peer_cost):
            return math.inf
        floor = self.UNKNOWN_TRAVERSAL_COST if freshness == "unknown" else 1.0
        return max(floor, peer_cost)
