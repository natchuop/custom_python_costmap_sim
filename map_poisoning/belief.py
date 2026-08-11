"""Per-robot direct belief state, intentionally separate from peer fusion."""
from __future__ import annotations

import math
import numpy as np

from .models import Cell, ClaimType, DirectObservation


class RobotBeliefMap:
    """A robot's local, directly observed view of the world."""

    UNKNOWN_TRAVERSAL_COST = 3.0
    # Free-space observations expire sooner than obstacle observations so an
    # old local reading cannot suppress a newer trusted peer blockage.
    DIRECT_FREE_MEMORY_STEPS = 40

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

    def direct_state(self, cell: Cell, step: int | None = None) -> ClaimType | None:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return ClaimType.BLOCKED
        item = self.direct.get(cell)
        if item is None:
            return None
        if (
            step is not None
            and item.claim == ClaimType.FREE
            and max(0, int(step) - int(item.step)) > self.DIRECT_FREE_MEMORY_STEPS
        ):
            return None
        return item.claim

    def has_direct_free(self, cell: Cell, step: int | None = None) -> bool:
        return self.direct_state(cell, step) == ClaimType.FREE

    def direct_free_strength(self, cell: Cell, step: int) -> float:
        item = self.direct.get(cell)
        if item is None or item.claim != ClaimType.FREE:
            return 0.0
        age = max(0, step - int(item.step))
        return 1.25 * math.exp(-0.01 * age)

    def is_blocked_for_planning(self, cell: Cell, fusion, step: int) -> bool:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return True
        direct = self.direct_state(cell, step)
        if direct == ClaimType.BLOCKED:
            return True
        if direct == ClaimType.FREE:
            # trust_fused treats a fresh local LiDAR reading as the highest-
            # trust source. A peer blocked claim must not override it until
            # this robot senses the cell itself again.
            if fusion.method == "trust_fused":
                return False
            if fusion.method != "trust_threshold":
                return fusion.footprint_hard_blocked([cell], step)
            return fusion._runner.blocked_support(cell, step) > self.direct_free_strength(cell, step)
        return fusion.footprint_hard_blocked([cell], step)

    def traversal_cost(self, cell: Cell, step: int, fusion) -> float:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return math.inf
        if self.is_blocked_for_planning(cell, fusion, step):
            return math.inf
        direct = self.direct_state(cell, step)
        if direct == ClaimType.FREE:
            if fusion.method == "trust_fused":
                return 1.0
            if fusion.method != "trust_threshold":
                return 1.0
            peer = fusion._runner.blocked_support(cell, step)
            local = self.direct_free_strength(cell, step)
            return 1.0 + fusion.routing_cost(cell, step) * min(1.0, peer / max(local, 1e-9))
        max_cost = self.UNKNOWN_TRAVERSAL_COST if direct is None else 1.0
        if (
            fusion.method == "trust_fused"
            and fusion.selected_claim(cell, step) == ClaimType.FREE
        ):
            return 1.0
        peer_cost = fusion.routing_cost(cell, step)
        if math.isinf(peer_cost):
            return math.inf
        return max(max_cost, peer_cost)
