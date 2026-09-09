"""Per-robot direct belief state, separate from peer fusion."""
from __future__ import annotations

import math
import numpy as np

from .models import Cell, ClaimType, DirectObservation
from .config import DEFAULT_UNKNOWN_TRAVERSAL_COST


class RobotBeliefMap:
    DEFAULT_MEMORY_STEPS = 300

    def __init__(
        self,
        static_grid: np.ndarray,
        memory_steps: int = DEFAULT_MEMORY_STEPS,
        unknown_traversal_cost: float = DEFAULT_UNKNOWN_TRAVERSAL_COST,
    ):
        self.static_grid = np.asarray(static_grid, dtype=np.uint8)
        self.rows, self.cols = self.static_grid.shape
        self.memory_steps = max(1, int(memory_steps))
        if float(unknown_traversal_cost) < 1.0:
            raise ValueError("unknown_traversal_cost must be >= 1")
        self.unknown_traversal_cost = float(unknown_traversal_cost)
        self.direct: dict[Cell, DirectObservation] = {}
        self.current_visible: set[Cell] = set()
        self.current_scan_step: int | None = None

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def begin_scan(self, step: int) -> None:
        self.current_scan_step = int(step)
        self.current_visible.clear()

    def observe(self, observation: DirectObservation) -> bool:
        if not self.in_bounds(observation.cell) or self.static_grid[observation.cell]:
            return False
        previous = self.direct.get(observation.cell)
        self.direct[observation.cell] = observation
        if self.current_scan_step == observation.step:
            self.current_visible.add(observation.cell)
        return previous is None or previous.claim != observation.claim

    def age_weight(self, cell: Cell, step: int) -> float:
        item = self.direct.get(cell)
        if item is None or item.step > step:
            return 0.0
        age = max(0, step - item.step)
        return max(0.0, 1.0 - age / float(self.memory_steps))

    def memory_strength(self, cell: Cell, step: int) -> float:
        item = self.direct.get(cell)
        if item is None:
            return 0.0
        return min(1.0, max(0.0, item.sensor_confidence * self.age_weight(cell, step)))

    def observation_status(self, cell: Cell, step: int | None = None) -> tuple[ClaimType | None, str]:
        """Return ``(claim, status)`` where status is current/memory/unknown."""
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return ClaimType.BLOCKED, "current"
        item = self.direct.get(cell)
        if item is None:
            return None, "unknown"
        if step is not None and item.step > step:
            return None, "unknown"
        if step is not None and step - item.step >= self.memory_steps:
            return None, "unknown"
        if step is not None and self.current_scan_step == step and cell in self.current_visible and item.step == step:
            return item.claim, "current"
        return item.claim, "memory"

    def direct_state(self, cell: Cell, step: int | None = None) -> ClaimType | None:
        claim, status = self.observation_status(cell, step)
        return claim if status == "current" else None

    def display_state(self, cell: Cell, step: int | None = None, *, max_age: int | None = None) -> ClaimType | None:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return ClaimType.BLOCKED
        item = self.direct.get(cell)
        if item is None:
            return None
        if step is not None:
            age_limit = self.memory_steps if max_age is None else max(1, int(max_age))
            if step - item.step >= age_limit:
                return None
        return item.claim

    def prune_expired(self, step: int, *, max_age: int | None = None) -> int:
        age_limit = self.memory_steps if max_age is None else max(1, int(max_age))
        removed = 0
        for cell in list(self.direct):
            if step - self.direct[cell].step >= age_limit:
                del self.direct[cell]
                self.current_visible.discard(cell)
                removed += 1
        return removed

    def has_direct_free(self, cell: Cell, step: int | None = None) -> bool:
        return self.direct_state(cell, step) == ClaimType.FREE

    def peer_free_supersedes_direct_block(self, cell: Cell, fusion, step: int) -> bool:
        """Whether a newer operational peer FREE claim supersedes stale local BLOCKED memory.

        Current direct LiDAR remains authoritative. This rule exists so an
        accepted False Clearance can actually clear an obstacle the victim saw
        earlier, while an older/stale peer FREE report cannot erase newer local
        blocked knowledge.
        """
        direct = self.direct.get(cell)
        if direct is None or direct.claim != ClaimType.BLOCKED:
            return False
        _, status = self.observation_status(cell, step)
        if status == "current":
            return False
        if fusion.evidence(cell, step) >= -1e-12:
            return False
        newest_free = fusion.newest_operational_claim_step(cell, step, claim=ClaimType.FREE)
        return newest_free is not None and int(newest_free) > int(direct.step)

    def is_blocked_for_planning(self, cell: Cell, fusion, step: int, *, hard_blocked_fn=None) -> bool:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return True
        claim, status = self.observation_status(cell, step)
        if status == "current":
            return claim == ClaimType.BLOCKED
        if status == "memory" and claim == ClaimType.BLOCKED:
            # A directly observed obstacle remains a hard wall for planning
            # throughout its direct-memory lifetime. Previously remembered
            # BLOCKED cells were converted to a finite traversal penalty, so a
            # robot could repeatedly choose a path back through a physical
            # obstacle it had already seen. A newer operational peer FREE
            # report may still supersede stale local BLOCKED memory (the False
            # Clearance attack), but current LiDAR can never be overridden.
            if not self.peer_free_supersedes_direct_block(cell, fusion, step):
                return True
        check = hard_blocked_fn or (lambda: fusion.footprint_hard_blocked([cell], step))
        return bool(check())

    def traversal_cost(self, cell: Cell, step: int, fusion, *, routing_cost_fn=None, hard_blocked_fn=None) -> float:
        if not self.in_bounds(cell) or self.static_grid[cell]:
            return math.inf
        claim, status = self.observation_status(cell, step)
        if status == "current":
            return math.inf if claim == ClaimType.BLOCKED else 1.0
        if self.is_blocked_for_planning(cell, fusion, step, hard_blocked_fn=hard_blocked_fn):
            return math.inf
        peer_cost = routing_cost_fn(cell, step) if routing_cost_fn is not None else fusion.routing_cost(cell, step)
        if math.isinf(peer_cost):
            return math.inf
        if status == "unknown" or claim is None:
            return max(self.unknown_traversal_cost, peer_cost)
        # An accepted, newer FREE report can clear stale local BLOCKED memory
        # (False Clearance). A current direct BLOCKED scan was handled above
        # and is never overridden.
        if claim == ClaimType.BLOCKED and self.peer_free_supersedes_direct_block(cell, fusion, step):
            return max(1.0, peer_cost)
        strength = self.memory_strength(cell, step)
        if claim == ClaimType.FREE:
            # Known free memory gradually returns toward unknown cost.
            direct_cost = self.unknown_traversal_cost - (self.unknown_traversal_cost - 1.0) * strength
        else:
            # Remembered obstacles are finite and fade toward unknown, avoiding
            # permanent ghost walls while retaining strong recent evidence.
            direct_cost = self.unknown_traversal_cost + fusion.cost_scale * (strength ** fusion.cost_exponent)
        return max(direct_cost, peer_cost)
