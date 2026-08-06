"""Canonical operational fusion for every modular defense method."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ClaimReport, ClaimType


@dataclass(frozen=True)
class StoredClaim:
    report: ClaimReport
    trust_at_report: float
    influence: float


class FusionEngine:
    """Stores immutable history plus one active claim per sender and cell."""
    def __init__(self, method: str, trust_score, *, decay_rate=.006, max_claim_age=900,
                 cost_scale=14., cost_exponent=1.5, blocked_probability_threshold=.70):
        self.method, self.trust_score = method, trust_score
        self.decay_rate, self.max_claim_age = decay_rate, max_claim_age
        self.cost_scale, self.cost_exponent = cost_scale, cost_exponent
        self.blocked_probability_threshold = blocked_probability_threshold
        self.report_history: dict[str, StoredClaim] = {}
        self.active_claims: dict[tuple[int, tuple[int, int]], StoredClaim] = {}

    @property
    def claims(self) -> dict[tuple[int, int], list[StoredClaim]]:
        """Compatibility view of active claims grouped by target cell."""
        grouped: dict[tuple[int, int], list[StoredClaim]] = {}
        for item in self.active_claims.values():
            grouped.setdefault(item.report.target_cell, []).append(item)
        return grouped

    def add(self, report: ClaimReport, influence: float = 1.) -> StoredClaim | None:
        if not 0. <= report.confidence <= 1.: raise ValueError("report confidence must be in [0, 1]")
        item = StoredClaim(report, self.trust_score(report.sender_id), influence)
        self.report_history[report.report_id] = item
        key = (report.sender_id, report.target_cell)
        previous = self.active_claims.get(key)
        # Older delayed packets are audit history but cannot replace a newer
        # operational claim from the same sender.
        if previous is None or (report.received_step, report.report_id) >= (previous.report.received_step, previous.report.report_id):
            self.active_claims[key] = item
        return previous

    def _active(self, cell: tuple[int, int], step: int):
        return [item for item in self.claims.get(cell, ()) if step - item.report.observation_step <= self.max_claim_age]

    @staticmethod
    def _impact(claim: ClaimType) -> float:
        return 1. if claim == ClaimType.BLOCKED else -1. if claim == ClaimType.FREE else 0.

    def _weight(self, item: StoredClaim, step: int) -> float:
        if self.method in {"full_trust", "hard_threshold", "soft_probability", "majority_vote"}:
            return item.influence * item.report.confidence
        if self.method == "time_decay":
            return item.influence * item.report.confidence * math.exp(-self.decay_rate * max(0, step - item.report.observation_step))
        if self.method == "trust_fused":
            return item.influence * item.report.confidence * item.trust_at_report
        return item.influence * item.report.confidence * self.trust_score(item.report.sender_id) * math.exp(-self.decay_rate * max(0, step - item.report.observation_step))

    def vote(self, cell: tuple[int, int], step: int) -> int:
        """Discrete peer majority; one latest active claim is one sender vote."""
        return sum(int(self._impact(item.report.claim)) for item in self._active(cell, step))

    def evidence(self, cell: tuple[int, int], step: int) -> float:
        if self.method == "majority_vote": return float(self.vote(cell, step))
        return sum(self._weight(item, step) * self._impact(item.report.claim) for item in self._active(cell, step))

    def probability(self, cell: tuple[int, int], step: int) -> float:
        if self.method == "majority_vote":
            vote = self.vote(cell, step)
            return 1. if vote > 0 else 0. if vote < 0 else .5
        return 1. / (1. + math.exp(-self.evidence(cell, step)))

    def blocked(self, cell: tuple[int, int], step: int) -> bool:
        if self.method == "majority_vote": return self.vote(cell, step) > 0
        return self.method == "hard_threshold" and self.probability(cell, step) > self.blocked_probability_threshold

    def routing_cost(self, cell: tuple[int, int], step: int) -> float:
        if self.blocked(cell, step): return math.inf
        if self.method == "majority_vote": return 1.
        risk = max(0., 2 * (self.probability(cell, step) - .5))
        return 1. + self.cost_scale * risk ** self.cost_exponent
