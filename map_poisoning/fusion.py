"""Peer-evidence fusion methods, including legacy method names."""
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
    def __init__(self, method: str, trust_score, *, decay_rate=.006, max_claim_age=900, cost_scale=14., cost_exponent=1.5, blocked_probability_threshold=.70):
        self.method, self.trust_score = method, trust_score
        self.decay_rate, self.max_claim_age = decay_rate, max_claim_age
        self.cost_scale, self.cost_exponent, self.blocked_probability_threshold = cost_scale, cost_exponent, blocked_probability_threshold
        self.claims: dict[tuple[int, int], list[StoredClaim]] = {}
    def add(self, report: ClaimReport, influence: float = 1.) -> None:
        self.claims.setdefault(report.target_cell, []).append(StoredClaim(report, self.trust_score(report.sender_id), influence))
    def _weight(self, item: StoredClaim, step: int) -> float:
        age = max(0, step - item.report.observation_step)
        if age > self.max_claim_age: return 0.
        if self.method in {"full_trust", "hard_threshold", "soft_probability", "majority_vote"}: return item.influence
        if self.method == "time_decay": return item.influence * math.exp(-self.decay_rate * age)
        if self.method == "trust_fused": return item.influence * item.trust_at_report
        return item.influence * self.trust_score(item.report.sender_id) * math.exp(-self.decay_rate * age)
    def evidence(self, cell: tuple[int, int], step: int) -> float:
        values = self.claims.get(cell, [])
        if self.method == "majority_vote":
            votes = sum(1 if x.report.claim == ClaimType.BLOCKED else -1 for x in values if self._weight(x, step)); return float(votes)
        return sum(self._weight(x, step) * (1 if x.report.claim == ClaimType.BLOCKED else -1 if x.report.claim == ClaimType.FREE else .5) for x in values)
    def probability(self, cell: tuple[int, int], step: int) -> float:
        if not self.claims.get(cell): return 0.
        return 1 / (1 + math.exp(-self.evidence(cell, step)))
    def blocked(self, cell: tuple[int, int], step: int) -> bool:
        return self.method == "hard_threshold" and self.probability(cell, step) > self.blocked_probability_threshold
    def routing_cost(self, cell: tuple[int, int], step: int) -> float:
        if self.blocked(cell, step): return math.inf
        risk = max(0., 2 * (self.probability(cell, step) - .5))
        return 1 + self.cost_scale * risk ** self.cost_exponent
