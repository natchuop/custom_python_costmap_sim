"""Defense methods for shared occupancy claims.

The simulator supplies callbacks for current source trust and Source Memory.
Primary comparison methods are ordered as:
    majority_vote, full_trust, trust_fused, source_memory
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple

Cell = Tuple[int, int]
FREE_CLAIM = 0
BLOCKED_CLAIM = 1
CONGESTED_CLAIM = 2

DEFENSE_METHODS = (
    "majority_vote",
    "full_trust",
    "trust_fused",
    "source_memory",
    "hard_threshold",
    "soft_probability",
    "time_decay",
    "trust_threshold",
)


@dataclass(frozen=True)
class StoredClaim:
    sender_id: int
    target_cell: Cell
    claim: int
    timestamp: int
    sensor_confidence: float
    trust_at_report: float
    is_malicious: bool = False

    @property
    def confidence(self) -> float:  # convenience for older analysis helpers
        return self.sensor_confidence


@dataclass
class DefenseConfig:
    method: str = "source_memory"
    trust_threshold: float = 0.50
    decay_rate: float = 0.006
    cost_scale: float = 40.0
    cost_exponent: float = 1.5
    blocked_probability_threshold: float = 0.70
    max_claim_age: int = 300
    congested_impact: float = 0.50
    duplicate_window_steps: int = 0
    majority_unknown_cost: float = 3.0

    def validate(self) -> None:
        if self.method not in DEFENSE_METHODS:
            raise ValueError(f"Unknown defense method {self.method!r}. Expected one of: {', '.join(DEFENSE_METHODS)}")
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if self.cost_scale < 0:
            raise ValueError("cost_scale must be non-negative")
        if self.cost_exponent <= 0:
            raise ValueError("cost_exponent must be positive")
        if not 0.0 <= self.blocked_probability_threshold <= 1.0:
            raise ValueError("blocked_probability_threshold must be in [0, 1]")
        if self.max_claim_age < 1:
            raise ValueError("max_claim_age must be at least 1")
        if not 0.0 <= self.trust_threshold <= 1.0:
            raise ValueError("trust_threshold must be in [0, 1]")
        if self.majority_unknown_cost < 1.0:
            raise ValueError("majority_unknown_cost must be >= 1")


class DefenseMethodRunner:
    def __init__(
        self,
        trust_score: Callable[[int], float],
        config: Optional[DefenseConfig] = None,
        *,
        trust_memory_score: Callable[[int], float] | None = None,
    ) -> None:
        self.trust_score = trust_score
        self.trust_memory_score = trust_memory_score or trust_score
        self.config = config or DefenseConfig()
        self.config.validate()
        self.claims_by_cell: DefaultDict[Cell, List[StoredClaim]] = defaultdict(list)
        self.active_claims: Dict[Tuple[int, Cell], StoredClaim] = {}
        self.claim_history: List[StoredClaim] = []
        self.current_timestamp = 0
        self.total_reports_seen = 0
        self.total_reports_stored = 0
        self.total_reports_pruned = 0

    @property
    def method(self) -> str:
        return self.config.method

    def set_time(self, timestamp: int) -> None:
        self.current_timestamp = int(timestamp)

    def clear(self) -> None:
        self.claims_by_cell.clear()
        self.active_claims.clear()
        self.claim_history.clear()

    def add_report(self, report) -> bool:
        self.total_reports_seen += 1
        cell = tuple(report.target_cell)
        sensor_confidence = float(getattr(report, "sensor_confidence", 1.0))
        sensor_confidence = min(1.0, max(0.0, sensor_confidence))
        timestamp = int(report.timestamp)
        sender_id = int(report.sender_id)
        claim = int(report.claim)
        stored = StoredClaim(
            sender_id=sender_id,
            target_cell=cell,
            claim=claim,
            timestamp=timestamp,
            sensor_confidence=sensor_confidence,
            trust_at_report=min(1.0, max(0.0, float(self.trust_score(sender_id)))),
            is_malicious=bool(getattr(report, "is_malicious", False)),
        )
        key = (stored.sender_id, cell)
        previous = self.active_claims.get(key)
        if previous is not None:
            if stored.timestamp < previous.timestamp:
                self.claim_history.append(stored)
                return False
            bucket = self.claims_by_cell[cell]
            self.claims_by_cell[cell] = [item for item in bucket if item is not previous]
        elif self._is_duplicate(stored):
            return False
        self.claims_by_cell[cell].append(stored)
        self.active_claims[key] = stored
        self.claim_history.append(stored)
        self.total_reports_stored += 1
        return True

    def _is_duplicate(self, candidate: StoredClaim) -> bool:
        window = int(self.config.duplicate_window_steps)
        if window < 0:
            return False
        for existing in reversed(self.claims_by_cell.get(candidate.target_cell, [])):
            if candidate.timestamp - existing.timestamp > window:
                break
            if existing.sender_id == candidate.sender_id and existing.claim == candidate.claim and existing.timestamp == candidate.timestamp:
                return True
        return False

    def prune(self, timestamp: Optional[int] = None) -> int:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        removed = 0
        for cell in list(self.claims_by_cell):
            retained = [claim for claim in self.claims_by_cell[cell] if now - claim.timestamp < self.config.max_claim_age]
            removed += len(self.claims_by_cell[cell]) - len(retained)
            if retained:
                self.claims_by_cell[cell] = retained
            else:
                del self.claims_by_cell[cell]
        self.active_claims = {
            key: claim for key, claim in self.active_claims.items()
            if now - claim.timestamp < self.config.max_claim_age
        }
        # History is for recent audit/debug only; expired claims are already
        # represented by counters/output logs and should not grow unbounded.
        self.claim_history = [claim for claim in self.claim_history if now - claim.timestamp < self.config.max_claim_age]
        self.total_reports_pruned += removed
        return removed

    def claims_for(self, cell: Cell) -> Tuple[StoredClaim, ...]:
        return tuple(self.claims_by_cell.get(tuple(cell), ()))

    def retract_active(self, sender_id: int, cell: Cell) -> bool:
        cell = tuple(cell)
        key = (int(sender_id), cell)
        claim = self.active_claims.pop(key, None)
        if claim is None:
            return False
        bucket = self.claims_by_cell.get(cell)
        if bucket:
            self.claims_by_cell[cell] = [item for item in bucket if item is not claim]
            if not self.claims_by_cell[cell]:
                del self.claims_by_cell[cell]
        return True

    def _claim_impact(self, claim: int) -> float:
        if claim == BLOCKED_CLAIM:
            return 1.0
        if claim == FREE_CLAIM:
            return -1.0
        if claim == CONGESTED_CLAIM:
            return self.config.congested_impact
        return 0.0

    def _linear_age_weight(self, claim: StoredClaim, timestamp: int) -> float:
        age = max(0, int(timestamp) - claim.timestamp)
        return max(0.0, 1.0 - age / float(self.config.max_claim_age))

    def _exp_age_weight(self, claim: StoredClaim, timestamp: int) -> float:
        age = max(0, int(timestamp) - claim.timestamp)
        return math.exp(-self.config.decay_rate * age)

    def _method_weight(self, claim: StoredClaim, timestamp: int, trust_override: Optional[float] = None) -> float:
        method = self.method
        c = claim.sensor_confidence
        if method == "majority_vote":
            return 1.0
        if method == "full_trust":
            return c * self._linear_age_weight(claim, timestamp)
        if method == "trust_fused":
            # Trust Fused freezes source credibility at report time. Reports
            # received while the sender is already distrusted remain stored for
            # audit, but have no operational occupancy influence. Older reports
            # that were received while trusted retain their original weighting.
            if claim.trust_at_report < self.config.trust_threshold:
                return 0.0
            return c * self._linear_age_weight(claim, timestamp) * claim.trust_at_report
        if method == "source_memory":
            current = min(1.0, max(0.0, float(self.trust_score(claim.sender_id) if trust_override is None else trust_override)))
            memory = min(1.0, max(0.0, float(self.trust_memory_score(claim.sender_id))))
            effective = min(claim.trust_at_report, current, memory)
            # Source Memory is retroactive: once the effective source memory is
            # below the distrust threshold, all claims from that source are
            # operationally ignored. They remain stored/logged so experiments
            # can audit what was received. Rehabilitation remains gradual
            # because memory must recover above the threshold before influence
            # resumes, and a report can never exceed trust_at_report.
            if effective < self.config.trust_threshold:
                return 0.0
            return c * self._linear_age_weight(claim, timestamp) * effective
        if method in ("hard_threshold", "soft_probability"):
            return c
        if method == "time_decay":
            return c * self._exp_age_weight(claim, timestamp)
        if method == "trust_threshold":
            current = min(1.0, max(0.0, float(self.trust_score(claim.sender_id) if trust_override is None else trust_override)))
            if current < self.config.trust_threshold:
                return 0.0
            return c * current * self._exp_age_weight(claim, timestamp)
        raise RuntimeError(f"Unhandled defense method: {method}")

    @staticmethod
    def _claim_included(claim, excluded_sender_id=None, excluded_claim_predicate=None):
        if excluded_sender_id is not None and claim.sender_id == int(excluded_sender_id):
            if excluded_claim_predicate is None or excluded_claim_predicate(claim):
                return False
        if excluded_sender_id is None and excluded_claim_predicate is not None and excluded_claim_predicate(claim):
            return False
        return True

    def _active_iter(self, cell: Cell, now: int, excluded_sender_id=None, excluded_claim_predicate=None):
        for claim in self.claims_by_cell.get(tuple(cell), ()):
            if now - claim.timestamp >= self.config.max_claim_age:
                continue
            if not self._claim_included(claim, excluded_sender_id, excluded_claim_predicate):
                continue
            yield claim

    def evidence(self, cell: Cell, timestamp: Optional[int] = None, excluded_sender_id=None, excluded_claim_predicate=None) -> float:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        if self.method == "majority_vote":
            votes = 0
            for claim in self._active_iter(cell, now, excluded_sender_id, excluded_claim_predicate):
                votes += 1 if claim.claim == BLOCKED_CLAIM else -1 if claim.claim == FREE_CLAIM else 0
            return float(votes)
        return sum(
            self._method_weight(claim, now) * self._claim_impact(claim.claim)
            for claim in self._active_iter(cell, now, excluded_sender_id, excluded_claim_predicate)
        )

    def active_claim_weight(self, sender_id: int, cell: Cell, timestamp: Optional[int] = None) -> float:
        """Return the operational weight of one sender's active claim for a cell."""
        now = self.current_timestamp if timestamp is None else int(timestamp)
        claim = self.active_claims.get((int(sender_id), tuple(cell)))
        if claim is None or now - claim.timestamp >= self.config.max_claim_age:
            return 0.0
        return self._method_weight(claim, now)

    def sender_route_risk(self, sender_id: int, cells: Iterable[Cell], timestamp: Optional[int] = None, trust_override: Optional[float] = None) -> float:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        total = 0.0
        for cell in {tuple(cell) for cell in cells}:
            for claim in self._active_iter(cell, now):
                if claim.sender_id != int(sender_id):
                    continue
                impact = self._claim_impact(claim.claim)
                if impact > 0:
                    total += self._method_weight(claim, now, trust_override=trust_override) * impact
        return total

    def occupancy_probability(self, cell: Cell, timestamp: Optional[int] = None, excluded_sender_id=None, excluded_claim_predicate=None) -> float:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        claims = tuple(self._active_iter(cell, now, excluded_sender_id, excluded_claim_predicate))
        if not claims:
            return 0.0
        value = self.evidence(cell, now, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate)
        return 1.0 / (1.0 + math.exp(-value))

    def normalized_occupied_risk(self, cell: Cell, timestamp: Optional[int] = None, excluded_sender_id=None, excluded_claim_predicate=None) -> float:
        probability = self.occupancy_probability(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate)
        return min(1.0, max(0.0, 2.0 * (probability - 0.5)))

    def routing_cost(self, cell: Cell, timestamp: Optional[int] = None, excluded_sender_id=None, excluded_claim_predicate=None) -> float:
        if self.method == "hard_threshold":
            return math.inf if self.is_hard_blocked(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate) else 1.0
        if self.method == "majority_vote":
            vote = self.evidence(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate)
            if vote > 0:
                return math.inf
            if vote < 0:
                return 1.0
            return self.config.majority_unknown_cost
        risk = self.normalized_occupied_risk(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate)
        return 1.0 + self.config.cost_scale * (risk ** self.config.cost_exponent)

    def is_hard_blocked(self, cell: Cell, timestamp: Optional[int] = None, excluded_sender_id=None, excluded_claim_predicate=None) -> bool:
        if self.method == "majority_vote":
            return self.evidence(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate) > 0.0
        if self.method == "trust_threshold":
            now = self.current_timestamp if timestamp is None else int(timestamp)
            return any(
                claim.claim == BLOCKED_CLAIM and self._method_weight(claim, now) > 0.0
                for claim in self._active_iter(cell, now, excluded_sender_id, excluded_claim_predicate)
            )
        if self.method != "hard_threshold":
            return False
        return self.occupancy_probability(cell, timestamp, excluded_sender_id=excluded_sender_id, excluded_claim_predicate=excluded_claim_predicate) > self.config.blocked_probability_threshold

    def footprint_cost(self, cells: Iterable[Cell], timestamp: Optional[int] = None) -> float:
        maximum = 1.0
        for cell in cells:
            cost = self.routing_cost(cell, timestamp)
            if math.isinf(cost):
                return math.inf
            maximum = max(maximum, cost)
        return maximum

    def footprint_is_hard_blocked(self, cells: Iterable[Cell], timestamp: Optional[int] = None) -> bool:
        return any(self.is_hard_blocked(cell, timestamp) for cell in cells)

    def snapshot(self, timestamp: Optional[int] = None) -> Dict[str, object]:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        active = [claim for claims in self.claims_by_cell.values() for claim in claims if now - claim.timestamp < self.config.max_claim_age]
        return {
            "method": self.method,
            "timestamp": now,
            "active_cells": len({claim.target_cell for claim in active}),
            "active_claims": len(active),
            "reports_seen": self.total_reports_seen,
            "reports_stored": self.total_reports_stored,
            "reports_pruned": self.total_reports_pruned,
            "active_senders": sorted({claim.sender_id for claim in active}),
        }


def build_defense_runner(method: str, trust_score: Callable[[int], float], *, trust_memory_score=None, **config_overrides) -> DefenseMethodRunner:
    config = DefenseConfig(method=method, **config_overrides)
    return DefenseMethodRunner(trust_score=trust_score, trust_memory_score=trust_memory_score, config=config)
