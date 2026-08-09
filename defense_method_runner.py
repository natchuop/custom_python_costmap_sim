"""Defense methods for trust-aware shared occupancy claims.

This module is intentionally independent of the simulator.  It accepts report-like
objects with these attributes:
    sender_id, target_cell, claim, timestamp, confidence

Claim values are interpreted as:
    0 = FREE, 1 = BLOCKED, 2 = CONGESTED

The simulator supplies a trust callback so current trust can retroactively change
old source-linked claims without creating a circular import.
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
    "full_trust",
    "majority_vote",
    "hard_threshold",
    "soft_probability",
    "time_decay",
    "trust_fused",
    "source_linked",
)


@dataclass(frozen=True)
class StoredClaim:
    """Immutable copy of one report retained by a defense method."""

    sender_id: int
    target_cell: Cell
    claim: int
    timestamp: int
    confidence: float
    trust_at_report: float
    is_malicious: bool = False

@dataclass(frozen=True)
class EffectivePeerCell:
    claim: int | None
    has_active_evidence: bool
    hard_blocked: bool
    routing_cost: float
    evidence: float


@dataclass
class DefenseConfig:
    """Shared parameters for the five occupancy-defense policies."""

    method: str = "source_linked"
    decay_rate: float = 0.01
    cost_scale: float = 8.0
    cost_exponent: float = 2.0
    blocked_probability_threshold: float = 0.70
    max_claim_age: int = 500
    congested_impact: float = 0.50
    duplicate_window_steps: int = 0

    def validate(self) -> None:
        if self.method not in DEFENSE_METHODS:
            raise ValueError(
                f"Unknown defense method {self.method!r}. "
                f"Expected one of: {', '.join(DEFENSE_METHODS)}"
            )
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


class DefenseMethodRunner:
    """Stores peer claims and converts them into hard blocks or routing costs.

    Policy behavior:
      hard_threshold:
          No trust or decay. Combined occupancy probability above the threshold
          becomes a hard wall.

      soft_probability:
          No trust or decay. Occupancy evidence creates a continuous cost.

      time_decay:
          Claim influence decays with age, but sender trust is ignored.

      trust_fused:
          Each report is weighted by trust at report time. Later trust changes do
          not revise old contributions.

      source_linked:
          Each report is weighted using current sender trust at planning time.
          Later trust changes therefore revise the influence of old claims.
    """

    def __init__(
        self,
        trust_score: Callable[[int], float],
        config: Optional[DefenseConfig] = None,
    ) -> None:
        self.trust_score = trust_score
        self.config = config or DefenseConfig()
        self.config.validate()

        self.claims_by_cell: DefaultDict[Cell, List[StoredClaim]] = defaultdict(list)
        # Operational fusion uses one current claim per sender and cell.  The
        # append-only history remains available for audit counters, but a sender
        # cannot amplify a claim merely by repeatedly refreshing it.
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
        """Store a report and return True when it can influence the method."""
        self.total_reports_seen += 1

        cell = tuple(report.target_cell)
        confidence = float(getattr(report, "confidence", 1.0))
        confidence = min(1.0, max(0.0, confidence))
        timestamp = int(report.timestamp)
        sender_id = int(report.sender_id)
        claim = int(report.claim)

        stored = StoredClaim(
            sender_id=sender_id,
            target_cell=cell,
            claim=claim,
            timestamp=timestamp,
            confidence=confidence,
            trust_at_report=min(1.0, max(0.0, float(self.trust_score(sender_id)))),
            is_malicious=bool(getattr(report, "is_malicious", False)),
        )

        key = (stored.sender_id, cell)
        previous = self.active_claims.get(key)
        if previous is not None:
            # An older delayed report is retained only as audit history; it
            # cannot replace the newer operational state.
            if stored.timestamp < previous.timestamp:
                self.claim_history.append(stored)
                return False
            self.claims_by_cell[cell].remove(previous)
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
            if (
                existing.sender_id == candidate.sender_id
                and existing.claim == candidate.claim
                and existing.timestamp == candidate.timestamp
            ):
                return True
        return False

    def prune(self, timestamp: Optional[int] = None) -> int:
        """Drop expired claims and return the number removed."""
        now = self.current_timestamp if timestamp is None else int(timestamp)
        removed = 0

        for cell in list(self.claims_by_cell):
            retained = [
                claim
                for claim in self.claims_by_cell[cell]
                if now - claim.timestamp <= self.config.max_claim_age
            ]
            removed += len(self.claims_by_cell[cell]) - len(retained)

            if retained:
                self.claims_by_cell[cell] = retained
            else:
                del self.claims_by_cell[cell]

        self.active_claims = {
            key: claim
            for key, claim in self.active_claims.items()
            if now - claim.timestamp <= self.config.max_claim_age
        }

        self.total_reports_pruned += removed
        return removed

    def claims_for(self, cell: Cell) -> Tuple[StoredClaim, ...]:
        return tuple(self.claims_by_cell.get(tuple(cell), ()))

    def _claim_impact(self, claim: int) -> float:
        if claim == BLOCKED_CLAIM:
            return 1.0
        if claim == FREE_CLAIM:
            return -1.0
        if claim == CONGESTED_CLAIM:
            return self.config.congested_impact
        return 0.0

    def _age_weight(self, claim: StoredClaim, timestamp: int) -> float:
        age = max(0, int(timestamp) - claim.timestamp)
        return math.exp(-self.config.decay_rate * age)

    def _method_weight(
        self,
        claim: StoredClaim,
        timestamp: int,
        trust_override: Optional[float] = None,
    ) -> float:
        method = self.method

        if method in ("full_trust", "majority_vote", "hard_threshold", "soft_probability"):
            return claim.confidence

        if method == "time_decay":
            return claim.confidence * self._age_weight(claim, timestamp)

        if method == "trust_fused":
            return claim.confidence * claim.trust_at_report

        if method == "source_linked":
            trust_value = (
                self.trust_score(claim.sender_id)
                if trust_override is None
                else trust_override
            )
            current_trust = min(1.0, max(0.0, float(trust_value)))
            return (
                claim.confidence
                * current_trust
                * self._age_weight(claim, timestamp)
            )

        raise RuntimeError(f"Unhandled defense method: {method}")

    def evidence(self, cell: Cell, timestamp: Optional[int] = None) -> float:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        if self.method == "majority_vote":
            votes = 0
            for claim in self.claims_by_cell.get(tuple(cell), ()):
                if now - claim.timestamp <= self.config.max_claim_age:
                    votes += 1 if claim.claim == BLOCKED_CLAIM else -1 if claim.claim == FREE_CLAIM else 0
            return float(votes)
        total = 0.0

        for claim in self.claims_by_cell.get(tuple(cell), ()):
            if now - claim.timestamp > self.config.max_claim_age:
                continue
            total += self._method_weight(claim, now) * self._claim_impact(claim.claim)

        return total

    def sender_route_risk(
        self,
        sender_id: int,
        cells: Iterable[Cell],
        timestamp: Optional[int] = None,
        trust_override: Optional[float] = None,
    ) -> float:
        """Return positive route evidence attributable to one sender.

        This is deliberately a raw additive risk-release measure rather than a
        final routing cost. The simulator uses it to decide whether a current
        trust change materially weakens old source-linked claims on the robot's
        near-term route. Duplicate footprint cells are counted once.
        """
        now = self.current_timestamp if timestamp is None else int(timestamp)
        sender_id = int(sender_id)
        total = 0.0

        for cell in {tuple(cell) for cell in cells}:
            for claim in self.claims_by_cell.get(cell, ()):
                if claim.sender_id != sender_id:
                    continue
                if now - claim.timestamp > self.config.max_claim_age:
                    continue
                impact = self._claim_impact(claim.claim)
                if impact <= 0.0:
                    continue
                total += (
                    self._method_weight(
                        claim,
                        now,
                        trust_override=trust_override,
                    )
                    * impact
                )

        return total

    def occupancy_probability(
        self,
        cell: Cell,
        timestamp: Optional[int] = None,
    ) -> float:
        """Return occupancy probability, with no claims treated as zero risk.

        The ordinary sigmoid gives sigma(0)=0.5.  That is mathematically tidy but
        would charge every unreported cell a cost.  Here, an empty claim set is
        explicitly treated as no peer-derived occupancy risk.
        """
        claims = self.claims_by_cell.get(tuple(cell), ())
        if not claims:
            return 0.0

        value = self.evidence(cell, timestamp)
        return 1.0 / (1.0 + math.exp(-value))

    def normalized_occupied_risk(
        self,
        cell: Cell,
        timestamp: Optional[int] = None,
    ) -> float:
        """Map neutral probability 0.5 to risk 0 and occupied certainty to 1."""
        probability = self.occupancy_probability(cell, timestamp)
        return min(1.0, max(0.0, 2.0 * (probability - 0.5)))

    def routing_cost(self, cell: Cell, timestamp: Optional[int] = None) -> float:
        """Return peer-derived traversal cost for a cell."""
        if self.method == "hard_threshold":
            return math.inf if self.is_hard_blocked(cell, timestamp) else 1.0
        if self.method == "majority_vote":
            # Majority is a discrete baseline, not a soft evidence method.
            # Positive votes block; free majorities, ties, and no votes add no
            # peer traversal penalty.
            return math.inf if self.evidence(cell, timestamp) > 0.0 else 1.0

        risk = self.normalized_occupied_risk(cell, timestamp)
        return 1.0 + self.config.cost_scale * (risk ** self.config.cost_exponent)

    def is_hard_blocked(self, cell: Cell, timestamp: Optional[int] = None) -> bool:
        if self.method == "majority_vote":
            return self.evidence(cell, timestamp) > 0.0
        if self.method != "hard_threshold":
            return False
        probability = self.occupancy_probability(cell, timestamp)
        return probability > self.config.blocked_probability_threshold

    def footprint_cost(
        self,
        cells: Iterable[Cell],
        timestamp: Optional[int] = None,
    ) -> float:
        """Use the maximum cell risk across the robot footprint."""
        maximum = 1.0
        for cell in cells:
            cost = self.routing_cost(cell, timestamp)
            if math.isinf(cost):
                return math.inf
            maximum = max(maximum, cost)
        return maximum

    def footprint_is_hard_blocked(
        self,
        cells: Iterable[Cell],
        timestamp: Optional[int] = None,
    ) -> bool:
        return any(self.is_hard_blocked(cell, timestamp) for cell in cells)

    def effective_cells(self, timestamp: Optional[int] = None) -> Dict[Cell, EffectivePeerCell]:
        """Read-only, method-aware peer state used by combined visualization."""
        now = self.current_timestamp if timestamp is None else int(timestamp)
        result = {}
        for cell, claims in self.claims_by_cell.items():
            active = [claim for claim in claims if now - claim.timestamp <= self.config.max_claim_age]
            if not active:
                continue
            evidence = self.evidence(cell, now)
            result[cell] = EffectivePeerCell(
                claim=max(active, key=lambda item: item.timestamp).claim,
                has_active_evidence=bool(evidence),
                hard_blocked=self.is_hard_blocked(cell, now),
                routing_cost=self.routing_cost(cell, now),
                evidence=evidence,
            )
        return result

    def snapshot(self, timestamp: Optional[int] = None) -> Dict[str, object]:
        now = self.current_timestamp if timestamp is None else int(timestamp)
        active_claims = sum(
            1
            for claims in self.claims_by_cell.values()
            for claim in claims
            if now - claim.timestamp <= self.config.max_claim_age
        )
        active_senders = sorted({
            claim.sender_id
            for claims in self.claims_by_cell.values()
            for claim in claims
            if now - claim.timestamp <= self.config.max_claim_age
        })
        return {
            "method": self.method,
            "timestamp": now,
            "active_cells": len(self.claims_by_cell),
            "active_claims": active_claims,
            "reports_seen": self.total_reports_seen,
            "reports_stored": self.total_reports_stored,
            "reports_pruned": self.total_reports_pruned,
            "active_senders": active_senders,
        }


def build_defense_runner(
    method: str,
    trust_score: Callable[[int], float],
    **config_overrides,
) -> DefenseMethodRunner:
    """Factory used by the simulator and standalone experiments."""
    config = DefenseConfig(method=method, **config_overrides)
    return DefenseMethodRunner(trust_score=trust_score, config=config)
