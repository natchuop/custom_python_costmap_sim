"""Operational fusion backed by the validated defense-method implementation."""
from __future__ import annotations

from dataclasses import dataclass

from defense_method_runner import DefenseMethodRunner, build_defense_runner

from .models import ClaimReport


@dataclass(frozen=True)
class StoredClaim:
    report: ClaimReport
    trust_at_report: float
    influence: float


class _RunnerReport:
    """Adapter so modular ``ClaimReport`` objects satisfy the runner protocol."""

    def __init__(self, report: ClaimReport, is_malicious: bool = False):
        self.sender_id = report.sender_id
        self.target_cell = report.target_cell
        self.claim = int(report.claim)
        self.timestamp = int(report.observation_step)
        self.confidence = float(report.confidence)
        self.is_malicious = is_malicious


class FusionEngine:
    """Modular API adapter for the validated fusion implementation."""

    def __init__(
        self,
        method: str,
        trust_score,
        *,
        decay_rate: float = 0.006,
        max_claim_age: int = 900,
        cost_scale: float = 14.0,
        cost_exponent: float = 1.5,
        blocked_probability_threshold: float = 0.70,
        congested_impact: float = 0.50,
        duplicate_window_steps: int = 0,
        trust_threshold: float = 0.55,
    ):
        self._runner: DefenseMethodRunner = build_defense_runner(
            method,
            trust_score,
            decay_rate=decay_rate,
            max_claim_age=max_claim_age,
            cost_scale=cost_scale,
            cost_exponent=cost_exponent,
            blocked_probability_threshold=blocked_probability_threshold,
            congested_impact=congested_impact,
            duplicate_window_steps=duplicate_window_steps,
            trust_threshold=trust_threshold,
        )
        self.decay_rate = decay_rate
        self.max_claim_age = max_claim_age
        self.cost_scale = cost_scale
        self.cost_exponent = cost_exponent
        self.blocked_probability_threshold = blocked_probability_threshold
        self.report_history: dict[str, StoredClaim] = {}
        self._active: dict[tuple[int, tuple[int, int]], StoredClaim] = {}
        self._claims_grouped: dict[tuple[int, int], list[StoredClaim]] | None = None

    @property
    def method(self) -> str:
        return self._runner.method

    def set_time(self, step: int) -> None:
        self._runner.set_time(step)

    def prune(self, step: int | None = None) -> int:
        removed = self._runner.prune(step)
        if removed:
            active_keys = set(self._runner.active_claims.keys())
            self._active = {key: item for key, item in self._active.items() if key in active_keys}
            self._invalidate_claims_cache()
        return removed

    def _invalidate_claims_cache(self) -> None:
        self._claims_grouped = None

    @property
    def claims(self) -> dict[tuple[int, int], list[StoredClaim]]:
        if self._claims_grouped is None:
            grouped: dict[tuple[int, int], list[StoredClaim]] = {}
            for item in self._active.values():
                grouped.setdefault(item.report.target_cell, []).append(item)
            self._claims_grouped = grouped
        return self._claims_grouped

    def claims_at(self, cell: tuple[int, int]) -> tuple[StoredClaim, ...]:
        """Return stored claims at one cell without rebuilding the grouped map."""
        cell = tuple(cell)
        return tuple(
            self._active[(claim.sender_id, cell)]
            for claim in self._runner.claims_for(cell)
            if (claim.sender_id, cell) in self._active
        )

    def retract(self, report: ClaimReport) -> bool:
        """Remove a contradicted claim from operational fusion state."""
        key = (report.sender_id, tuple(report.target_cell))
        if key not in self._active:
            return False
        del self._active[key]
        self._invalidate_claims_cache()
        return self._runner.retract_active(report.sender_id, report.target_cell)

    def add(self, report: ClaimReport, influence: float = 1.0, is_malicious: bool = False) -> StoredClaim | None:
        if not 0.0 <= report.confidence <= 1.0:
            raise ValueError("report confidence must be in [0, 1]")
        item = StoredClaim(report, self._runner.trust_score(report.sender_id), influence)
        self.report_history[report.report_id] = item
        applied = self._runner.add_report(_RunnerReport(report, is_malicious))
        previous = self._active.get((report.sender_id, report.target_cell))
        if applied:
            self._active[(report.sender_id, report.target_cell)] = item
            self._invalidate_claims_cache()
            return previous
        return previous

    def sender_route_risk(
        self,
        sender_id: int,
        cells,
        step: int,
        trust_override: float | None = None,
    ) -> float:
        self.set_time(step)
        return self._runner.sender_route_risk(sender_id, cells, step, trust_override=trust_override)

    def evidence(self, cell: tuple[int, int], step: int) -> float:
        self.set_time(step)
        return self._runner.evidence(cell, step)

    def probability(self, cell: tuple[int, int], step: int) -> float:
        self.set_time(step)
        return self._runner.occupancy_probability(cell, step)

    def blocked(self, cell: tuple[int, int], step: int) -> bool:
        self.set_time(step)
        return self._runner.is_hard_blocked(cell, step)

    def routing_cost(self, cell: tuple[int, int], step: int) -> float:
        self.set_time(step)
        return self._runner.routing_cost(cell, step)

    def soft_routing_cost(self, cell: tuple[int, int], step: int) -> float:
        """Peer-influence cost without trust-threshold hard blocks."""
        self.set_time(step)
        if self.method != "trust_threshold":
            return self.routing_cost(cell, step)
        risk = self._runner.normalized_occupied_risk(cell, step)
        return 1.0 + self.cost_scale * (risk ** self.cost_exponent)

    def routing_cost_excluding_sender(
        self,
        cell: tuple[int, int],
        step: int,
        sender_id: int,
        predicate=None,
    ) -> float:
        """Counterfactual routing cost with selected source claims removed."""
        self.set_time(step)
        return self._runner.routing_cost(
            cell,
            step,
            excluded_sender_id=sender_id,
            excluded_claim_predicate=predicate,
        )

    def vote(self, cell: tuple[int, int], step: int) -> int:
        self.set_time(step)
        return int(self._runner.evidence(cell, step))

    def footprint_hard_blocked(self, cells, step: int) -> bool:
        self.set_time(step)
        return self._runner.footprint_is_hard_blocked(cells, step)
