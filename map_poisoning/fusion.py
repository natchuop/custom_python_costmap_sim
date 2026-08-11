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
        )
        self.decay_rate = decay_rate
        self.max_claim_age = max_claim_age
        self.cost_scale = cost_scale
        self.cost_exponent = cost_exponent
        self.blocked_probability_threshold = blocked_probability_threshold
        self.report_history: dict[str, StoredClaim] = {}
        self._active: dict[tuple[int, tuple[int, int]], StoredClaim] = {}

    @property
    def method(self) -> str:
        return self._runner.method

    def set_time(self, step: int) -> None:
        self._runner.set_time(step)

    def prune(self, step: int | None = None) -> int:
        return self._runner.prune(step)

    @property
    def claims(self) -> dict[tuple[int, int], list[StoredClaim]]:
        grouped: dict[tuple[int, int], list[StoredClaim]] = {}
        for item in self._active.values():
            grouped.setdefault(item.report.target_cell, []).append(item)
        return grouped

    def add(self, report: ClaimReport, influence: float = 1.0, is_malicious: bool = False) -> StoredClaim | None:
        if not 0.0 <= report.confidence <= 1.0:
            raise ValueError("report confidence must be in [0, 1]")
        item = StoredClaim(report, self._runner.trust_score(report.sender_id), influence)
        self.report_history[report.report_id] = item
        applied = self._runner.add_report(_RunnerReport(report, is_malicious))
        previous = self._active.get((report.sender_id, report.target_cell))
        if applied:
            self._active[(report.sender_id, report.target_cell)] = item
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

    def vote(self, cell: tuple[int, int], step: int) -> int:
        self.set_time(step)
        return int(self._runner.evidence(cell, step))

    def selected_claim(self, cell: tuple[int, int], step: int) -> int | None:
        self.set_time(step)
        return self._runner.selected_claim(cell, step)

    def footprint_hard_blocked(self, cells, step: int) -> bool:
        self.set_time(step)
        return self._runner.footprint_is_hard_blocked(cells, step)
