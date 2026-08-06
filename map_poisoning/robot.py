"""Independent benign robot state and behavior aligned with legacy ``GridRobot``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, FrozenSet
import math

from .admission import decide
from .belief import RobotBeliefMap
from .fusion import FusionEngine
from .models import ClaimReport, ClaimType, DeliveryTask, DirectObservation, VerificationOutcome
from .planning import astar
from .sensing import lidar_observations
from .trust import TrustModel

PATH_INVALID_REPLAN_COOLDOWN_STEPS = 8
HONEST_REPORT_REFRESH_STEPS = 80
SOURCE_LINKED_REPLAN_COOLDOWN_STEPS = 25
SOURCE_LINKED_MIN_TRUST_DELTA = 0.10
SOURCE_LINKED_MIN_ROUTE_RISK_DROP = 0.20
SOURCE_LINKED_ROUTE_LOOKAHEAD_ANCHORS = 40
DEFENSE_PRUNE_PERIOD_STEPS = 20
SPAWN_COLLISION_GRACE_STEPS = 100


class _PlanningBelief:
    """Adapter so legacy fallback planning can use modular belief + fusion."""

    def __init__(self, belief: RobotBeliefMap, fusion: FusionEngine, step: int):
        self._belief = belief
        self._fusion = fusion
        self._step = step

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return self._belief.in_bounds(cell)

    def is_blocked_for_planning(self, cell: tuple[int, int]) -> bool:
        return self._belief.is_blocked_for_planning(cell, self._fusion, self._step)

    def traversal_cost(self, cell: tuple[int, int]) -> float:
        return self._belief.traversal_cost(cell, self._step, self._fusion)


@dataclass
class ReplanRecord:
    step: int
    reason: str
    old_path_cost: float | None
    new_path_cost: float | None
    old_path_length: int
    new_path_length: int
    changed: bool


@dataclass
class ModularRobot:
    robot_id: int
    start: tuple[int, int]
    tasks: tuple[DeliveryTask, ...]
    belief: RobotBeliefMap
    trust: TrustModel
    fusion: FusionEngine
    trust_threshold: float
    admission_policy: str
    position: tuple[int, int] = field(init=False)
    task_index: int = 0
    carrying: bool = False
    completed: bool = False
    path: list[tuple[int, int]] | None = None
    inbox: list[ClaimReport] = field(default_factory=list)
    pending: dict[str, ClaimReport] = field(default_factory=dict)
    deliveries_completed: int = 0
    no_path_steps: int = 0
    movement_steps: int = 0
    total_distance: float = 0.0
    total_replans: int = 0
    blocked_moves: int = 0
    productive_replans: int = 0
    replan_records: list[ReplanRecord] = field(default_factory=list)
    no_path_causes: dict[str, int] = field(default_factory=dict)
    last_path_invalid_replan_step: int = -10**9
    last_source_linked_replan_step: int = -10**9
    last_shared_claim: dict[tuple[int, int], ClaimType] = field(default_factory=dict)
    last_shared_step: dict[tuple[int, int], int] = field(default_factory=dict)
    defense_replan_needed: bool = False
    source_linked_replan_context: dict | None = None

    def __post_init__(self):
        self.position = self.start

    @property
    def goal(self) -> tuple[int, int]:
        task = self.tasks[self.task_index]
        return task.dropoff if self.carrying else task.pickup

    def receive(self, report: ClaimReport) -> None:
        self.inbox.append(report)

    def remaining_route(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.path or ())

    def route_evidence(self, step: int) -> float:
        return sum(self.fusion.evidence(cell, step) for cell in self.remaining_route())

    def should_share_observation(self, cell: tuple[int, int], claim: ClaimType, step: int) -> bool:
        previous = self.last_shared_claim.get(cell)
        last_step = self.last_shared_step.get(cell, -10**9)
        if previous == claim and step - last_step < HONEST_REPORT_REFRESH_STEPS:
            return False
        self.last_shared_claim[cell] = claim
        self.last_shared_step[cell] = step
        return True

    def _source_linked_route_cells(self, step: int) -> list[tuple[int, int]]:
        if not self.path:
            return []
        anchors = self.path[:SOURCE_LINKED_ROUTE_LOOKAHEAD_ANCHORS]
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for anchor in anchors:
            if self.belief.has_direct_free(anchor):
                continue
            if anchor not in seen:
                seen.add(anchor)
                cells.append(anchor)
        return cells

    def reports_affect_remaining_route(
        self,
        reports: Iterable[tuple[ClaimReport, object]],
        malicious_ids: FrozenSet[str],
    ) -> bool:
        if not self.path:
            return False
        remaining = list(self.path)
        for report, _ in reports:
            target = tuple(report.target_cell)
            if self.belief.has_direct_free(target):
                continue
            is_malicious = report.report_id in malicious_ids
            if (
                report.claim == ClaimType.FREE
                and not is_malicious
                and self.fusion.method != "majority_vote"
            ):
                continue
            if target in remaining:
                return True
        return False

    def path_invalid_or_empty(self, step: int) -> bool:
        if not self.path:
            return True
        for cell in self.path:
            if self.belief.is_blocked_for_planning(cell, self.fusion, step):
                return True
        return False

    def should_replan_for_path_state(self, step: int) -> bool:
        if self.completed:
            return False
        if not self.path:
            return self.position != self.goal
        if not self.path_invalid_or_empty(step):
            return False
        return step - self.last_path_invalid_replan_step >= PATH_INVALID_REPLAN_COOLDOWN_STEPS

    def process_inbox(self, step: int, malicious_ids: FrozenSet[str] = frozenset()):
        accepted = []
        route_affected = False
        for report in self.inbox:
            policy = decide(self.admission_policy, self.trust.score(report.sender_id), self.trust_threshold)
            if not policy.accepted:
                continue
            is_malicious = report.report_id in malicious_ids
            previous = self.fusion.add(report, policy.influence, is_malicious=is_malicious)
            if previous is not None:
                self.pending.pop(previous.report.report_id, None)
            self.pending[report.report_id] = report
            accepted.append((report, policy))
            if self._report_affects_route(report, malicious_ids):
                route_affected = True
        self.inbox.clear()
        return accepted, route_affected

    def _report_affects_route(self, report: ClaimReport, malicious_ids: FrozenSet[str]) -> bool:
        target = tuple(report.target_cell)
        if self.belief.has_direct_free(target):
            return False
        is_malicious = report.report_id in malicious_ids
        if report.claim == ClaimType.FREE and not is_malicious and self.fusion.method != "majority_vote":
            return False
        return target in self.remaining_route()

    def sense(self, world, step: int, other_positions: Iterable[tuple[int, int]]) -> list[DirectObservation]:
        if step % DEFENSE_PRUNE_PERIOD_STEPS == 0:
            self.fusion.prune(step)
        truth = world.truth_grid(step)
        raw = lidar_observations(truth, self.position, other_positions)
        observations: list[DirectObservation] = []
        for cell, claim in raw.items():
            obs = DirectObservation(self.robot_id, cell, claim, step)
            self.belief.observe(obs)
            observations.append(obs)
        return observations

    def verify(self, observations: Iterable[DirectObservation], step: int):
        results = []
        by_cell = {item.cell: item for item in observations}
        verified_by_sender: dict[int, list[tuple[ClaimReport, bool]]] = {}
        old_trust_by_sender: dict[int, float] = {}

        for report_id, report in list(self.pending.items()):
            observed = by_cell.get(report.target_cell)
            if observed is None:
                continue
            truth_matches = observed.claim == report.claim
            sender_id = report.sender_id
            if sender_id not in old_trust_by_sender:
                old_trust_by_sender[sender_id] = self.trust.score(sender_id)
            outcome = (
                VerificationOutcome.CONFIRMED
                if truth_matches
                else VerificationOutcome.CONTRADICTED_FRESH
            )
            evidence_before = self.fusion.evidence(report.target_cell, step)
            probability_before = self.fusion.probability(report.target_cell, step)
            old_trust, new_trust = self.trust.update(sender_id, outcome)
            evidence_after = self.fusion.evidence(report.target_cell, step)
            probability_after = self.fusion.probability(report.target_cell, step)
            results.append(
                (
                    report,
                    outcome,
                    old_trust,
                    new_trust,
                    evidence_before,
                    evidence_after,
                    probability_before,
                    probability_after,
                )
            )
            verified_by_sender.setdefault(sender_id, []).append((report, truth_matches))
            del self.pending[report_id]

        if self.fusion.method == "source_linked" and verified_by_sender:
            route_cells = self._source_linked_route_cells(step)
            for sender_id, items in verified_by_sender.items():
                old_trust = float(old_trust_by_sender[sender_id])
                new_trust = float(self.trust.score(sender_id))
                trust_delta = old_trust - new_trust
                if trust_delta <= 0.0:
                    continue
                if trust_delta < SOURCE_LINKED_MIN_TRUST_DELTA:
                    continue
                if step - self.last_source_linked_replan_step < SOURCE_LINKED_REPLAN_COOLDOWN_STEPS:
                    continue
                risk_before = self.fusion.sender_route_risk(sender_id, route_cells, step, trust_override=old_trust)
                risk_after = self.fusion.sender_route_risk(sender_id, route_cells, step, trust_override=new_trust)
                risk_drop = risk_before - risk_after
                if risk_drop < SOURCE_LINKED_MIN_ROUTE_RISK_DROP:
                    continue
                self.defense_replan_needed = True
                self.source_linked_replan_context = {
                    "sender_id": sender_id,
                    "old_trust": old_trust,
                    "new_trust": new_trust,
                    "trust_delta": trust_delta,
                    "route_risk_before": risk_before,
                    "route_risk_after": risk_after,
                    "route_risk_drop": risk_drop,
                }
                self.last_source_linked_replan_step = step
                break

        return results

    def replan(self, step: int, reason: str) -> bool:
        old = list(self.path or ())
        old_cost = sum(self.fusion.routing_cost(cell, step) for cell in old) if old else None
        self.fusion.set_time(step)
        adapter = _PlanningBelief(self.belief, self.fusion, step)
        self.path = astar(
            self.position,
            self.goal,
            lambda cell: adapter.traversal_cost(cell),
        )
        if self.path is None:
            try:
                import sim2

                if sim2.ENABLE_FALLBACK_EXPLORATION:
                    self.path, _ = sim2.plan_to_reachable_fallback(
                        adapter,
                        self.position,
                        self.goal,
                    )
            except RuntimeError:
                self.path = None
        new = list(self.path or ())
        new_cost = sum(self.fusion.routing_cost(cell, step) for cell in new) if new else None
        changed = old != new
        self.total_replans += 1
        self.productive_replans += int(changed)
        if "path_invalid" in reason or reason == "path_invalid_or_empty":
            self.last_path_invalid_replan_step = step
        self.replan_records.append(
            ReplanRecord(step, reason, old_cost, new_cost, len(old), len(new), changed)
        )
        return bool(self.path)

    def classify_no_path(self, world, step: int) -> str:
        truth = astar(
            self.position,
            self.goal,
            lambda cell: math.inf if world.state(cell, step) == ClaimType.BLOCKED else 1.0,
        )
        if truth is None:
            cause = "truth_disconnected"
        else:
            direct = astar(
                self.position,
                self.goal,
                lambda cell: math.inf if self.belief.direct_state(cell) == ClaimType.BLOCKED else 1.0,
            )
            if direct is None:
                cause = "direct_belief_disconnected"
            else:
                operational = astar(
                    self.position,
                    self.goal,
                    lambda cell: self.belief.traversal_cost(cell, step, self.fusion),
                )
                cause = (
                    "peer_fusion_disconnected"
                    if operational is None
                    else "planner_or_state_error"
                )
        self.no_path_causes[cause] = self.no_path_causes.get(cause, 0) + 1
        return cause

    def move(self, world, step: int, occupied: set[tuple[int, int]]) -> str:
        if self.completed:
            return "idle"
        if self.position == self.goal:
            if self.carrying:
                self.carrying = False
                self.deliveries_completed += 1
                self.task_index += 1
                if self.task_index >= len(self.tasks):
                    self.completed = True
                    self.path = None
                    return "task_transition"
                self.path = None
                return "task_transition"
            self.carrying = True
            self.path = None
            return "task_transition"
        if not self.path:
            self.no_path_steps += 1
            return "no_path"
        next_cell = self.path[0]
        if (
            step >= SPAWN_COLLISION_GRACE_STEPS
            and next_cell in occupied
            and next_cell != self.position
        ):
            self.blocked_moves += 1
            self.path = None
            return "blocked_move"
        if world.state(next_cell, step) == ClaimType.BLOCKED:
            self.blocked_moves += 1
            self.belief.observe(DirectObservation(self.robot_id, next_cell, ClaimType.BLOCKED, step))
            self.path = None
            return "blocked_move"
        self.path.pop(0)
        self.position = next_cell
        self.movement_steps += 1
        self.total_distance += 1.0
        return "move"
