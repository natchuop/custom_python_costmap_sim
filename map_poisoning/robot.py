"""Independent robot state and behavior for modular manifest replay."""
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
SOURCE_LINKED_MIN_TRUST_DELTA = 0.05
SOURCE_LINKED_MIN_ROUTE_RISK_DROP = 0.20
SOURCE_LINKED_ROUTE_LOOKAHEAD_ANCHORS = 40
DEFENSE_PRUNE_PERIOD_STEPS = 20
TRAFFIC_REROUTE_AFTER_WAITS = 4


class _PlanningBelief:
    """Adapter exposing modular belief and fusion to the A* planner."""

    def __init__(self, belief: RobotBeliefMap, fusion: FusionEngine, step: int, temporarily_blocked=()):
        self._belief = belief
        self._fusion = fusion
        self._step = step
        self._temporarily_blocked = {tuple(cell) for cell in temporarily_blocked}

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return self._belief.in_bounds(cell)

    def is_blocked_for_planning(self, cell: tuple[int, int]) -> bool:
        if cell in self._temporarily_blocked:
            return True
        claim, freshness = self._belief.observation_status(cell, self._step)
        if self._fusion.method == "trust_threshold" and freshness == "unknown":
            return False
        return self._belief.is_blocked_for_planning(cell, self._fusion, self._step)

    def traversal_cost(self, cell: tuple[int, int]) -> float:
        if cell in self._temporarily_blocked:
            return math.inf
        claim, freshness = self._belief.observation_status(cell, self._step)
        if self._fusion.method == "trust_threshold" and freshness == "unknown":
            peer_cost = self._fusion.soft_routing_cost(cell, self._step)
            if math.isinf(peer_cost):
                return math.inf
            return max(self._belief.UNKNOWN_TRAVERSAL_COST, peer_cost)
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
    confirmation_cooldown_steps: int = 10
    position: tuple[int, int] = field(init=False)
    task_index: int = 0
    carrying: bool = False
    completed: bool = False
    path: list[tuple[int, int]] | None = None
    inbox: list[ClaimReport] = field(default_factory=list)
    pending: dict[str, ClaimReport] = field(default_factory=dict)
    deliveries_completed: int = 0
    delivery_durations: list[int] = field(default_factory=list)
    delivery_start_step: int | None = None
    no_path_steps: int = 0
    movement_steps: int = 0
    total_distance: float = 0.0
    total_replans: int = 0
    blocked_moves: int = 0
    traffic_wait_steps: int = 0
    consecutive_traffic_waits: int = 0
    traffic_replans: int = 0
    productive_replans: int = 0
    replan_records: list[ReplanRecord] = field(default_factory=list)
    no_path_causes: dict[str, int] = field(default_factory=dict)
    last_path_invalid_replan_step: int = -10**9
    last_source_linked_replan_step: int = -10**9
    last_shared_claim: dict[tuple[int, int], ClaimType] = field(default_factory=dict)
    last_shared_step: dict[tuple[int, int], int] = field(default_factory=dict)
    verified_reports: set[str] = field(default_factory=set)
    last_positive_trust_update_step: dict[int, int] = field(default_factory=dict)
    defense_replan_needed: bool = False
    source_linked_replan_context: dict | None = None
    accepted_reports: int = 0
    rejected_reports: int = 0
    # Traffic coordination (sim2 parity, single-cell robots).
    traffic_mode: str = "NORMAL"
    traffic_blocked_by: int | None = None
    active_yield_target: tuple[int, int] | None = None
    yield_blocked_cell: tuple[int, int] | None = None
    yield_conflict_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    saved_yield_path: list[tuple[int, int]] | None = None
    saved_yield_goal: tuple[int, int] | None = None
    traffic_deadlock_active: bool = False
    active_deadlock_id: str | None = None
    position_history: list[tuple[int, int]] = field(default_factory=list)
    traffic_yield_count: int = 0

    def __post_init__(self):
        self.position = self.start
        self.position_history.append(tuple(self.start))

    def proposed_next_cell(self) -> tuple[int, int] | None:
        return tuple(self.path[0]) if self.path else None

    def reserved_cells(self) -> set[tuple[int, int]]:
        """Current body plus the cell this robot intends to step into."""
        cells = {tuple(self.position)}
        nxt = self.proposed_next_cell()
        if nxt is not None:
            cells.add(tuple(nxt))
        return cells

    def record_position(self) -> None:
        if not self.position_history or self.position_history[-1] != self.position:
            self.position_history.append(tuple(self.position))

    @property
    def goal(self) -> tuple[int, int]:
        if self.completed or self.task_index >= len(self.tasks):
            return tuple(self.position)
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
            if self.belief.has_direct_free(anchor, step):
                continue
            if anchor not in seen:
                seen.add(anchor)
                cells.append(anchor)
        return cells

    def reports_affect_remaining_route(
        self,
        reports: Iterable[tuple[ClaimReport, object]],
        malicious_ids: FrozenSet[str],
        step: int,
    ) -> bool:
        if not self.path:
            return False
        remaining = list(self.path)
        for report, _ in reports:
            target = tuple(report.target_cell)
            if self.belief.has_direct_free(target, step):
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
                self.rejected_reports += 1
                continue
            is_malicious = report.report_id in malicious_ids
            previous = self.fusion.add(report, policy.influence, is_malicious=is_malicious)
            if previous is not None:
                self.pending.pop(previous.report.report_id, None)
            self.pending[report.report_id] = report
            accepted.append((report, policy))
            self.accepted_reports += 1
            if self._report_affects_route(report, malicious_ids, step):
                route_affected = True
        self.inbox.clear()
        return accepted, route_affected

    def _report_affects_route(self, report: ClaimReport, malicious_ids: FrozenSet[str], step: int) -> bool:
        target = tuple(report.target_cell)
        if self.belief.has_direct_free(target, step):
            return False
        is_malicious = report.report_id in malicious_ids
        if report.claim == ClaimType.FREE and not is_malicious and self.fusion.method != "majority_vote":
            return False
        return target in self.remaining_route()

    def sense(self, world, step: int, other_positions: Iterable[tuple[int, int]]) -> list[DirectObservation]:
        if step % DEFENSE_PRUNE_PERIOD_STEPS == 0:
            self.fusion.prune(step)
            self.belief.prune_expired(step, max_age=self.fusion.max_claim_age)
        truth = world.truth_grid(step)
        raw = lidar_observations(truth, self.position, other_positions)
        observations: list[DirectObservation] = []
        for cell, claim in raw.items():
            obs = DirectObservation(self.robot_id, cell, claim, step)
            self.belief.observe(obs)
            observations.append(obs)
        return observations

    def _resolve_observation(self, report: ClaimReport, by_cell: dict, step: int) -> DirectObservation | None:
        observed = by_cell.get(report.target_cell)
        if observed is not None:
            return observed
        claim, freshness = self.belief.observation_status(report.target_cell, step)
        if freshness != "fresh" or claim is None:
            return None
        return DirectObservation(self.robot_id, report.target_cell, claim, step)

    def _verify_report(
        self,
        report: ClaimReport,
        observed: DirectObservation,
        step: int,
        *,
        results: list,
        verified_by_sender: dict[int, list[tuple[ClaimReport, bool]]],
        old_trust_by_sender: dict[int, float],
        processed: set[str],
    ) -> None:
        if report.report_id in processed or report.report_id in self.verified_reports:
            return
        if report.received_step is not None and report.received_step == step:
            return
        processed.add(report.report_id)
        self.verified_reports.add(report.report_id)
        self.pending.pop(report.report_id, None)
        truth_matches = observed.claim == report.claim
        sender_id = report.sender_id
        if sender_id not in old_trust_by_sender:
            old_trust_by_sender[sender_id] = self.trust.score(sender_id)
        stale_honest_report = (
            report.scenario_event_id is None
            and step - report.observation_step > self.belief.memory_steps
        )
        if stale_honest_report:
            outcome = VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED
        else:
            outcome = (
                VerificationOutcome.CONFIRMED
                if truth_matches
                else VerificationOutcome.CONTRADICTED_FRESH
            )
        evidence_before = self.fusion.evidence(report.target_cell, step)
        probability_before = self.fusion.probability(report.target_cell, step)
        trust_severity = 1.0
        if outcome == VerificationOutcome.CONFIRMED:
            last_credit = self.last_positive_trust_update_step.get(sender_id, -10**9)
            if step - last_credit < self.confirmation_cooldown_steps:
                # Keep the validation event, but do not let one lidar sweep
                # provide dozens of independent reputation credits.
                trust_severity = 0.0
            else:
                self.last_positive_trust_update_step[sender_id] = step
        old_trust, new_trust = self.trust.update(sender_id, outcome, severity=trust_severity)
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
        if outcome == VerificationOutcome.CONTRADICTED_FRESH:
            if report.claim == ClaimType.BLOCKED and observed.claim == ClaimType.FREE:
                self.belief.observe(
                    DirectObservation(self.robot_id, report.target_cell, ClaimType.FREE, step)
                )
            elif report.claim == ClaimType.FREE and observed.claim == ClaimType.BLOCKED:
                self.belief.observe(
                    DirectObservation(self.robot_id, report.target_cell, ClaimType.BLOCKED, step)
                )
            self.fusion.retract(report)
        elif outcome == VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED:
            # The direct reading is current, but an old honest peer report is
            # not reliable evidence that the sender was wrong when it spoke.
            if not truth_matches:
                self.fusion.retract(report)

    def verify(self, observations: Iterable[DirectObservation], step: int):
        results = []
        by_cell = {item.cell: item for item in observations}
        verified_by_sender: dict[int, list[tuple[ClaimReport, bool]]] = {}
        old_trust_by_sender: dict[int, float] = {}
        processed: set[str] = set()

        for report in list(self.pending.values()):
            observed = self._resolve_observation(report, by_cell, step)
            if observed is None:
                continue
            self._verify_report(
                report,
                observed,
                step,
                results=results,
                verified_by_sender=verified_by_sender,
                old_trust_by_sender=old_trust_by_sender,
                processed=processed,
            )

        # Stored peer BLOCKED claims can outlive the one-shot pending queue.
        # Re-check them whenever fresh direct sensing shows the cell is clear.
        for cell, observed in by_cell.items():
            if observed.claim != ClaimType.FREE:
                continue
            for item in self.fusion.claims_at(cell):
                report = item.report
                if report.claim != ClaimType.BLOCKED:
                    continue
                if report.received_step is not None and report.received_step == step:
                    continue
                self._verify_report(
                    report,
                    observed,
                    step,
                    results=results,
                    verified_by_sender=verified_by_sender,
                    old_trust_by_sender=old_trust_by_sender,
                    processed=processed,
                )

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
                # A false claim may already have forced the robot onto a detour,
                # so it is no longer present on the *current* route.  A verified
                # trust decrease must still release that earlier detour.
                sender_has_active_blocked_claim = any(
                    item.report.sender_id == sender_id and item.report.claim == ClaimType.BLOCKED
                    for items in self.fusion.claims.values() for item in items
                )
                if risk_drop < SOURCE_LINKED_MIN_ROUTE_RISK_DROP and not sender_has_active_blocked_claim:
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

        if self.fusion.method == "trust_threshold" and verified_by_sender:
            for sender_id in verified_by_sender:
                if old_trust_by_sender[sender_id] >= self.trust_threshold > self.trust.score(sender_id):
                    self.defense_replan_needed = True
                    break

        return results

    def replan(self, step: int, reason: str, temporarily_blocked=()) -> bool:
        old = list(self.path or ())
        old_cost = sum(self.fusion.routing_cost(cell, step) for cell in old) if old else None
        self.fusion.set_time(step)
        adapter = _PlanningBelief(self.belief, self.fusion, step, temporarily_blocked)
        planned = astar(
            self.position,
            self.goal,
            lambda cell: adapter.traversal_cost(cell),
        )
        if planned is None and reason.startswith("traffic_") and old:
            # A failed traffic detour should not erase a still-valid waiting route.
            self.path = old
        else:
            self.path = planned
        # A missing path is a real no-path condition in the modular engine.
        new = list(self.path or ())
        new_cost = sum(self.fusion.routing_cost(cell, step) for cell in new) if new else None
        changed = old != new
        self.total_replans += 1
        if reason.startswith("traffic_"):
            self.traffic_replans += 1
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
                if self.delivery_start_step is not None:
                    self.delivery_durations.append(step - self.delivery_start_step)
                self.delivery_start_step = None
                self.task_index += 1
                if self.task_index >= len(self.tasks):
                    self.completed = True
                    self.path = None
                    return "task_transition"
                self.path = None
                return "task_transition"
            self.carrying = True
            self.delivery_start_step = step
            self.path = None
            return "task_transition"
        if not self.path:
            self.no_path_steps += 1
            return "no_path"
        next_cell = self.path[0]
        if next_cell in occupied and next_cell != self.position:
            # Another robot is a transient traffic constraint, not evidence of
            # a physical obstacle. Preserve the route and record a wait.
            self.traffic_wait_steps += 1
            self.consecutive_traffic_waits += 1
            return "traffic_wait"
        if world.state(next_cell, step) == ClaimType.BLOCKED:
            self.blocked_moves += 1
            self.belief.observe(DirectObservation(self.robot_id, next_cell, ClaimType.BLOCKED, step))
            self.path = None
            return "blocked_move"
        self.path.pop(0)
        self.position = next_cell
        self.movement_steps += 1
        self.total_distance += 1.0
        self.consecutive_traffic_waits = 0
        return "move"
