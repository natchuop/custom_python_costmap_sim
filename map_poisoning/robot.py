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
DEFENSE_PRUNE_PERIOD_STEPS = 20
TRAFFIC_REROUTE_AFTER_WAITS = 4


class _PlanningBelief:
    def __init__(self, belief: RobotBeliefMap, fusion: FusionEngine, step: int, temporarily_blocked=()):
        self._belief = belief
        self._fusion = fusion
        self._step = step
        self._temporarily_blocked = {tuple(cell) for cell in temporarily_blocked}
        # A* may inspect the same cell multiple times. At a fixed simulation
        # step the operational map is immutable for the duration of one plan,
        # so cache traversal/blocking results inside this adapter.
        self._blocked_cache: dict[tuple[int, int], bool] = {}
        self._cost_cache: dict[tuple[int, int], float] = {}

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        return self._belief.in_bounds(cell)

    def is_blocked_for_planning(self, cell: tuple[int, int]) -> bool:
        cell = tuple(cell)
        cached = self._blocked_cache.get(cell)
        if cached is not None:
            return cached
        if cell in self._temporarily_blocked:
            value = True
        else:
            claim, status = self._belief.observation_status(cell, self._step)
            if self._fusion.method == "trust_threshold" and status == "unknown":
                value = False
            else:
                value = self._belief.is_blocked_for_planning(cell, self._fusion, self._step)
        self._blocked_cache[cell] = value
        return value

    def traversal_cost(self, cell: tuple[int, int]) -> float:
        cell = tuple(cell)
        if cell in self._cost_cache:
            return self._cost_cache[cell]
        if cell in self._temporarily_blocked:
            value = math.inf
        else:
            claim, status = self._belief.observation_status(cell, self._step)
            if self._fusion.method == "trust_threshold" and status == "unknown":
                peer_cost = self._fusion.soft_routing_cost(cell, self._step)
                value = math.inf if math.isinf(peer_cost) else max(self._belief.UNKNOWN_TRAVERSAL_COST, peer_cost)
            else:
                value = self._belief.traversal_cost(cell, self._step, self._fusion)
        self._cost_cache[cell] = value
        return value


@dataclass
class ReplanRecord:
    step: int
    reason: str
    old_path_cost: float | None
    new_path_cost: float | None
    old_path_length: int
    new_path_length: int
    changed: bool
    adopted: bool = True


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
    confidence_resend_delta: float = 0.10
    environment_change_period_steps: int = 150
    lidar_range_cells: int = 5
    position: tuple[int, int] = field(init=False)
    task_index: int = 0
    carrying: bool = False
    completed: bool = False
    path: list[tuple[int, int]] | None = None
    inbox: list[ClaimReport] = field(default_factory=list)
    pending: dict[str, ClaimReport] = field(default_factory=dict)
    pending_by_cell: dict[tuple[int, int], set[str]] = field(default_factory=dict)
    pending_exact_ids: set[str] = field(default_factory=set)
    pending_index_size: int = 0
    deliveries_completed: int = 0
    # ``delivery_durations`` is retained for compatibility and measures the
    # loaded pickup-to-dropoff leg. ``delivery_cycle_durations`` measures the
    # complete task cycle from task activation (or prior dropoff) to dropoff.
    delivery_durations: list[int] = field(default_factory=list)
    delivery_cycle_durations: list[int] = field(default_factory=list)
    delivery_start_step: int | None = None
    delivery_cycle_start_step: int = 0
    no_path_steps: int = 0
    movement_steps: int = 0
    total_distance: float = 0.0
    total_replans: int = 0  # actual path changes, retained name for report compatibility
    planning_checks: int = 0
    path_changes: int = 0
    blocked_moves: int = 0
    traffic_wait_steps: int = 0
    consecutive_traffic_waits: int = 0
    traffic_replans: int = 0
    productive_replans: int = 0
    replan_records: list[ReplanRecord] = field(default_factory=list)
    no_path_causes: dict[str, int] = field(default_factory=dict)
    last_path_invalid_replan_step: int = -10**9
    last_shared_claim: dict[tuple[int, int], ClaimType] = field(default_factory=dict)
    last_shared_step: dict[tuple[int, int], int] = field(default_factory=dict)
    last_shared_sensor_confidence: dict[tuple[int, int], float] = field(default_factory=dict)
    # report_id -> verification step. Only the operational memory horizon is
    # retained; expired report IDs cannot affect fusion or trust anymore.
    verified_reports: dict[str, int] = field(default_factory=dict)
    last_trust_batches: list[dict] = field(default_factory=list)
    last_route_affecting_report_ids: set[str] = field(default_factory=set)
    previous_scan_observations: dict[tuple[int, int], DirectObservation] = field(default_factory=dict)
    current_scan_observations: dict[tuple[int, int], DirectObservation] = field(default_factory=dict)
    accepted_reports: int = 0
    rejected_reports: int = 0
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

    def _add_pending(self, report: ClaimReport) -> None:
        self.pending[report.report_id] = report
        cell = tuple(report.target_cell)
        self.pending_by_cell.setdefault(cell, set()).add(report.report_id)
        self.pending_index_size += 1
        remembered = self.belief.direct.get(cell)
        if remembered is not None and remembered.step == report.observation_step:
            self.pending_exact_ids.add(report.report_id)

    def _remove_pending(self, report_id: str) -> ClaimReport | None:
        report = self.pending.pop(report_id, None)
        self.pending_exact_ids.discard(report_id)
        if report is not None:
            self.pending_index_size = max(0, self.pending_index_size - 1)
        if report is not None:
            cell = tuple(report.target_cell)
            ids = self.pending_by_cell.get(cell)
            if ids is not None:
                ids.discard(report_id)
                if not ids:
                    self.pending_by_cell.pop(cell, None)
        return report

    def receive(self, report: ClaimReport) -> None:
        self.inbox.append(report)

    def remaining_route(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.path or ())

    def route_evidence(self, step: int) -> float:
        return sum(self.fusion.evidence(cell, step) for cell in self.remaining_route())

    def remaining_route_cost(self, step: int) -> float | None:
        """Current operational cost of the remaining route without running A*."""
        path = list(self.path or ())
        if not path:
            return None
        self.fusion.set_time(step)
        adapter = _PlanningBelief(self.belief, self.fusion, step)
        return self._path_cost(path, adapter)

    def should_share_observation(
        self,
        cell: tuple[int, int],
        claim: ClaimType,
        step: int,
        sensor_confidence: float = 1.0,
    ) -> bool:
        previous = self.last_shared_claim.get(cell)
        previous_conf = self.last_shared_sensor_confidence.get(cell)
        last_step = self.last_shared_step.get(cell, -10**9)
        claim_changed = previous != claim
        confidence_changed = previous_conf is None or abs(float(sensor_confidence) - previous_conf) + 1e-12 >= self.confidence_resend_delta
        refresh_due = step - last_step >= HONEST_REPORT_REFRESH_STEPS
        if not claim_changed and not confidence_changed and not refresh_due:
            return False
        self.last_shared_claim[cell] = claim
        self.last_shared_sensor_confidence[cell] = float(sensor_confidence)
        self.last_shared_step[cell] = step
        return True

    def reports_affect_remaining_route(self, reports: Iterable[tuple[ClaimReport, object]], step: int) -> bool:
        if not self.path:
            return False
        return any(report.report_id in self.last_route_affecting_report_ids for report, _ in reports)

    def path_invalid_or_empty(self, step: int) -> bool:
        if not self.path:
            return True
        return any(self.belief.is_blocked_for_planning(cell, self.fusion, step) for cell in self.path)

    def should_replan_for_path_state(self, step: int) -> bool:
        if self.completed:
            return False
        if not self.path:
            return self.position != self.goal
        if not self.path_invalid_or_empty(step):
            return False
        # Current LiDAR is authoritative and must never be suppressed by the
        # generic path-invalid cooldown.  The cooldown only throttles repeated
        # replans caused by persistent remembered/peer state; a newly visible
        # physical obstacle on any remaining path cell requires an immediate
        # same-step A* check so the robot does not attempt to drive into it.
        if any(
            self.belief.observation_status(cell, step) == (ClaimType.BLOCKED, "current")
            for cell in self.path
        ):
            return True
        return step - self.last_path_invalid_replan_step >= PATH_INVALID_REPLAN_COOLDOWN_STEPS

    def process_inbox(self, step: int, malicious_ids: FrozenSet[str] = frozenset(), runtime_observer=None):
        accepted = []
        self.last_route_affecting_report_ids.clear()
        remaining = set(self.path or ())
        for report in self.inbox:
            policy = decide(self.admission_policy, self.trust.score(report.sender_id), self.trust_threshold)
            if not policy.accepted:
                self.rejected_reports += 1
                continue
            target = tuple(report.target_cell)
            route_candidate = target in remaining and not self.belief.has_direct_free(target, step)
            before_cost = self.belief.traversal_cost(target, step, self.fusion) if route_candidate else None
            # The malicious label is passed only for offline counterfactual
            # metrics; it never changes fusion weighting or robot decisions.
            is_malicious = report.report_id in malicious_ids
            before_count = self.fusion.active_claim_count() if runtime_observer is not None else None
            if runtime_observer is not None:
                import time
                started_ns = time.perf_counter_ns()
            previous = self.fusion.add(report, policy.influence, is_malicious=is_malicious)
            if runtime_observer is not None:
                runtime_observer(report, started_ns, time.perf_counter_ns(), before_count, self.fusion.active_claim_count())
            if previous is not None:
                self._remove_pending(previous.report.report_id)
            self._add_pending(report)
            accepted.append((report, policy))
            self.accepted_reports += 1
            if route_candidate:
                after_cost = self.belief.traversal_cost(target, step, self.fusion)
                materially_changed = (
                    math.isinf(before_cost) != math.isinf(after_cost)
                    or (not math.isinf(before_cost) and not math.isinf(after_cost) and abs(after_cost - before_cost) >= 0.10)
                )
                if materially_changed:
                    self.last_route_affecting_report_ids.add(report.report_id)
        self.inbox.clear()
        return accepted

    def sense(self, world, step: int, other_positions: Iterable[tuple[int, int]], truth_grid=None) -> list[DirectObservation]:
        if step % DEFENSE_PRUNE_PERIOD_STEPS == 0:
            self.fusion.prune(step)
            self.belief.prune_expired(step)
            # Pending validation should not outlive operational report memory.
            self.pending = {
                report_id: report for report_id, report in self.pending.items()
                if step - report.observation_step < self.fusion.max_claim_age
            }
            self.pending_by_cell = {}
            for report_id, report in self.pending.items():
                self.pending_by_cell.setdefault(tuple(report.target_cell), set()).add(report_id)
            self.pending_exact_ids.intersection_update(self.pending.keys())
            self.pending_index_size = len(self.pending)
            self.verified_reports = {
                report_id: verified_step for report_id, verified_step in self.verified_reports.items()
                if step - verified_step < self.fusion.max_claim_age
            }
        truth = world.truth_grid(step) if truth_grid is None else truth_grid
        self.belief.begin_scan(step)
        raw = lidar_observations(truth, self.position, other_positions, radius=self.lidar_range_cells)
        observations: list[DirectObservation] = []
        for cell, reading in raw.items():
            obs = DirectObservation(
                self.robot_id,
                cell,
                reading.claim,
                step,
                reading.sensor_confidence,
            )
            self.belief.observe(obs)
            observations.append(obs)
        self.previous_scan_observations = self.current_scan_observations
        self.current_scan_observations = {obs.cell: obs for obs in observations}
        return observations

    def _same_environment_epoch(self, earlier_step: int, later_step: int) -> bool:
        """Return whether temporary-obstacle truth is guaranteed unchanged.

        Temporary obstacles are authored in fixed change windows.  A peer
        report may remain useful occupancy evidence for the full 300-step
        memory horizon, but trust should only be rewarded/penalized when the
        recipient can compare observations from the same physical-world epoch.
        This avoids blaming an honest sender because a pallet moved later.
        """
        period = max(1, int(self.environment_change_period_steps))
        return int(earlier_step) // period == int(later_step) // period

    def _resolve_observation(
        self, report: ClaimReport, by_cell: dict, step: int
    ) -> tuple[DirectObservation, bool] | None:
        # Prefer the previous scan when it is exactly contemporaneous with the
        # peer report.  This matters when a temporary obstacle changes at the
        # current step: the exact step-N snapshot is stronger evidence than the
        # new step-(N+1) world state.
        exact = self.previous_scan_observations.get(report.target_cell)
        if exact is not None and exact.step == report.observation_step:
            return exact, True
        observed = by_cell.get(report.target_cell)
        if observed is not None:
            return observed, self._same_environment_epoch(report.observation_step, step)
        # Reports arrive after sensing, so a report cannot be validated until
        # the next simulation step.  An exact same-time direct snapshot is
        # always comparable even if verification itself happens later.  Do NOT
        # compare against arbitrary older direct memory: in a dynamic world
        # that can turn an honest report into a false contradiction simply
        # because a pallet moved.
        remembered = self.belief.direct.get(report.target_cell)
        if remembered is None or remembered.step != report.observation_step:
            return None
        if step - remembered.step >= self.belief.memory_steps:
            return None
        return remembered, True

    def _report_age_weight(self, report: ClaimReport, step: int) -> float:
        age = max(0, step - report.observation_step)
        return max(0.0, 1.0 - age / float(self.fusion.max_claim_age))

    def verify(self, observations: Iterable[DirectObservation], step: int):
        """Validate peer reports and perform one trust update per sender/scan."""
        by_cell = {item.cell: item for item in observations}
        processed: set[str] = set()
        candidates: list[tuple[ClaimReport, DirectObservation, VerificationOutcome, float, float]] = []

        def collect(report: ClaimReport, observed: DirectObservation, temporally_comparable: bool) -> None:
            if report.report_id in processed or report.report_id in self.verified_reports:
                return
            processed.add(report.report_id)
            self.verified_reports[report.report_id] = step
            self._remove_pending(report.report_id)
            age_weight = self._report_age_weight(report, step)
            if age_weight <= 0.0 or not temporally_comparable:
                outcome = VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED
            else:
                outcome = VerificationOutcome.CONFIRMED if observed.claim == report.claim else VerificationOutcome.CONTRADICTED_FRESH
            evidence_before = self.fusion.evidence(report.target_cell, step)
            probability_before = self.fusion.probability(report.target_cell, step)
            candidates.append((report, observed, outcome, evidence_before, probability_before))

        # Tests/legacy callers may insert directly into ``pending``. Rebuild
        # the auxiliary index only when its size proves it is out of sync; the
        # normal production path uses _add_pending/_remove_pending and avoids
        # scanning the whole pending set each step.
        if self.pending_index_size != len(self.pending):
            self.pending_by_cell = {}
            self.pending_exact_ids = set()
            for report_id, report in self.pending.items():
                cell = tuple(report.target_cell)
                self.pending_by_cell.setdefault(cell, set()).add(report_id)
                remembered = self.belief.direct.get(cell)
                if remembered is not None and remembered.step == report.observation_step:
                    self.pending_exact_ids.add(report_id)
            self.pending_index_size = len(self.pending)

        candidate_ids = set(self.pending_exact_ids)
        for cell in by_cell:
            candidate_ids.update(self.pending_by_cell.get(cell, ()))
        for cell in self.previous_scan_observations:
            candidate_ids.update(self.pending_by_cell.get(cell, ()))
        for report_id in candidate_ids:
            report = self.pending.get(report_id)
            if report is None:
                continue
            resolved = self._resolve_observation(report, by_cell, step)
            if resolved is not None:
                observed, comparable = resolved
                collect(report, observed, comparable)

        # A current direct observation supersedes any contradictory retained
        # peer claim.  If the report and current sighting are from different
        # temporary-obstacle epochs, retract the stale map evidence without a
        # trust penalty; the physical world may simply have changed.
        for cell, observed in by_cell.items():
            for item in list(self.fusion.claims_at(cell)):
                report = item.report
                if report.claim == observed.claim:
                    continue
                if report.report_id not in processed and report.report_id not in self.verified_reports:
                    collect(report, observed, self._same_environment_epoch(report.observation_step, step))
                elif report.report_id not in processed:
                    self.fusion.retract(report)
                    self._remove_pending(report.report_id)

        by_sender: dict[int, list[tuple[ClaimReport, DirectObservation, VerificationOutcome]]] = {}
        for report, observed, outcome, _, _ in candidates:
            by_sender.setdefault(report.sender_id, []).append((report, observed, outcome))

        sender_trust: dict[int, tuple[float, float]] = {}
        self.last_trust_batches = []
        for sender_id, items in by_sender.items():
            weighted = []
            for report, observed, outcome in items:
                if outcome not in (VerificationOutcome.CONFIRMED, VerificationOutcome.CONTRADICTED_FRESH):
                    continue
                q = (
                    max(0.0, min(1.0, report.sensor_confidence))
                    * self._report_age_weight(report, step)
                    * max(0.0, min(1.0, observed.sensor_confidence))
                )
                if q > 0.0:
                    weighted.append((outcome, q))
            if not weighted:
                current = self.trust.score(sender_id)
                sender_trust[sender_id] = (current, current)
                continue
            total_q = sum(q for _, q in weighted)
            average_quality = total_q / len(weighted)
            confirmed_fraction = sum(q for outcome, q in weighted if outcome == VerificationOutcome.CONFIRMED) / total_q
            contradicted_fraction = sum(q for outcome, q in weighted if outcome == VerificationOutcome.CONTRADICTED_FRESH) / total_q
            confirmed_weight = confirmed_fraction * average_quality
            contradicted_weight = contradicted_fraction * average_quality
            old_trust, new_trust = self.trust.update_batch(sender_id, confirmed_weight, contradicted_weight)
            sender_trust[sender_id] = (old_trust, new_trust)
            self.last_trust_batches.append({
                "step": step,
                "sender_id": sender_id,
                "old_trust": old_trust,
                "new_trust": new_trust,
                "confirmed_weight": confirmed_weight,
                "contradicted_weight": contradicted_weight,
                "validated_reports": len(items),
                "source_memory": self.trust.memory_score(sender_id),
                "report_ids": [report.report_id for report, _, _ in items],
            })

        results = []
        for report, observed, outcome, evidence_before, probability_before in candidates:
            old_trust, new_trust = sender_trust.get(report.sender_id, (self.trust.score(report.sender_id), self.trust.score(report.sender_id)))
            truth_matches = observed.claim == report.claim
            if outcome == VerificationOutcome.CONTRADICTED_FRESH:
                self.fusion.retract(report)
            elif outcome == VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED and not truth_matches:
                self.fusion.retract(report)
            evidence_after = self.fusion.evidence(report.target_cell, step)
            probability_after = self.fusion.probability(report.target_cell, step)
            results.append((
                report,
                outcome,
                old_trust,
                new_trust,
                evidence_before,
                evidence_after,
                probability_before,
                probability_after,
            ))
        return results

    def _path_cost(self, path, adapter: _PlanningBelief) -> float | None:
        if path is None:
            return None
        total = 0.0
        for cell in path:
            cost = adapter.traversal_cost(cell)
            if math.isinf(cost):
                return math.inf
            total += cost
        return total

    def replan(
        self,
        step: int,
        reason: str,
        temporarily_blocked=(),
        *,
        only_if_improved: bool = False,
        improvement_epsilon: float = 0.01,
    ) -> bool:
        old = list(self.path or ())
        self.fusion.set_time(step)
        adapter = _PlanningBelief(self.belief, self.fusion, step, temporarily_blocked)
        old_cost = self._path_cost(old, adapter) if old else None
        planned = astar(self.position, self.goal, lambda cell: adapter.traversal_cost(cell))
        new_cost = self._path_cost(planned, adapter)
        self.planning_checks += 1

        adopt = True
        if only_if_improved and old and planned is not None and old_cost is not None and not math.isinf(old_cost):
            adopt = new_cost is not None and (old_cost - new_cost) > improvement_epsilon
        if planned is None and reason.startswith("traffic_") and old:
            adopt = False
        if adopt:
            self.path = planned
        new = list(self.path or ())
        changed = old != new
        if changed:
            self.path_changes += 1
            self.total_replans += 1
            self.productive_replans += 1
            if reason.startswith("traffic_"):
                self.traffic_replans += 1
        if "path_invalid" in reason or reason == "path_invalid_or_empty":
            self.last_path_invalid_replan_step = step
        self.replan_records.append(ReplanRecord(
            step,
            reason,
            old_cost,
            self._path_cost(new, adapter) if new else None,
            len(old),
            len(new),
            changed,
            adopt,
        ))
        return bool(self.path)

    def classify_no_path(self, world, step: int) -> str:
        truth = astar(self.position, self.goal, lambda cell: math.inf if world.state(cell, step) == ClaimType.BLOCKED else 1.0)
        if truth is None:
            cause = "truth_disconnected"
        else:
            direct = astar(
                self.position,
                self.goal,
                lambda cell: math.inf if self.belief.observation_status(cell, step) == (ClaimType.BLOCKED, "current") else 1.0,
            )
            if direct is None:
                cause = "direct_belief_disconnected"
            else:
                operational = astar(self.position, self.goal, lambda cell: self.belief.traversal_cost(cell, step, self.fusion))
                cause = "peer_fusion_disconnected" if operational is None else "planner_or_state_error"
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
                self.delivery_cycle_durations.append(step - self.delivery_cycle_start_step)
                self.delivery_start_step = None
                self.task_index += 1
                if self.task_index >= len(self.tasks):
                    self.completed = True
                    self.path = None
                    return "task_transition"
                self.delivery_cycle_start_step = step
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
            self.traffic_wait_steps += 1
            self.consecutive_traffic_waits += 1
            return "traffic_wait"
        if world.state(next_cell, step) == ClaimType.BLOCKED:
            self.blocked_moves += 1
            # Physical contact is authoritative and maximum confidence.
            self.belief.observe(DirectObservation(self.robot_id, next_cell, ClaimType.BLOCKED, step, 1.0))
            self.path = None
            return "blocked_move"
        self.path.pop(0)
        self.position = next_cell
        self.movement_steps += 1
        self.total_distance += 1.0
        self.consecutive_traffic_waits = 0
        return "move"
