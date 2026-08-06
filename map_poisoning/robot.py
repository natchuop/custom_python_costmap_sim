"""Independent benign robot state and behavior for the modular simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

from .admission import decide
from .belief import RobotBeliefMap
from .fusion import FusionEngine
from .models import ClaimReport, ClaimType, DeliveryTask, DirectObservation, VerificationOutcome
from .planning import astar
from .trust import TrustModel


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
    sensor_radius: int = 4
    position: tuple[int, int] = field(init=False)
    task_index: int = 0
    carrying: bool = False
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

    def __post_init__(self): self.position = self.start
    @property
    def goal(self) -> tuple[int, int]:
        task = self.tasks[self.task_index]
        return task.dropoff if self.carrying else task.pickup
    def receive(self, report: ClaimReport) -> None: self.inbox.append(report)
    def remaining_route(self) -> tuple[tuple[int, int], ...]: return tuple(self.path or ())
    def route_evidence(self, step: int) -> float: return sum(self.fusion.evidence(cell, step) for cell in self.remaining_route())

    def process_inbox(self, step: int):
        accepted=[]; route_affected=False
        for report in self.inbox:
            policy=decide(self.admission_policy, self.trust.score(report.sender_id), self.trust_threshold)
            if policy.accepted:
                previous=self.fusion.add(report, policy.influence)
                if previous is not None: self.pending.pop(previous.report.report_id, None)
                self.pending[report.report_id]=report; accepted.append((report, policy))
                route_affected = route_affected or report.target_cell in self.remaining_route()
        self.inbox.clear()
        return accepted, route_affected

    def sense(self, world, step: int) -> list[DirectObservation]:
        observations=[]
        for row in range(self.position[0]-self.sensor_radius, self.position[0]+self.sensor_radius+1):
            for col in range(self.position[1]-self.sensor_radius, self.position[1]+self.sensor_radius+1):
                cell=(row,col)
                if abs(row-self.position[0])+abs(col-self.position[1]) > self.sensor_radius or not self.belief.in_bounds(cell): continue
                obs=DirectObservation(self.robot_id, cell, world.state(cell,step), step)
                self.belief.observe(obs); observations.append(obs)
        return observations

    def verify(self, observations: Iterable[DirectObservation], step: int):
        results=[]
        by_cell={item.cell:item for item in observations}
        for report_id, report in list(self.pending.items()):
            observed=by_cell.get(report.target_cell)
            if observed is None: continue
            age=step-report.observation_step
            outcome=VerificationOutcome.CONFIRMED if observed.claim == report.claim else (VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED if age > self.fusion.max_claim_age else VerificationOutcome.CONTRADICTED_FRESH)
            evidence_before=self.fusion.evidence(report.target_cell,step); probability_before=self.fusion.probability(report.target_cell,step)
            old_trust,new_trust=self.trust.update(report.sender_id,outcome)
            evidence_after=self.fusion.evidence(report.target_cell,step); probability_after=self.fusion.probability(report.target_cell,step)
            results.append((report, outcome, old_trust, new_trust, evidence_before, evidence_after, probability_before, probability_after))
            del self.pending[report_id]
        return results

    def replan(self, step: int, reason: str) -> bool:
        old=list(self.path or ()); old_cost=sum(self.fusion.routing_cost(cell,step) for cell in old) if old else None
        self.path=astar(self.position,self.goal,lambda cell:self.belief.traversal_cost(cell,step,self.fusion))
        new=list(self.path or ()); new_cost=sum(self.fusion.routing_cost(cell,step) for cell in new) if new else None
        changed=old != new; self.total_replans += 1; self.productive_replans += int(changed)
        self.replan_records.append(ReplanRecord(step,reason,old_cost,new_cost,len(old),len(new),changed)); return bool(self.path)

    def classify_no_path(self, world, step: int) -> str:
        """Explain a no-path event without consulting attack audit labels."""
        truth = astar(self.position, self.goal, lambda cell: math.inf if world.state(cell, step) == ClaimType.BLOCKED else 1.)
        if truth is None:
            cause = "truth_disconnected"
        else:
            direct = astar(self.position, self.goal, lambda cell: math.inf if self.belief.direct_state(cell) == ClaimType.BLOCKED else 1.)
            if direct is None:
                cause = "direct_belief_disconnected"
            else:
                operational = astar(self.position, self.goal, lambda cell: self.belief.traversal_cost(cell, step, self.fusion))
                cause = "peer_fusion_disconnected" if operational is None else "planner_or_state_error"
        self.no_path_causes[cause] = self.no_path_causes.get(cause, 0) + 1
        return cause

    def move(self, world, step: int) -> str:
        if self.position == self.goal:
            if self.carrying:
                self.carrying=False; self.deliveries_completed+=1; self.task_index=(self.task_index+1)%len(self.tasks)
            else: self.carrying=True
            self.path=None; return "task_transition"
        if not self.path:
            self.no_path_steps+=1; return "no_path"
        next_cell=self.path.pop(0)
        if world.state(next_cell,step)==ClaimType.BLOCKED:
            self.blocked_moves+=1; self.path=None; return "blocked_move"
        self.position=next_cell; self.movement_steps+=1; self.total_distance+=1.; return "move"
