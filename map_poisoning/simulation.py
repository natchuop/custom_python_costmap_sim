"""Headless replay loop.  Attack authoring is intentionally absent from this module."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .admission import decide
from .config import SimulationConfig
from .fusion import FusionEngine
from .metrics import CsvMetrics
from .models import ClaimReport, ClaimType, SimulationPhase, VerificationOutcome
from .planning import astar
from .scenario import ScenarioManifest
from .trust import make_trust_model
from .world import World, demo_grid

def phase_at(step: int, manifest: ScenarioManifest) -> SimulationPhase:
    if step < manifest.phase_boundaries["reconnaissance_end"]: return SimulationPhase.RECONNAISSANCE
    if step < manifest.phase_boundaries["attack_end"]: return SimulationPhase.ATTACK
    return SimulationPhase.RECOVERY

@dataclass
class RunResult:
    output_directory: Path
    method: str
    manifest: ScenarioManifest
    summary: dict

@dataclass
class BenignRobotState:
    robot_id: int
    position: tuple[int, int]
    goals: tuple[tuple[int, int], ...]
    goal_index: int = 0
    path: list[tuple[int, int]] | None = None
    deliveries_completed: int = 0
    no_path_steps: int = 0
    movement_steps: int = 0
    total_distance: float = 0.0
    total_replans: int = 0

    @property
    def goal(self): return self.goals[self.goal_index]

    def finish_delivery(self):
        self.deliveries_completed += 1
        self.goal_index = (self.goal_index + 1) % len(self.goals)
        self.path = None

def replay(config: SimulationConfig, manifest: ScenarioManifest, method: str, output_directory: Path) -> RunResult:
    grid=np.asarray(manifest.static_grid, dtype=np.uint8); world=World(grid, manifest.obstacle_episodes)
    trust=make_trust_model(config.trust.model, config.trust.prior_alpha, config.trust.prior_beta)
    fusion=FusionEngine(method, trust.score, decay_rate=config.fusion.decay_rate, max_claim_age=config.fusion.max_claim_age, cost_scale=config.fusion.cost_scale, cost_exponent=config.fusion.cost_exponent, blocked_probability_threshold=config.fusion.blocked_probability_threshold)
    metrics=CsvMetrics(); attacks={event.step:event for event in manifest.attack_events}; pending=[]; received=accepted=contradicted=0
    total=min(config.total_steps, manifest.phase_boundaries["total"])
    fusion_effect_count = 0
    fusion_absolute_delta_total = 0.0
    malicious_fusion_effect_count = 0
    malicious_fusion_absolute_delta_total = 0.0
    starts=((2,2),(grid.shape[0]-3,grid.shape[1]-3),(2,grid.shape[1]-3))
    all_goals=((grid.shape[0]-3,2),(2,grid.shape[1]-3),(grid.shape[0]-3,grid.shape[1]-3))
    robots={robot_id: BenignRobotState(robot_id, starts[index], tuple(all_goals[(index+offset) % len(all_goals)] for offset in (0,1))) for index,robot_id in enumerate(manifest.benign_robot_ids)}
    replan_requested = {robot_id: True for robot_id in robots}
    for step in range(total):
        phase=phase_at(step,manifest)
        if step in (manifest.phase_boundaries["reconnaissance_end"], manifest.phase_boundaries["attack_end"]): metrics.event(step,"phase_transition",phase=phase.value)
        event=attacks.get(step)
        if event:
            metrics.event(step,"attack_action_injected",scenario_event_id=event.event_id,attack_type=event.attack_type.value,target_cell=event.cells[0],sender_id=event.sender_id,recipients=";".join(map(str,event.recipients)))
            for cell, report_id in zip(event.cells,event.report_ids):
                report=ClaimReport(report_id,event.sender_id,cell,event.claim,event.observation_step,step,step,scenario_event_id=event.event_id)
                # Audit values stay in logging only; fusion receives the operational report.
                metrics.event(step,"report_sent",report_id=report_id,scenario_event_id=event.event_id,attack_type=event.attack_type.value,target_cell=cell,claim=event.claim.name,sender_id=event.sender_id,recipients=";".join(map(str,event.recipients)),malicious_audit=True)
                for recipient in event.recipients:
                    received += 1; admission=decide(config.fusion.admission_policy,trust.score(report.sender_id),config.trust.threshold)
                    metrics.event(step,"admission",report_id=report_id,recipient_id=recipient,accepted=admission.accepted,influence=admission.influence,reason=admission.reason)
                    if admission.accepted: fusion.add(report,admission.influence); accepted += 1; pending.append((step+1,report,recipient))
                    replan_requested[recipient] = True
        # Honest attacker reports during reconnaissance/recovery demonstrate normal reporting continues.
        if phase != SimulationPhase.ATTACK and step % config.communication_period_steps == 0:
            cell=(2 + (step//config.communication_period_steps) % (grid.shape[0]-4), 2)
            claim=world.state(cell,step); report=ClaimReport(f"honest-{step:05}",manifest.malicious_robot_id,cell,claim,step,step,step)
            metrics.event(step,"honest_report_sent",report_id=report.report_id,sender_id=report.sender_id,target_cell=cell,claim=claim.name,phase=phase.value)
            for recipient in manifest.benign_robot_ids:
                fusion.add(report,1.); pending.append((step+1,report,recipient)); replan_requested[recipient] = True
        for due,report,recipient in [x for x in pending if x[0] == step]:
            actual=world.state(report.target_cell,step)
            age=step-report.observation_step
            outcome=VerificationOutcome.CONFIRMED if actual == report.claim else (VerificationOutcome.HONEST_STALE_OR_EXPIRED if age > 20 else VerificationOutcome.CONTRADICTED_FRESH)
            evidence_before = fusion.evidence(report.target_cell, step)
            probability_before = fusion.probability(report.target_cell, step)
            old,new=trust.update(report.sender_id,outcome)
            evidence_after = fusion.evidence(report.target_cell, step)
            probability_after = fusion.probability(report.target_cell, step)
            evidence_delta = evidence_after - evidence_before
            if evidence_delta != 0.0:
                fusion_effect_count += 1
                fusion_absolute_delta_total += abs(evidence_delta)
                if report.scenario_event_id:
                    malicious_fusion_effect_count += 1
                    malicious_fusion_absolute_delta_total += abs(evidence_delta)
                # Only source-linked fusion changes stored evidence after a trust update.
                replan_requested[recipient] = True
            if outcome == VerificationOutcome.CONTRADICTED_FRESH: contradicted += 1
            metrics.trust_update(step, method=method, report_id=report.report_id,
                                 sender_id=report.sender_id, recipient_id=recipient,
                                 outcome=outcome.value, old_trust=old, new_trust=new)
            metrics.fusion_effect(step, method=method, report_id=report.report_id,
                                  sender_id=report.sender_id, recipient_id=recipient,
                                  cell=report.target_cell, evidence_before=evidence_before,
                                  evidence_after=evidence_after,
                                  probability_before=probability_before,
                                  probability_after=probability_after, outcome=outcome.value,
                                  phase=phase.value, observation_age=age,
                                  scenario_event_id=report.scenario_event_id)
        pending[:]=[x for x in pending if x[0] > step]
        for robot in robots.values():
            if robot.position == robot.goal:
                robot.finish_delivery()
                replan_requested[robot.robot_id] = True
            if replan_requested[robot.robot_id] or not robot.path:
                def traversal_cost(cell):
                    if world.state(cell, step) == ClaimType.BLOCKED: return float("inf")
                    return fusion.routing_cost(cell, step)
                robot.path = astar(robot.position, robot.goal, traversal_cost)
                robot.total_replans += 1
                replan_requested[robot.robot_id] = False
            if robot.path:
                robot.position = robot.path.pop(0)
                robot.movement_steps += 1
                robot.total_distance += 1.0
            else:
                robot.no_path_steps += 1
        if step % config.logging.timeseries_period_steps == 0:
            active_cells = tuple(fusion.claims)
            active_evidence = sum(fusion.evidence(cell, step) for cell in active_cells)
            mean_probability = (sum(fusion.probability(cell, step) for cell in active_cells) / len(active_cells)) if active_cells else 0.0
            for robot in robots.values():
                metrics.sample(step=step,phase=phase.value,method=method,robot_id=robot.robot_id,position=robot.position,goal=robot.goal,deliveries_completed=robot.deliveries_completed,benign_no_path_steps=robot.no_path_steps,benign_movement_steps=robot.movement_steps,benign_total_distance=robot.total_distance,benign_total_replans=robot.total_replans,attacker_trust=trust.score(manifest.malicious_robot_id),active_claim_cells=len(active_cells),active_peer_evidence=active_evidence,mean_peer_occupancy_probability=mean_probability,reports_received=received)
    summary={"method":method,"seed":config.seed,"steps_completed":total,"attack_actions":len([e for e in manifest.attack_events if e.step < total]),"benign_total_deliveries_completed":sum(robot.deliveries_completed for robot in robots.values()),"benign_no_path_steps":sum(robot.no_path_steps for robot in robots.values()),"benign_movement_steps":sum(robot.movement_steps for robot in robots.values()),"benign_total_distance":sum(robot.total_distance for robot in robots.values()),"benign_total_replans":sum(robot.total_replans for robot in robots.values()),"reports_received":received,"reports_accepted":accepted,"fresh_contradictions":contradicted,"trust_updates_with_retroactive_evidence_change":fusion_effect_count,"total_absolute_retroactive_evidence_delta":fusion_absolute_delta_total,"malicious_report_retroactive_evidence_changes":malicious_fusion_effect_count,"malicious_report_absolute_retroactive_evidence_delta":malicious_fusion_absolute_delta_total,"final_attacker_trust":trust.score(manifest.malicious_robot_id),"manifest_hash":manifest.map_hash}
    output_directory.mkdir(parents=True,exist_ok=True); metrics.write(output_directory,summary)
    return RunResult(output_directory,method,manifest,summary)
