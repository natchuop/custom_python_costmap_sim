"""Headless manifest replay with independent benign robot belief and behavior."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import SimulationConfig
from .fusion import FusionEngine
from .metrics import CsvMetrics
from .models import ClaimReport, ClaimType, DeliveryTask, SimulationPhase
from .robot import ModularRobot
from .scenario import ScenarioManifest
from .trust import make_trust_model
from .world import World


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


def _build_robots(config: SimulationConfig, world: World, method: str, ids: tuple[int, ...]) -> dict[int, ModularRobot]:
    rows, cols=world.static_grid.shape
    starts=((2,2),(rows-3,cols-3),(2,cols-3))
    targets=((rows-3,2),(2,cols-3),(rows-3,cols-3),(2,2))
    robots={}
    for index, robot_id in enumerate(ids):
        tasks=tuple(DeliveryTask(f"r{robot_id}-task-{task}", targets[(index+task)%4],targets[(index+task+2)%4]) for task in range(config.deliveries_per_robot))
        trust=make_trust_model(config.trust.model,config.trust.prior_alpha,config.trust.prior_beta)
        fusion=FusionEngine(method,trust.score,decay_rate=config.fusion.decay_rate,max_claim_age=config.fusion.max_claim_age,cost_scale=config.fusion.cost_scale,cost_exponent=config.fusion.cost_exponent,blocked_probability_threshold=config.fusion.blocked_probability_threshold)
        from .belief import RobotBeliefMap
        robots[robot_id]=ModularRobot(robot_id,starts[index],tasks,RobotBeliefMap(world.static_grid),trust,fusion,config.trust.threshold,config.fusion.admission_policy)
    return robots


def replay(config: SimulationConfig, manifest: ScenarioManifest, method: str, output_directory: Path) -> RunResult:
    grid=np.asarray(manifest.static_grid,dtype=np.uint8); world=World(grid,manifest.obstacle_episodes)
    robots=_build_robots(config,world,method,manifest.benign_robot_ids)
    if manifest.robot_starts and manifest.task_queues:
        # A manifest controls the fixed starting/task conditions.  Rebuild only
        # the live benign robot state; no object is shared across recipients.
        for robot_id, robot in robots.items():
            robot.start=manifest.robot_starts[robot_id]; robot.position=robot.start; robot.tasks=manifest.task_queues[robot_id]
    metrics=CsvMetrics(); attacks={event.step:event for event in manifest.attack_events}; audit={report_id:event for event in manifest.attack_events for report_id in event.report_ids}
    total=min(config.total_steps,manifest.phase_boundaries["total"]); received=accepted=contradicted=0
    retroactive_changes=retroactive_delta=0.; malicious_retroactive=malicious_delta=0.
    last_shared_direct: dict[tuple[int, tuple[int, int]], ClaimType] = {}
    honest_by_step={report.sent_step: report for report in manifest.honest_attacker_reports}
    no_path_active: dict[int, str | None] = {robot_id: None for robot_id in robots}
    for step in range(total):
        phase=phase_at(step,manifest)
        if step in (manifest.phase_boundaries["reconnaissance_end"],manifest.phase_boundaries["attack_end"]): metrics.event(step,"phase_transition",method=method,phase=phase.value)
        # Direct sensing is authoritative and verification is recipient-specific.
        observations={robot_id:robot.sense(world,step) for robot_id,robot in robots.items()}
        for robot_id,robot in robots.items():
            for report,outcome,old,new,ev_before,ev_after,p_before,p_after in robot.verify(observations[robot_id],step):
                delta=ev_after-ev_before; malicious=report.report_id in audit
                if outcome.value=="contradicted_fresh": contradicted+=1
                if delta:
                    retroactive_changes+=1; retroactive_delta+=abs(delta)
                    if malicious: malicious_retroactive+=1; malicious_delta+=abs(delta)
                metrics.trust_update(step,method=method,report_id=report.report_id,sender_id=report.sender_id,recipient_id=robot_id,outcome=outcome.value,old_trust=old,new_trust=new)
                metrics.fusion_effect(step,method=method,report_id=report.report_id,sender_id=report.sender_id,recipient_id=robot_id,cell=report.target_cell,evidence_before=ev_before,evidence_after=ev_after,probability_before=p_before,probability_after=p_after,outcome=outcome.value,phase=phase.value,observation_age=step-report.observation_step,scenario_event_id=report.scenario_event_id)
        # Honest benign robots share only their direct observations at the communication cadence.
        if step % config.communication_period_steps == 0:
            for sender_id,items in observations.items():
                for item in items:
                    share_key=(sender_id,item.cell)
                    if last_shared_direct.get(share_key) == item.claim:
                        continue
                    last_shared_direct[share_key]=item.claim
                    report=ClaimReport(f"direct-{sender_id}-{step}-{item.cell[0]}-{item.cell[1]}",sender_id,item.cell,item.claim,step,step,step)
                    for recipient_id,recipient in robots.items():
                        if recipient_id != sender_id: recipient.receive(report); received+=1
            # The entire attacker stream is fixed in the manifest.  Honest
            # reports are replayed in all phases; malicious events are added
            # only during the attack phase by the manifest scheduler below.
            report=honest_by_step.get(step)
            if report is not None:
                metrics.event(step,"honest_report_sent",method=method,report_id=report.report_id,sender_id=report.sender_id,target_cell=report.target_cell,claim=report.claim.name,phase=phase.value)
                for recipient in robots.values(): recipient.receive(report); received+=1
        event=attacks.get(step)
        if event:
            metrics.event(step,"attack_action_injected",method=method,scenario_event_id=event.event_id,attack_type=event.attack_type.value,target_cell=event.cells[0],sender_id=event.sender_id,recipients=";".join(map(str,event.recipients)))
            for cell,report_id in zip(event.cells,event.report_ids):
                report=ClaimReport(report_id,event.sender_id,cell,event.claim,event.observation_step,step,step,scenario_event_id=event.event_id)
                metrics.event(step,"report_sent",method=method,report_id=report_id,scenario_event_id=event.event_id,attack_type=event.attack_type.value,target_cell=cell,claim=event.claim.name,sender_id=event.sender_id,recipients=";".join(map(str,event.recipients)),malicious_audit=True)
                for recipient_id in event.recipients: robots[recipient_id].receive(report); received+=1
        # Each recipient independently admits reports, decides whether its own route is affected, and moves.
        for robot_id,robot in robots.items():
            inbox_accepted,route_affected=robot.process_inbox(step); accepted+=len(inbox_accepted)
            reason="initial_or_empty" if not robot.path else ""
            if route_affected: reason="peer_claim_on_route"
            if not robot.path or route_affected: robot.replan(step,reason)
            action=robot.move(world,step)
            if action == "no_path":
                cause=robot.classify_no_path(world,step)
                if no_path_active[robot_id] is None:
                    metrics.event(step,"no_path_started",method=method,robot_id=robot_id,cause=cause)
                no_path_active[robot_id]=cause
            elif no_path_active[robot_id] is not None:
                metrics.event(step,"no_path_ended",method=method,robot_id=robot_id,cause=no_path_active[robot_id])
                no_path_active[robot_id]=None
            if action=="blocked_move": robot.replan(step,"direct_blocked_move")
            if action=="task_transition": robot.replan(step,"task_transition")
            if robot.replan_records and robot.replan_records[-1].step==step:
                record=robot.replan_records[-1]
                metrics.event(step,"replan",method=method,robot_id=robot_id,reason=record.reason,old_path_cost=record.old_path_cost,new_path_cost=record.new_path_cost,old_path_length=record.old_path_length,new_path_length=record.new_path_length,path_changed=record.changed)
            metrics.event(step,"robot_action",method=method,robot_id=robot_id,action=action,position=robot.position,goal=robot.goal)
        if step % config.logging.timeseries_period_steps == 0:
            for robot in robots.values():
                active_cells=tuple(robot.fusion.claims); evidence=sum(robot.fusion.evidence(cell,step) for cell in active_cells)
                metrics.sample(step=step,phase=phase.value,method=method,robot_id=robot.robot_id,position=robot.position,goal=robot.goal,carrying=robot.carrying,deliveries_completed=robot.deliveries_completed,benign_no_path_steps=robot.no_path_steps,no_path_truth_disconnected=robot.no_path_causes.get("truth_disconnected",0),no_path_direct_belief_disconnected=robot.no_path_causes.get("direct_belief_disconnected",0),no_path_peer_fusion_disconnected=robot.no_path_causes.get("peer_fusion_disconnected",0),no_path_planner_or_state_error=robot.no_path_causes.get("planner_or_state_error",0),benign_movement_steps=robot.movement_steps,benign_total_distance=robot.total_distance,benign_total_replans=robot.total_replans,productive_replans=robot.productive_replans,blocked_moves=robot.blocked_moves,attacker_trust=robot.trust.score(manifest.malicious_robot_id),active_claim_cells=len(active_cells),active_peer_evidence=evidence,reports_received=received)
    summary={"method":method,"seed":config.seed,"steps_completed":total,"attack_actions":len(manifest.attack_events),"benign_total_deliveries_completed":sum(r.deliveries_completed for r in robots.values()),"benign_no_path_steps":sum(r.no_path_steps for r in robots.values()),"benign_no_path_truth_disconnected":sum(r.no_path_causes.get("truth_disconnected",0) for r in robots.values()),"benign_no_path_direct_belief_disconnected":sum(r.no_path_causes.get("direct_belief_disconnected",0) for r in robots.values()),"benign_no_path_peer_fusion_disconnected":sum(r.no_path_causes.get("peer_fusion_disconnected",0) for r in robots.values()),"benign_no_path_planner_or_state_error":sum(r.no_path_causes.get("planner_or_state_error",0) for r in robots.values()),"benign_movement_steps":sum(r.movement_steps for r in robots.values()),"benign_total_distance":sum(r.total_distance for r in robots.values()),"benign_total_replans":sum(r.total_replans for r in robots.values()),"benign_productive_replans":sum(r.productive_replans for r in robots.values()),"benign_blocked_moves":sum(r.blocked_moves for r in robots.values()),"reports_received":received,"reports_accepted":accepted,"fresh_contradictions":contradicted,"trust_updates_with_retroactive_evidence_change":retroactive_changes,"total_absolute_retroactive_evidence_delta":retroactive_delta,"malicious_report_retroactive_evidence_changes":malicious_retroactive,"malicious_report_absolute_retroactive_evidence_delta":malicious_delta,"final_attacker_trust_mean":sum(r.trust.score(manifest.malicious_robot_id) for r in robots.values())/len(robots),"manifest_hash":manifest.map_hash}
    output_directory.mkdir(parents=True,exist_ok=True); metrics.write(output_directory,summary)
    return RunResult(output_directory,method,manifest,summary)
