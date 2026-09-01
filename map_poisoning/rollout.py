"""Native modular manifest replay.

This is the only simulation engine used by the public application.  A manifest
contains the complete attack stream, while each robot owns an independent
belief map, fusion engine, and trust model.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np

from .belief import RobotBeliefMap
from .config import SimulationConfig
from .fusion import FusionEngine
from .metrics import CsvMetrics
from .models import ClaimReport, ClaimType, VerificationOutcome
from .planning import astar
from .robot import ModularRobot, TRAFFIC_REROUTE_AFTER_WAITS
from .scenario import ScenarioManifest, scenario_manifest_hash
from .trust import make_trust_model
from .traffic import TrafficState, coordinate_robot_intents, summarize_traffic_events
from .world import World


def _phase(config: SimulationConfig, step: int) -> str:
    if step < config.phases.recon_steps:
        return "RECONNAISSANCE"
    if step < config.phases.recon_steps + config.phases.attack_steps:
        return "ATTACK"
    return "RECOVERY"


def _make_robots(config: SimulationConfig, manifest: ScenarioManifest, method: str):
    static = np.asarray(manifest.static_grid, dtype=np.uint8)
    robots = []
    for robot_id in (manifest.malicious_robot_id, *manifest.benign_robot_ids):
        tasks = (manifest.task_queues or {}).get(robot_id)
        start = (manifest.robot_starts or {}).get(robot_id)
        if not tasks or start is None:
            raise ValueError("manifest is missing modular robot starts or task queues")
        trust = make_trust_model(
            config.trust.model,
            config.trust.prior_alpha,
            config.trust.prior_beta,
            evidence_cap=config.trust.evidence_cap,
            confirmation_multiplier=config.trust.confirmation_multiplier,
            contradiction_multiplier=config.trust.contradiction_multiplier,
            memory_recovery_rate=config.trust.source_memory_recovery_rate,
        )
        fusion = FusionEngine(
            method,
            trust.score,
            trust_memory_score=trust.memory_score,
            decay_rate=config.fusion.decay_rate,
            max_claim_age=config.fusion.max_claim_age,
            cost_scale=config.fusion.cost_scale,
            cost_exponent=config.fusion.cost_exponent,
            blocked_probability_threshold=config.fusion.blocked_probability_threshold,
            congested_impact=config.fusion.congested_impact,
            duplicate_window_steps=config.fusion.duplicate_window_steps,
            trust_threshold=config.trust.threshold,
            majority_unknown_cost=config.fusion.majority_unknown_cost,
        )
        robots.append(
            ModularRobot(
                robot_id,
                tuple(start),
                tuple(tasks),
                RobotBeliefMap(static, memory_steps=config.observation_lifetime_steps),
                trust,
                fusion,
                config.trust.threshold,
                config.fusion.admission_policy,
                confidence_resend_delta=config.confidence_resend_delta,
                environment_change_period_steps=config.temporary_blockage_change_period_steps,
                lidar_range_cells=config.lidar_range_cells,
            )
        )
    return robots

def _route_attacker_cost(
    robot: ModularRobot,
    attacker_id: int,
    step: int,
    minimum_cost_delta: float,
) -> tuple[float, bool]:
    impact = _route_attacker_impact(robot, attacker_id, step, minimum_cost_delta)
    return impact["attack_route_penalty"], impact["attack_induced_path_change"]


def _route_attacker_impact(
    robot: ModularRobot,
    attacker_id: int,
    step: int,
    minimum_cost_delta: float,
) -> dict:
    """Measure the navigation penalty caused by malicious peer claims.

    The comparison replans from the robot's current position with and without
    malicious claims.  Comparing path shapes alone is insufficient: two
    equally good A* tie paths are not attacker influence.
    """
    if not robot.path or robot.completed:
        return {
            "attack_route_penalty": 0.0,
            "attack_extra_path_length": 0,
            "attack_induced_path_change": False,
            "with_attacker_path_length": 0,
            "without_attacker_path_length": 0,
        }

    malicious_claim = lambda claim: claim.is_malicious
    cost_cache: dict[tuple[bool, tuple[int, int]], float] = {}

    def planning_cost(cell, *, exclude_malicious: bool):
        cell = tuple(cell)
        key = (exclude_malicious, cell)
        if key in cost_cache:
            return cost_cache[key]
        if exclude_malicious:
            value = robot.belief.traversal_cost(
                cell,
                step,
                robot.fusion,
                routing_cost_fn=lambda item, now: robot.fusion.routing_cost_excluding_sender(
                    item, now, attacker_id, malicious_claim
                ),
                hard_blocked_fn=lambda: math.isinf(
                    robot.fusion.routing_cost_excluding_sender(
                        cell, step, attacker_id, malicious_claim
                    )
                ),
            )
        else:
            value = robot.belief.traversal_cost(cell, step, robot.fusion)
        cost_cache[key] = value
        return value

    with_attacker = astar(
        robot.position, robot.goal,
        lambda cell: planning_cost(cell, exclude_malicious=False),
    )
    without_attacker = astar(
        robot.position, robot.goal,
        lambda cell: planning_cost(cell, exclude_malicious=True),
    )
    if with_attacker is None:
        return {
            "attack_route_penalty": math.inf if without_attacker is not None else 0.0,
            "attack_extra_path_length": None,
            "attack_induced_path_change": without_attacker is not None,
            "with_attacker_path_length": None,
            "without_attacker_path_length": None if without_attacker is None else max(0, len(without_attacker) - 1),
        }
    if without_attacker is None:
        return {
            "attack_route_penalty": 0.0,
            "attack_extra_path_length": 0,
            "attack_induced_path_change": False,
            "with_attacker_path_length": max(0, len(with_attacker) - 1),
            "without_attacker_path_length": None,
        }

    def path_cost(path, cost):
        return sum(cost(cell) for cell in path[1:])

    with_cost = path_cost(
        with_attacker, lambda cell: planning_cost(cell, exclude_malicious=False)
    )
    without_cost = path_cost(
        without_attacker, lambda cell: planning_cost(cell, exclude_malicious=True)
    )
    penalty = max(0.0, with_cost - without_cost)
    route_changed = tuple(with_attacker) != tuple(without_attacker)
    # A retained claim can add a small soft cost while leaving the chosen route
    # untouched.  That is map influence, but not navigation influence.
    with_length = max(0, len(with_attacker) - 1)
    without_length = max(0, len(without_attacker) - 1)
    return {
        "attack_route_penalty": penalty,
        "attack_extra_path_length": max(0, with_length - without_length),
        "attack_induced_path_change": route_changed and penalty >= minimum_cost_delta,
        "with_attacker_path_length": with_length,
        "without_attacker_path_length": without_length,
    }


def _map_error(robot: ModularRobot, world: World, step: int, truth_grid=None) -> float:
    """Operational disagreement over cells that still have valid knowledge."""
    errors = 0
    total = 0
    truth = world.truth_grid(step) if truth_grid is None else truth_grid
    candidate_cells = set(robot.belief.direct)
    candidate_cells.update(robot.fusion._runner.claims_by_cell.keys())
    for cell in candidate_cells:
        if robot.belief.static_grid[cell]:
            continue
        claim, status = robot.belief.observation_status(cell, step)
        if status == "current":
            predicted_blocked = claim == ClaimType.BLOCKED
        elif status == "memory" and robot.belief.memory_strength(cell, step) >= 0.5:
            predicted_blocked = claim == ClaimType.BLOCKED
        else:
            predicted_blocked = robot.fusion.probability(cell, step) > 0.5
        errors += predicted_blocked != bool(truth[cell])
        total += 1
    return errors / total if total else 0.0

def _log_replan(log: dict, method: str, robot: ModularRobot, step: int) -> None:
    record = robot.replan_records[-1]
    log["events"].append({
        "step": step, "kind": "replan", "method": method,
        "robot_id": robot.robot_id, "reason": record.reason,
        "old_path_cost": record.old_path_cost, "new_path_cost": record.new_path_cost,
        "old_path_length": record.old_path_length,
        "new_path_length": record.new_path_length, "changed": record.changed,
        "path": list(robot.path or ()),
    })


def run_manifest_rollout(
    config: SimulationConfig,
    manifest: ScenarioManifest,
    method: str,
    *,
    show_progress: bool = True,
    capture_reference_state: bool = False,
) -> tuple[World, list[ModularRobot], dict]:
    """Replay a manifest with standardized sensing, trust, and replanning."""
    world = World(np.asarray(manifest.static_grid, dtype=np.uint8), manifest.obstacle_episodes)
    robots = _make_robots(config, manifest, method)
    attacker = manifest.malicious_robot_id
    malicious_ids = frozenset(
        report_id for event in manifest.attack_events for report_id in event.report_ids
    )
    log = {
        "engine": "modular_native",
        "defense_method": method,
        "malicious_robot_id": attacker,
        "phase": [],
        "events": [],
        # Detailed report rows are retained for malicious reports only. Honest
        # report volume is tracked by counters to prevent long-run memory growth.
        "reports": [],
        "report_count_total": 0,
        "malicious_report_count_total": 0,
        "timeseries": [],
        "trust_events": [],
        "attack_injection_steps": [],
        "false_acceptance_count": 0,
        "malicious_report_deliveries": 0,
        "malicious_reports_accepted": 0,
        "malicious_reports_influential": 0,
        "malicious_reports_operationally_ignored": 0,
        "traffic_events": [],
        "attack_relevance": [],
        # Populated only by the attack-free authoring rollout.  It is not
        # written to result CSVs and cannot influence a defense replay.
        "reference_states": {} if capture_reference_state else None,
    }
    traffic_state = TrafficState()
    if config.visualization.animation:
        from .live_view import init_live_log
        init_live_log(log, world, robots, config, manifest)
    max_steps = config.total_steps
    serial = 0
    live_note = "heatmap then live maps" if config.visualization.animation else "no live animation"
    if show_progress:
        print(f"Simulating {max_steps} steps with {method} ({live_note})...", flush=True)
    route_eval_period = max(1, int(config.visualization.route_impact_eval_period_steps))
    latest_route_metrics: dict[int, tuple[float, bool]] = {
        robot.robot_id: (0.0, False) for robot in robots
    }
    attacks_by_step: dict[int, list] = {}
    attack_type_by_report_id: dict[str, str] = {}
    attack_metadata_by_event_id = {
        str(item.get("event_id", item.get("candidate_id"))): item
        for item in (manifest.candidate_metadata or ())
        if item.get("event_id") or item.get("candidate_id")
    }
    for event in manifest.attack_events:
        attacks_by_step.setdefault(int(event.step), []).append(event)
        for report_id in event.report_ids:
            attack_type_by_report_id[str(report_id)] = event.attack_type.value

    for step in range(max_steps):
        if show_progress and (step == 0 or (step + 1) % 200 == 0 or step + 1 == max_steps):
            print(f"  step {step + 1}/{max_steps}", flush=True)
        phase = _phase(config, step)
        log["phase"].append(phase)
        positions = {robot.robot_id: robot.position for robot in robots}
        # A temporary obstacle yields at onset while any robot occupies its
        # footprint; otherwise a discrete rectangle can materialize around a
        # robot and create an artificial full-episode no-path stall.
        truth_grid = world.begin_step(step, positions.values())
        observations_by_robot = {}
        direct_verification_replan: dict[int, bool] = {}
        trust_verification_route_change: dict[int, bool] = {}
        trust_threshold_crossing: dict[int, bool] = {}
        planned_this_step: set[int] = set()
        route_metric_dirty: set[int] = set()

        # Current LiDAR is authoritative. Reports received later in this step
        # cannot be validated until the next step because verification happens
        # only here, before new deliveries are processed.
        for robot in robots:
            other_positions = [position for rid, position in positions.items() if rid != robot.robot_id]
            observations_by_robot[robot.robot_id] = robot.sense(world, step, other_positions, truth_grid=truth_grid)
        if capture_reference_state:
            log["reference_states"][step] = {
                robot.robot_id: {
                    "position": tuple(robot.position),
                    "goal": None if robot.completed else tuple(robot.goal),
                    "path": tuple(robot.path or ()),
                    "visible_cells": tuple(robot.current_scan_observations),
                }
                for robot in robots
            }

        for robot in robots:
            route_cost_before_verification = robot.remaining_route_cost(step)
            results = robot.verify(observations_by_robot[robot.robot_id], step)
            route_cost_after_verification = robot.remaining_route_cost(step)
            if route_cost_before_verification is not None and route_cost_after_verification is not None:
                materially_changed = (
                    math.isinf(route_cost_before_verification) != math.isinf(route_cost_after_verification)
                    or (
                        not math.isinf(route_cost_before_verification)
                        and not math.isinf(route_cost_after_verification)
                        and abs(route_cost_after_verification - route_cost_before_verification) >= 0.10
                    )
                )
                if materially_changed:
                    trust_verification_route_change[robot.robot_id] = True
            for batch in robot.last_trust_batches:
                if (
                    float(batch["old_trust"]) >= config.trust.threshold
                    and float(batch["new_trust"]) < config.trust.threshold
                ):
                    trust_threshold_crossing[robot.robot_id] = True
                trust_event = {
                    "step": step,
                    "kind": "trust_update",
                    "method": method,
                    "sender_id": batch["sender_id"],
                    "recipient_id": robot.robot_id,
                    "old_trust": batch["old_trust"],
                    "new_trust": batch["new_trust"],
                    "confirmed_weight": batch["confirmed_weight"],
                    "contradicted_weight": batch["contradicted_weight"],
                    "validated_reports": batch["validated_reports"],
                    "source_memory": batch["source_memory"],
                    "report_ids": batch["report_ids"],
                    "outcome": (
                        "contradicted_fresh"
                        if batch["contradicted_weight"] > batch["confirmed_weight"]
                        else "confirmed"
                    ),
                }
                log["events"].append(trust_event)
                log["trust_events"].append(trust_event)
            for result in results:
                report, outcome, old, new, evidence_before, evidence_after, probability_before, probability_after = result
                # Fusion effects are useful for attacks and contradictions but
                # need not duplicate every ordinary confirmed report in memory.
                if report.scenario_event_id is not None or outcome != VerificationOutcome.CONFIRMED:
                    log["events"].append({
                        "step": step,
                        "kind": "fusion_effect",
                        "method": method,
                        "report_id": report.report_id,
                        "sender_id": report.sender_id,
                        "recipient_id": robot.robot_id,
                        "target_cell": report.target_cell,
                        "evidence_before": evidence_before,
                        "evidence_after": evidence_after,
                        "probability_before": probability_before,
                        "probability_after": probability_after,
                        "outcome": outcome.value,
                        "phase": phase,
                        "observation_age": step - report.observation_step,
                        "scenario_event_id": report.scenario_event_id,
                        "sensor_confidence": report.sensor_confidence,
                    })
                if report.scenario_event_id is not None:
                    route_metric_dirty.add(robot.robot_id)
                if outcome == VerificationOutcome.CONTRADICTED_FRESH and report.target_cell in set(robot.path or ()):
                    direct_verification_replan[robot.robot_id] = True

        deliveries: dict[int, list[ClaimReport]] = {robot.robot_id: [] for robot in robots}

        # Honest sharing. Sensor confidence is inherited from the LiDAR reading.
        for robot in robots:
            for observation in observations_by_robot[robot.robot_id]:
                if not robot.should_share_observation(
                    observation.cell,
                    observation.claim,
                    step,
                    observation.sensor_confidence,
                ):
                    continue
                serial += 1
                report = ClaimReport(
                    f"peer-{step:06}-{robot.robot_id}-{serial:06}",
                    robot.robot_id,
                    observation.cell,
                    observation.claim,
                    step,
                    sensor_confidence=observation.sensor_confidence,
                )
                log["report_count_total"] += 1
                for recipient in robots:
                    if recipient.robot_id != robot.robot_id:
                        deliveries[recipient.robot_id].append(report)

        # Attack reports retain the manifest's observation_step; stale attacks
        # therefore carry their original blocked-observation timestamp.
        for event in manifest.attack_events:
            if event.step != step:
                continue
            log["attack_injection_steps"].append(step)
            relevance = dict(attack_metadata_by_event_id.get(event.event_id, {}))
            for field in (
                "route_overlap",
                "victim_distance",
                "remaining_obstacle_lifetime",
                "age_since_clearance",
                "target_visible_to_victim",
            ):
                relevance.setdefault(field, None)
            relevance.update({
                "step": step,
                "kind": "attack_relevance",
                "method": method,
                "scenario_event_id": event.event_id,
                "attack_type": event.attack_type.value,
                "observation_step": event.observation_step,
                "target_visible_to_victim": relevance["target_visible_to_victim"],
            })
            log["events"].append(relevance)
            log["attack_relevance"].append(relevance)
            for cell, report_id in zip(event.cells, event.report_ids):
                report = ClaimReport(
                    report_id,
                    event.sender_id,
                    cell,
                    event.claim,
                    event.observation_step,
                    sensor_confidence=1.0,
                    scenario_event_id=event.event_id,
                )
                log["report_count_total"] += 1
                log["malicious_report_count_total"] += 1
                for recipient_id in event.recipients:
                    deliveries[recipient_id].append(report)
                log["reports"].append({
                    "step": step,
                    "report_id": report_id,
                    "sender_id": event.sender_id,
                    "target_cell": cell,
                    "claim": int(event.claim),
                    "sensor_confidence": report.sensor_confidence,
                    "is_malicious": True,
                    "attack_type": event.attack_type.value,
                    "scenario_event_id": event.event_id,
                    "recipient_ids": list(event.recipients),
                })

        # Admission/fusion and immediate replanning are identical in structure
        # for all methods. No operational decision receives the malicious label.
        for robot in robots:
            for report in deliveries[robot.robot_id]:
                robot.receive(report)
            accepted = robot.process_inbox(step, malicious_ids)
            accepted_ids = {report.report_id for report, _ in accepted}
            for report in deliveries[robot.robot_id]:
                if report.report_id not in malicious_ids:
                    continue
                accepted_here = report.report_id in accepted_ids
                evidence = robot.fusion.evidence(report.target_cell, step) if accepted_here else None
                operational_weight = robot.fusion.operational_weight(report, step) if accepted_here else 0.0
                operationally_ignored = bool(accepted_here and operational_weight <= 1e-12)
                log["events"].append({
                    "step": step,
                    "kind": "report_received",
                    "method": method,
                    "report_id": report.report_id,
                    "sender_id": report.sender_id,
                    "recipient_id": robot.robot_id,
                    "target_cell": report.target_cell,
                    "claim": int(report.claim),
                    "sensor_confidence": report.sensor_confidence,
                    "accepted": accepted_here,
                    "operational_weight": operational_weight,
                    "operationally_ignored": operationally_ignored,
                    "is_malicious": True,
                    "evidence_after": evidence,
                    "scenario_event_id": report.scenario_event_id,
                })
                log["malicious_report_deliveries"] += 1
                log["malicious_reports_accepted"] += int(accepted_here)
                log["malicious_reports_influential"] += int(accepted_here and operational_weight > 1e-12)
                log["malicious_reports_operationally_ignored"] += int(operationally_ignored)
                if accepted_here and operational_weight > 1e-12:
                    log["false_acceptance_count"] += 1
                if accepted_here:
                    route_metric_dirty.add(robot.robot_id)

            route_affected = robot.reports_affect_remaining_route(accepted, step)
            route_affecting_ids = set(robot.last_route_affecting_report_ids)
            affecting_attack_types = {
                attack_type_by_report_id[report_id]
                for report_id in route_affecting_ids
                if report_id in attack_type_by_report_id
            }
            path_state = robot.should_replan_for_path_state(step)
            current_route = tuple(robot.path or ())
            temporary_obstacle_on_route = bool(path_state and any(
                not bool(world.static_grid[cell])
                and bool(truth_grid[cell])
                and robot.belief.observation_status(cell, step) == (ClaimType.BLOCKED, "current")
                for cell in current_route
            ))
            other_robot_on_route = bool(path_state and any(
                cell in set(positions.values()) - {robot.position}
                and robot.belief.observation_status(cell, step) == (ClaimType.BLOCKED, "current")
                for cell in current_route
            ))
            trust_route_changed = trust_verification_route_change.get(robot.robot_id, False)
            threshold_crossed = trust_threshold_crossing.get(robot.robot_id, False)
            if path_state or route_affected or direct_verification_replan.get(robot.robot_id, False) or trust_route_changed or threshold_crossed:
                reasons = []
                if route_affected:
                    reasons.append("peer_report_on_route")
                if affecting_attack_types:
                    reasons.append("malicious_report_on_route")
                    reasons.extend(
                        f"{attack_type}_report_on_route"
                        for attack_type in sorted(affecting_attack_types)
                    )
                if direct_verification_replan.get(robot.robot_id, False):
                    reasons.append("direct_verification")
                if trust_route_changed:
                    reasons.append("trust_or_verification_route_change")
                if threshold_crossed:
                    reasons.append("trust_threshold_crossing")
                if path_state:
                    if temporary_obstacle_on_route:
                        reasons.append("temporary_physical_obstacle_on_route")
                    if other_robot_on_route:
                        reasons.append("robot_detected_on_route")
                    if not temporary_obstacle_on_route and not other_robot_on_route:
                        reasons.append("path_invalid_or_empty")
                robot.replan(step, "+".join(reasons) or "map_change")
                _log_replan(log, method, robot, step)
                planned_this_step.add(robot.robot_id)

            # Event-level counterfactual: the same current belief is planned
            # with and without the malicious sender's claims.  This runs only
            # on authored attack steps, not on every simulation step.
            for attack_event in attacks_by_step.get(step, ()):
                if robot.robot_id not in attack_event.recipients:
                    continue
                impact = _route_attacker_impact(
                    robot,
                    attacker,
                    step,
                    config.visualization.route_impact_min_cost_delta,
                )
                log["events"].append({
                    "step": step,
                    "kind": "attack_route_impact",
                    "method": method,
                    "scenario_event_id": attack_event.event_id,
                    "attack_type": attack_event.attack_type.value,
                    "recipient_id": robot.robot_id,
                    **impact,
                })

        # Common periodic optimization catches gradual age/trust-memory changes.
        if step > 0 and step % config.periodic_route_check_steps == 0:
            for robot in robots:
                if robot.completed or robot.robot_id in planned_this_step:
                    continue
                robot.replan(
                    step,
                    "periodic_route_optimization",
                    only_if_improved=True,
                    improvement_epsilon=config.periodic_route_improvement_epsilon,
                )
                _log_replan(log, method, robot, step)
                planned_this_step.add(robot.robot_id)

        # Multi-robot traffic uses frozen intents before any robot moves.
        approved, traffic_events = coordinate_robot_intents(robots, world, step, traffic_state)
        log["traffic_events"].extend(traffic_events)
        recovered = {
            event["robot_id"]
            for event in traffic_events
            if event.get("event_type") in {"traffic_deadlock_recovered", "traffic_yield_completed"}
        }
        occupied = {robot.position for robot in robots}
        for robot in robots:
            if robot.robot_id in recovered:
                robot.replan(step, "traffic_yield_or_deadlock_completed")
                _log_replan(log, method, robot, step)
        for robot in sorted(robots, key=lambda item: item.robot_id):
            before = robot.position
            deliveries_before_move = robot.deliveries_completed
            others = [other for other in robots if other is not robot]
            other_cells = {other.position for other in others}
            other_reserved = set().union(*(other.reserved_cells() for other in others)) if others else set()
            if not approved.get(robot.robot_id, True):
                robot.traffic_wait_steps += 1
                robot.consecutive_traffic_waits += 1
                action = "traffic_wait"
            else:
                action = robot.move(world, step, occupied - {before})
                if action == "move":
                    occupied.discard(before)
                    occupied.add(robot.position)
                    robot.consecutive_traffic_waits = 0
            no_path_cause = robot.classify_no_path(world, step) if action == "no_path" else None
            robot.record_position()
            if action == "blocked_move" or (action == "task_transition" and not robot.completed):
                robot.replan(step, action)
                _log_replan(log, method, robot, step)
            elif action == "traffic_wait" and robot.consecutive_traffic_waits >= TRAFFIC_REROUTE_AFTER_WAITS:
                robot.replan(step, "traffic_wait_reroute", other_reserved | other_cells)
                _log_replan(log, method, robot, step)
            if robot.deliveries_completed > deliveries_before_move:
                log["events"].append({
                    "step": step,
                    "kind": "delivery_completed",
                    "method": method,
                    "robot_id": robot.robot_id,
                    "delivery_number": robot.deliveries_completed,
                    "phase": phase,
                    "loaded_delivery_duration_steps": robot.delivery_durations[-1] if robot.delivery_durations else None,
                    "delivery_cycle_duration_steps": robot.delivery_cycle_durations[-1] if robot.delivery_cycle_durations else None,
                })
            log["events"].append({
                "step": step,
                "kind": "robot_action",
                "method": method,
                "robot_id": robot.robot_id,
                "action": action,
                "reason": no_path_cause,
                "phase": phase,
                "position": robot.position,
                "goal": robot.goal if not robot.completed else None,
            })

        for robot in robots:
            if robot.robot_id != attacker and (robot.robot_id in route_metric_dirty or step % route_eval_period == 0):
                latest_route_metrics[robot.robot_id] = _route_attacker_cost(
                    robot,
                    attacker,
                    step,
                    config.visualization.route_impact_min_cost_delta,
                )
            route_cost, route_affected = latest_route_metrics[robot.robot_id]
            active_fake_claims = robot.fusion.active_malicious_claim_count()
            sample = {
                "step": step,
                "phase": phase,
                "method": method,
                "robot_id": robot.robot_id,
                "position": robot.position,
                "goal": None if robot.completed else robot.goal,
                "deliveries_completed": robot.deliveries_completed,
                "mean_delivery_time_steps": (
                    sum(robot.delivery_durations) / len(robot.delivery_durations)
                    if robot.delivery_durations else None
                ),
                "mean_delivery_cycle_time_steps": (
                    sum(robot.delivery_cycle_durations) / len(robot.delivery_cycle_durations)
                    if robot.delivery_cycle_durations else None
                ),
                "benign_no_path_steps": robot.no_path_steps,
                "benign_traffic_wait_steps": robot.traffic_wait_steps,
                "benign_movement_steps": robot.movement_steps,
                "benign_total_distance": robot.total_distance,
                "benign_total_replans": robot.total_replans,
                "planning_checks": robot.planning_checks,
                "path_changes": robot.path_changes,
                "attacker_trust": robot.trust.score(attacker),
                "attacker_source_memory": robot.trust.memory_score(attacker),
                "attacker_is_trusted": robot.trust.score(attacker) >= config.trust.threshold,
                "active_fake_claim_count": active_fake_claims,
                "influential_fake_claim_count": active_fake_claims if route_affected else 0,
                "attacker_route_cost_delta": route_cost,
                "route_affected_by_attacker": route_affected,
                "attacker_attributable_cost_on_route": route_cost,
                "preferred_route_affected_by_attacker": route_affected,
                "map_error": (
                    _map_error(robot, world, step, truth_grid=truth_grid)
                    if step % config.logging.timeseries_period_steps == 0 or step == max_steps - 1
                    else None
                ),
            }
            log["timeseries"].append(sample)
        if config.visualization.animation:
            from .live_view import record_live_frame
            record_live_frame(log, world, robots, step, phase)
    return world, robots, log

def _persist_event(event: dict, attacker_id: int) -> bool:
    """Keep the CSV trace small enough for report generation on full runs."""
    kind = event.get("kind")
    if kind == "report_received":
        return bool(event.get("is_malicious"))
    if kind == "fusion_effect":
        return event.get("scenario_event_id") is not None or event.get("outcome") != "confirmed"
    if kind == "trust_update":
        return event.get("sender_id") == attacker_id or event.get("outcome") != "confirmed"
    return True


def collect_rollout_metrics(config: SimulationConfig, manifest: ScenarioManifest, method: str, world, robots, log: dict) -> tuple[dict, CsvMetrics]:
    collector = CsvMetrics()
    attacker_id = log.get("malicious_robot_id", manifest.malicious_robot_id)
    for event in log["events"]:
        if not _persist_event(event, attacker_id):
            continue
        payload = dict(event)
        collector.event(payload.pop("step"), payload.pop("kind"), **payload)
    benign = [robot for robot in robots if robot.robot_id in manifest.benign_robot_ids]
    samples = log["timeseries"]
    for sample in samples:
        if sample["step"] % config.logging.timeseries_period_steps == 0:
            collector.sample(**sample)
    benign_samples = [sample for sample in samples if sample["robot_id"] in manifest.benign_robot_ids]
    map_errors = [sample["map_error"] for sample in benign_samples if sample["map_error"] is not None]
    final_errors = [sample["map_error"] for sample in benign_samples if sample["step"] == config.total_steps - 1 and sample["map_error"] is not None]
    last_injection = max(log["attack_injection_steps"], default=None)
    ever_affected = any(sample["route_affected_by_attacker"] for sample in benign_samples)
    recovery = None
    if ever_affected and last_injection is not None:
        for step in range(last_injection, config.total_steps):
            step_samples = [sample for sample in benign_samples if sample["step"] == step]
            if step_samples and all(not sample["route_affected_by_attacker"] for sample in step_samples):
                recovery = step - last_injection
                break
    trust_events = log["trust_events"]
    traffic_counts = summarize_traffic_events(log.get("traffic_events", []))
    initial_trust = config.trust.prior_alpha / max(1e-12, config.trust.prior_alpha + config.trust.prior_beta)
    distrust_steps_by_robot: dict[int, int | None] = {}
    attacker_min_trust_by_robot: dict[int, float] = {}
    attacker_min_source_memory_by_robot: dict[int, float] = {}
    for robot in benign:
        recipient_events = [
            event for event in trust_events
            if event.get("sender_id") == manifest.malicious_robot_id
            and event.get("recipient_id") == robot.robot_id
        ]
        distrust_steps_by_robot[robot.robot_id] = next(
            (int(event["step"]) for event in recipient_events if float(event.get("new_trust", 1.0)) < config.trust.threshold),
            None,
        )
        attacker_min_trust_by_robot[robot.robot_id] = min(
            [initial_trust] + [float(event.get("new_trust", initial_trust)) for event in recipient_events]
        )
        attacker_min_source_memory_by_robot[robot.robot_id] = min(
            [initial_trust] + [float(event.get("source_memory", initial_trust)) for event in recipient_events]
        )
    distrusted_count = sum(step is not None for step in distrust_steps_by_robot.values())
    all_benign_distrust = distrusted_count == len(benign) and bool(benign)
    distrust = min((step for step in distrust_steps_by_robot.values() if step is not None), default=None)
    time_to_all_distrust = max(distrust_steps_by_robot.values()) if all_benign_distrust else None
    delivery_events = [
        event for event in log["events"]
        if event.get("kind") == "delivery_completed"
        and event.get("robot_id") in manifest.benign_robot_ids
    ]
    delivery_after_attack = sum(
        1 for event in delivery_events if int(event.get("step", -1)) >= config.phases.recon_steps
    )
    delivery_after_distrust = sum(
        1 for event in delivery_events
        if distrust_steps_by_robot.get(int(event.get("robot_id"))) is not None
        and int(event.get("step", -1)) >= int(distrust_steps_by_robot[int(event.get("robot_id"))])
    )
    delivery_durations = [duration for robot in benign for duration in robot.delivery_durations]
    delivery_cycle_durations = [duration for robot in benign for duration in robot.delivery_cycle_durations]

    def percentile(values, quantile):
        return float(np.percentile(values, quantile)) if values else None

    impact_events = [event for event in log["events"] if event.get("kind") == "attack_route_impact"]
    benign_replan_events = [
        event for event in log["events"]
        if event.get("kind") == "replan"
        and event.get("robot_id") in manifest.benign_robot_ids
    ]

    def replan_count(reason_token: str, *, changed_only: bool = False) -> int:
        return sum(
            1 for event in benign_replan_events
            if reason_token in str(event.get("reason", ""))
            and (not changed_only or bool(event.get("changed")))
        )
    impacts_by_attack: dict[str, list[dict]] = {}
    for event in impact_events:
        impacts_by_attack.setdefault(str(event.get("scenario_event_id")), []).append(event)
    per_attack_penalties = []
    per_attack_extra_lengths = []
    attack_ids_with_path_change = set()
    for attack_id, events in impacts_by_attack.items():
        finite_penalties = [float(event["attack_route_penalty"]) for event in events if event.get("attack_route_penalty") is not None]
        extra_lengths = [int(event["attack_extra_path_length"]) for event in events if event.get("attack_extra_path_length") is not None]
        if finite_penalties:
            per_attack_penalties.append(max(finite_penalties))
        if extra_lengths:
            per_attack_extra_lengths.append(max(extra_lengths))
        if any(bool(event.get("attack_induced_path_change")) for event in events):
            attack_ids_with_path_change.add(attack_id)
    affected_steps = {
        int(sample["step"])
        for sample in benign_samples
        if bool(sample.get("route_affected_by_attacker"))
    }
    summary = {
        "method": method, "engine": "modular_native", "seed": config.seed,
        "steps_completed": config.total_steps, "attack_actions": len([report for report in log["reports"] if report["is_malicious"]]),
        "benign_total_deliveries_completed": sum(robot.deliveries_completed for robot in benign),
        "benign_delivery_time_mean_steps": (
            sum(delivery_durations) / len(delivery_durations) if delivery_durations else None
        ),
        "benign_loaded_delivery_duration_mean_steps": (
            sum(delivery_durations) / len(delivery_durations) if delivery_durations else None
        ),
        "benign_loaded_delivery_duration_median_steps": percentile(delivery_durations, 50),
        "benign_loaded_delivery_duration_p95_steps": percentile(delivery_durations, 95),
        "benign_delivery_cycle_duration_mean_steps": (
            sum(delivery_cycle_durations) / len(delivery_cycle_durations) if delivery_cycle_durations else None
        ),
        "benign_delivery_cycle_duration_median_steps": percentile(delivery_cycle_durations, 50),
        "benign_delivery_cycle_duration_p95_steps": percentile(delivery_cycle_durations, 95),
        "benign_success_rate": sum(robot.deliveries_completed for robot in benign) / max(1, len(benign) * config.deliveries_per_robot),
        "benign_deliveries_after_attack": delivery_after_attack,
        "benign_deliveries_after_distrust": delivery_after_distrust,
        "benign_no_path_steps": sum(robot.no_path_steps for robot in benign),
        "benign_no_path_causes": {
            cause: sum(robot.no_path_causes.get(cause, 0) for robot in benign)
            for cause in (
                "truth_disconnected",
                "direct_belief_disconnected",
                "peer_fusion_disconnected",
                "planner_or_state_error",
            )
        },
        "benign_movement_steps": sum(robot.movement_steps for robot in benign),
        "benign_total_distance": sum(robot.total_distance for robot in benign),
        "benign_total_replans": sum(robot.total_replans for robot in benign),
        "benign_planning_checks": sum(robot.planning_checks for robot in benign),
        "benign_path_changes": sum(robot.path_changes for robot in benign),
        "temporary_obstacle_replan_checks": replan_count("temporary_physical_obstacle_on_route"),
        "temporary_obstacle_path_changes": replan_count("temporary_physical_obstacle_on_route", changed_only=True),
        "temporary_obstacle_deferred_activation_steps": world.deferred_activation_steps,
        "temporary_obstacle_episodes_deferred": len(world.deferred_episode_ids),
        "attack_events_while_obstacle_deferred": sum(
            1
            for event in manifest.attack_events
            if event.obstacle_episode_id is not None
            and event.obstacle_episode_id in world.deferred_episode_ids
            and (
                world.activation_step(event.obstacle_episode_id) is None
                or int(event.step) < int(world.activation_step(event.obstacle_episode_id))
            )
        ),
        "robot_on_route_replan_checks": replan_count("robot_detected_on_route"),
        "robot_on_route_path_changes": replan_count("robot_detected_on_route", changed_only=True),
        "malicious_report_replan_checks": replan_count("malicious_report_on_route"),
        "malicious_report_path_changes": replan_count("malicious_report_on_route", changed_only=True),
        "fake_obstacle_replan_checks": replan_count("fake_obstacle_report_on_route"),
        "fake_obstacle_path_changes": replan_count("fake_obstacle_report_on_route", changed_only=True),
        "false_clearance_replan_checks": replan_count("false_clearance_report_on_route"),
        "false_clearance_path_changes": replan_count("false_clearance_report_on_route", changed_only=True),
        "stale_reassertion_replan_checks": replan_count("stale_reassertion_report_on_route"),
        "stale_reassertion_path_changes": replan_count("stale_reassertion_report_on_route", changed_only=True),
        "benign_productive_replans": sum(robot.productive_replans for robot in benign),
        "benign_blocked_world": sum(robot.blocked_moves for robot in benign),
        "benign_blocked_moves": sum(robot.blocked_moves for robot in benign),
        "benign_traffic_wait_steps": sum(robot.traffic_wait_steps for robot in benign),
        "traffic_event_counts": traffic_counts,
        **traffic_counts,
        "traffic_replans": sum(robot.traffic_replans for robot in benign),
        "intent_commit_mismatches": 0,
        "corridor_entry_denied": traffic_counts.get("corridor_entry_denied", 0),
        "corridor_reservations_started": 0,
        "corridor_reservations_released": 0,
        "traffic_replans_suppressed": 0,
        "idle_parking_events": 0,
        "per_robot_idle_steps": {},
        "prevented_robot_conflicts": traffic_counts["vertex_conflicts_detected"]
        + traffic_counts["head_on_swap_conflicts_detected"]
        + traffic_counts["reservation_conflicts_detected"],
        "time_to_distrust_malicious_robot": distrust,
        "time_to_all_benign_distrust": time_to_all_distrust,
        "distrusted_benign_robot_count": distrusted_count,
        "all_benign_distrust_attacker": int(all_benign_distrust),
        "time_to_distrust_by_robot": distrust_steps_by_robot,
        "attacker_min_trust_by_robot": attacker_min_trust_by_robot,
        "attacker_min_source_memory_by_robot": attacker_min_source_memory_by_robot,
        "attacker_min_trust_mean": sum(attacker_min_trust_by_robot.values()) / max(1, len(attacker_min_trust_by_robot)),
        "attacker_min_source_memory_mean": sum(attacker_min_source_memory_by_robot.values()) / max(1, len(attacker_min_source_memory_by_robot)),
        "benign_deliveries_by_robot": {robot.robot_id: robot.deliveries_completed for robot in benign},
        "malicious_verified_false_reports": sum(
            1 for event in log["events"]
            if event.get("kind") == "fusion_effect"
            and event.get("sender_id") == manifest.malicious_robot_id
            and event.get("outcome") == "contradicted_fresh"
        ),
        "fresh_contradictions": sum(
            1 for event in log["events"]
            if event.get("kind") == "fusion_effect" and event.get("outcome") == "contradicted_fresh"
        ),
        "malicious_contradiction_batches": sum(
            1 for event in trust_events
            if event.get("sender_id") == manifest.malicious_robot_id
            and float(event.get("contradicted_weight", 0.0)) > 0.0
        ),
        "final_attacker_trust_mean": sum(robot.trust.score(manifest.malicious_robot_id) for robot in benign) / max(1, len(benign)),
        "map_error_mean": sum(map_errors) / max(1, len(map_errors)),
        "map_error_final": sum(final_errors) / max(1, len(final_errors)),
        "false_acceptance_count": log["false_acceptance_count"],
        "false_acceptance_rate": log["false_acceptance_count"] / max(1, log["malicious_report_deliveries"]),
        "malicious_report_deliveries": log["malicious_report_deliveries"],
        "malicious_reports_accepted": log["malicious_reports_accepted"],
        "malicious_reports_influential": log["malicious_reports_influential"],
        "malicious_reports_operationally_ignored": log["malicious_reports_operationally_ignored"],
        "attack_route_penalty_mean": (
            sum(per_attack_penalties) / len(per_attack_penalties) if per_attack_penalties else 0.0
        ),
        "attack_route_penalty_max": max(per_attack_penalties, default=0.0),
        "attack_route_penalty_total": sum(per_attack_penalties),
        "attack_extra_path_length_mean": (
            sum(per_attack_extra_lengths) / len(per_attack_extra_lengths) if per_attack_extra_lengths else 0.0
        ),
        "attack_extra_path_length_max": max(per_attack_extra_lengths, default=0),
        "attack_extra_path_length_total": sum(per_attack_extra_lengths),
        "attack_induced_path_changes": len(attack_ids_with_path_change),
        "attacks_evaluated_for_route_impact": len(impacts_by_attack),
        "steps_route_affected_by_attacker": len(affected_steps),
        "recovery_time_steps": recovery,
        "manifest_hash": manifest.map_hash, "map_hash": manifest.map_hash,
        "scenario_manifest_hash": scenario_manifest_hash(manifest),
    }
    return summary, collector


def replay_manifest(config: SimulationConfig, manifest: ScenarioManifest, method: str, output_directory: Path, *, show_animation: bool = False) -> SimpleNamespace:
    world, robots, log = run_manifest_rollout(config, manifest, method)
    if show_animation:
        print("Opening attack-free reference heatmap, then live belief-map windows...", flush=True)
        from .live_view import show_live_windows
        show_live_windows(log, world, robots, block=True)
    summary, collector = collect_rollout_metrics(config, manifest, method, world, robots, log)
    output_directory.mkdir(parents=True, exist_ok=True)
    print("Writing CSVs and diagrams...", flush=True)
    collector.write(output_directory, summary)
    if config.logging.generate_plots:
        try:
            from .reporting import generate_run_report
            generate_run_report(output_directory, formats=(config.logging.plot_format,))
        except Exception as exc:
            warning = f"Warning: plot generation failed for {output_directory}: {exc}"
            print(warning)
            (output_directory / "plot_generation_error.txt").write_text(warning + "\n", encoding="utf-8")
    if manifest.reconnaissance_heatmap is not None:
        from .recon_authoring import save_traffic_heatmap_artifacts
        save_traffic_heatmap_artifacts(
            output_directory,
            np.asarray(manifest.reconnaissance_heatmap, dtype=np.int32),
            title=f"Attack-free reference traffic heatmap | seed {config.seed}",
        )
    elif log.get("live", {}).get("recon_heatmap") is not None:
        from .recon_authoring import save_traffic_heatmap_artifacts
        save_traffic_heatmap_artifacts(
            output_directory,
            np.asarray(log["live"]["recon_heatmap"], dtype=np.int32),
            title=f"Attack-free reference traffic heatmap | seed {config.seed}",
        )
    effective = config.to_dict() | {"effective_method": method, "engine": "modular_native"}
    CsvMetrics.config(output_directory / "effective_config.json", effective)
    return SimpleNamespace(output_directory=output_directory, method=method, summary=summary, world=world, robots=robots, log=log)
