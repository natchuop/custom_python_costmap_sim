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
from .models import ClaimReport, ClaimType
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
        trust = make_trust_model(config.trust.model, config.trust.prior_alpha, config.trust.prior_beta)
        fusion = FusionEngine(
            method,
            trust.score,
            decay_rate=config.fusion.decay_rate,
            max_claim_age=config.fusion.max_claim_age,
            cost_scale=config.fusion.cost_scale,
            cost_exponent=config.fusion.cost_exponent,
            blocked_probability_threshold=config.fusion.blocked_probability_threshold,
            congested_impact=config.fusion.congested_impact,
            duplicate_window_steps=config.fusion.duplicate_window_steps,
            trust_threshold=config.trust.threshold,
        )
        robots.append(
            ModularRobot(
                robot_id,
                tuple(start),
                tuple(tasks),
                RobotBeliefMap(static, memory_steps=config.direct_memory_steps),
                trust,
                fusion,
                config.trust.threshold,
                config.fusion.admission_policy,
            )
        )
    return robots


def _route_attacker_cost(
    robot: ModularRobot,
    attacker_id: int,
    step: int,
    minimum_cost_delta: float,
) -> tuple[float, bool]:
    """Measure the navigation penalty caused by malicious peer claims.

    The comparison replans from the robot's current position with and without
    malicious claims.  Comparing path shapes alone is insufficient: two
    equally good A* tie paths are not attacker influence.
    """
    if not robot.path or robot.completed:
        return 0.0, False

    malicious_claim = lambda claim: claim.is_malicious

    def planning_cost(cell, *, exclude_malicious: bool):
        if exclude_malicious:
            return robot.belief.traversal_cost(
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
        return robot.belief.traversal_cost(cell, step, robot.fusion)

    with_attacker = astar(
        robot.position, robot.goal,
        lambda cell: planning_cost(cell, exclude_malicious=False),
    )
    without_attacker = astar(
        robot.position, robot.goal,
        lambda cell: planning_cost(cell, exclude_malicious=True),
    )
    if with_attacker is None:
        return (math.inf, without_attacker is not None)
    if without_attacker is None:
        return 0.0, False

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
    return penalty, route_changed and penalty >= minimum_cost_delta


def _map_error(robot: ModularRobot, world: World, step: int) -> float:
    """Operational map disagreement over non-static cells (unknown is free)."""
    errors = 0
    total = 0
    candidate_cells = set(robot.belief.direct)
    candidate_cells.update(robot.fusion._runner.claims_by_cell.keys())
    for cell in candidate_cells:
        if robot.belief.static_grid[cell]:
            continue
            direct = robot.belief.direct_state(cell, step)
            predicted_blocked = (
                direct == ClaimType.BLOCKED
                if direct is not None
                else robot.fusion.probability(cell, step) >= 0.5
            )
            errors += predicted_blocked != (world.state(cell, step) == ClaimType.BLOCKED)
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
) -> tuple[World, list[ModularRobot], dict]:
    """Replay a manifest without importing the retired simulator."""
    world = World(np.asarray(manifest.static_grid, dtype=np.uint8), manifest.obstacle_episodes)
    robots = _make_robots(config, manifest, method)
    attacker = manifest.malicious_robot_id
    malicious_ids = frozenset(
        report_id for event in manifest.attack_events for report_id in event.report_ids
    )
    event_by_report = {
        report_id: event for event in manifest.attack_events for report_id in event.report_ids
    }
    log = {
        "engine": "modular_native",
        "defense_method": method,
        "malicious_robot_id": attacker,
        "phase": [],
        "events": [],
        "reports": [],
        "timeseries": [],
        "trust_events": [],
        "attack_injection_steps": [],
        "false_acceptance_count": 0,
        "malicious_report_deliveries": 0,
        "malicious_reports_accepted": 0,
        "traffic_events": [],
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

    for step in range(max_steps):
        if show_progress and (step == 0 or (step + 1) % 200 == 0 or step + 1 == max_steps):
            print(f"  step {step + 1}/{max_steps}", flush=True)
        phase = _phase(config, step)
        log["phase"].append(phase)
        truth = world.truth_grid(step)
        positions = {robot.robot_id: robot.position for robot in robots}

        # Direct sensing and verification are independent for each robot.
        verification_replan: dict[int, bool] = {}
        observations_by_robot = {}

        def apply_verification(targets=None) -> None:
            for robot in targets or robots:
                for result in robot.verify(observations_by_robot[robot.robot_id], step):
                    report, outcome, old, new, evidence_before, evidence_after, probability_before, probability_after = result
                    trust_event = {
                        "step": step, "kind": "trust_update", "method": method,
                        "report_id": report.report_id, "sender_id": report.sender_id,
                        "recipient_id": robot.robot_id, "outcome": outcome.value,
                        "old_trust": old, "new_trust": new,
                    }
                    log["events"].append(trust_event)
                    log["trust_events"].append(trust_event)
                    log["events"].append({
                        "step": step, "kind": "fusion_effect", "method": method,
                        "report_id": report.report_id, "sender_id": report.sender_id,
                        "recipient_id": robot.robot_id, "target_cell": report.target_cell,
                        "evidence_before": evidence_before, "evidence_after": evidence_after,
                        "probability_before": probability_before, "probability_after": probability_after,
                        "outcome": outcome.value, "phase": phase,
                        "observation_age": step - report.observation_step,
                        "scenario_event_id": report.scenario_event_id,
                    })
                    if outcome.value == "contradicted_fresh":
                        verification_replan[robot.robot_id] = True

        for robot in robots:
            other_positions = [position for rid, position in positions.items() if rid != robot.robot_id]
            observations_by_robot[robot.robot_id] = robot.sense(world, step, other_positions)
        apply_verification()

        deliveries: dict[int, list[ClaimReport]] = {robot.robot_id: [] for robot in robots}
        # Robots share local observations throughout all phases.
        for robot in robots:
            for observation in observations_by_robot[robot.robot_id]:
                if not robot.should_share_observation(observation.cell, observation.claim, step):
                    continue
                serial += 1
                report = ClaimReport(
                    f"peer-{step:06}-{robot.robot_id}-{serial:06}", robot.robot_id,
                    observation.cell, observation.claim, step, step, step,
                )
                for recipient in robots:
                    if recipient.robot_id != robot.robot_id:
                        deliveries[recipient.robot_id].append(report)
                log["reports"].append({
                    "step": step, "report_id": report.report_id, "sender_id": robot.robot_id,
                    "target_cell": report.target_cell, "claim": int(report.claim),
                    "is_malicious": False, "scenario_event_id": None,
                    "recipient_ids": [peer.robot_id for peer in robots if peer.robot_id != robot.robot_id],
                })

        # Attack reports are delivered only to the recipients captured in the manifest.
        for event in manifest.attack_events:
            if event.step != step:
                continue
            log["attack_injection_steps"].append(step)
            for cell, report_id in zip(event.cells, event.report_ids):
                report = ClaimReport(
                    report_id, event.sender_id, cell, event.claim, event.observation_step,
                    event.step, step, scenario_event_id=event.event_id,
                )
                for recipient_id in event.recipients:
                    deliveries[recipient_id].append(report)
                log["reports"].append({
                    "step": step, "report_id": report_id, "sender_id": event.sender_id,
                    "target_cell": cell, "claim": int(event.claim), "is_malicious": True,
                    "attack_type": event.attack_type.value, "scenario_event_id": event.event_id,
                    "recipient_ids": list(event.recipients),
                })

        # Each recipient independently accepts and fuses its delivered reports.
        for robot in robots:
            for report in deliveries[robot.robot_id]:
                robot.receive(report)
            accepted, _ = robot.process_inbox(step, malicious_ids)
            accepted_ids = {report.report_id for report, _ in accepted}
            for report in deliveries[robot.robot_id]:
                is_malicious = report.report_id in malicious_ids
                accepted_here = report.report_id in accepted_ids
                evidence = robot.fusion.evidence(report.target_cell, step) if accepted_here else None
                log["events"].append({
                    "step": step, "kind": "report_received", "method": method,
                    "report_id": report.report_id, "sender_id": report.sender_id,
                    "recipient_id": robot.robot_id, "target_cell": report.target_cell,
                    "claim": int(report.claim), "accepted": accepted_here,
                    "is_malicious": is_malicious, "evidence_after": evidence,
                    "scenario_event_id": report.scenario_event_id,
                })
                if is_malicious:
                    log["malicious_report_deliveries"] += 1
                    log["malicious_reports_accepted"] += int(accepted_here)
                    if accepted_here and evidence is not None and abs(evidence) > 1e-12:
                        log["false_acceptance_count"] += 1

            apply_verification((robot,))
            route_affected = robot.reports_affect_remaining_route(accepted, malicious_ids, step)
            if robot.should_replan_for_path_state(step) or route_affected or robot.defense_replan_needed or verification_replan.get(robot.robot_id, False):
                reasons = []
                if route_affected:
                    reasons.append("peer_report_on_route")
                if robot.defense_replan_needed:
                    reasons.append("source_linked_trust_reweight")
                if verification_replan.get(robot.robot_id, False):
                    reasons.append("direct_verification")
                if robot.should_replan_for_path_state(step):
                    reasons.append("path_invalid_or_empty")
                robot.replan(step, "+".join(reasons))
                _log_replan(log, method, robot, step)
                robot.defense_replan_needed = False
                robot.source_linked_replan_context = None

        # Multi-robot traffic uses frozen intents before any robot moves.
        approved, traffic_events = coordinate_robot_intents(robots, world, step, traffic_state)
        log["traffic_events"].extend(traffic_events)
        recovered = {
            event["robot_id"]
            for event in traffic_events
            if event.get("event_type") == "traffic_deadlock_recovered"
        }
        occupied = {robot.position for robot in robots}
        for robot in robots:
            if robot.robot_id in recovered:
                robot.replan(step, "traffic_deadlock_recovered")
                _log_replan(log, method, robot, step)
        for robot in sorted(robots, key=lambda item: item.robot_id):
            before = robot.position
            others = [other for other in robots if other is not robot]
            other_cells = {other.position for other in others}
            other_reserved = set().union(*(other.reserved_cells() for other in others)) if others else set()
            if not approved.get(robot.robot_id, True):
                robot.traffic_wait_steps += 1
                action = "traffic_wait"
            else:
                action = robot.move(world, step, occupied - {before})
                if action == "move":
                    occupied.discard(before)
                    occupied.add(robot.position)
                    robot.consecutive_traffic_waits = 0
            robot.record_position()
            if action == "blocked_move" or (action == "task_transition" and not robot.completed):
                robot.replan(step, action)
                _log_replan(log, method, robot, step)
            elif action == "traffic_wait" and robot.consecutive_traffic_waits >= TRAFFIC_REROUTE_AFTER_WAITS:
                robot.replan(step, "traffic_wait_reroute", other_reserved | other_cells)
                _log_replan(log, method, robot, step)
            log["events"].append({
                "step": step, "kind": "robot_action", "method": method,
                "robot_id": robot.robot_id, "action": action, "phase": phase,
                "position": robot.position, "goal": robot.goal if not robot.completed else None,
            })

        for robot in robots:
            if step < route_eval_period or step % route_eval_period == 0 or step == max_steps - 1:
                latest_route_metrics[robot.robot_id] = _route_attacker_cost(
                    robot,
                    attacker,
                    step,
                    config.visualization.route_impact_min_cost_delta,
                )
            route_cost, route_affected = latest_route_metrics[robot.robot_id]
            active_fake_claims = sum(
                1
                for item in robot.fusion.report_history.values()
                if item.report.report_id in malicious_ids
            )
            sample = {
                "step": step, "phase": phase, "method": method, "robot_id": robot.robot_id,
                "position": robot.position, "goal": None if robot.completed else robot.goal,
                "deliveries_completed": robot.deliveries_completed,
                "mean_delivery_time_steps": (
                    sum(robot.delivery_durations) / len(robot.delivery_durations)
                    if robot.delivery_durations else None
                ),
                "benign_no_path_steps": robot.no_path_steps,
                "benign_traffic_wait_steps": robot.traffic_wait_steps,
                "benign_movement_steps": robot.movement_steps,
                "benign_total_distance": robot.total_distance,
                "benign_total_replans": robot.total_replans,
                "attacker_trust": robot.trust.score(attacker),
                "attacker_is_trusted": robot.trust.score(attacker) >= config.trust.threshold,
                "active_fake_claim_count": active_fake_claims,
                # Preserve the report schema used by the analysis plots.  A
                # malicious claim is influential only while it changes this
                # robot's preferred route; the count makes that relationship
                # explicit rather than treating every retained claim as
                # navigation-relevant.
                "influential_fake_claim_count": active_fake_claims if route_affected else 0,
                "attacker_route_cost_delta": route_cost,
                "route_affected_by_attacker": route_affected,
                "attacker_attributable_cost_on_route": route_cost,
                "preferred_route_affected_by_attacker": route_affected,
                "map_error": (
                    _map_error(robot, world, step)
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
    distrust = next((event["step"] for event in trust_events if event["sender_id"] == manifest.malicious_robot_id and event["new_trust"] < config.trust.threshold), None)
    delivery_after_attack = sum(robot.deliveries_completed for robot in benign)
    delivery_durations = [duration for robot in benign for duration in robot.delivery_durations]
    summary = {
        "method": method, "engine": "modular_native", "seed": config.seed,
        "steps_completed": config.total_steps, "attack_actions": len([report for report in log["reports"] if report["is_malicious"]]),
        "benign_total_deliveries_completed": sum(robot.deliveries_completed for robot in benign),
        "benign_delivery_time_mean_steps": (
            sum(delivery_durations) / len(delivery_durations) if delivery_durations else None
        ),
        "benign_success_rate": sum(robot.deliveries_completed for robot in benign) / max(1, len(benign) * config.deliveries_per_robot),
        "benign_deliveries_after_attack": delivery_after_attack,
        "benign_deliveries_after_distrust": delivery_after_attack if distrust is not None else 0,
        "benign_no_path_steps": sum(robot.no_path_steps for robot in benign),
        "benign_movement_steps": sum(robot.movement_steps for robot in benign),
        "benign_total_distance": sum(robot.total_distance for robot in benign),
        "benign_total_replans": sum(robot.total_replans for robot in benign),
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
        "malicious_verified_false_reports": sum(1 for event in trust_events if event["sender_id"] == manifest.malicious_robot_id and event["outcome"] == "contradicted_fresh"),
        "fresh_contradictions": sum(event["outcome"] == "contradicted_fresh" for event in trust_events),
        "final_attacker_trust_mean": sum(robot.trust.score(manifest.malicious_robot_id) for robot in benign) / max(1, len(benign)),
        "map_error_mean": sum(map_errors) / max(1, len(map_errors)),
        "map_error_final": sum(final_errors) / max(1, len(final_errors)),
        "false_acceptance_count": log["false_acceptance_count"],
        "false_acceptance_rate": log["false_acceptance_count"] / max(1, log["malicious_report_deliveries"]),
        "malicious_report_deliveries": log["malicious_report_deliveries"],
        "malicious_reports_accepted": log["malicious_reports_accepted"],
        "recovery_time_steps": recovery,
        "manifest_hash": manifest.map_hash, "map_hash": manifest.map_hash,
        "scenario_manifest_hash": scenario_manifest_hash(manifest),
    }
    return summary, collector


def replay_manifest(config: SimulationConfig, manifest: ScenarioManifest, method: str, output_directory: Path, *, show_animation: bool = False) -> SimpleNamespace:
    world, robots, log = run_manifest_rollout(config, manifest, method)
    if show_animation:
        print("Opening reconnaissance heatmap, then live belief-map windows...", flush=True)
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
            title=f"Reconnaissance traffic heatmap | seed {config.seed}",
        )
    elif log.get("live", {}).get("recon_heatmap") is not None:
        from .recon_authoring import save_traffic_heatmap_artifacts
        save_traffic_heatmap_artifacts(
            output_directory,
            np.asarray(log["live"]["recon_heatmap"], dtype=np.int32),
            title=f"Reconnaissance traffic heatmap | seed {config.seed}",
        )
    effective = config.to_dict() | {"effective_method": method, "engine": "modular_native"}
    CsvMetrics.config(output_directory / "effective_config.json", effective)
    return SimpleNamespace(output_directory=output_directory, method=method, summary=summary, world=world, robots=robots, log=log)
