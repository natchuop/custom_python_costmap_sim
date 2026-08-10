"""Shared manifest rollout for modular simulation replay."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sim2

from .config import SimulationConfig
from .metrics import CsvMetrics
from .scenario import ScenarioManifest, scenario_manifest_hash


def _defense_config_dict(config: SimulationConfig) -> dict:
    return {
        "decay_rate": config.fusion.decay_rate,
        "cost_scale": config.fusion.cost_scale,
        "cost_exponent": config.fusion.cost_exponent,
        "blocked_probability_threshold": config.fusion.blocked_probability_threshold,
        "max_claim_age": config.fusion.max_claim_age,
        "congested_impact": config.fusion.congested_impact,
        "duplicate_window_steps": config.fusion.duplicate_window_steps,
        "trust_threshold": config.trust.threshold,
    }


def _lock_legacy_globals(config: SimulationConfig):
    old_phase = (
        sim2.MIN_RECON_STEPS,
        sim2.MAX_RECON_STEPS,
        sim2.ATTACK_BURST_DURATION_STEPS,
    )
    old_trust = (
        sim2.TRUST_MODEL_NAME,
        sim2.TRUST_INITIAL_VALUE,
        sim2.TRUST_ACCEPT_THRESHOLD,
        sim2.TRUST_REWARD,
        sim2.TRUST_PENALTY,
        sim2.TRUST_BAYES_PRIOR_ALPHA,
        sim2.TRUST_BAYES_PRIOR_BETA,
    )
    sim2.MIN_RECON_STEPS = config.phases.recon_steps
    sim2.MAX_RECON_STEPS = config.phases.recon_steps
    sim2.ATTACK_BURST_DURATION_STEPS = config.phases.attack_steps
    sim2.TRUST_MODEL_NAME = config.trust.model
    sim2.TRUST_INITIAL_VALUE = config.trust.prior_alpha / (
        config.trust.prior_alpha + config.trust.prior_beta
    )
    sim2.TRUST_ACCEPT_THRESHOLD = config.trust.threshold
    sim2.TRUST_REWARD = 0.02
    sim2.TRUST_PENALTY = 0.06
    sim2.TRUST_BAYES_PRIOR_ALPHA = config.trust.prior_alpha
    sim2.TRUST_BAYES_PRIOR_BETA = config.trust.prior_beta
    return old_phase, old_trust


def _restore_legacy_globals(old_phase, old_trust):
    sim2.MIN_RECON_STEPS, sim2.MAX_RECON_STEPS, sim2.ATTACK_BURST_DURATION_STEPS = old_phase
    (
        sim2.TRUST_MODEL_NAME,
        sim2.TRUST_INITIAL_VALUE,
        sim2.TRUST_ACCEPT_THRESHOLD,
        sim2.TRUST_REWARD,
        sim2.TRUST_PENALTY,
        sim2.TRUST_BAYES_PRIOR_ALPHA,
        sim2.TRUST_BAYES_PRIOR_BETA,
    ) = old_trust


def _manifest_task_queues(manifest: ScenarioManifest) -> dict | None:
    if not manifest.task_queues:
        return None
    return {
        robot_id: [
            sim2.DeliveryTask(pickup=tuple(task.pickup), dropoff=tuple(task.dropoff))
            for task in tasks
        ]
        for robot_id, tasks in manifest.task_queues.items()
    }


def run_manifest_rollout(
    config: SimulationConfig,
    manifest: ScenarioManifest,
    method: str,
) -> tuple[sim2.GridWorld, list, dict]:
    """Run the validated continuous-motion loop on a fixed manifest."""
    if manifest.malicious_robot_id != 0 or tuple(manifest.benign_robot_ids) != (1, 2):
        raise ValueError(
            "Invalid team roles: Robot 0 must be malicious and Robots 1 and 2 must be benign."
        )
    max_steps = config.max_steps or config.phases.total_steps
    static_grid = np.asarray(manifest.static_grid, dtype=np.uint8)
    obstacle_episodes = manifest.obstacle_episodes if manifest.obstacle_episodes else None
    old_phase, old_trust = _lock_legacy_globals(config)
    try:
        world, robots, log = sim2.run_simulation(
            grid=static_grid,
            prior_grid=static_grid,
            defense_method=method,
            defense_config=_defense_config_dict(config),
            tasks_per_robot=config.deliveries_per_robot,
            max_steps=max_steps,
            random_seed=config.seed,
            experiment_mode="attack",
            attack_events=manifest.attack_events,
            obstacle_episodes=obstacle_episodes,
            manifest_robot_starts=manifest.robot_starts,
            manifest_task_queues=_manifest_task_queues(manifest),
            manifest_malicious_robot_id=manifest.malicious_robot_id,
            map_view=config.visualization.map_view,
        )
    finally:
        _restore_legacy_globals(old_phase, old_trust)
    return world, robots, log


def collect_rollout_metrics(
    config: SimulationConfig,
    manifest: ScenarioManifest,
    method: str,
    world,
    robots,
    log: dict,
) -> tuple[dict, CsvMetrics]:
    calculated = sim2.compute_experiment_metrics(robots, log)
    malicious = log["malicious_robot_id"]
    benign = [robot for robot in robots if not robot.is_malicious]
    collector = CsvMetrics()

    for report in log["reports"]:
        collector.event(
            report["step"],
            "report_sent",
            method=method,
            **{key: value for key, value in report.items() if key != "step"},
        )
    for event in log.get("trust_events", []):
        collector.event(
            event["step"],
            "trust_update",
            method=method,
            **{key: value for key, value in event.items() if key != "step"},
        )
    for event in log.get("traffic_events", []):
        collector.event(event.get("step", 0), "traffic", method=method, **{key: value for key, value in event.items() if key != "step"})

    for robot in robots:
        rid = robot.robot_id
        rlog = log["robots"][rid]
        for step, event in enumerate(rlog["events"]):
            collector.event(step, "robot_action", method=method, robot_id=rid, action=event)
        for replan in robot.replan_events:
            collector.event(
                replan["step"],
                "replan",
                method=method,
                robot_id=rid,
                reason=replan["reason"],
                next_five_changed=replan["next_five_changed"],
                old_path_length=replan["old_path_length"],
                new_path_length=replan["new_path_length"],
            )
        for step in range(0, len(rlog["position"]), config.logging.timeseries_period_steps):
            trust_snapshot = rlog["trust"][step]
            trust_value = trust_snapshot.get(malicious) if isinstance(trust_snapshot, dict) else None
            if isinstance(trust_value, dict):
                trust_value = trust_value.get("score")
            collector.sample(
                step=step,
                phase=log["phase"][step],
                method=method,
                robot_id=rid,
                position=rlog["position"][step],
                goal=rlog["current_goal"][step],
                deliveries_completed=rlog["completed_tasks"][step],
                benign_no_path_steps=sum(event == "no_path" for event in rlog["events"][: step + 1]),
                benign_movement_steps=sum(
                    event in ("moved_cell", "moved_continuous") for event in rlog["events"][: step + 1]
                ),
                benign_total_distance=sim2.compute_path_distance(rlog["position"][: step + 1]),
                benign_total_replans=rlog["replan_count"][step],
                attacker_trust=trust_value,
                attacker_is_trusted=(trust_value is not None and float(trust_value) >= config.trust.threshold),
                trust_threshold=config.trust.threshold,
                malicious_claim_cells_on_route=rlog["malicious_claim_cells_on_route"][step],
                traffic_waits=rlog.get("traffic_waits", [0])[step],
                traffic_deadlock_active=rlog.get("traffic_deadlock_active", [False])[step],
                active_deadlock_id=rlog.get("active_deadlock_id", [None])[step],
                traffic_mode=rlog.get("traffic_mode", ["NORMAL"])[step],
                traffic_replans=rlog.get("traffic_replans", [0])[step],
            )

    contradicted = sum(
        1
        for event in log.get("trust_events", [])
        if event.get("truth_matches") is False
    )

    summary = {
        "method": method,
        "engine": "modular",
        "seed": config.seed,
        "steps_completed": len(log["truth_grid"]),
        "attack_actions": sum(bool(report.get("is_malicious")) for report in log["reports"]),
        "benign_total_deliveries_completed": calculated["benign_total_completed_deliveries"],
        "benign_success_rate": calculated["benign_delivery_success_rate"],
        "benign_deliveries_after_attack": calculated["benign_deliveries_after_attack"],
        "benign_deliveries_after_distrust": calculated["benign_deliveries_after_distrust"],
        "benign_no_path_steps": sum(calculated["no_path_count_per_robot"][r.robot_id] for r in benign),
        "benign_movement_steps": calculated["benign_total_movement_steps"],
        "benign_total_distance": calculated["benign_total_grid_distance"],
        "benign_total_replans": calculated["benign_total_replans"],
        "benign_productive_replans": calculated["benign_next_five_changed_replans"],
        "benign_malicious_report_replans": sum(
            "malicious_report" in event["reason"]
            for robot in benign
            for event in robot.replan_events
        ),
        "benign_malicious_route_replans": sum(
            "malicious_report_on_route" in event["reason"]
            for robot in benign
            for event in robot.replan_events
        ),
        "benign_blocked_moves": sum(calculated["blocked_moves_per_robot"][r.robot_id] for r in benign),
        "time_to_distrust_malicious_robot": calculated["time_to_distrust_malicious_robot"],
        "malicious_verified_false_reports": calculated["malicious_verified_false_reports"],
        "fresh_contradictions": contradicted,
        "final_attacker_trust_mean": (
            sum(robot.trust_for(malicious) for robot in benign) / len(benign) if benign else 0.0
        ),
        "manifest_hash": manifest.map_hash,
        "scenario_manifest_hash": scenario_manifest_hash(manifest),
        "benign_traffic_wait_steps": calculated.get("benign_traffic_wait_steps", 0),
        "traffic_replans": calculated.get("traffic_replans", 0),
        "vertex_conflicts_detected": calculated.get("vertex_conflicts_detected", 0),
        "head_on_swap_conflicts_detected": calculated.get("head_on_swap_conflicts_detected", 0),
        "reservation_conflicts_detected": calculated.get("reservation_conflicts_detected", 0),
        "traffic_yield_events": calculated.get("traffic_yield_events", 0),
        "deadlocks_detected": calculated.get("deadlocks_detected", 0),
        "deadlocks_recovered": calculated.get("deadlocks_recovered", 0),
        "robot_overlap_violations": calculated.get("robot_overlap_violations", 0),
        "recovery_start_step": calculated.get("recovery_start_step"),
        "recovery_metrics_per_robot": calculated.get("recovery_metrics_per_robot", {}),
        "recovery_trust_gain_r1": calculated.get("recovery_metrics_per_robot", {}).get(1, {}).get("recovery_trust_gain"),
        "recovery_trust_gain_r2": calculated.get("recovery_metrics_per_robot", {}).get(2, {}).get("recovery_trust_gain"),
        "retrust_latency_r1": calculated.get("recovery_metrics_per_robot", {}).get(1, {}).get("retrust_latency_steps"),
        "retrust_latency_r2": calculated.get("recovery_metrics_per_robot", {}).get(2, {}).get("retrust_latency_steps"),
    }
    return summary, collector


def replay_manifest(
    config: SimulationConfig,
    manifest: ScenarioManifest,
    method: str,
    output_directory: Path,
    *,
    show_animation: bool = False,
) -> SimpleNamespace:
    world, robots, log = run_manifest_rollout(config, manifest, method)
    if show_animation:
        sim2.show_recon_heatmap(world, log)
        sim2.animate(world, robots, log, map_view=config.visualization.map_view)
    summary, collector = collect_rollout_metrics(
        config, manifest, method, world, robots, log
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    collector.write(output_directory, summary)
    from .reporting import generate_run_report
    generate_run_report(output_directory, formats=(config.logging.plot_format,) if config.logging.generate_plots else ())
    effective = config.to_dict() | {
        "effective_method": method,
        "defense_config": _defense_config_dict(config),
    }
    CsvMetrics.config(output_directory / "effective_config.json", effective)
    return SimpleNamespace(
        output_directory=output_directory,
        method=method,
        summary=summary,
        world=world,
        robots=robots,
        log=log,
    )
