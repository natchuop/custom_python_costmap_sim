"""Controlled end-to-end checks for the native modular experiment path."""
from dataclasses import replace

import numpy as np
import pytest

from map_poisoning.config import AttackConfig, FusionConfig, PhaseConfig, SimulationConfig, TrustConfig
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import AttackEvent, AttackType, ClaimReport, ClaimType, DeliveryTask, DirectObservation, TemporaryObstacleEpisode, VerificationOutcome
from map_poisoning.robot import ModularRobot
from map_poisoning.rollout import _map_error, collect_rollout_metrics, run_manifest_rollout
from map_poisoning.scenario import ScenarioManifest, author_manifest
from map_poisoning.trust import ScalarTrustModel
from map_poisoning.world import World


def _grid():
    grid = np.zeros((17, 17), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    return grid


def _manifest(attack_type=AttackType.FAKE_OBSTACLE, claim=ClaimType.BLOCKED, *, episode=(), step=1):
    grid = _grid()
    target = (7, 8)
    event = AttackEvent(
        "attack-0", step, attack_type, (target,), claim, step, 0, (1, 2), ("attack-report",),
        episode[0].episode_id if episode else None,
    )
    return ScenarioManifest(
        2, 1, {}, "test-map", tuple(grid.shape), tuple(tuple(int(v) for v in row) for row in grid),
        {"reconnaissance_end": 1, "attack_end": 5, "total": 8}, 0, (1, 2), tuple(episode), (event,),
        robot_starts={0: (1, 1), 1: (7, 1), 2: (7, 15)},
        task_queues={
            0: (DeliveryTask("a", (1, 2), (1, 3)),),
            1: (DeliveryTask("b", (7, 15), (15, 15)),),
            2: (DeliveryTask("c", (7, 14), (15, 1)),),
        },
    )


def _config(*, trust=(7, 3)):
    return SimulationConfig(
        seed=1,
        phases=PhaseConfig(1, 4, 3),
        attacks=AttackConfig(enabled=(AttackType.FAKE_OBSTACLE.value,), interval_min=1, interval_max=1),
        trust=TrustConfig(model="scalar", prior_alpha=trust[0], prior_beta=trust[1]),
        fusion=FusionConfig(method="source_memory"),
        deliveries_per_robot=1,
        max_steps=8,
    )


def test_map_error_counts_active_peer_disagreement():
    grid = np.zeros((5, 5), dtype=np.uint8)
    world = World(grid, ())
    trust = ScalarTrustModel(initial=0.80)
    fusion = FusionEngine("full_trust", trust.score)
    fusion.add(ClaimReport("peer", 0, (2, 2), ClaimType.BLOCKED, 0, 1.0))
    robot = type(
        "Robot",
        (),
        {"belief": RobotBeliefMap(grid), "fusion": fusion},
    )()

    assert _map_error(robot, world, 0) == 1.0


def test_trust_updates_once_per_sender_per_scan_without_cooldown():
    grid = np.zeros((8, 8), dtype=np.uint8)
    task = (DeliveryTask("t", (1, 2), (6, 6)),)
    trust = ScalarTrustModel(initial=0.80)
    robot = ModularRobot(
        1, (1, 1), task, RobotBeliefMap(grid), trust,
        FusionEngine("trust_threshold", trust.score, trust_threshold=0.50),
        0.50, "accept_all",
    )
    for rid, cell in (("free-a", (2, 2)), ("free-b", (2, 3))):
        robot.receive(ClaimReport(rid, 0, cell, ClaimType.FREE, 0, 1.0))
    robot.process_inbox(0)
    robot.verify([
        DirectObservation(1, (2, 2), ClaimType.FREE, 1, 1.0),
        DirectObservation(1, (2, 3), ClaimType.FREE, 1, 1.0),
    ], 1)
    assert len(robot.last_trust_batches) == 1
    after_first = trust.score(0)
    assert 0.80 < after_first < 0.83

    robot.receive(ClaimReport("free-c", 0, (2, 4), ClaimType.FREE, 2, 1.0))
    robot.process_inbox(2)
    robot.verify([DirectObservation(1, (2, 4), ClaimType.FREE, 3, 1.0)], 3)
    assert len(robot.last_trust_batches) == 1
    assert trust.score(0) > after_first

def test_aged_report_can_be_contradicted_with_age_weighted_evidence():
    grid = np.zeros((8, 8), dtype=np.uint8)
    task = (DeliveryTask("t", (1, 2), (6, 6)),)
    trust = ScalarTrustModel(initial=0.80)
    robot = ModularRobot(
        1, (1, 1), task, RobotBeliefMap(grid, memory_steps=300), trust,
        FusionEngine("trust_threshold", trust.score, trust_threshold=0.50, max_claim_age=300),
        0.50, "accept_all",
    )
    report = ClaimReport("aged", 0, (2, 2), ClaimType.BLOCKED, 0, 1.0)
    robot.receive(report)
    robot.process_inbox(0)
    results = robot.verify([DirectObservation(1, (2, 2), ClaimType.FREE, 20, 1.0)], 20)
    assert results[0][1] == VerificationOutcome.CONTRADICTED_FRESH
    assert trust.score(0) < 0.80
    assert (2, 2) not in robot.fusion.claims

def _received(log, report_id="attack-report"):
    return [event for event in log["events"] if event.get("kind") == "report_received" and event.get("report_id") == report_id]


def test_peer_reports_are_delivered_and_fused_per_recipient():
    manifest = _manifest()
    _, robots, log = run_manifest_rollout(_config(), manifest, "full_trust")
    assert log["report_count_total"] > log["malicious_report_count_total"]
    assert robots[2].accepted_reports > 0
    assert any(item.report.sender_id == 1 for item in robots[2].fusion.report_history.values())
    assert not any(item.report.sender_id == 1 for item in robots[1].fusion.report_history.values())

def test_high_trust_fake_obstacle_changes_route_and_low_trust_does_not():
    manifest = _manifest()
    _, _, high_log = run_manifest_rollout(_config(trust=(7, 3)), manifest, "source_memory")
    _, _, low_log = run_manifest_rollout(_config(trust=(1, 9)), manifest, "source_memory")
    high_replan = next(
        event
        for event in high_log["events"]
        if event.get("kind") == "replan"
        and event["robot_id"] == 1
        and event["step"] == 1
        and "peer_report_on_route" in str(event.get("reason", ""))
    )
    low_peer_replans = [
        event for event in low_log["events"]
        if event.get("kind") == "replan"
        and event["robot_id"] == 1
        and event["step"] == 1
        and "peer_report_on_route" in str(event.get("reason", ""))
    ]
    assert "peer_report_on_route" in high_replan["reason"]
    assert "malicious_report_on_route" in high_replan["reason"]
    assert "fake_obstacle_report_on_route" in high_replan["reason"]
    assert not low_peer_replans
    high_sample = next(
        row for row in high_log["timeseries"] if row["robot_id"] == 1 and row["step"] == 1
    )
    low_sample = next(
        row for row in low_log["timeseries"] if row["robot_id"] == 1 and row["step"] == 1
    )
    assert high_sample["attacker_route_cost_delta"] > 0
    assert high_sample["attacker_route_cost_delta"] > low_sample["attacker_route_cost_delta"]
    assert not low_sample["route_affected_by_attacker"]
    impact = next(event for event in high_log["events"] if event.get("kind") == "attack_route_impact" and event.get("recipient_id") == 1)
    assert impact["scenario_event_id"] == "attack-0"
    assert impact["attack_route_penalty"] >= 0
    assert "attack_extra_path_length" in impact
    assert "attack_induced_path_change" in impact


def test_loaded_leg_and_full_delivery_cycle_durations_are_separate():
    grid = np.zeros((8, 8), dtype=np.uint8)
    tasks = (
        DeliveryTask("one", (1, 1), (1, 2)),
        DeliveryTask("two", (1, 2), (1, 3)),
    )
    trust = ScalarTrustModel(initial=0.80)
    robot = ModularRobot(
        1, (1, 1), tasks, RobotBeliefMap(grid), trust,
        FusionEngine("full_trust", trust.score), 0.50, "accept_all",
    )
    robot.move(None, 0, set())  # pick up task one
    robot.position = (1, 2)
    robot.move(None, 5, set())  # deliver task one; task two activates
    robot.move(None, 6, set())  # pick up task two
    robot.position = (1, 3)
    robot.move(None, 10, set())
    assert robot.delivery_durations == [5, 4]
    assert robot.delivery_cycle_durations == [5, 5]


def test_each_attack_has_expected_fusion_and_verification_behavior():
    # Fake obstacle and stale reassertion are blocked claims against a free cell.
    for kind in (AttackType.FAKE_OBSTACLE, AttackType.STALE_REASSERTION):
        cleared = (TemporaryObstacleEpisode("cleared", ((7, 8),), 0, 1),) if kind == AttackType.STALE_REASSERTION else ()
        manifest = _manifest(kind, ClaimType.BLOCKED, episode=cleared)
        _, _, log = run_manifest_rollout(_config(), manifest, "source_memory")
        deliveries = _received(log)
        assert len(deliveries) == 2 and all(event["accepted"] and event["evidence_after"] > 0 for event in deliveries)
        victim_replan = next(
            event
            for event in log["events"]
            if event.get("kind") == "replan"
            and event["robot_id"] == 1
            and event["step"] == 1
            and "peer_report_on_route" in str(event.get("reason", ""))
        )
        assert "peer_report_on_route" in victim_replan["reason"]
        assert any(
            event.get("kind") == "fusion_effect"
            and event.get("report_id") == "attack-report"
            and event.get("outcome") == "contradicted_fresh"
            for event in log["events"]
        )

    # False clearance claims FREE while the physical temporary blockage exists.
    episode = TemporaryObstacleEpisode("temp", ((7, 8),), 0, 5)
    manifest = _manifest(AttackType.FALSE_CLEARANCE, ClaimType.FREE, episode=(episode,))
    _, _, log = run_manifest_rollout(_config(), manifest, "source_memory")
    deliveries = _received(log)
    assert len(deliveries) == 2 and all(event["accepted"] and event["evidence_after"] < 0 for event in deliveries)
    # A FREE report that does not materially change the current route cost no
    # longer forces a redundant A* run. It is still fused and later validated.
    assert not any(
        event.get("kind") == "replan"
        and event.get("robot_id") == 1
        and event.get("step") == 1
        and "peer_report_on_route" in str(event.get("reason", ""))
        for event in log["events"]
    )
    assert any(event.get("kind") == "fusion_effect" and event.get("report_id") == "attack-report" and event.get("outcome") == "contradicted_fresh" for event in log["events"])


def test_active_fake_claim_metric_excludes_retracted_claims():
    manifest = _manifest()
    config = _config()
    _, _, log = run_manifest_rollout(config, manifest, "source_memory")
    final = next(
        row
        for row in reversed(log["timeseries"])
        if row["robot_id"] == 1
    )
    assert final["active_fake_claim_count"] == 0


def test_baselines_and_requested_metrics_run_on_one_manifest():
    manifest = _manifest()
    for method in ("majority_vote", "full_trust", "trust_fused", "source_memory"):
        world, robots, log = run_manifest_rollout(_config(), manifest, method)
        summary, _ = collect_rollout_metrics(_config(), manifest, method, world, robots, log)
        assert summary["engine"] == "modular_native"
        assert summary["malicious_report_deliveries"] == 2
        assert summary["false_acceptance_count"] == 2
        assert 0.0 <= summary["map_error_mean"] <= 1.0
        assert "benign_delivery_cycle_duration_mean_steps" in summary
        assert "benign_delivery_cycle_duration_median_steps" in summary
        assert "benign_delivery_cycle_duration_p95_steps" in summary
        assert "attack_route_penalty_mean" in summary
        assert "attack_induced_path_changes" in summary
        assert "steps_route_affected_by_attacker" in summary
        assert 0.0 <= summary["map_error_final"] <= 1.0
        assert "recovery_time_steps" in summary
        if method == "source_memory":
            assert summary["malicious_verified_false_reports"] > 0


def test_delivery_time_and_no_path_metrics_are_counted_from_robot_state():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    manifest = ScenarioManifest(
        2, 2, {}, "delivery-map", tuple(grid.shape), tuple(tuple(int(v) for v in row) for row in grid),
        {"reconnaissance_end": 1, "attack_end": 2, "total": 6}, 0, (1, 2), (), (),
        robot_starts={0: (1, 1), 1: (3, 1), 2: (5, 1)},
        task_queues={
            0: (DeliveryTask("a", (1, 1), (1, 3)),),
            1: (DeliveryTask("b", (3, 1), (3, 3)),),
            2: (DeliveryTask("c", (5, 1), (5, 3)),),
        },
    )
    config = replace(_config(), seed=2, phases=PhaseConfig(1, 1, 4), max_steps=6)
    world, robots, log = run_manifest_rollout(config, manifest, "full_trust")
    summary, _ = collect_rollout_metrics(config, manifest, "full_trust", world, robots, log)
    assert summary["benign_total_deliveries_completed"] == 2
    assert summary["benign_delivery_time_mean_steps"] == 3.0
    assert summary["benign_no_path_steps"] == 0
    assert any(
        event.get("kind") == "replan" and "task_transition" in str(event.get("reason", ""))
        for event in log["events"]
    )

    disconnected = grid.copy()
    disconnected[1:-1, 3] = 1
    no_path_manifest = ScenarioManifest(
        2, 3, {}, "no-path-map", tuple(disconnected.shape), tuple(tuple(int(v) for v in row) for row in disconnected),
        {"reconnaissance_end": 1, "attack_end": 2, "total": 4}, 0, (1, 2), (), (),
        robot_starts={0: (1, 1), 1: (3, 1), 2: (5, 1)},
        task_queues={
            0: (DeliveryTask("a", (1, 1), (1, 2)),),
            1: (DeliveryTask("b", (3, 5), (3, 5)),),
            2: (DeliveryTask("c", (5, 2), (5, 1)),),
        },
    )
    no_path_config = replace(config, seed=3, phases=PhaseConfig(1, 1, 2), max_steps=4)
    world, robots, log = run_manifest_rollout(no_path_config, no_path_manifest, "full_trust")
    summary, _ = collect_rollout_metrics(no_path_config, no_path_manifest, "full_trust", world, robots, log)
    assert summary["benign_no_path_steps"] > 0


def test_robot_contention_is_a_traffic_wait_not_a_blocked_world_move():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    manifest = ScenarioManifest(
        2, 4, {}, "traffic-map", tuple(grid.shape), tuple(tuple(int(v) for v in row) for row in grid),
        {"reconnaissance_end": 1, "attack_end": 2, "total": 3}, 0, (1, 2), (), (),
        robot_starts={0: (3, 1), 1: (3, 3), 2: (5, 1)},
        task_queues={
            0: (DeliveryTask("a", (3, 2), (3, 2)),),
            1: (DeliveryTask("b", (3, 2), (3, 2)),),
            2: (DeliveryTask("c", (5, 2), (5, 3)),),
        },
    )
    config = replace(_config(), seed=4, phases=PhaseConfig(1, 1, 1), max_steps=3)
    world, robots, log = run_manifest_rollout(config, manifest, "full_trust")
    summary, _ = collect_rollout_metrics(config, manifest, "full_trust", world, robots, log)
    assert summary["benign_traffic_wait_steps"] > 0
    assert summary["benign_blocked_moves"] == 0


def test_temporary_obstacle_route_effect_is_labeled_and_counted():
    grid = np.zeros((9, 12), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    episode = TemporaryObstacleEpisode("physical", ((4, 5),), 1, 5)
    manifest = ScenarioManifest(
        2, 5, {}, "physical-map", tuple(grid.shape), tuple(tuple(int(v) for v in row) for row in grid),
        {"reconnaissance_end": 1, "attack_end": 4, "total": 5}, 0, (1, 2), (episode,), (),
        robot_starts={0: (1, 1), 1: (4, 2), 2: (7, 1)},
        task_queues={
            0: (DeliveryTask("a", (1, 2), (1, 3)),),
            1: (DeliveryTask("b", (4, 9), (4, 9)),),
            2: (DeliveryTask("c", (7, 2), (7, 3)),),
        },
    )
    config = replace(_config(), seed=5, phases=PhaseConfig(1, 3, 1), max_steps=5)
    world, robots, log = run_manifest_rollout(config, manifest, "full_trust")
    summary, _ = collect_rollout_metrics(config, manifest, "full_trust", world, robots, log)
    physical = [
        event for event in log["events"]
        if event.get("kind") == "replan"
        and event.get("robot_id") == 1
        and "temporary_physical_obstacle_on_route" in str(event.get("reason", ""))
    ]
    assert physical and physical[0]["changed"]
    assert summary["temporary_obstacle_replan_checks"] >= 1
    assert summary["temporary_obstacle_path_changes"] >= 1


def test_recovery_is_none_when_the_attacker_never_changes_a_route():
    manifest = _manifest()
    world, robots, log = run_manifest_rollout(_config(trust=(1, 9)), manifest, "source_memory")
    summary, _ = collect_rollout_metrics(_config(trust=(1, 9)), manifest, "source_memory", world, robots, log)
    assert not any(
        row["route_affected_by_attacker"]
        for row in log["timeseries"]
        if row["robot_id"] in manifest.benign_robot_ids
    )
    assert summary["recovery_time_steps"] is None


def test_fresh_lidar_keeps_a_seen_free_cell_on_the_route():
    manifest = replace(
        _manifest(),
        robot_starts={0: (1, 1), 1: (7, 6), 2: (7, 15)},
        task_queues={
            0: (DeliveryTask("a", (1, 2), (1, 3)),),
            1: (DeliveryTask("b", (7, 15), (7, 15)),),
            2: (DeliveryTask("c", (7, 14), (15, 1)),),
        },
    )
    _, _, log = run_manifest_rollout(_config(trust=(7, 3)), manifest, "source_memory")
    first = next(event for event in log["events"] if event.get("kind") == "replan" and event["robot_id"] == 1)
    assert (7, 8) in first["path"]
    sample = next(row for row in log["timeseries"] if row["robot_id"] == 1 and row["step"] == 1)
    assert not sample["route_affected_by_attacker"]


def test_manifest_author_includes_all_requested_attack_types_for_supported_maps():
    config = replace(
        _config(),
        phases=PhaseConfig(200, 500, 50),
        attacks=AttackConfig(
            enabled=tuple(kind.value for kind in AttackType), interval_min=50, interval_max=50,
        ),
        max_steps=None,
    )
    manifest = author_manifest(config, _grid())
    assert {event.attack_type for event in manifest.attack_events} == set(AttackType)


def test_trust_threshold_drops_when_a_fake_obstacle_is_observed():
    manifest = replace(_manifest(), robot_starts={0: (1, 1), 1: (7, 7), 2: (7, 15)})
    _, _, log = run_manifest_rollout(_config(), manifest, "trust_threshold")
    updates = [
        event
        for event in log["events"]
        if event.get("kind") == "trust_update"
        and event.get("recipient_id") == 1
        and event.get("sender_id") == 0
    ]
    assert updates
    assert any(
        float(event.get("new_trust", 0.0)) < float(event.get("old_trust", 0.0))
        for event in updates
    )
    assert any(
        event.get("kind") == "fusion_effect"
        and event.get("recipient_id") == 1
        and event.get("sender_id") == 0
        and event.get("outcome") == "contradicted_fresh"
        for event in log["events"]
    )
