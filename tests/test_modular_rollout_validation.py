"""Controlled end-to-end checks for the native modular experiment path."""
from dataclasses import replace

import numpy as np

from map_poisoning.config import AttackConfig, FusionConfig, PhaseConfig, SimulationConfig, TrustConfig
from map_poisoning.models import AttackEvent, AttackType, ClaimType, DeliveryTask, TemporaryObstacleEpisode
from map_poisoning.rollout import collect_rollout_metrics, run_manifest_rollout
from map_poisoning.scenario import ScenarioManifest, author_manifest


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
        fusion=FusionConfig(method="source_linked"),
        deliveries_per_robot=1,
        max_steps=8,
    )


def _received(log, report_id="attack-report"):
    return [event for event in log["events"] if event.get("kind") == "report_received" and event.get("report_id") == report_id]


def test_peer_reports_are_delivered_and_fused_per_recipient():
    manifest = _manifest()
    _, robots, log = run_manifest_rollout(_config(), manifest, "full_trust")
    peer_deliveries = [
        event for event in log["events"]
        if event.get("kind") == "report_received" and event["sender_id"] == 1 and event["recipient_id"] == 2
    ]
    assert peer_deliveries and all(event["accepted"] for event in peer_deliveries)
    delivered = peer_deliveries[0]
    stored = robots[2].fusion.report_history[delivered["report_id"]]
    assert stored.report.sender_id == 1
    assert stored.report.target_cell == delivered["target_cell"]
    assert robots[2].fusion.evidence(stored.report.target_cell, 7) < 0
    assert not any(item.report.sender_id == 1 for item in robots[1].fusion.report_history.values())


def test_high_trust_fake_obstacle_changes_route_and_low_trust_does_not():
    manifest = _manifest()
    _, _, high_log = run_manifest_rollout(_config(trust=(7, 3)), manifest, "source_linked")
    _, _, low_log = run_manifest_rollout(_config(trust=(1, 9)), manifest, "source_linked")
    high_replan = next(event for event in high_log["events"] if event.get("kind") == "replan" and event["robot_id"] == 1 and event["step"] == 1)
    low_replan = next(event for event in low_log["events"] if event.get("kind") == "replan" and event["robot_id"] == 1 and event["step"] == 1)
    assert (7, 8) not in high_replan["path"]
    assert (7, 8) in low_replan["path"]
    high_sample = next(
        row for row in high_log["timeseries"] if row["robot_id"] == 1 and row["step"] == 1
    )
    low_sample = next(
        row for row in low_log["timeseries"] if row["robot_id"] == 1 and row["step"] == 1
    )
    assert high_sample["route_affected_by_attacker"]
    assert high_sample["attacker_route_cost_delta"] > 0
    assert not low_sample["route_affected_by_attacker"]
    assert 0 < low_sample["attacker_route_cost_delta"] < high_sample["attacker_route_cost_delta"]


def test_each_attack_has_expected_fusion_and_verification_behavior():
    # Fake obstacle and stale reassertion are blocked claims against a free cell.
    for kind in (AttackType.FAKE_OBSTACLE, AttackType.STALE_REASSERTION):
        cleared = (TemporaryObstacleEpisode("cleared", ((7, 8),), 0, 1),) if kind == AttackType.STALE_REASSERTION else ()
        manifest = _manifest(kind, ClaimType.BLOCKED, episode=cleared)
        _, _, log = run_manifest_rollout(_config(), manifest, "source_linked")
        deliveries = _received(log)
        assert len(deliveries) == 2 and all(event["accepted"] and event["evidence_after"] > 0 for event in deliveries)
        victim_replan = next(event for event in log["events"] if event.get("kind") == "replan" and event["robot_id"] == 1 and event["step"] == 1)
        assert (7, 8) not in victim_replan["path"]
        assert any(event["kind"] == "trust_update" and event["report_id"] == "attack-report" and event["outcome"] == "contradicted_fresh" for event in log["events"])

    # False clearance claims FREE while the physical temporary blockage exists.
    episode = TemporaryObstacleEpisode("temp", ((7, 8),), 0, 5)
    manifest = _manifest(AttackType.FALSE_CLEARANCE, ClaimType.FREE, episode=(episode,))
    _, _, log = run_manifest_rollout(_config(), manifest, "source_linked")
    deliveries = _received(log)
    assert len(deliveries) == 2 and all(event["accepted"] and event["evidence_after"] < 0 for event in deliveries)
    victim_replan = next(event for event in log["events"] if event.get("kind") == "replan" and event["robot_id"] == 1 and event["step"] == 1)
    assert (7, 8) in victim_replan["path"]
    assert any(event["kind"] == "trust_update" and event["report_id"] == "attack-report" and event["outcome"] == "contradicted_fresh" for event in log["events"])


def test_baselines_and_requested_metrics_run_on_one_manifest():
    manifest = _manifest()
    for method in ("full_trust", "majority_vote", "trust_fused", "source_linked"):
        world, robots, log = run_manifest_rollout(_config(), manifest, method)
        summary, _ = collect_rollout_metrics(_config(), manifest, method, world, robots, log)
        assert summary["engine"] == "modular_native"
        assert summary["malicious_report_deliveries"] == 2
        assert summary["false_acceptance_count"] == 2
        assert 0.0 <= summary["map_error_mean"] <= 1.0
        assert 0.0 <= summary["map_error_final"] <= 1.0
        assert summary["recovery_time_steps"] is not None


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


def test_recovery_is_none_when_the_attacker_never_changes_a_route():
    manifest = _manifest()
    world, robots, log = run_manifest_rollout(_config(trust=(1, 9)), manifest, "source_linked")
    summary, _ = collect_rollout_metrics(_config(trust=(1, 9)), manifest, "source_linked", world, robots, log)
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
    _, _, log = run_manifest_rollout(_config(trust=(7, 3)), manifest, "source_linked")
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
