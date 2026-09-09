from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from map_poisoning.audit import audit_manifest
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.config import AttackConfig, FusionConfig, PhaseConfig, SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.map_io import load_movingai, packaged_movingai_map_path
from map_poisoning.models import ClaimReport, ClaimType, DeliveryTask, DirectObservation
from map_poisoning.obstacles import anchors_remain_connected, footprint_finite_detour_score
from map_poisoning.recon_authoring import author_warehouse_manifest, run_clean_reference_rollout
from map_poisoning.robot import ModularRobot
from map_poisoning.scenario import ScenarioManifest, _hash, author_manifest, scenario_manifest_hash
from map_poisoning.trust import BayesianTrustModel


def _open_grid(size=18):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    return grid


def _manifest_for_reference(grid, phases):
    static = tuple(tuple(int(value) for value in row) for row in grid)
    return ScenarioManifest(
        3, 1, {}, _hash(grid), tuple(grid.shape), static,
        {
            "reconnaissance_end": phases.recon_steps,
            "attack_end": phases.recon_steps + phases.attack_steps,
            "total": phases.total_steps,
        },
        0, (1, 2), (), (),
        robot_starts={0: (2, 2), 1: (2, 3), 2: (2, 4)},
        task_queues={
            rid: (DeliveryTask(f"t{rid}", (3, 3), (4, 4)),)
            for rid in (0, 1, 2)
        },
    )


def _verification_batch(cell_count, *, report_confidence=1.0, observation_confidence=1.0):
    grid = _open_grid(24)
    trust = BayesianTrustModel()
    robot = ModularRobot(
        1,
        (1, 1),
        (DeliveryTask("task", (2, 2), (20, 20)),),
        RobotBeliefMap(grid),
        trust,
        FusionEngine("full_trust", trust.score, max_claim_age=300),
        0.5,
        "accept_all",
    )
    reports = []
    observations = []
    for index in range(cell_count):
        cell = (3 + index // 10, 3 + index % 10)
        report = ClaimReport(
            f"r{index}", 0, cell, ClaimType.BLOCKED, 0,
            sensor_confidence=report_confidence,
        )
        reports.append(report)
        observations.append(
            DirectObservation(1, cell, ClaimType.FREE, 0, observation_confidence)
        )
        robot.receive(report)
    robot.process_inbox(0)
    before = trust.score(0)
    robot.verify(observations, 0)
    return before, trust.score(0), robot.last_trust_batches[-1], trust


def test_default_phase_configuration_is_500_2000_500():
    phases = PhaseConfig()
    assert (phases.recon_steps, phases.attack_steps, phases.recovery_steps) == (500, 2000, 500)
    assert phases.total_steps == 3000
    assert SimulationConfig(max_steps=125).total_steps == 125


def test_clean_reference_rollout_is_structurally_recon_only(monkeypatch):
    phases = PhaseConfig(40, 80, 20)
    config = SimulationConfig(phases=phases, max_steps=140)
    manifest = _manifest_for_reference(_open_grid(), phases)
    seen = []

    def fake_rollout(reference_config, manifest_arg, method, **kwargs):
        seen.append((reference_config.max_steps, reference_config.total_steps, method, kwargs.get("capture_reference_state")))
        return None, [], {"events": [], "timeseries": [], "reference_states": {}}

    monkeypatch.setattr("map_poisoning.recon_authoring.run_manifest_rollout", fake_rollout)
    heatmap, _, _ = run_clean_reference_rollout(config, manifest)
    assert seen == [(phases.recon_steps, phases.recon_steps, "full_trust", True)]
    assert int(heatmap.sum()) == 0


def test_default_warehouse_recon_metadata_never_references_post_recon_state():
    config = SimulationConfig(
        seed=15,
        phases=PhaseConfig(60, 120, 30),
        attacks=AttackConfig(interval_min=30, interval_max=30),
        deliveries_per_robot=2,
    )
    manifest = author_warehouse_manifest(config)
    assert manifest.candidate_metadata
    assert all(item["heatmap_reference_steps"] == 60 for item in manifest.candidate_metadata)
    assert all(0 <= item["reference_step"] < 60 for item in manifest.candidate_metadata)
    assert all("reconnaissance" in item["selection_basis"] for item in manifest.candidate_metadata)


def test_default_warehouse_has_three_shared_valid_permanent_obstacles():
    config = SimulationConfig(
        seed=15,
        phases=PhaseConfig(60, 120, 30),
        attacks=AttackConfig(enabled=()),
        deliveries_per_robot=2,
    )
    manifest = author_warehouse_manifest(config)
    grid = np.asarray(manifest.static_grid, dtype=np.uint8)
    protected = set(manifest.robot_starts.values())
    for queue in manifest.task_queues.values():
        for task in queue:
            protected.update((task.pickup, task.dropoff))
    assert len(manifest.permanent_obstacles) == 3
    all_cells = set()
    for obstacle in manifest.permanent_obstacles:
        assert obstacle.cells
        assert not set(obstacle.cells) & all_cells
        assert not set(obstacle.cells) & protected
        assert all(grid[cell] == 0 for cell in obstacle.cells)
        assert footprint_finite_detour_score(grid, obstacle.cells) > 0
        all_cells.update(obstacle.cells)


def test_seed10_physical_obstacle_schedule_preserves_all_mission_connectivity():
    """Regression for a true-world disconnect found during full-seed QA."""
    from map_poisoning.map_io import WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS

    config = SimulationConfig(seed=10)
    manifest = author_warehouse_manifest(config)
    grid = np.asarray(manifest.static_grid, dtype=np.uint8)
    groups = [tuple(WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS)]
    for robot_id, start in manifest.robot_starts.items():
        anchors = {tuple(start)}
        for task in manifest.task_queues[robot_id]:
            anchors.update((tuple(task.pickup), tuple(task.dropoff)))
        groups.append(tuple(anchors))
    permanent = {
        tuple(cell)
        for obstacle in manifest.permanent_obstacles
        for cell in obstacle.cells
    }
    for step in range(
        0,
        config.phases.total_steps,
        config.temporary_blockage_change_period_steps,
    ):
        active = {
            tuple(cell)
            for episode in manifest.obstacle_episodes
            if episode.appearance_step <= step < episode.clearance_step
            for cell in episode.cells
        }
        assert all(
            anchors_remain_connected(grid, permanent | active, anchors)
            for anchors in groups
        ), step


def test_false_clearance_can_target_a_permanent_obstacle_and_audit_truth_is_blocked():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(20, 120, 20),
            attacks=AttackConfig(enabled=("false_clearance",), interval_min=20, interval_max=20),
            deliveries_per_robot=2,
        ),
        _open_grid(),
    )
    permanent_ids = {obstacle.obstacle_id for obstacle in manifest.permanent_obstacles}
    permanent_events = [
        event for event in manifest.attack_events
        if event.obstacle_episode_id in permanent_ids
    ]
    assert permanent_events
    labels = {label.report_id: label for label in manifest.report_audit_labels}
    assert all(labels[report_id].actual_state_at_observation == ClaimType.BLOCKED for event in permanent_events for report_id in event.report_ids)
    assert audit_manifest(manifest)["passed"]


def test_fake_obstacles_never_report_statically_or_physically_blocked_cells():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(20, 120, 20),
            attacks=AttackConfig(enabled=("fake_obstacle",), interval_min=20, interval_max=20),
            deliveries_per_robot=2,
        ),
        _open_grid(),
    )
    static = np.asarray(manifest.static_grid, dtype=np.uint8)
    permanent = {cell for obstacle in manifest.permanent_obstacles for cell in obstacle.cells}
    for event in manifest.attack_events:
        for cell in event.cells:
            assert static[cell] == 0
            assert cell not in permanent
    assert audit_manifest(manifest)["passed"]


def test_recovery_phase_contains_no_new_attack_injections():
    phases = PhaseConfig(20, 80, 40)
    manifest = author_manifest(
        SimulationConfig(seed=15, phases=phases, deliveries_per_robot=2),
        _open_grid(),
    )
    attack_end = phases.recon_steps + phases.attack_steps
    assert all(phases.recon_steps <= event.step < attack_end for event in manifest.attack_events)


def test_cell_weighted_scan_evidence_scales_with_number_of_contradicted_cells():
    _, one_after, one_batch, _ = _verification_batch(1)
    _, four_after, four_batch, _ = _verification_batch(4)
    assert one_batch["contradicted_weight"] == pytest.approx(1.0)
    assert four_batch["contradicted_weight"] == pytest.approx(4.0)
    assert four_after < one_after


def test_cell_weighted_scan_evidence_scales_with_confidence_and_cap():
    before, strong_after, strong_batch, _ = _verification_batch(2, report_confidence=1.0, observation_confidence=1.0)
    _, weak_after, weak_batch, _ = _verification_batch(2, report_confidence=0.5, observation_confidence=0.5)
    assert strong_batch["contradicted_weight"] == pytest.approx(2.0)
    assert weak_batch["contradicted_weight"] == pytest.approx(0.5)
    assert strong_after < weak_after < before
    _, _, _, trust = _verification_batch(18)
    alpha, beta = trust.values[0]
    assert alpha + beta <= 12.0 + 1e-9


def test_movingai_sources_preserve_exact_dimensions_and_author_deterministically():
    expected = {"room-32-32-4": (32, 32), "den312d": (81, 65)}
    for map_id, shape in expected.items():
        path = packaged_movingai_map_path(map_id)
        grid = load_movingai(path)
        assert grid.shape == shape
        config = SimulationConfig(
            seed=23,
            phases=PhaseConfig(20, 60, 20),
            map_movingai=path,
            deliveries_per_robot=2,
        )
        first = author_manifest(config, grid)
        second = author_manifest(config, grid)
        assert scenario_manifest_hash(first) == scenario_manifest_hash(second)
        assert len(first.permanent_obstacles) == 3
        assert audit_manifest(first)["passed"]
        assert all(grid[cell] == 0 for cell in first.robot_starts.values())
        for queue in first.task_queues.values():
            for task in queue:
                assert grid[task.pickup] == 0
            assert grid[task.dropoff] == 0


def test_room32_fake_and_false_attacks_are_authored_from_recon_only():
    path = packaged_movingai_map_path("room-32-32-4")
    grid = load_movingai(path)
    phases = PhaseConfig(40, 120, 30)
    config = SimulationConfig(
        seed=15,
        phases=phases,
        attacks=AttackConfig(
            enabled=("fake_obstacle", "false_clearance"),
            interval_min=20,
            interval_max=20,
        ),
        map_movingai=path,
        deliveries_per_robot=2,
    )
    manifest = author_manifest(config, grid)

    assert manifest.reconnaissance_data is not None
    assert manifest.reconnaissance_data.recon_steps == phases.recon_steps
    assert {event.attack_type.value for event in manifest.attack_events} <= {
        "fake_obstacle",
        "false_clearance",
    }
    assert all(
        phases.recon_steps <= event.step < phases.recon_steps + phases.attack_steps
        for event in manifest.attack_events
    )
    assert manifest.candidate_metadata
    assert all(
        item["reference_step"] < phases.recon_steps
        and item["heatmap_reference_steps"] == phases.recon_steps
        and "frozen_reconnaissance" in item["selection_basis"]
        for item in manifest.candidate_metadata
    )
    labels = {label.report_id: label for label in manifest.report_audit_labels}
    assert all(
        labels[report_id].actual_state_at_observation == (
            ClaimType.FREE
            if event.attack_type.value == "fake_obstacle"
            else ClaimType.BLOCKED
        )
        for event in manifest.attack_events
        for report_id in event.report_ids
    )
    assert audit_manifest(manifest)["passed"]


def test_false_clearance_authors_both_permanent_and_temporary_targets():
    grid = _open_grid()
    manifest = author_manifest(
        SimulationConfig(
            seed=1,
            phases=PhaseConfig(20, 200, 20),
            attacks=AttackConfig(enabled=("false_clearance",), interval_min=20, interval_max=20),
            deliveries_per_robot=2,
        ),
        grid,
    )
    permanent_ids = {obstacle.obstacle_id for obstacle in manifest.permanent_obstacles}
    temporary_ids = {episode.episode_id for episode in manifest.obstacle_episodes}
    events = list(manifest.attack_events)
    assert any(event.obstacle_episode_id in permanent_ids for event in events)
    assert any(event.obstacle_episode_id in temporary_ids for event in events)
    labels = {label.report_id: label for label in manifest.report_audit_labels}
    assert all(
        labels[report_id].actual_state_at_observation == ClaimType.BLOCKED
        for event in events for report_id in event.report_ids
    )
