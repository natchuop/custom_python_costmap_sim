from dataclasses import replace
import contextlib
import io
from collections import Counter
from map_poisoning.config import AttackConfig, FusionConfig, PhaseConfig, SimulationConfig, VisualizationConfig
from map_poisoning.models import ClaimType
from map_poisoning.scenario import author_manifest
from map_poisoning.audit import audit_manifest
from map_poisoning.models import AttackType
from map_poisoning.rollout import run_manifest_rollout
from map_poisoning.sensing import lidar_observations
from map_poisoning.world import demo_grid
import sim2
from sim2 import CellState, TemporaryBlockageManager
import numpy as np

def test_same_seed_same_manifest():
    assert author_manifest(SimulationConfig()).to_dict() == author_manifest(SimulationConfig()).to_dict()


def test_seed_selects_attack_types_per_event():
    seen = set()
    for seed in range(12):
        config = SimulationConfig(
            seed=seed,
            phases=PhaseConfig(20, 80, 20),
            attacks=AttackConfig(interval_min=10, interval_max=10),
        )
        manifest = author_manifest(config)
        assert manifest.attack_events
        assert {event.attack_type for event in manifest.attack_events} <= set(AttackType)
        seen.update(event.attack_type for event in manifest.attack_events)
    assert seen == set(AttackType)


def test_lidar_detects_temporary_obstacle_in_view():
    grid = np.zeros((10, 10), dtype=int)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    grid[5, 4] = int(CellState.TEMPORARILY_BLOCKED)
    observations = lidar_observations(grid, (5, 2))
    assert observations[(5, 4)] == ClaimType.BLOCKED


def test_temporary_objects_use_seeded_movement_modes():
    manager = TemporaryBlockageManager(demo_grid(24, 30), active_count=1, change_period=10, seed=3)
    before_index = next(iter(manager.active_indices))
    before = manager.current_footprints[before_index][0]
    manager.refresh_active_blockages()
    after_index = next(iter(manager.active_indices))
    after = manager.current_footprints[after_index][0]
    assert after_index == before_index
    assert manager.movement_decisions[after_index] in {"shift", "teleport", "unchanged"}
    assert len(before) == len(after)
    assert manager.current_footprints[after_index][1] == manager.pool[after_index][1]


def test_each_attack_type_reaches_benign_replanning():
    for seed in (0, 2, 4):
        config = SimulationConfig(
            seed=seed,
            phases=PhaseConfig(20, 80, 20),
            attacks=AttackConfig(interval_min=10, interval_max=10),
            visualization=VisualizationConfig(False),
            max_steps=100,
            deliveries_per_robot=1,
        )
        manifest = author_manifest(config)
        _, robots, log = run_manifest_rollout(config, manifest, "source_linked")
        assert any(report["is_malicious"] for report in log["reports"])
        first_attack_step = manifest.attack_events[0].step
        assert any(
            record["step"] >= first_attack_step
            for robot in robots if not robot.is_malicious
            for record in robot.replan_events
        )


def test_simulation_roles_are_fixed_to_robot_zero_attacker():
    _, robots, log = sim2.run_simulation(
        max_steps=5,
        random_seed=12,
        experiment_mode="clean",
    )
    assert log["malicious_robot_id"] == 0
    assert [robot.robot_id for robot in robots if robot.is_malicious] == [0]
    assert [robot.robot_id for robot in robots if not robot.is_malicious] == [1, 2]


def test_fake_obstacles_use_enlarged_footprints():
    config = SimulationConfig(
        seed=0,
        phases=PhaseConfig(20, 80, 20),
        attacks=AttackConfig(interval_min=10, interval_max=10),
    )
    manifest = author_manifest(config)
    fake_events = [event for event in manifest.attack_events if event.attack_type == AttackType.FAKE_OBSTACLE]
    assert fake_events
    assert max(len(event.cells) for event in fake_events) >= 15
    assert all(not manifest.static_grid[r][c] for event in fake_events for r, c in event.cells)


def test_fake_obstacle_dimensions_are_bounded_by_seven_cells():
    rng = np.random.default_rng(7)
    dimensions = [sim2.sample_fake_obstacle_dimensions(rng) for _ in range(200)]
    assert all(1 <= height <= 7 and 1 <= width <= 7 for height, width in dimensions)
    assert any(height == 7 or width == 7 for height, width in dimensions)


def test_attack_labels_and_peer_delivery_provenance_cover_all_attack_types():
    config = SimulationConfig(
        seed=0,
        phases=PhaseConfig(20, 120, 20),
        attacks=AttackConfig(interval_min=10, interval_max=10),
        deliveries_per_robot=1,
        max_steps=160,
        visualization=VisualizationConfig(False),
    )
    manifest = author_manifest(config)
    assert audit_manifest(manifest)["passed"]
    assert {event.attack_type for event in manifest.attack_events} == set(AttackType)

    with contextlib.redirect_stdout(io.StringIO()):
        _, robots, log = run_manifest_rollout(config, manifest, "source_linked")

    malicious_sent = [report for report in log["reports"] if report["is_malicious"]]
    malicious_deliveries = [
        delivery for delivery in log["report_deliveries"]
        if delivery["is_malicious"]
    ]
    malicious_counts = Counter(report["attack_type"] for report in malicious_sent)
    assert set(malicious_counts) == {item.value for item in AttackType}
    assert all(count > 0 for count in malicious_counts.values())
    assert len(malicious_deliveries) == 2 * len(malicious_sent)
    assert {delivery["recipient_id"] for delivery in malicious_deliveries} == {1, 2}

    processed = [
        item for item in log["report_processing"] if item["is_malicious"]
    ]
    assert len(processed) == len(malicious_deliveries)
    assert all(item["accepted"] for item in processed)
    assert any(
        "malicious_report_on_route" in event["reason"]
        for robot in robots if not robot.is_malicious
        for event in robot.replan_events
    )

def test_benign_shared_blocked_observations_reach_combined_map():
    for method in ("source_linked", "soft_probability"):
        with contextlib.redirect_stdout(io.StringIO()):
            _, robots, log = sim2.run_simulation(
                tasks_per_robot=100,
                max_steps=300,
                random_seed=15,
                experiment_mode="clean",
                defense_method=method,
                map_view="combined",
            )

        for receiver, sender in ((1, 2), (2, 1)):
            accepted_blocked = {
                claim.target_cell
                for claims in robots[receiver].defense_runner.claims_by_cell.values()
                for claim in claims
                if claim.sender_id == sender and claim.claim == sim2.ClaimType.BLOCKED
            }
            displayed_orange = {
                tuple(cell)
                for frame in log["robots"][receiver]["combined_belief"]
                for cell in np.argwhere(frame == sim2.DISPLAY_PEER_BELIEF)
            }
            assert accepted_blocked & displayed_orange
