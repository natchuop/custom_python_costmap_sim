from collections import defaultdict

import numpy as np

from map_poisoning.audit import audit_manifest
from map_poisoning.config import AttackConfig, PhaseConfig, SimulationConfig
from map_poisoning.map_io import default_warehouse_map
from map_poisoning.models import AttackType, ClaimType, TemporaryObstacleEpisode
from map_poisoning.obstacles import FAKE_MIN_REPORT_CELLS, TEMP_ACTIVE_COUNT, TEMP_MIN_AREA
from map_poisoning.scenario import author_manifest
from map_poisoning.world import World


def _open_grid():
    grid = np.zeros((17, 17), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    return grid


def test_authored_temporary_obstacles_are_multi_cell_and_concurrent():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(200, 500, 50),
            attacks=AttackConfig(interval_min=50, interval_max=50),
            deliveries_per_robot=2,
            temporary_blockage_change_period_steps=200,
        ),
        _open_grid(),
    )
    assert manifest.obstacle_episodes
    assert all(len(episode.cells) >= TEMP_MIN_AREA for episode in manifest.obstacle_episodes)
    active = [episode for episode in manifest.obstacle_episodes if episode.appearance_step <= 0 < episode.clearance_step]
    assert len(active) >= 2
    world = World(np.asarray(manifest.static_grid, dtype=np.uint8), manifest.obstacle_episodes)
    dynamic = int((world.truth_grid(0) != np.asarray(manifest.static_grid)).sum())
    assert dynamic >= TEMP_MIN_AREA * 2


def test_temporary_obstacle_activation_yields_until_robot_leaves_footprint():
    grid = np.zeros((7, 7), dtype=np.uint8)
    episode = TemporaryObstacleEpisode("moving", ((3, 3), (3, 4)), 1, 4)
    world = World(grid, (episode,))

    assert world.begin_step(1, ((3, 3),))[3, 3] == 0
    assert world.state((3, 4), 1) == ClaimType.FREE
    assert world.deferred_activation_steps == 1
    assert world.deferred_episode_ids == {"moving"}

    assert world.begin_step(2, ((2, 3),))[3, 3] == 1
    assert world.state((3, 4), 2) == ClaimType.BLOCKED
    assert world.activation_step("moving") == 2
    assert world.begin_step(4, ((2, 3),))[3, 3] == 0


def test_authored_fake_obstacles_cover_a_rectangle_of_free_cells():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(200, 500, 50),
            attacks=AttackConfig(
                enabled=(AttackType.FAKE_OBSTACLE.value,),
                interval_min=50,
                interval_max=50,
            ),
            deliveries_per_robot=2,
        ),
        _open_grid(),
    )
    fakes = [event for event in manifest.attack_events if event.attack_type == AttackType.FAKE_OBSTACLE]
    assert fakes
    assert all(len(event.cells) >= FAKE_MIN_REPORT_CELLS for event in fakes)
    assert all(len(event.cells) == len(event.report_ids) for event in fakes)
    assert audit_manifest(manifest)["passed"]


def test_clearance_and_stale_attacks_cover_the_whole_temp_footprint():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(200, 500, 50),
            attacks=AttackConfig(interval_min=50, interval_max=50),
            deliveries_per_robot=2,
            temporary_blockage_change_period_steps=200,
        ),
        _open_grid(),
    )
    by_id = {episode.episode_id: episode for episode in manifest.obstacle_episodes}
    for event in manifest.attack_events:
        if event.attack_type not in {AttackType.FALSE_CLEARANCE, AttackType.STALE_REASSERTION}:
            continue
        episode = by_id[event.obstacle_episode_id]
        assert tuple(event.cells) == tuple(episode.cells)
    assert audit_manifest(manifest)["passed"]


def test_default_warehouse_gets_visible_temp_and_fake_footprints():
    grid = default_warehouse_map()
    manifest = author_manifest(
        SimulationConfig(seed=15, deliveries_per_robot=2, max_steps=None),
        grid,
    )
    active_by_step = defaultdict(int)
    for episode in manifest.obstacle_episodes:
        active_by_step[episode.appearance_step] += 1
        assert len(episode.cells) >= TEMP_MIN_AREA
    assert max(active_by_step.values()) >= min(TEMP_ACTIVE_COUNT, 3)
    fakes = [event for event in manifest.attack_events if event.attack_type == AttackType.FAKE_OBSTACLE]
    assert fakes and min(len(event.cells) for event in fakes) >= FAKE_MIN_REPORT_CELLS
    assert audit_manifest(manifest)["passed"]


def test_temporary_obstacles_shift_or_teleport_between_windows():
    manifest = author_manifest(
        SimulationConfig(
            seed=15,
            phases=PhaseConfig(80, 80, 80),
            attacks=AttackConfig(enabled=()),
            deliveries_per_robot=2,
            temporary_blockage_change_period_steps=80,
        ),
        _open_grid(),
    )
    first = {episode.episode_id: set(episode.cells) for episode in manifest.obstacle_episodes if episode.appearance_step == 0}
    second = {episode.episode_id: set(episode.cells) for episode in manifest.obstacle_episodes if episode.appearance_step == 80}
    assert first and second
    assert any(cells not in first.values() for cells in second.values())
