from dataclasses import replace

from map_poisoning.config import AttackConfig, PhaseConfig, SimulationConfig
from map_poisoning.map_io import (
    default_warehouse_map,
    WAREHOUSE_ATTACKER_START,
    WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS,
    WAREHOUSE_CORRIDOR_DELIVERY,
    WAREHOUSE_NARROW_CORRIDOR_CELLS,
)
from map_poisoning.obstacles import TEMP_ACTIVE_COUNT, TEMP_MIN_AREA
from map_poisoning.recon_authoring import author_warehouse_manifest, recon_heatmap_attack_candidates
from map_poisoning.warehouse_layout import build_warehouse_layout


def test_default_warehouse_manifest_includes_reconnaissance_heatmap():
    config = replace(
        SimulationConfig(),
        seed=14,
        phases=PhaseConfig(80, 120, 40),
        attacks=AttackConfig(enabled=("fake_obstacle",), interval_min=40, interval_max=40),
        deliveries_per_robot=2,
        max_steps=240,
    )
    manifest = author_warehouse_manifest(config, default_warehouse_map())
    assert manifest.reconnaissance_heatmap is not None
    # One shared heatmap covers the complete deterministic attack-free
    # reference horizon, not only the reconnaissance prefix.
    assert sum(map(sum, manifest.reconnaissance_heatmap)) == len(manifest.benign_robot_ids) * config.phases.total_steps
    assert manifest.obstacle_episodes
    assert sum(1 for episode in manifest.obstacle_episodes if episode.appearance_step == 0) >= min(
        TEMP_ACTIVE_COUNT, 3
    )
    assert all(len(episode.cells) >= TEMP_MIN_AREA for episode in manifest.obstacle_episodes)
    assert manifest.attack_events
    assert any(item.get("traffic_score") is not None for item in manifest.candidate_metadata)
    assert all(item.get("heatmap_reference_steps") == config.phases.total_steps for item in manifest.candidate_metadata)
    assert all(item.get("reference_step") >= config.phases.recon_steps for item in manifest.candidate_metadata)
    assert all(item.get("target_visible_to_victim") is False for item in manifest.candidate_metadata)
    assert all(item.get("intended_victim_id") in manifest.benign_robot_ids for item in manifest.candidate_metadata)


def test_attack_free_warehouse_manifest_skips_candidate_requirement():
    config = replace(
        SimulationConfig(),
        phases=PhaseConfig(3, 4, 3),
        attacks=AttackConfig(enabled=()),
        deliveries_per_robot=1,
        max_steps=10,
    )
    manifest = author_warehouse_manifest(config, default_warehouse_map())
    assert manifest.attack_events == ()
    assert manifest.candidate_metadata == ()
    assert sum(map(sum, manifest.reconnaissance_heatmap)) == 20


def test_warehouse_layout_and_candidates_are_reachable():
    grid = default_warehouse_map()
    starts, goals, tasks = build_warehouse_layout(grid, deliveries_per_robot=2)
    assert len(starts) == 3
    assert len(goals) == 14
    assert all(tasks[robot_id] for robot_id in starts)
    assert starts[0] == WAREHOUSE_ATTACKER_START
    assert starts[0] in WAREHOUSE_NARROW_CORRIDOR_CELLS
    assert set(goals) & WAREHOUSE_NARROW_CORRIDOR_CELLS == {WAREHOUSE_CORRIDOR_DELIVERY}
    assert not (set(list(starts.values())[1:]) & WAREHOUSE_NARROW_CORRIDOR_CELLS)
    assert all(
        min(abs(start[0] - cell[0]) + abs(start[1] - cell[1]) for cell in WAREHOUSE_NARROW_CORRIDOR_CELLS) >= 4
        for start in list(starts.values())[1:]
    )
    assert max(col for _, col in goals) >= 40
    assert max(row for row, _ in goals) >= 24
    assert min(col for _, col in goals) <= 8
    candidates = recon_heatmap_attack_candidates(
        grid,
        goals,
        [],
        __import__("numpy").zeros(grid.shape, dtype=int),
        require_route_overlap=False,
    )
    assert isinstance(candidates, list)


def test_warehouse_temporary_obstacles_keep_the_corridor_open():
    from map_poisoning.models import ClaimType
    from map_poisoning.obstacles import author_temporary_obstacle_episodes
    from map_poisoning.rng import named_rng
    from map_poisoning.world import World

    grid = default_warehouse_map()
    starts, goals, _ = build_warehouse_layout(grid, deliveries_per_robot=2)
    forbidden = set(starts.values()) | set(WAREHOUSE_NARROW_CORRIDOR_CELLS)
    episodes = author_temporary_obstacle_episodes(
        grid,
        named_rng(15, "temporary_obstacles"),
        total_steps=300,
        period=150,
        forbidden_cells=forbidden,
        required_anchors=WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS,
    )
    assert episodes
    assert sum(1 for episode in episodes if episode.appearance_step == 0) >= min(TEMP_ACTIVE_COUNT, 3)
    first = sorted([set(episode.cells) for episode in episodes if episode.appearance_step == 0], key=sorted)
    second = sorted([set(episode.cells) for episode in episodes if episode.appearance_step == 150], key=sorted)
    assert first and second
    assert first != second
    world = World(grid, episodes)
    for step in (0, 150, 299):
        blocked = {
            cell
            for cell in WAREHOUSE_NARROW_CORRIDOR_CELLS
            if world.state(cell, step) == ClaimType.BLOCKED
        }
        assert not blocked
        truth = world.truth_grid(step)
        for start in starts.values():
            assert not truth[start]
