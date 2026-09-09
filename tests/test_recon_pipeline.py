from dataclasses import replace

import numpy as np

from map_poisoning.config import PhaseConfig, SimulationConfig
from map_poisoning.map_io import load_movingai, packaged_movingai_map_path
from map_poisoning.models import (
    AttackType,
    DeliveryTask,
    FrozenReconData,
    TemporaryObstacleEpisode,
)
from map_poisoning.recon_authoring import run_reconnaissance
from map_poisoning.scenario import ScenarioManifest, _hash, scenario_manifest_hash


def _manifest(grid, *, scenario_id="recon-test", episodes=()):
    free = [
        (row, col)
        for row in range(grid.shape[0])
        for col in range(grid.shape[1])
        if not grid[row, col]
    ]
    starts = {0: free[0], 1: free[1], 2: free[2]}
    tasks = {
        rid: (DeliveryTask(f"r{rid}-task", free[3 + rid], free[6 + rid]),)
        for rid in starts
    }
    phases = PhaseConfig(8, 12, 4)
    return ScenarioManifest(
        3,
        7,
        {},
        _hash(grid),
        tuple(grid.shape),
        tuple(tuple(int(value) for value in row) for row in grid),
        {"reconnaissance_end": 8, "attack_end": 20, "total": 24},
        0,
        (1, 2),
        tuple(episodes),
        (),
        scenario_id=scenario_id,
        robot_starts=starts,
        task_queues=tasks,
    )


def test_default_and_room32_use_the_same_frozen_recon_contract():
    default = np.zeros((14, 14), dtype=np.uint8)
    room32 = load_movingai(packaged_movingai_map_path("room-32-32-4"))
    for grid in (default, room32):
        frozen = run_reconnaissance(SimulationConfig(phases=PhaseConfig(8, 12, 4)), _manifest(grid))
        assert isinstance(frozen, FrozenReconData)
        assert frozen.recon_steps == 8
        assert len(frozen.traffic_heatmap) == grid.shape[0]
        assert all(snapshot.step < frozen.recon_steps for snapshot in frozen.mission_snapshots)
        assert all(snapshot.step < frozen.recon_steps for snapshot in frozen.route_snapshots)


def test_frozen_recon_data_ignores_post_recon_scenario_changes():
    grid = np.zeros((14, 14), dtype=np.uint8)
    first = _manifest(grid, scenario_id="first")
    changed_after_recon = replace(
        first,
        scenario_id="radically-different-after-recon",
        obstacle_episodes=(TemporaryObstacleEpisode("late", ((8, 8),), 12, 16),),
    )
    first_recon = run_reconnaissance(SimulationConfig(phases=PhaseConfig(8, 12, 4)), first)
    second_recon = run_reconnaissance(SimulationConfig(phases=PhaseConfig(8, 12, 4)), changed_after_recon)
    assert first_recon.to_dict() == second_recon.to_dict()


def test_recon_manifest_round_trip_preserves_the_frozen_dataset(tmp_path):
    from map_poisoning.scenario import load_manifest, save_manifest

    grid = np.zeros((14, 14), dtype=np.uint8)
    manifest = _manifest(grid)
    frozen = run_reconnaissance(SimulationConfig(phases=PhaseConfig(8, 12, 4)), manifest)
    persisted = replace(manifest, reconnaissance_data=frozen, reconnaissance_heatmap=frozen.traffic_heatmap)
    path = tmp_path / "manifest.json"
    save_manifest(persisted, path)
    loaded = load_manifest(path)
    assert scenario_manifest_hash(persisted) == scenario_manifest_hash(loaded)
    assert loaded.reconnaissance_data == frozen

