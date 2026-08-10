from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from map_poisoning.config import AttackConfig, PhaseConfig, SimulationConfig
from map_poisoning.map_io import load_npy
from map_poisoning.scenario import author_manifest, scenario_manifest_hash
from map_poisoning.scenario_presets import PRESETS, validate_fixed_preset


ROOT = Path(__file__).parents[1]
MAPS = {
    "warehouse_002": ROOT / "converted_maps/maps_002_map/static_grid.npy",
    "warehouse_005": ROOT / "converted_maps/maps_005_map/static_grid.npy",
    "warehouse_005_rotated": ROOT / "converted_maps/maps_005_map_rotated/static_grid.npy",
}


@pytest.mark.parametrize("preset_id", tuple(PRESETS))
def test_known_preset_geometry_validates(preset_id):
    path = MAPS[preset_id]
    if not path.exists():
        pytest.skip("converted experimental maps are not present")
    validate_fixed_preset(load_npy(path), PRESETS[preset_id])


def test_preset_validation_rejects_wrong_shape():
    preset = PRESETS["warehouse_005"]
    with pytest.raises(ValueError, match="expected shape"):
        validate_fixed_preset(np.zeros((4, 4), dtype=np.uint8), preset)


def test_fixed_manifest_geometry_is_seed_and_method_independent():
    path = MAPS["warehouse_005"]
    if not path.exists():
        pytest.skip("converted experimental maps are not present")
    grid = load_npy(path)
    base = SimulationConfig(
        scenario_preset="warehouse_005",
        map_npy=str(path),
        phases=PhaseConfig(1, 3, 1),
        attacks=AttackConfig(enabled=("fake_obstacle",), interval_min=1, interval_max=1),
        deliveries_per_robot=2,
    )
    first = author_manifest(replace(base, seed=15), grid)
    second = author_manifest(replace(base, seed=16), grid)
    assert first.robot_starts == second.robot_starts
    assert first.task_queues == second.task_queues
    assert scenario_manifest_hash(first) != scenario_manifest_hash(second)
