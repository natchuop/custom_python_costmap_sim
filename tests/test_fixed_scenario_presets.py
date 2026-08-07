from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from map_poisoning.config import FusionConfig, LoggingConfig, SimulationConfig
from map_poisoning.cli import config_from_args, parser
from map_poisoning.scenario import _nominal_route_cells, author_manifest
from map_poisoning.scenario_presets import PRESETS, validate_fixed_preset
from map_poisoning.map_io import load_npy
from map_poisoning.ui import validate_gui_map_preset


ROOT = Path(__file__).parents[1]
MAPS = {
    "warehouse_002": ROOT / "converted_maps/maps_002_map/static_grid.npy",
    "warehouse_005": ROOT / "converted_maps/maps_005_map/static_grid.npy",
    "warehouse_005_rotated": ROOT / "converted_maps/maps_005_map_rotated/static_grid.npy",
}


def test_known_presets_have_valid_shapes_and_cells():
    for preset_id, path in MAPS.items():
        validate_fixed_preset(load_npy(path), PRESETS[preset_id])


def test_fixed_preset_is_seed_independent(tmp_path):
    path = MAPS["warehouse_005"]
    base = SimulationConfig(scenario_preset="warehouse_005", map_npy=str(path), deliveries_per_robot=2)
    first = author_manifest(replace(base, seed=15), load_npy(path))
    second = author_manifest(replace(base, seed=16), load_npy(path))
    assert first.robot_starts == second.robot_starts
    assert first.task_queues == second.task_queues


def test_fixed_preset_is_method_independent():
    path = MAPS["warehouse_005"]
    grid = load_npy(path)
    first = author_manifest(SimulationConfig(scenario_preset="warehouse_005", map_npy=str(path), fusion=FusionConfig(method="source_linked")), grid)
    second = author_manifest(SimulationConfig(scenario_preset="warehouse_005", map_npy=str(path), fusion=FusionConfig(method="full_trust")), grid)
    assert first.robot_starts == second.robot_starts
    assert first.task_queues == second.task_queues


def test_nominal_routes_use_preset_geometry():
    grid = load_npy(MAPS["warehouse_005"])
    preset = PRESETS["warehouse_005"]
    routes = _nominal_route_cells(grid, tuple(preset.robot_starts.values()), preset.delivery_points)
    assert routes
    assert (2, 2) not in routes


def test_blocked_preset_cell_fails_loudly():
    grid = load_npy(MAPS["warehouse_005"])
    preset = PRESETS["warehouse_005"]
    broken = grid.copy()
    broken[preset.delivery_points[0]] = 1
    with pytest.raises(ValueError, match="blocked"):
        validate_fixed_preset(broken, replace(preset, expected_map_hash=None))


def test_unreachable_preset_cell_fails_loudly():
    grid = load_npy(MAPS["warehouse_005"])
    preset = PRESETS["warehouse_005"]
    broken = grid.copy()
    goal = preset.delivery_points[0]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        broken[goal[0] + dr, goal[1] + dc] = 1
    with pytest.raises(ValueError, match="cannot reach"):
        validate_fixed_preset(broken, replace(preset, expected_map_hash=None))


def test_gui_config_propagates_scenario_preset_and_map():
    args = parser().parse_args([
        "--headless",
        "--map-npy", str(MAPS["warehouse_005"]),
        "--scenario-preset", "warehouse_005",
    ])
    config = config_from_args(args)
    assert config.map_npy == str(MAPS["warehouse_005"])
    assert config.scenario_preset == "warehouse_005"


def test_gui_rejects_known_map_without_matching_preset():
    with pytest.raises(ValueError, match="select that preset"):
        validate_gui_map_preset(str(MAPS["warehouse_005"]), None)
    validate_gui_map_preset(str(MAPS["warehouse_005"]), "warehouse_005")
