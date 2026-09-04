from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from map_poisoning.config import FusionConfig, LoggingConfig, SimulationConfig
from map_poisoning.cli import config_from_args, parser
from map_poisoning.scenario import _nominal_route_cells, author_manifest
from map_poisoning.scenario_presets import PRESETS, validate_fixed_preset
from map_poisoning.map_io import load_npy
from map_poisoning.ui import is_physical_ai_method_selection, run_physical_ai_workflow, validate_gui_map_preset


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
    first = author_manifest(SimulationConfig(scenario_preset="warehouse_005", map_npy=str(path), fusion=FusionConfig(method="source_memory")), grid)
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


def test_attacker_receives_repeating_fixed_queue_in_short_runs():
    path = MAPS["warehouse_005"]
    manifest = author_manifest(
        SimulationConfig(
            scenario_preset="warehouse_005",
            map_npy=str(path),
            deliveries_per_robot=1,
        ),
        load_npy(path),
    )
    assert len(manifest.task_queues[0]) > 1
    assert all(task.pickup != task.dropoff for task in manifest.task_queues[0])


def test_cli_preset_without_explicit_map_resolves_packaged_map():
    from map_poisoning.cli import config_from_args, parser
    from map_poisoning.map_io import load_npy
    config = config_from_args(parser().parse_args(["--headless", "--scenario-preset", "warehouse_002", "--no-plots"]))
    assert config.map_npy is not None
    assert load_npy(config.map_npy).shape == PRESETS["warehouse_002"].expected_shape


def test_gui_packaged_map_options_are_cwd_independent():
    from map_poisoning.ui import MAP_OPTIONS
    for label in ("Map 002 (converted)", "Map 005 (converted)", "Map 005 rotated (converted)"):
        path, preset_id = MAP_OPTIONS[label]
        assert Path(path).is_absolute()
        assert Path(path).exists()
        validate_fixed_preset(load_npy(path), PRESETS[preset_id])


def test_gui_four_method_selection_identifies_physical_ai_reporting_workflow():
    assert is_physical_ai_method_selection(("full_trust", "majority_vote", "trust_fused", "source_memory"))
    assert not is_physical_ai_method_selection(("full_trust", "majority_vote"))
    assert not is_physical_ai_method_selection(("full_trust", "majority_vote", "trust_fused", "source_memory", "latest_report"))


def test_gui_physical_ai_report_delegates_to_canonical_reference_reporter(monkeypatch, tmp_path):
    import map_poisoning.ui as ui

    calls = []
    monkeypatch.setattr(ui, "generate_reference_report", lambda path: calls.append(path) or {"generated": []})
    result = ui.generate_physical_ai_report(tmp_path)
    assert result == {"generated": []}
    assert calls == [tmp_path]


def test_gui_physical_ai_default_mode_runs_only_normal_batch(monkeypatch, tmp_path):
    import map_poisoning.batch as batch
    import map_poisoning.ui as ui

    calls = []
    monkeypatch.setattr(batch, "run_multiseed", lambda *args, **kwargs: calls.append(("normal", args, kwargs)))
    monkeypatch.setattr(ui, "generate_physical_ai_report", lambda root: calls.append(("report", root)) or {})
    config = SimulationConfig(logging=LoggingConfig(output_directory=str(tmp_path)))

    run_physical_ai_workflow(config, (2, 7))

    assert [call[0] for call in calls] == ["normal", "report"]
    assert calls[0][1][1] == (2, 7)
    assert calls[1][1] == str(tmp_path)


def test_gui_full_physical_ai_mode_runs_sweeps_before_report(monkeypatch, tmp_path):
    import map_poisoning.reference_experiments as experiments
    import map_poisoning.ui as ui

    calls = []
    monkeypatch.setattr(
        experiments,
        "run_reference_suite",
        lambda config, seeds, root, **kwargs: calls.append(("suite", config, seeds, root, kwargs)),
    )
    monkeypatch.setattr(ui, "generate_physical_ai_report", lambda root: calls.append(("report", root)) or {})
    config = SimulationConfig(logging=LoggingConfig(output_directory=str(tmp_path)))

    run_physical_ai_workflow(config, (3, 5), full_suite=True)

    assert [call[0] for call in calls] == ["suite", "report"]
    assert calls[0][2] == (3, 5)
    assert calls[0][3] == str(tmp_path)
    assert calls[0][4]["include_sweeps"] is True
    assert calls[0][4]["generate_report"] is False
    assert calls[1][1] == str(tmp_path)
