from pathlib import Path

from pathlib import Path

from map_poisoning.cli import (
    attack_slug,
    config_from_args,
    geometry_slug,
    parser,
    result_location_message,
    suggested_output_directory,
)


def test_geometry_slug_uses_preset_then_map_then_default():
    assert geometry_slug(scenario_preset="warehouse_005") == "warehouse_005"
    assert geometry_slug(map_npy=r"converted_maps\maps_005_map\static_grid.npy") == "maps_005_map"
    assert geometry_slug() == "default_warehouse"


def test_suggested_paths_separate_run_compare_and_multiseed():
    single = suggested_output_directory(method="full_trust", seed=15)
    compare = suggested_output_directory(method="full_trust", seed=15, compare=True)
    multi = suggested_output_directory(method="full_trust", seed=15, seeds="1,2")
    assert Path(single) == Path("outputs") / "runs" / "full_trust_seed15_default_warehouse_all_attacks"
    assert Path(compare) == Path("outputs") / "comparisons" / "seed15_default_warehouse_all_attacks"
    assert Path(multi) == Path("outputs") / "multiseed" / "full_trust_seeds1,2_default_warehouse_all_attacks"


def test_cli_ports_main_visualization_and_trust_controls():
    config = config_from_args(parser().parse_args([
        "--headless",
        "--defense-method", "trust_threshold",
        "--trust-threshold", "0.62",
        "--map-view", "local",
        "--temp-obstacle-interval", "150",
        "--no-animation",
    ]))
    assert config.fusion.method == "trust_threshold"
    assert config.trust.threshold == 0.62
    assert config.visualization.map_view == "local"
    assert config.temporary_blockage_change_period_steps == 150
    assert config.visualization.animation is False


def test_cli_comparison_methods_are_selectable():
    config = config_from_args(parser().parse_args([
        "--headless",
        "--comparison-methods", "trust_threshold,full_trust",
        "--no-animation",
    ]))
    assert config.comparison_methods == ("trust_threshold", "full_trust")
    assert config.fusion.method == "source_memory"


def test_cli_default_output_is_named_unless_overridden():
    auto = config_from_args(parser().parse_args(["--headless", "--defense-method", "full_trust", "--seed", "15"]))
    assert Path(auto.logging.output_directory) == Path("outputs") / "runs" / "full_trust_seed15_default_warehouse_all_attacks"
    custom = config_from_args(parser().parse_args(["--headless", "--output-directory", str(Path("outputs") / "custom")]))
    assert Path(custom.logging.output_directory) == Path("outputs") / "custom"


def test_result_message_points_at_plots():
    text = result_location_message(r"outputs\runs\full_trust_seed15_default_warehouse")
    assert "plots" in text
    assert "full_trust_seed15_default_warehouse" in text


def test_attack_slug_and_subset_output_path():
    assert attack_slug(()) == "no_attacks"
    assert attack_slug(("fake_obstacle", "false_clearance", "stale_reassertion")) == "all_attacks"
    assert attack_slug(("stale_reassertion", "fake_obstacle")) == "fake_obstacle+stale_reassertion"
    path = suggested_output_directory(
        method="full_trust", seed=15, enabled_attacks=("fake_obstacle", "stale_reassertion"),
    )
    assert Path(path) == Path("outputs") / "runs" / "full_trust_seed15_default_warehouse_fake_obstacle+stale_reassertion"
