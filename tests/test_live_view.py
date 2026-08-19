import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

from map_poisoning.config import VisualizationConfig
from map_poisoning.live_view import show_belief_maps, show_traffic_heatmap
from map_poisoning.rollout import run_manifest_rollout

from tests.test_modular_rollout_validation import _config, _manifest


def test_live_recording_builds_heatmap_and_four_map_axes():
    config = replace(_config(), visualization=VisualizationConfig(animation=True))
    world, robots, log = run_manifest_rollout(config, _manifest(), "full_trust")
    live = log["live"]
    assert len(live["truth"]) == config.total_steps
    assert set(live["beliefs"]) == {robot.robot_id for robot in robots}
    assert live["heatmap"].sum() > 0
    heat = show_traffic_heatmap(log, show=False)
    maps = show_belief_maps(log, world, robots, show=False, interval_ms=1)
    assert heat is not None
    fig, anim = maps
    map_axes = [ax for ax in fig.axes if ax.get_xlabel() == "col"]
    assert len(map_axes) == 1 + len(robots)
    anim._func(0)
    heat.clf()
    fig.clf()
