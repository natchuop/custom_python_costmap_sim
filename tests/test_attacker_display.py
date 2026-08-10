import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import sim2


def test_fake_history_ignores_honest_and_other_attack_provenance():
    history = {
        1: [
            {"attacker_id": 1, "attack_type": "fake_obstacle", "cells": [(2, 2)], "expires_step": 50},
            {"attacker_id": 0, "attack_type": "false_clearance", "cells": [(3, 3)], "expires_step": 50},
            {"attacker_id": 0, "attack_type": "fake_obstacle", "cells": [(4, 4)], "expires_step": 50},
        ]
    }
    assert sim2.fake_obstacle_history_cells(history, 1, 10) == [(4, 4)]


def test_fake_history_ttl_boundary():
    history = {1: [{"attacker_id": 0, "attack_type": "fake_obstacle", "cells": [(4, 4)], "expires_step": 50}]}
    assert sim2.fake_obstacle_history_cells(history, 1, 49) == [(4, 4)]
    assert sim2.fake_obstacle_history_cells(history, 1, 50) == []


def test_fake_outline_is_dotted_red():
    fig, ax = plt.subplots()
    try:
        patches = sim2.draw_attack_outlines(ax, [(4, 4)])
        assert patches[0].get_linestyle() == ":"
        edge = tuple(patches[0].get_edgecolor()[:3])
        assert edge[0] > edge[1] and edge[0] > edge[2]
    finally:
        plt.close(fig)


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything")
def test_animation_uses_four_map_axes_and_dedicated_status_regions(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    _, robots, log = sim2.run_simulation(max_steps=1, tasks_per_robot=1, random_seed=15, experiment_mode="clean")
    animation = sim2.animate(sim2.GridWorld(log["truth_grid"][0]), robots, log)
    figure = animation._fig
    try:
        animation._func(0)
        titles = [axis.get_title() for axis in figure.axes]
        assert sum("Ground Truth Map" in title for title in titles) == 1
        assert sum("Robot 0" in title for title in titles) == 1
        assert sum("Robot 1" in title for title in titles) == 1
        assert sum("Robot 2" in title for title in titles) == 1
        assert any(
            axis.get_title(loc="left") == "Robot trust level | trust_threshold"
            for axis in figure.axes
        )
        for robot_id in (0, 1, 2):
            map_axis = next(
                axis
                for axis in figure.axes
                if axis.get_title().startswith(f"Robot {robot_id} |")
            )
            own_ray_lines = [
                line
                for line in map_axis.lines
                if line.get_color() == sim2.ROBOT_COLORS[robot_id]
            ]
            assert len(own_ray_lines) >= sim2.LIDAR_NUM_RAYS
            assert any(
                patch.get_radius() == sim2.LIDAR_RANGE_CELLS
                for patch in map_axis.patches
            )
        assert len(figure.axes) >= 8
    finally:
        animation.event_source.stop()
        plt.close(figure)
