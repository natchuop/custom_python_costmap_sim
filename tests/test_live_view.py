import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

import numpy as np

from map_poisoning.config import VisualizationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.live_view import DISPLAY_R2, combined_display_grid
from map_poisoning.models import ClaimReport, ClaimType, DirectObservation
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.live_view import show_belief_maps, show_traffic_heatmap
from map_poisoning.rollout import run_manifest_rollout
from map_poisoning.trust import ScalarTrustModel

from tests.test_modular_rollout_validation import _config, _manifest


def test_stale_explored_clear_does_not_hide_trusted_peer_blocked_on_map():
    grid = np.zeros((8, 8), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {"malicious_robot_id": 0}
    observer = type(
        "Robot",
        (),
        {
            "robot_id": 1,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "trust_threshold": 0.55,
            "belief": RobotBeliefMap(grid, memory_steps=2),
            "trust": ScalarTrustModel(initial=0.70),
            "fusion": FusionEngine("trust_threshold", lambda _: 0.70, trust_threshold=0.55),
        },
    )()
    peer = type(
        "Robot",
        (),
        {
            "robot_id": 2,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "belief": RobotBeliefMap(grid, memory_steps=2),
        },
    )()
    observer.belief.observe(DirectObservation(1, (3, 3), ClaimType.FREE, 0))
    peer.belief.observe(DirectObservation(2, (3, 3), ClaimType.BLOCKED, 5))
    observer.fusion.add(
        ClaimReport("peer-blocked", 2, (3, 3), ClaimType.BLOCKED, 5, 5, 5),
    )
    arr = combined_display_grid(observer, world, log, step=10, robots=(observer, peer))
    assert arr[3, 3] == DISPLAY_R2


def test_trusted_attacker_fake_obstacles_paint_on_victim_map():
    from map_poisoning.models import AttackEvent, AttackType

    grid = np.zeros((8, 8), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {
        "malicious_robot_id": 0,
        "attack_events": (
            AttackEvent(
                "attack-0000",
                5,
                AttackType.FAKE_OBSTACLE,
                ((2, 2), (2, 3)),
                ClaimType.BLOCKED,
                5,
                0,
                (1, 2),
                ("report-0", "report-1"),
            ),
        ),
    }
    victim = type(
        "Robot",
        (),
        {
            "robot_id": 1,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "trust_threshold": 0.55,
            "belief": RobotBeliefMap(grid, memory_steps=2),
            "trust": ScalarTrustModel(initial=0.80),
            "fusion": FusionEngine("trust_threshold", lambda _: 0.80, trust_threshold=0.55),
        },
    )()
    attacker = type(
        "Robot",
        (),
        {
            "robot_id": 0,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "belief": RobotBeliefMap(grid, memory_steps=2),
        },
    )()
    from map_poisoning.live_view import DISPLAY_R0
    from map_poisoning.models import ClaimReport

    victim.fusion.add(
        ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 5, 5),
        is_malicious=True,
    )
    victim.fusion.add(
        ClaimReport("report-1", 0, (2, 3), ClaimType.BLOCKED, 5, 5, 5),
        is_malicious=True,
    )
    arr = combined_display_grid(victim, world, log, step=6, robots=(attacker, victim))

    assert arr[2, 2] == DISPLAY_R0
    assert arr[2, 3] == DISPLAY_R0


def test_validated_fake_obstacle_clears_from_victim_map():
    from map_poisoning.live_view import DISPLAY_FREE, DISPLAY_R0
    from map_poisoning.models import AttackEvent, AttackType, ClaimReport, DirectObservation

    grid = np.zeros((8, 8), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {
        "malicious_robot_id": 0,
        "attack_events": (
            AttackEvent(
                "attack-0000",
                5,
                AttackType.FAKE_OBSTACLE,
                ((2, 2),),
                ClaimType.BLOCKED,
                5,
                0,
                (1,),
                ("report-0",),
            ),
        ),
    }
    victim = type(
        "Robot",
        (),
        {
            "robot_id": 1,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "trust_threshold": 0.55,
            "belief": RobotBeliefMap(grid, memory_steps=12),
            "trust": ScalarTrustModel(initial=0.80),
            "fusion": FusionEngine("trust_threshold", lambda _: 0.80, trust_threshold=0.55),
        },
    )()
    attacker = type(
        "Robot",
        (),
        {
            "robot_id": 0,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "belief": RobotBeliefMap(grid, memory_steps=12),
        },
    )()
    victim.fusion.add(
        ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 5, 5),
        is_malicious=True,
    )
    assert combined_display_grid(victim, world, log, step=6, robots=(attacker, victim))[2, 2] == DISPLAY_R0

    victim.belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 6))
    victim.fusion.retract(ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 5, 5))
    assert combined_display_grid(victim, world, log, step=6, robots=(attacker, victim))[2, 2] == DISPLAY_FREE


def test_live_recording_builds_heatmap_and_four_map_axes():
    config = replace(_config(), visualization=VisualizationConfig(animation=True))
    world, robots, log = run_manifest_rollout(config, _manifest(), "full_trust")
    live = log["live"]
    assert len(live["truth"]) == config.total_steps
    assert set(live["beliefs"]) == {robot.robot_id for robot in robots}
    assert len(live["combined_beliefs"][robots[0].robot_id]) == config.total_steps
    assert len(live["local_beliefs"][robots[0].robot_id]) == config.total_steps
    assert live["heatmap"].sum() > 0
    assert live["threshold"] == config.trust.threshold
    heat = show_traffic_heatmap(log, show=False)
    maps = show_belief_maps(log, world, robots, show=False, interval_ms=1)
    assert heat is not None
    fig, anim = maps
    titled = [ax.get_title() for ax in fig.axes]
    assert any("Ground Truth Map" in title for title in titled)
    assert "seed 1" in heat.axes[0].get_title()
    assert "seed 1" in (fig.get_suptitle() or "")
    assert any(ax.get_title(loc="left") == "Robot trust level | full_trust" for ax in fig.axes)
    assert any("Threshold:" in (text.get_text() or "") for ax in fig.axes for text in ax.texts)
    anim._func(0)
    heat.clf()
    fig.clf()
