import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

import numpy as np

from map_poisoning.config import VisualizationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.live_view import DISPLAY_DYNAMIC, DISPLAY_FREE, DISPLAY_R1, DISPLAY_R2, combined_display_grid
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
        ClaimReport("peer-blocked", 2, (3, 3), ClaimType.BLOCKED, 5, 1.0),
    )
    arr = combined_display_grid(observer, world, log, step=10, robots=(observer, peer))
    assert arr[3, 3] == DISPLAY_R2


def test_trusted_peer_free_clears_older_peer_blocked_on_combined_map():
    grid = np.zeros((8, 8), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {"malicious_robot_id": 0}
    scores = {1: 0.80, 2: 0.80}
    observer = type(
        "Robot",
        (),
        {
            "robot_id": 0,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "belief": RobotBeliefMap(grid, memory_steps=300),
            "fusion": FusionEngine("trust_threshold", scores.get, trust_threshold=0.55),
        },
    )()
    orange = type("Robot", (), {"robot_id": 1})()
    blue = type("Robot", (), {"robot_id": 2})()
    observer.fusion.add(ClaimReport("old-blocked", 1, (3, 3), ClaimType.BLOCKED, 5, 1.0))
    observer.fusion.add(ClaimReport("new-free", 2, (3, 3), ClaimType.FREE, 6, 1.0))
    assert combined_display_grid(observer, world, log, step=6, robots=(observer, orange, blue))[3, 3] == DISPLAY_FREE
    scores[2] = 0.40
    assert combined_display_grid(observer, world, log, step=7, robots=(observer, orange, blue))[3, 3] == DISPLAY_R1


def test_accepted_false_clearance_report_paints_white():
    grid = np.zeros((6, 6), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {"malicious_robot_id": 0}
    victim = type(
        "Robot",
        (),
        {
            "robot_id": 1,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "belief": RobotBeliefMap(grid, memory_steps=300),
            "fusion": FusionEngine("trust_threshold", lambda _: 0.80, trust_threshold=0.55),
        },
    )()
    attacker = type("Robot", (), {"robot_id": 0})()
    victim.fusion.add(ClaimReport("false-clear", 0, (2, 2), ClaimType.FREE, 5, 1.0))
    assert combined_display_grid(victim, world, log, step=5, robots=(attacker, victim))[2, 2] == DISPLAY_FREE


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
        ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 1.0, "attack-0000"),
        is_malicious=True,
    )
    victim.fusion.add(
        ClaimReport("report-1", 0, (2, 3), ClaimType.BLOCKED, 5, 1.0, "attack-0000"),
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
        ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 1.0, "attack-0000"),
        is_malicious=True,
    )
    assert combined_display_grid(victim, world, log, step=6, robots=(attacker, victim))[2, 2] == DISPLAY_R0

    victim.belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 6))
    victim.fusion.retract(ClaimReport("report-0", 0, (2, 2), ClaimType.BLOCKED, 5, 1.0, "attack-0000"))
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


def test_source_memory_combined_map_hides_blocked_claim_after_distrust():
    from map_poisoning.live_view import DISPLAY_R0, DISPLAY_UNKNOWN

    grid = np.zeros((8, 8), dtype=np.uint8)
    world = type("World", (), {"static_grid": grid})()
    log = {"malicious_robot_id": 0, "attack_events": ()}
    trust = {0: .8}
    memory = {0: .8}
    victim = type(
        "Robot",
        (),
        {
            "robot_id": 1,
            "completed": True,
            "tasks": (),
            "task_index": 0,
            "carrying": False,
            "trust_threshold": .5,
            "belief": RobotBeliefMap(grid, memory_steps=300),
            "trust": type("Trust", (), {"score": lambda self, sender: trust[sender]})(),
            "fusion": FusionEngine(
                "source_memory",
                lambda sender: trust[sender],
                trust_memory_score=lambda sender: memory[sender],
                trust_threshold=.5,
            ),
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
            "belief": RobotBeliefMap(grid, memory_steps=300),
        },
    )()
    report = ClaimReport("malicious", 0, (2, 2), ClaimType.BLOCKED, 5, 1.0, "attack")
    victim.fusion.add(report, is_malicious=True)
    assert combined_display_grid(victim, world, log, step=5, robots=(attacker, victim))[2, 2] == DISPLAY_R0
    trust[0] = .4
    memory[0] = .4
    assert victim.fusion.operational_weight(report, 6) == 0.0
    assert combined_display_grid(victim, world, log, step=6, robots=(attacker, victim))[2, 2] == DISPLAY_UNKNOWN


def test_source_memory_popup_state_uses_memory_gate():
    config = replace(_config(), visualization=VisualizationConfig(animation=True))
    world, robots, log = run_manifest_rollout(config, _manifest(), "source_memory")
    live = log["live"]
    # Force a display-only snapshot where current trust recovered but Source
    # Memory is still below the operational threshold.
    live["pairwise_trust"][0][1][0] = 0.80
    live["pairwise_source_memory"][0][1][0] = 0.40
    fig, anim = show_belief_maps(log, world, robots, show=False, interval_ms=1)
    anim._func(0)
    trust_axes = [ax for ax in fig.axes if ax.get_title(loc="left").startswith("Robot trust level")]
    assert trust_axes
    table = trust_axes[0].tables[0]
    headers = [table[(0, col)].get_text().get_text() for col in range(5)]
    assert headers == ["Reporter", "Observed by", "Trust", "Memory", "State"]
    target_row = None
    for row in range(1, 7):
        if table[(row, 0)].get_text().get_text() == "R0" and table[(row, 1)].get_text().get_text() == "R1":
            target_row = row
            break
    assert target_row is not None
    assert table[(target_row, 2)].get_text().get_text() == "0.80"
    assert table[(target_row, 3)].get_text().get_text() == "0.40"
    assert table[(target_row, 4)].get_text().get_text() == "IGNORED"
    fig.clf()


def test_reference_heatmap_background_excludes_one_step_dynamic_obstacles():
    config = replace(_config(), visualization=VisualizationConfig(animation=True))
    world, _, log = run_manifest_rollout(config, _manifest(), "source_memory")
    live = log["live"]
    dynamic_cell = tuple(_manifest().obstacle_episodes[0].cells[0]) if _manifest().obstacle_episodes else (2, 2)
    # Force a dynamic color into the recorded truth frame. The reference
    # heatmap must still use static geometry rather than this momentary state.
    live["truth"][0][dynamic_cell] = DISPLAY_DYNAMIC
    fig = show_traffic_heatmap(log, show=False)
    background = np.asarray(fig.axes[0].images[0].get_array())
    assert background[dynamic_cell] != DISPLAY_DYNAMIC
    fig.clf()
