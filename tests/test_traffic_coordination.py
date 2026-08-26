import numpy as np

from map_poisoning.models import DeliveryTask
from map_poisoning.robot import ModularRobot
from map_poisoning.traffic import TrafficState, coordinate_robot_intents, summarize_traffic_events
from map_poisoning.world import World


def _robot(robot_id, start, target):
    grid = np.zeros((12, 12), dtype=np.uint8)
    belief = __import__("map_poisoning.belief", fromlist=["RobotBeliefMap"]).RobotBeliefMap(grid)
    trust = __import__("map_poisoning.trust", fromlist=["ScalarTrustModel"]).ScalarTrustModel()
    fusion = __import__("map_poisoning.fusion", fromlist=["FusionEngine"]).FusionEngine("full_trust", trust.score)
    robot = ModularRobot(
        robot_id,
        start,
        (DeliveryTask("t", target, target),),
        belief,
        trust,
        fusion,
        0.55,
        "accept_all",
    )
    robot.path = [target]
    return robot


def test_same_destination_has_one_approved_intent():
    world = World(np.zeros((12, 12), dtype=np.uint8), ())
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 6), (5, 5))
    approved, events = coordinate_robot_intents([first, second], world, 0)
    assert sum(approved.values()) == 1
    assert any(event["event_type"] == "traffic_vertex_conflict" for event in events)


def test_narrow_corridor_is_single_file():
    grid = np.ones((9, 9), dtype=np.uint8)
    grid[1:8, 1] = 0
    grid[1:8, 7] = 0
    grid[4, 1:8] = 0
    world = World(grid, ())
    inside = _robot(0, (4, 4), (4, 5))
    waiting = _robot(1, (4, 1), (4, 2))
    approved, events = coordinate_robot_intents([inside, waiting], world, 0)
    assert approved[0] is True
    assert approved[1] is False
    assert any(event["event_type"] == "traffic_corridor_conflict" for event in events)


def test_second_robot_already_in_corridor_reverses_out():
    grid = np.ones((9, 9), dtype=np.uint8)
    grid[1:8, 1] = 0
    grid[1:8, 7] = 0
    grid[4, 1:8] = 0
    world = World(grid, ())
    owner = _robot(0, (4, 4), (4, 5))
    intruder = _robot(1, (4, 3), (4, 4))
    approved, events = coordinate_robot_intents([owner, intruder], world, 0)
    assert any(event["event_type"] == "traffic_yield_started" and event["robot_id"] == 1 for event in events)
    assert owner.traffic_mode == "NORMAL"
    assert intruder.traffic_mode == "YIELDING"
    assert approved[0] is True


def test_head_on_swap_yields_immediately():
    world = World(np.zeros((12, 12), dtype=np.uint8), ())
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 5), (5, 4))
    approved, events = coordinate_robot_intents([first, second], world, 0)
    assert any(event["event_type"] == "traffic_swap_conflict" for event in events)
    assert any(event["event_type"] == "traffic_yield_started" for event in events)
    yielding = [robot for robot in (first, second) if robot.traffic_mode == "YIELDING"]
    assert len(yielding) == 1
    assert sum(1 for robot in (first, second) if robot.traffic_mode == "NORMAL") == 1


def test_yielded_swap_unparks_after_partner_moves():
    world = World(np.zeros((12, 12), dtype=np.uint8), ())
    state = TrafficState()
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 5), (5, 4))
    coordinate_robot_intents([first, second], world, 0, state)
    yielder = first if first.traffic_mode == "YIELDING" else second
    partner = second if yielder is first else first
    assert yielder.active_yield_target is not None
    yielder.position = yielder.active_yield_target
    yielder.path = None
    partner.position = (5, 4) if partner.robot_id == 1 else (5, 5)
    partner.path = [(5, 3)] if partner.robot_id == 1 else [(5, 6)]
    _, events = coordinate_robot_intents([first, second], world, 1, state)
    assert yielder.traffic_mode == "NORMAL"
    assert any(event["event_type"] == "traffic_yield_completed" for event in events)


def test_deadlock_summary_counts_unique_paired_episodes_only():
    events = [
        {"event_type": "traffic_deadlock_recovered", "deadlock_id": "orphan"},
        {"event_type": "traffic_deadlock_detected", "deadlock_id": "d1"},
        {"event_type": "traffic_deadlock_detected", "deadlock_id": "d1"},
        {"event_type": "traffic_deadlock_recovered", "deadlock_id": "d1"},
        {"event_type": "traffic_deadlock_detected", "deadlock_id": "d2"},
    ]
    summary = summarize_traffic_events(events)
    assert summary["deadlocks_detected"] == 2
    assert summary["deadlocks_recovered"] == 1
