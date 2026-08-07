import numpy as np

import sim2


def _robot(robot_id, start, target):
    robot = sim2.GridRobot(
        robot_id=robot_id,
        initial_grid=np.zeros((12, 12), dtype=np.uint8),
        start_cell=start,
        task_queue=[sim2.DeliveryTask(pickup=target, dropoff=target)],
    )
    robot.path = [tuple(start), tuple(target)]
    robot.path_index = 0
    robot.current_step = 0
    robot.current_phase = "TEST"
    return robot


def test_lidar_robot_detection_is_not_environment_obstacle_evidence():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    observations, _, visible = world.observe_cells_lidar(
        sim2.cell_to_xy((5, 5)), robot_positions={(5, 7)}
    )
    assert (5, 7) in visible
    assert observations.get((5, 7)) != sim2.CellState.OCCUPIED_DYNAMIC


def test_same_destination_has_one_approved_intent():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 6), (5, 5))
    approved, events = sim2.coordinate_robot_intents([first, second], world, 0)
    assert sum(approved.values()) == 1
    assert any(event["event_type"] == "traffic_vertex_conflict" for event in events)


def test_head_on_swap_is_prevented():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 5), (5, 4))
    approved, events = sim2.coordinate_robot_intents([first, second], world, 0)
    assert sum(approved.values()) == 1
    assert any(event["event_type"] == "traffic_swap_conflict" for event in events)


def test_traffic_wait_does_not_pollute_belief_map():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    first = _robot(0, (5, 4), (5, 5))
    before = first.belief_map.belief.copy(), first.belief_map.source.copy()
    moved, event = first.move_one_cell(world, {(5, 5)})
    assert not moved and event == "traffic_wait"
    assert np.array_equal(first.belief_map.belief, before[0])
    assert np.array_equal(first.belief_map.source, before[1])


def test_repeated_traffic_wait_detects_and_recovers_deadlock():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    first = _robot(0, (5, 4), (5, 5))
    second = _robot(1, (5, 6), (5, 5))
    all_events = []
    traffic_state = {}
    for step in range(sim2.TRAFFIC_DEADLOCK_WAIT_THRESHOLD):
        _, events = sim2.coordinate_robot_intents([first, second], world, step, traffic_state)
        all_events.extend(events)
    assert any(event["event_type"] == "traffic_deadlock_detected" for event in all_events)

    waiting = first if first.traffic_deadlock_active else second
    waiting.path_index = 1  # its route has cleared; it can now yield and recover
    _, events = sim2.coordinate_robot_intents([first, second], world, 20, traffic_state)
    assert any(
        event["event_type"] == "traffic_deadlock_recovered"
        and event["robot_id"] == waiting.robot_id
        for event in events
    )
