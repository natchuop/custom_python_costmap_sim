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
    assert sum(approved.values()) == 0
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
    waiting.position = waiting.active_yield_target  # emulate its approved yield move
    _, parked_events = sim2.coordinate_robot_intents([first, second], world, 20, traffic_state)
    assert waiting.traffic_mode == "YIELDING_PARKED"
    assert not any(event["event_type"] == "traffic_deadlock_recovered" for event in parked_events)

    blocker = second if waiting is first else first
    blocker.position = (4, 6)
    blocker.path = [(4, 6), (4, 7)]
    _, events = sim2.coordinate_robot_intents([first, second], world, 21, traffic_state)
    assert any(
        event["event_type"] == "traffic_deadlock_recovered"
        and event["robot_id"] == waiting.robot_id
        for event in events
    )


def test_yield_target_prefers_distant_branch_over_corridor_cell():
    grid = np.ones((12, 14), dtype=np.uint8)
    for col in range(1, 12):
        grid[6, col] = 0
    for row in range(2, 11):
        grid[row, 1] = 0
    world = sim2.GridWorld(grid)
    robot = _robot(0, (6, 10), (6, 11))
    robot.position_history.clear()
    for col in range(10, 1, -1):
        robot.position_history.append((6, col))
    other = _robot(1, (6, 11), (6, 10))
    target = sim2._traffic_yield_target(robot, [robot, other], world)
    assert target == (6, 1)


def test_completed_robot_is_parked_when_blocking_active_checkpoint():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    idle = _robot(0, (5, 5), (5, 6))
    idle.completed = True
    active = _robot(1, (5, 4), (5, 5))
    approved, events = sim2.coordinate_robot_intents([idle, active], world, 0, {})
    assert any(event.get("reason") == "completed_robot_parking" for event in events)
    assert idle.traffic_mode == "YIELDING"
    assert approved[active.robot_id] is True


def test_commit_uses_frozen_approved_target_after_path_changes():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    robot = _robot(0, (5, 4), (5, 5))
    intent = robot.propose_move_intent()
    robot.path = [(5, 4), (4, 4)]  # stale mutation after coordination
    moved, event = robot.commit_move_intent(intent, world, True)
    assert moved and event == "moved_cell"
    assert robot.position == (5, 5)
    assert robot.intent_commit_mismatches == 0


def test_corridor_topology_identifies_narrow_segment():
    grid = np.ones((12, 14), dtype=np.uint8)
    for col in range(1, 12):
        grid[6, col] = 0
    for row in range(2, 11):
        grid[row, 1] = 0
    by_cell, segments = sim2.build_narrow_corridor_topology(grid)
    assert segments
    assert by_cell[(6, 5)] in segments


def test_real_world_blockage_cancels_frozen_commit_without_substitution():
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[5, 5] = sim2.CellState.OCCUPIED_STATIC
    world = sim2.GridWorld(grid)
    robot = _robot(0, (5, 4), (5, 5))
    intent = robot.propose_move_intent()
    moved, event = robot.commit_move_intent(intent, world, True)
    assert not moved and event == "blocked_world"
    assert robot.position == (5, 4)


def test_perceived_blockage_does_not_override_physical_traffic_commit():
    world = sim2.GridWorld(np.zeros((12, 12), dtype=np.uint8))
    robot = _robot(0, (5, 4), (5, 5))
    robot.belief_map.belief[5, 5] = sim2.CellState.TEMPORARILY_BLOCKED
    intent = robot.propose_move_intent()
    moved, event = robot.commit_move_intent(intent, world, True)
    assert moved and event == "moved_cell"
