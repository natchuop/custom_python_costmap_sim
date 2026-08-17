import sim2


def test_legacy_direct_free_suppresses_peer_route_cost():
    grid = sim2.make_demo_static_grid()
    robot = sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(2, 2),
        goal_cell=(2, 3),
        defense_method="full_trust",
    )
    cell = (2, 3)
    robot.belief_map.set_planning_time(0)
    report = sim2.PeerReport(0, cell, sim2.ClaimType.BLOCKED, 0)
    robot.defense_runner.add_report(report)
    assert robot.belief_map.traversal_cost(cell) > 1.0
    robot.belief_map.update_from_sensor({cell: sim2.CellState.FREE}, 1)
    robot.belief_map.set_planning_time(1)
    assert robot.belief_map.traversal_cost(cell) == 1.0


def test_majority_direct_free_suppresses_peer_hard_block():
    grid = sim2.make_demo_static_grid()
    robot = sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(2, 2),
        goal_cell=(2, 3),
        defense_method="majority_vote",
    )
    cell = (2, 3)
    robot.belief_map.set_planning_time(0)
    report = sim2.PeerReport(0, cell, sim2.ClaimType.BLOCKED, 0, is_malicious=True)
    robot.defense_runner.add_report(report)
    assert robot.belief_map.is_blocked_for_planning(cell)
    robot.belief_map.update_from_sensor({cell: sim2.CellState.FREE}, 1)
    robot.belief_map.set_planning_time(1)
    assert not robot.belief_map.is_blocked_for_planning(cell)
    assert robot.belief_map.traversal_cost(cell) == 1.0


def test_source_linked_replan_risk_ignores_directly_free_route_cell():
    grid = sim2.make_demo_static_grid()
    robot = sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(2, 2),
        goal_cell=(2, 3),
        defense_method="source_linked",
    )
    robot.path = [(2, 3)]
    robot.path_index = 0
    robot.belief_map.update_from_sensor({(2, 3): sim2.CellState.FREE}, 1)
    assert robot._source_linked_route_cells() == []
