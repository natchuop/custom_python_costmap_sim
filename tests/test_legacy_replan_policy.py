import sim2


def test_path_invalid_replan_uses_cooldown():
    grid = sim2.make_demo_static_grid()
    robot = sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(2, 2),
        goal_cell=(2, 3),
        defense_method="full_trust",
    )
    robot.current_step = 10
    robot.path = []
    robot.path_index = 0
    assert robot.should_replan_for_path_state() is True
    robot.plan_path(reason="path_invalid_or_empty", timestamp=10)
    assert robot.replan_count == 1
    robot.current_step = 11
    robot.replanned_this_step = False
    assert robot.should_replan_for_path_state() is False


def test_direct_free_report_does_not_affect_remaining_route():
    grid = sim2.make_demo_static_grid()
    robot = sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(2, 2),
        goal_cell=(2, 4),
        defense_method="source_linked",
    )
    robot.path = [(2, 2), (2, 3), (2, 4)]
    robot.path_index = 0
    robot.belief_map.update_from_sensor({(2, 3): sim2.CellState.FREE}, 1)
    report = sim2.PeerReport(0, (2, 3), sim2.ClaimType.BLOCKED, 0, is_malicious=True)
    assert robot.reports_affect_remaining_route([report]) is False
