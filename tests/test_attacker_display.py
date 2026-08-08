import numpy as np
import matplotlib.pyplot as plt

import sim2


def _victim(method="source_linked"):
    grid = np.zeros((11, 11), dtype=np.uint8)
    return sim2.GridRobot(
        robot_id=1,
        initial_grid=grid,
        start_cell=(5, 2),
        goal_cell=(5, 8),
        task_queue=[],
        is_malicious=False,
        defense_method=method,
    )


def _fake_claim(victim, cell=(5, 6)):
    report = sim2.PeerReport(
        sender_id=0,
        target_cell=cell,
        claim=sim2.ClaimType.BLOCKED,
        timestamp=0,
        is_malicious=True,
    )
    assert victim.defense_runner.add_report(report)
    victim.belief_map.set_planning_time(0)


def _claim(victim, sender_id, cell, *, malicious, claim=sim2.ClaimType.BLOCKED):
    report = sim2.PeerReport(
        sender_id=sender_id,
        target_cell=cell,
        claim=claim,
        timestamp=0,
        is_malicious=malicious,
    )
    assert victim.defense_runner.add_report(report)
    victim.belief_map.set_planning_time(0)


def test_influential_fake_cell_uses_planner_cost_boundary():
    victim = _victim()
    _fake_claim(victim)

    assert sim2.count_influential_fake_claim_cells(victim, {(5, 6)}, threshold=0.10) == 1
    assert sim2.count_influential_fake_claim_cells(victim, {(5, 6)}, threshold=100.0) == 0


def test_source_linked_influence_tracks_current_trust_not_report_trust():
    victim = _victim("source_linked")
    _fake_claim(victim)

    assert sim2.count_influential_fake_claim_cells(victim, {(5, 6)}) == 1
    victim.trust_model.values[0] = 0.0
    assert sim2.count_influential_fake_claim_cells(victim, {(5, 6)}) == 0


def test_honest_overlap_does_not_create_attacker_attribution():
    victim = _victim()
    _claim(victim, 0, (5, 6), malicious=True)
    _claim(victim, 2, (5, 6), malicious=False)
    victim.trust_model.values[0] = 0.0
    victim.trust_model.values[2] = 1.0

    assert victim.belief_map.traversal_cost((5, 6)) > 1.0
    assert sim2.attacker_cell_cost_delta(victim, (5, 6), 0) == 0.0
    assert sim2.count_influential_fake_claim_cells(
        victim, {(5, 6)}, attacker_id=0
    ) == 0


def test_unknown_and_physical_costs_do_not_create_attacker_attribution():
    victim = _victim()
    _claim(victim, 0, (5, 6), malicious=True)
    victim.trust_model.values[0] = 0.0
    victim.belief_map.belief[5, 6] = sim2.CellState.UNKNOWN
    victim.belief_map.source[5, 6] = "unknown"
    assert victim.belief_map.traversal_cost((5, 6)) >= 3.0
    assert sim2.attacker_cell_cost_delta(victim, (5, 6), 0) == 0.0

    physical = _victim()
    physical.belief_map.initial_prior[5, 6] = sim2.CellState.OCCUPIED_STATIC
    physical.belief_map.belief[5, 6] = sim2.CellState.OCCUPIED_STATIC
    _claim(physical, 0, (5, 6), malicious=True)
    physical.trust_model.values[0] = 0.0
    assert sim2.attacker_cell_cost_delta(physical, (5, 6), 0) == 0.0


def test_route_metrics_do_not_credit_suboptimal_route_without_attack():
    victim = _victim()
    victim.path = [(5, 2), (5, 3), (5, 4), (4, 4), (4, 5), (5, 5), (5, 6), (5, 7), (5, 8)]
    stored_delta, preferred = sim2.route_impact_from_fake_claims(victim, attacker_id=0)
    assert stored_delta == 0.0
    assert preferred is False


def test_source_linked_exact_zero_clears_repeated_fake_attribution():
    victim = _victim()
    for timestamp in range(5):
        report = sim2.PeerReport(
            sender_id=0,
            target_cell=(5, 6),
            claim=sim2.ClaimType.BLOCKED,
            timestamp=timestamp,
            is_malicious=True,
        )
        assert victim.defense_runner.add_report(report)
    victim.trust_model.values[0] = 0.0
    victim.belief_map.set_planning_time(5)
    assert sim2.attacker_cell_cost_delta(victim, (5, 6), 0) == 0.0
    assert sim2.count_influential_fake_claim_cells(
        victim, {(5, 6)}, attacker_id=0
    ) == 0
    victim.path = [(5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)]
    assert sim2.route_impact_from_fake_claims(victim, attacker_id=0) == (0.0, False)


def test_active_fake_claim_ground_truth_is_separate_from_belief():
    log = {
        "reports": [
            {"step": 2, "sender_id": 0, "target_cell": (4, 4), "claim": int(sim2.ClaimType.BLOCKED), "is_malicious": True},
            {"step": 2, "sender_id": 0, "target_cell": (4, 5), "claim": int(sim2.ClaimType.FREE), "is_malicious": True},
        ]
    }
    assert sim2.active_fake_claim_cells(log, 0, 2, max_claim_age=10) == {(4, 4)}
    assert sim2.active_fake_claim_cells(log, 0, 20, max_claim_age=10) == set()


def test_newer_attacker_free_claim_replaces_old_fake_block():
    log = {
        "reports": [
            {"step": 2, "sender_id": 0, "target_cell": (4, 4), "claim": int(sim2.ClaimType.BLOCKED), "is_malicious": True},
            {"step": 3, "sender_id": 0, "target_cell": (4, 4), "claim": int(sim2.ClaimType.FREE), "is_malicious": True},
        ]
    }
    assert sim2.active_fake_claim_cells(log, 0, 3, max_claim_age=10) == set()


def test_animation_trust_panel_has_dedicated_grid_row(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    world, robots, log = sim2.run_simulation(
        max_steps=1,
        tasks_per_robot=1,
        random_seed=15,
    )
    animation = sim2.animate(world, robots, log)
    figure = animation._fig
    try:
        animation._func(0)
        assert len(figure.axes) == 2 + len(robots)
        trust_axis = figure.axes[0]
        map_axes = figure.axes[1:]
        assert not trust_axis.axison
        assert trust_axis.get_position().y0 > max(axis.get_position().y1 for axis in map_axes)
        assert any(text.get_text().startswith("ATTACKER") for text in trust_axis.texts)
    finally:
        plt.close(figure)
