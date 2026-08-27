import math
from dataclasses import replace

import numpy as np
import pytest

from map_poisoning.cli import config_from_args, parser
from map_poisoning.config import PRIMARY_METHODS
from map_poisoning.models import ClaimReport, ClaimType, DeliveryTask, DirectObservation
from map_poisoning.sensing import lidar_observations, sensor_confidence_for_distance
from map_poisoning.trust import BayesianTrustModel
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.fusion import FusionEngine
from map_poisoning.robot import ModularRobot


def test_primary_method_order_and_headless_compare_defaults():
    assert PRIMARY_METHODS == ("latest_report", "majority_vote", "full_trust", "trust_fused", "source_memory")
    args = parser().parse_args(["--headless", "--compare", "--no-plots"])
    config = config_from_args(args)
    assert config.comparison_methods == PRIMARY_METHODS


def test_lidar_is_circular_and_occluded_by_first_obstacle():
    grid = np.zeros((15, 15), dtype=np.uint8)
    origin = (7, 7)
    grid[7, 9] = 1
    readings = lidar_observations(grid, origin)
    assert (7, 9) in readings
    assert readings[(7, 9)].claim == ClaimType.BLOCKED
    assert (7, 10) not in readings
    assert (7, 12) not in readings
    assert (10, 11) in readings  # 3-4-5 Euclidean boundary
    assert (11, 11) not in readings  # sqrt(32) > 5


def test_lidar_confidence_reaches_point_six_at_range_five():
    assert sensor_confidence_for_distance(1.0) == pytest.approx(1.0)
    assert sensor_confidence_for_distance(2.0) == pytest.approx(.9)
    assert sensor_confidence_for_distance(3.0) == pytest.approx(.8)
    assert sensor_confidence_for_distance(4.0) == pytest.approx(.7)
    assert sensor_confidence_for_distance(5.0) == pytest.approx(.6)


def test_confidence_change_can_trigger_resend_without_claim_change():
    grid = np.zeros((8, 8), dtype=np.uint8)
    trust = BayesianTrustModel()
    robot = ModularRobot(
        1, (1, 1), (DeliveryTask("t", (1, 2), (6, 6)),),
        RobotBeliefMap(grid), trust,
        FusionEngine("full_trust", trust.score), .5, "accept_all",
        confidence_resend_delta=.10,
    )
    cell = (2, 2)
    assert robot.should_share_observation(cell, ClaimType.FREE, 0, .60)
    assert not robot.should_share_observation(cell, ClaimType.FREE, 1, .65)
    assert robot.should_share_observation(cell, ClaimType.FREE, 2, .70)


def test_bayesian_evidence_is_capped_and_two_strong_batches_distrust():
    trust = BayesianTrustModel(alpha0=9, beta0=1, evidence_cap=12, confirmation_multiplier=1, contradiction_multiplier=6)
    for _ in range(30):
        trust.update_batch(0, 1.0, 0.0)
    alpha, beta = trust.values[0]
    assert alpha + beta == pytest.approx(12.0)
    assert trust.score(0) > .90
    trust.update_batch(0, 0.0, 1.0)
    assert trust.score(0) > .5
    trust.update_batch(0, 0.0, 1.0)
    # The tuned default should distrust after about two strong pure batches;
    # mixed real scans commonly require a few more batches.
    assert trust.score(0) < .5


def test_source_memory_drops_immediately_and_recovers_slowly():
    trust = BayesianTrustModel(memory_recovery_rate=.05)
    sender = 0
    # Materially lower current trust first.
    trust.update_batch(sender, 0.0, 1.0)
    low = trust.score(sender)
    assert trust.memory_score(sender) == pytest.approx(low)
    old_memory = trust.memory_score(sender)
    trust.update_batch(sender, 1.0, 0.0)
    current = trust.score(sender)
    memory = trust.memory_score(sender)
    assert old_memory < memory < current


def test_direct_memory_softens_then_expires_at_300():
    grid = np.zeros((7, 7), dtype=np.uint8)
    belief = RobotBeliefMap(grid, memory_steps=300)
    fusion = FusionEngine("full_trust", lambda _: 1.0, max_claim_age=300)
    belief.begin_scan(0)
    belief.observe(DirectObservation(1, (3, 3), ClaimType.BLOCKED, 0, 1.0))
    assert math.isinf(belief.traversal_cost((3, 3), 0, fusion))
    belief.begin_scan(1)
    early = belief.traversal_cost((3, 3), 1, fusion)
    late = belief.traversal_cost((3, 3), 299, fusion)
    assert math.isfinite(early)
    assert early > late >= 3.0
    assert belief.observation_status((3, 3), 300)[1] == "unknown"


def test_peer_claim_expires_at_exact_lifetime_boundary():
    engine = FusionEngine("majority_vote", lambda _: 1.0, max_claim_age=300, majority_unknown_cost=3)
    engine.add(ClaimReport("r", 0, (2, 2), ClaimType.BLOCKED, 0, 1.0))
    assert engine.vote((2, 2), 299) == 1
    assert engine.vote((2, 2), 300) == 0
    assert engine.routing_cost((2, 2), 300) == 3


def _validation_robot():
    grid = np.zeros((8, 8), dtype=np.uint8)
    trust = BayesianTrustModel()
    robot = ModularRobot(
        1, (1, 1), (DeliveryTask("t", (1, 2), (6, 6)),),
        RobotBeliefMap(grid, memory_steps=300), trust,
        FusionEngine("source_memory", trust.score, trust_memory_score=trust.memory_score, max_claim_age=300),
        .5, "accept_all",
    )
    return robot


def test_next_step_validation_can_use_same_time_sensor_snapshot():
    robot = _validation_robot()
    cell = (3, 3)
    robot.belief.begin_scan(10)
    robot.belief.observe(DirectObservation(1, cell, ClaimType.FREE, 10, .8))
    report = ClaimReport("peer", 0, cell, ClaimType.FREE, 10, .9)
    robot.pending[report.report_id] = report
    results = robot.verify([], 11)
    assert len(results) == 1
    assert results[0][1].value == "confirmed"


def test_validation_does_not_compare_report_to_older_direct_memory():
    robot = _validation_robot()
    cell = (3, 3)
    robot.belief.begin_scan(9)
    robot.belief.observe(DirectObservation(1, cell, ClaimType.BLOCKED, 9, .8))
    report = ClaimReport("peer", 0, cell, ClaimType.FREE, 10, .9)
    robot.pending[report.report_id] = report
    assert robot.verify([], 11) == []
    assert report.report_id in robot.pending


def test_current_lidar_block_on_route_triggers_immediate_replan():
    grid = np.zeros((7, 7), dtype=np.uint8)
    trust = BayesianTrustModel()
    robot = ModularRobot(
        1, (3, 1), (DeliveryTask("t", (3, 5), (5, 5)),),
        RobotBeliefMap(grid, memory_steps=300), trust,
        FusionEngine("full_trust", trust.score, max_claim_age=300),
        .5, "accept_all",
    )
    assert robot.replan(0, "initial")
    assert robot.path and robot.path[0] == (3, 2)
    robot.belief.begin_scan(1)
    robot.belief.observe(DirectObservation(1, (3, 2), ClaimType.BLOCKED, 1, 1.0))
    assert robot.should_replan_for_path_state(1)
    robot.replan(1, "path_invalid_or_empty")
    assert robot.path and robot.path[0] != (3, 2)


def test_current_lidar_block_bypasses_path_invalid_replan_cooldown():
    grid = np.zeros((7, 7), dtype=np.uint8)
    trust = BayesianTrustModel()
    robot = ModularRobot(
        1, (3, 1), (DeliveryTask("t", (3, 5), (5, 5)),),
        RobotBeliefMap(grid, memory_steps=300), trust,
        FusionEngine("full_trust", trust.score, max_claim_age=300),
        .5, "accept_all",
    )
    assert robot.replan(0, "initial")
    robot.last_path_invalid_replan_step = 5
    robot.belief.begin_scan(6)
    robot.belief.observe(DirectObservation(1, robot.path[0], ClaimType.BLOCKED, 6, 1.0))
    assert robot.should_replan_for_path_state(6)


def test_live_map_uses_white_for_valid_free_and_gray_for_unknown_or_expired():
    from map_poisoning.live_view import DISPLAY_FREE, DISPLAY_UNKNOWN, local_display_grid
    from map_poisoning.world import World
    grid = np.zeros((7, 7), dtype=np.uint8)
    robot = _validation_robot()
    world = World(grid, ())
    robot.belief.begin_scan(0)
    robot.belief.observe(DirectObservation(1, (3, 3), ClaimType.FREE, 0, 1.0))
    now = local_display_grid(robot, world, 0)
    assert now[3, 3] == DISPLAY_FREE
    assert now[4, 4] == DISPLAY_UNKNOWN
    expired = local_display_grid(robot, world, 300)
    assert expired[3, 3] == DISPLAY_UNKNOWN


def test_observation_lifetime_cli_controls_direct_and_peer_memory():
    args = parser().parse_args(["--headless", "--observation-lifetime", "150", "--no-plots"])
    config = config_from_args(args)
    assert config.observation_lifetime_steps == 150
    assert config.fusion.max_claim_age == 150


def test_attack_audit_label_does_not_change_validation_or_trust():
    def run(event_id):
        robot = _validation_robot()
        cell = (3, 3)
        report = ClaimReport("peer", 0, cell, ClaimType.BLOCKED, 10, .9, event_id)
        robot.pending[report.report_id] = report
        observed = DirectObservation(1, cell, ClaimType.FREE, 11, .9)
        robot.belief.begin_scan(11)
        robot.belief.observe(observed)
        result = robot.verify([observed], 11)
        return result[0][1], robot.trust.score(0)
    assert run(None) == run("attack-event")


def test_cross_epoch_mismatch_retracts_stale_claim_without_punishing_sender():
    robot = _validation_robot()
    robot.environment_change_period_steps = 150
    cell = (3, 3)
    report = ClaimReport("honest-old", 2, cell, ClaimType.BLOCKED, 149, .9)
    robot.fusion.add(report)
    robot.pending[report.report_id] = report
    before = robot.trust.score(2)
    observed = DirectObservation(1, cell, ClaimType.FREE, 150, .9)
    robot.belief.begin_scan(150)
    robot.belief.observe(observed)
    results = robot.verify([observed], 150)
    assert len(results) == 1
    assert results[0][1].value == "temporally_ambiguous_or_expired"
    assert robot.trust.score(2) == pytest.approx(before)
    assert not robot.fusion.claims_at(cell)


def test_same_epoch_mismatch_is_attributable_and_penalizes_sender():
    robot = _validation_robot()
    robot.environment_change_period_steps = 150
    cell = (3, 3)
    report = ClaimReport("false-fresh", 0, cell, ClaimType.BLOCKED, 120, .9)
    robot.fusion.add(report)
    robot.pending[report.report_id] = report
    before = robot.trust.score(0)
    observed = DirectObservation(1, cell, ClaimType.FREE, 130, .9)
    robot.belief.begin_scan(130)
    robot.belief.observe(observed)
    results = robot.verify([observed], 130)
    assert len(results) == 1
    assert results[0][1].value == "contradicted_fresh"
    assert robot.trust.score(0) < before
    assert not robot.fusion.claims_at(cell)


def test_exact_previous_scan_wins_over_changed_current_epoch_state():
    robot = _validation_robot()
    robot.environment_change_period_steps = 150
    cell = (3, 3)
    # Simulate a recipient that independently saw BLOCKED at the report time.
    exact = DirectObservation(1, cell, ClaimType.BLOCKED, 149, .9)
    robot.previous_scan_observations[cell] = exact
    report = ClaimReport("same-time", 2, cell, ClaimType.BLOCKED, 149, .9)
    robot.fusion.add(report)
    robot.pending[report.report_id] = report
    before = robot.trust.score(2)
    # The world changes at 150 and the current scan now sees FREE.
    current = DirectObservation(1, cell, ClaimType.FREE, 150, .9)
    results = robot.verify([current], 150)
    assert len(results) == 1
    assert results[0][1].value == "confirmed"
    assert robot.trust.score(2) > before
