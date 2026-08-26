import pytest
import numpy as np

from map_poisoning.config import SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType, DeliveryTask, DirectObservation, VerificationOutcome
from map_poisoning.scenario import author_manifest
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.robot import ModularRobot
from map_poisoning.trust import BayesianTrustModel, ScalarTrustModel


def test_same_seed_same_manifest():
    assert author_manifest(SimulationConfig()).to_dict() == author_manifest(SimulationConfig()).to_dict()


def test_source_memory_is_retroactive_and_trust_fused_is_not():
    trust = {0: .7}
    memory = {0: .7}
    score = lambda sender: trust[sender]
    report = ClaimReport("r", 0, (1, 1), ClaimType.BLOCKED, 0, 1.0)
    source = FusionEngine("source_memory", score, trust_memory_score=lambda sender: memory[sender])
    fused = FusionEngine("trust_fused", score)
    source.add(report)
    fused.add(report)
    before_source = source.evidence((1, 1), 0)
    before_fused = fused.evidence((1, 1), 0)
    trust[0] = .1
    memory[0] = .1
    assert source.evidence((1, 1), 0) < before_source
    assert fused.evidence((1, 1), 0) == pytest.approx(before_fused)


def test_source_memory_ignores_distrusted_source_then_recovers_to_report_ceiling():
    trust = {0: .7}
    memory = {0: .7}
    engine = FusionEngine("source_memory", lambda s: trust[s], trust_memory_score=lambda s: memory[s], trust_threshold=.5)
    report = ClaimReport("r", 0, (1, 1), ClaimType.BLOCKED, 0, 1.0)
    engine.add(report)
    trust[0] = .2; memory[0] = .2
    assert engine.evidence((1, 1), 0) == 0.0
    assert engine.operational_weight(report, 0) == 0.0
    trust[0] = .9; memory[0] = .35
    assert engine.evidence((1, 1), 0) == 0.0
    memory[0] = .55
    assert engine.evidence((1, 1), 0) == pytest.approx(.55)
    memory[0] = .9
    assert engine.evidence((1, 1), 0) == pytest.approx(.7)


def test_trust_fused_ignores_new_reports_received_while_distrusted_only():
    trust = {0: .7}
    engine = FusionEngine("trust_fused", lambda s: trust[s], trust_threshold=.5)
    old_report = ClaimReport("old", 0, (1, 1), ClaimType.BLOCKED, 0, 1.0)
    engine.add(old_report)
    old_weight = engine.operational_weight(old_report, 0)
    trust[0] = .2
    assert engine.operational_weight(old_report, 0) == pytest.approx(old_weight)
    new_report = ClaimReport("new", 0, (1, 2), ClaimType.BLOCKED, 1, 1.0)
    engine.add(new_report)
    assert engine.operational_weight(new_report, 1) == 0.0
    assert engine.evidence((1, 2), 1) == 0.0


def test_recipients_keep_independent_belief_and_trust_state():
    grid = np.zeros((8, 8), dtype=np.uint8)
    task = (DeliveryTask("t", (1, 2), (6, 6)),)
    first_trust, second_trust = BayesianTrustModel(), BayesianTrustModel()
    first = ModularRobot(1, (1, 1), task, RobotBeliefMap(grid), first_trust,
                         FusionEngine("source_memory", first_trust.score, trust_memory_score=first_trust.memory_score), .5, "accept_all")
    second = ModularRobot(2, (1, 1), task, RobotBeliefMap(grid), second_trust,
                          FusionEngine("source_memory", second_trust.score, trust_memory_score=second_trust.memory_score), .5, "accept_all")
    report = ClaimReport("r", 0, (3, 3), ClaimType.BLOCKED, 0, 1.0)
    first.receive(report); first.process_inbox(0)
    assert first.fusion.claims[(3, 3)]
    assert (3, 3) not in second.fusion.claims
    first.trust.update(0, VerificationOutcome.CONTRADICTED_FRESH)
    assert first.trust.score(0) < second.trust.score(0)


def test_scalar_batch_validation_changes_trust_once_per_sender():
    grid = np.zeros((8, 8), dtype=np.uint8)
    task = (DeliveryTask("t", (1, 2), (6, 6)),)
    trust = ScalarTrustModel(initial=0.80)
    robot = ModularRobot(1, (1, 1), task, RobotBeliefMap(grid), trust,
                         FusionEngine("trust_threshold", trust.score, trust_threshold=0.5), .5, "accept_all")
    for idx, cell in enumerate(((2, 2), (2, 3), (2, 4))):
        robot.receive(ClaimReport(f"free-{idx}", 0, cell, ClaimType.FREE, 0, 1.0))
    robot.process_inbox(0)
    observations = [DirectObservation(1, cell, ClaimType.FREE, 1, 1.0) for cell in ((2, 2), (2, 3), (2, 4))]
    robot.verify(observations, 1)
    assert len(robot.last_trust_batches) == 1
    assert robot.trust.score(0) > .80
