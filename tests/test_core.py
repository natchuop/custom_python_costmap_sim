from dataclasses import replace
from map_poisoning.config import FusionConfig, SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType, DeliveryTask, VerificationOutcome
from map_poisoning.scenario import author_manifest
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.robot import ModularRobot
from map_poisoning.trust import BayesianTrustModel
import numpy as np

def test_same_seed_same_manifest():
    assert author_manifest(SimulationConfig()).to_dict() == author_manifest(SimulationConfig()).to_dict()

def test_source_linked_is_retroactive_and_trust_fused_is_not():
    trust={0:.7}; score=lambda sender: trust[sender]
    report=ClaimReport("r",0,(1,1),ClaimType.BLOCKED,0,0)
    linked=FusionEngine("source_linked",score); fused=FusionEngine("trust_fused",score)
    linked.add(report); fused.add(report); before_linked=linked.evidence((1,1),0); before_fused=fused.evidence((1,1),0)
    trust[0]=.1
    assert linked.evidence((1,1),0) < before_linked
    assert fused.evidence((1,1),0) == before_fused

def test_fusion_effect_delta_can_be_collected():
    trust={0:.7}; score=lambda sender: trust[sender]
    report=ClaimReport("r",0,(1,1),ClaimType.BLOCKED,0,0)
    linked=FusionEngine("source_linked",score); fused=FusionEngine("trust_fused",score)
    linked.add(report); fused.add(report)
    linked_before, fused_before = linked.evidence((1,1),0), fused.evidence((1,1),0)
    trust[0]=.1
    assert linked.evidence((1,1),0) - linked_before < 0
    assert fused.evidence((1,1),0) - fused_before == 0

def test_recipients_keep_independent_belief_and_trust_state():
    grid=np.zeros((8,8),dtype=np.uint8)
    task=(DeliveryTask("t",(1,2),(6,6)),)
    first_trust, second_trust=BayesianTrustModel(), BayesianTrustModel()
    first=ModularRobot(1,(1,1),task,RobotBeliefMap(grid),first_trust,FusionEngine("source_linked",first_trust.score),.55,"auto_soft")
    second=ModularRobot(2,(1,1),task,RobotBeliefMap(grid),second_trust,FusionEngine("source_linked",second_trust.score),.55,"auto_soft")
    report=ClaimReport("r",0,(3,3),ClaimType.BLOCKED,0,0)
    first.receive(report); first.process_inbox(0)
    assert first.fusion.claims[(3,3)]
    assert (3,3) not in second.fusion.claims
    first.trust.update(0, VerificationOutcome.CONTRADICTED_FRESH)
    assert first.trust.score(0) < second.trust.score(0)
