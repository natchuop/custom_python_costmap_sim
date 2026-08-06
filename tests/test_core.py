from dataclasses import replace
from map_poisoning.config import FusionConfig, SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType
from map_poisoning.scenario import author_manifest

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
