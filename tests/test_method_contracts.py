import math
import numpy as np

from map_poisoning.belief import RobotBeliefMap
from map_poisoning.config import SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType, DirectObservation, VerificationOutcome
from map_poisoning.scenario import author_manifest
from map_poisoning.trust import BayesianTrustModel
from map_poisoning.map_io import default_warehouse_map
from map_poisoning.planning import astar


def report(report_id, sender, claim, step=0, confidence=1.0):
    return ClaimReport(report_id, sender, (2, 2), claim, step, confidence)


def test_trust_threshold_method_zeros_untrusted_influence():
    trust = {0: 0.80}
    engine = FusionEngine("trust_threshold", lambda sender: trust[sender], trust_threshold=0.55)
    engine.add(report("blocked", 0, ClaimType.BLOCKED))
    assert engine.evidence((2, 2), 0) > 0
    assert engine.blocked((2, 2), 0)
    trust[0] = 0.20
    assert engine.evidence((2, 2), 0) == 0
    assert not engine.blocked((2, 2), 0)


def test_primary_weighting_contracts_and_active_replacement():
    trust = {0: .7}; memory = {0: .7}
    engines = {
        "full_trust": FusionEngine("full_trust", lambda s: trust[s]),
        "trust_fused": FusionEngine("trust_fused", lambda s: trust[s]),
        "source_memory": FusionEngine("source_memory", lambda s: trust[s], trust_memory_score=lambda s: memory[s]),
    }
    for engine in engines.values():
        engine.add(report("one", 0, ClaimType.BLOCKED))
        engine.add(report("two", 0, ClaimType.BLOCKED, 1))
        assert len(engine.claims[(2, 2)]) == 1
    before = {name: engine.evidence((2, 2), 1) for name, engine in engines.items()}
    trust[0] = .1; memory[0] = .1
    assert engines["full_trust"].evidence((2, 2), 1) == before["full_trust"]
    assert engines["trust_fused"].evidence((2, 2), 1) == before["trust_fused"]
    assert engines["source_memory"].evidence((2, 2), 1) < before["source_memory"]


def test_primary_continuous_methods_share_linear_aging():
    trust = {0: .8}; memory = {0: .8}
    engines = [
        FusionEngine("full_trust", lambda s: trust[s], max_claim_age=300),
        FusionEngine("trust_fused", lambda s: trust[s], max_claim_age=300),
        FusionEngine("source_memory", lambda s: trust[s], trust_memory_score=lambda s: memory[s], max_claim_age=300),
    ]
    for engine in engines:
        engine.add(report("r", 0, ClaimType.BLOCKED, 0))
        assert engine.evidence((2, 2), 150) < engine.evidence((2, 2), 0)
        assert engine.evidence((2, 2), 300) == 0


def test_majority_is_one_vote_per_sender_discrete_and_tie_unknown():
    engine = FusionEngine("majority_vote", lambda _: .1, max_claim_age=300, majority_unknown_cost=3)
    engine.add(report("a", 0, ClaimType.BLOCKED))
    engine.add(report("b", 0, ClaimType.BLOCKED, 1))
    engine.add(report("c", 1, ClaimType.FREE, 1))
    assert engine.vote((2, 2), 1) == 0
    assert not engine.blocked((2, 2), 1)
    assert engine.routing_cost((2, 2), 1) == 3
    engine.add(report("d", 2, ClaimType.BLOCKED, 1))
    assert math.isinf(engine.routing_cost((2, 2), 1))
    assert engine.vote((2, 2), 301) == 0


def test_current_direct_observation_is_authoritative_then_becomes_memory():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=300)
    fusion = FusionEngine("full_trust", lambda _: 1.)
    fusion.add(report("r", 0, ClaimType.BLOCKED))
    belief.begin_scan(1)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 1, 1.0))
    assert belief.observation_status((2, 2), 1) == (ClaimType.FREE, "current")
    assert belief.traversal_cost((2, 2), 1, fusion) == 1
    belief.begin_scan(2)
    assert belief.observation_status((2, 2), 2) == (ClaimType.FREE, "memory")
    assert belief.traversal_cost((2, 2), 2, fusion) > 1


def test_remembered_block_is_soft_and_expires():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=300)
    fusion = FusionEngine("full_trust", lambda _: 1.)
    belief.begin_scan(0)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.BLOCKED, 0, 1.0))
    assert math.isinf(belief.traversal_cost((2, 2), 0, fusion))
    belief.begin_scan(1)
    cost1 = belief.traversal_cost((2, 2), 1, fusion)
    cost150 = belief.traversal_cost((2, 2), 150, fusion)
    assert math.isfinite(cost1) and cost1 > cost150 > 3
    assert belief.display_state((2, 2), 300) is None


def test_manifest_has_exact_three_robot_team():
    manifest = author_manifest(SimulationConfig())
    assert manifest.malicious_robot_id == 0
    assert manifest.benign_robot_ids == (1, 2)


def test_ambiguous_verification_does_not_reward_bayesian_trust():
    trust = BayesianTrustModel()
    before = trust.score(0)
    trust.update(0, VerificationOutcome.TEMPORALLY_AMBIGUOUS_OR_EXPIRED)
    assert trust.score(0) == before


def test_default_map_has_an_attacker_escape_corridor():
    grid = default_warehouse_map()
    assert not grid[10:13, 8].any()
    path = astar((6, 8), (14, 8), lambda cell: float("inf") if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]) or grid[cell] else 1.)
    assert path is not None
