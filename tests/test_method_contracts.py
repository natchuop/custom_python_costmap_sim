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


def report(report_id, sender, claim, step=0):
    return ClaimReport(report_id, sender, (2, 2), claim, step, step, step)


def test_trust_threshold_method_zeros_untrusted_influence():
    trust = {0: 0.80}
    engine = FusionEngine("trust_threshold", lambda sender: trust[sender], trust_threshold=0.55)
    engine.add(report("blocked", 0, ClaimType.BLOCKED))
    high = engine.evidence((2, 2), 0)
    assert high > 0
    assert engine.blocked((2, 2), 0)
    trust[0] = 0.20
    assert engine.evidence((2, 2), 0) == 0
    assert not engine.blocked((2, 2), 0)


def test_trust_threshold_trusted_blocked_report_hard_blocks():
    engine = FusionEngine("trust_threshold", lambda _: 0.70, trust_threshold=0.55)
    engine.add(report("blocked", 1, ClaimType.BLOCKED))
    assert engine.blocked((2, 2), 0)


def test_primary_weighting_contracts_and_active_replacement():
    trust = {0: .7}
    score = lambda sender: trust[sender]
    engines = {name: FusionEngine(name, score, decay_rate=.01) for name in ("full_trust", "trust_fused", "source_linked")}
    for engine in engines.values():
        engine.add(report("one", 0, ClaimType.BLOCKED))
        engine.add(report("two", 0, ClaimType.BLOCKED, 1))
        assert len(engine.claims[(2, 2)]) == 1
    before = {name: engine.evidence((2, 2), 1) for name, engine in engines.items()}
    trust[0] = .1
    assert engines["full_trust"].evidence((2, 2), 1) == before["full_trust"]
    assert engines["trust_fused"].evidence((2, 2), 1) == before["trust_fused"]
    assert engines["source_linked"].evidence((2, 2), 1) < before["source_linked"]
    assert before["full_trust"] >= before["trust_fused"]


def test_majority_is_one_vote_per_sender_and_discrete():
    engine = FusionEngine("majority_vote", lambda _: .1)
    engine.add(report("a", 0, ClaimType.BLOCKED))
    engine.add(report("b", 0, ClaimType.BLOCKED, 1))
    engine.add(report("c", 1, ClaimType.FREE, 1))
    assert engine.vote((2, 2), 1) == 0
    assert not engine.blocked((2, 2), 1)
    engine.add(report("d", 2, ClaimType.BLOCKED, 1))
    assert engine.blocked((2, 2), 1)
    assert math.isinf(engine.routing_cost((2, 2), 1))


def test_direct_observations_ignore_future_steps():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8))
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 5))
    assert belief.observation_status((2, 2), 2) == (None, "unknown")
    assert belief.observation_status((2, 2), 5) == (ClaimType.FREE, "fresh")


def test_direct_free_and_blocked_override_peer_evidence():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8))
    fusion = FusionEngine("full_trust", lambda _: 1.)
    fusion.add(report("r", 0, ClaimType.BLOCKED))
    assert fusion.routing_cost((2, 2), 0) > 1
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 1))
    assert belief.traversal_cost((2, 2), 1, fusion) == 1
    belief.observe(DirectObservation(1, (2, 2), ClaimType.BLOCKED, 2))
    assert math.isinf(belief.traversal_cost((2, 2), 2, fusion))


def test_stale_direct_free_allows_peer_blocked_cost():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=2)
    fusion = FusionEngine("source_linked", lambda _: 1.0, cost_scale=40.0)
    fusion.add(report("r", 0, ClaimType.BLOCKED), is_malicious=True)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 0))
    assert belief.traversal_cost((2, 2), 2, fusion) == 1.0
    assert belief.traversal_cost((2, 2), 3, fusion) > 1.0
    assert not belief.has_direct_free((2, 2), 3)


def test_display_state_outlasts_planning_memory():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=2)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 0))
    assert belief.direct_state((2, 2), 5) is None
    assert belief.display_state((2, 2), 5, max_age=10) == ClaimType.FREE
    assert belief.display_state((2, 2), 11, max_age=10) is None


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
