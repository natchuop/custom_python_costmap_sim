import math
import numpy as np
import pytest

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


def test_primary_trust_methods_use_the_same_linear_age_factor():
    for method in ("full_trust", "trust_fused", "source_memory"):
        engine = FusionEngine(
            method,
            lambda _: 1.0,
            trust_memory_score=lambda _: 1.0,
            max_claim_age=100,
        )
        item = report(f"age-{method}", 0, ClaimType.BLOCKED)
        engine.add(item)
        initial = engine.operational_weight(item, 0)
        midpoint = engine.operational_weight(item, 50)
        assert initial > 0.0
        assert midpoint == pytest.approx(initial * 0.5)


def test_majority_is_one_vote_per_sender_discrete_and_tie_unknown():
    engine = FusionEngine("majority_vote", lambda _: .1, max_claim_age=300, unknown_traversal_cost=3)
    engine.add(report("a", 0, ClaimType.BLOCKED))
    engine.add(report("b", 0, ClaimType.BLOCKED, 1))
    engine.add(report("c", 1, ClaimType.FREE, 1))
    assert engine.vote((2, 2), 1) == 0
    assert not engine.blocked((2, 2), 1)
    assert engine.routing_cost((2, 2), 1) == 3
    engine.add(report("d", 2, ClaimType.BLOCKED, 1))
    assert math.isinf(engine.routing_cost((2, 2), 1))
    assert engine.vote((2, 2), 301) == 0


def test_latest_report_is_categorical_trust_agnostic_and_tie_unknown():
    trust = {0: .01, 1: .99}
    engine = FusionEngine("latest_report", lambda sender: trust[sender], max_claim_age=300, unknown_traversal_cost=3)
    engine.add(report("old-free", 1, ClaimType.FREE, 4))
    engine.add(report("new-blocked", 0, ClaimType.BLOCKED, 5))
    assert math.isinf(engine.routing_cost((2, 2), 5))
    trust[0] = 0.0
    assert math.isinf(engine.routing_cost((2, 2), 5))
    engine.add(report("newer-free", 1, ClaimType.FREE, 6))
    assert engine.routing_cost((2, 2), 6) == 1.0
    engine.add(report("same-time-blocked", 0, ClaimType.BLOCKED, 6))
    assert engine.routing_cost((2, 2), 6) == 3.0
    assert not engine.blocked((2, 2), 6)
    assert engine.evidence((2, 2), 306) == 0.0


def test_current_lidar_overrides_latest_report():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=300)
    fusion = FusionEngine("latest_report", lambda _: 0.0)
    fusion.add(report("blocked", 0, ClaimType.BLOCKED, 1))
    belief.begin_scan(1)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.FREE, 1, 1.0))
    assert belief.traversal_cost((2, 2), 1, fusion) == 1.0


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


def test_remembered_direct_block_stays_hard_until_cleared_or_expired():
    belief = RobotBeliefMap(np.zeros((6, 6), dtype=np.uint8), memory_steps=300)
    fusion = FusionEngine("full_trust", lambda _: 1.)
    belief.begin_scan(0)
    belief.observe(DirectObservation(1, (2, 2), ClaimType.BLOCKED, 0, 1.0))
    assert math.isinf(belief.traversal_cost((2, 2), 0, fusion))
    belief.begin_scan(1)
    assert math.isinf(belief.traversal_cost((2, 2), 1, fusion))
    assert math.isinf(belief.traversal_cost((2, 2), 150, fusion))
    assert belief.display_state((2, 2), 300) is None
    assert not math.isinf(belief.traversal_cost((2, 2), 300, fusion))


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


def test_primary_probabilistic_methods_hard_block_fresh_high_probability_blocked_evidence():
    for method in ("full_trust", "trust_fused", "source_memory"):
        engine = FusionEngine(
            method,
            lambda _: 1.0,
            trust_memory_score=lambda _: 1.0,
            blocked_probability_threshold=0.70,
            max_claim_age=300,
        )
        engine.add(report(f"{method}-blocked", 0, ClaimType.BLOCKED, 0, 1.0))
        assert engine.probability((2, 2), 0) > 0.70
        assert engine.blocked((2, 2), 0)
        assert math.isinf(engine.routing_cost((2, 2), 0)) is False  # soft cost is still available for scoring
        # Linear aging eventually drops the same report below the hard-block
        # threshold without changing its underlying fusion definition.
        assert not engine.blocked((2, 2), 100)


def test_default_primary_probability_block_threshold_matches_sensor_confidence_scale():
    from map_poisoning.config import FusionConfig
    assert FusionConfig().blocked_probability_threshold == 0.50


def test_default_unknown_traversal_cost_is_shared_by_primary_methods():
    from map_poisoning.belief import RobotBeliefMap
    from map_poisoning.config import FusionConfig
    belief = RobotBeliefMap(np.zeros((2, 2), dtype=np.uint8))
    assert FusionConfig().unknown_traversal_cost == belief.unknown_traversal_cost == 1.5


def test_default_trust_rehabilitation_is_deliberately_slower_than_contradiction():
    from map_poisoning.config import TrustConfig
    from map_poisoning.trust import BayesianTrustModel

    cfg = TrustConfig()
    assert cfg.confirmation_multiplier == 0.025
    assert cfg.contradiction_multiplier == 5.0
    trust = BayesianTrustModel(
        confirmation_multiplier=cfg.confirmation_multiplier,
        contradiction_multiplier=cfg.contradiction_multiplier,
        evidence_cap=cfg.evidence_cap,
    )
    trust.update_batch(0, 0.0, 2.7)  # several strong false cells in one scan
    after_attack = trust.score(0)
    trust.update_batch(0, 8.0, 0.0)
    trust.update_batch(0, 8.0, 0.0)
    assert after_attack < 0.50
    assert trust.score(0) < 0.50  # two good scans do not instantly rehabilitate


def test_fake_claim_has_no_hard_expiration_but_still_age_decays():
    from map_poisoning.fusion import FusionEngine
    from map_poisoning.models import ClaimReport, ClaimType

    engine = FusionEngine("full_trust", lambda _: 1.0, max_claim_age=10)
    persistent = ClaimReport(
        "fake-persistent", 0, (2, 2), ClaimType.BLOCKED, 0, 1.0,
        "attack-fake", True,
    )
    ordinary = ClaimReport("ordinary", 1, (3, 3), ClaimType.BLOCKED, 0, 1.0)
    engine.add(persistent, is_malicious=True)
    engine.add(ordinary)
    initial = engine.operational_weight(persistent, 0)
    later = engine.operational_weight(persistent, 20)
    engine.prune(20)

    assert initial == 1.0
    assert 0.0 < later < initial
    assert engine.is_active_report(persistent)
    assert not engine.is_active_report(ordinary)


def test_primary_methods_and_trust_threshold_share_direct_lidar_authority_and_false_clearance_contract():
    import math
    import numpy as np
    from map_poisoning.belief import RobotBeliefMap
    from map_poisoning.fusion import FusionEngine
    from map_poisoning.models import ClaimReport, ClaimType, DirectObservation

    methods = ("majority_vote", "full_trust", "trust_fused", "source_memory", "trust_threshold")
    for method in methods:
        grid = np.zeros((6, 6), dtype=np.uint8)
        belief = RobotBeliefMap(grid, memory_steps=300)
        fusion = FusionEngine(method, lambda _: 0.9, trust_memory_score=lambda _: 0.9, trust_threshold=0.5)
        belief.begin_scan(5)
        belief.observe(DirectObservation(1, (2, 2), ClaimType.BLOCKED, 5, 1.0))
        belief.begin_scan(10)
        fusion.add(ClaimReport("clear-" + method, 0, (2, 2), ClaimType.FREE, 10, 1.0, "attack-clear"), is_malicious=True)
        assert not math.isinf(belief.traversal_cost((2, 2), 10, fusion)), method

        belief.begin_scan(11)
        belief.observe(DirectObservation(1, (2, 2), ClaimType.BLOCKED, 11, 1.0))
        assert math.isinf(belief.traversal_cost((2, 2), 11, fusion)), method
