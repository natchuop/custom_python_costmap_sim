import math
from types import SimpleNamespace

from defense_method_runner import build_defense_runner


def report(report_id, sender, claim, step=0):
    return SimpleNamespace(
        report_id=report_id,
        sender_id=sender,
        target_cell=(2, 2),
        claim=claim,
        timestamp=step,
        confidence=1.0,
    )


def test_primary_weighting_contracts_and_active_replacement():
    trust = {0: 0.7}
    score = lambda sender: trust[sender]
    runners = {
        name: build_defense_runner(name, score, decay_rate=0.01)
        for name in ("full_trust", "trust_fused", "source_linked")
    }
    for runner in runners.values():
        runner.add_report(report("one", 0, 1))
        runner.add_report(report("two", 0, 1, 1))
        assert len(runner.claims_for((2, 2))) == 1
    before = {name: runner.evidence((2, 2), 1) for name, runner in runners.items()}
    trust[0] = 0.1
    assert runners["full_trust"].evidence((2, 2), 1) == before["full_trust"]
    assert runners["trust_fused"].evidence((2, 2), 1) < before["trust_fused"]
    assert runners["source_linked"].evidence((2, 2), 1) < before["source_linked"]
    assert before["full_trust"] >= before["trust_fused"]


def test_trust_threshold_is_dynamic_but_retains_below_threshold_reports():
    trust = {0: 0.8}
    runner = build_defense_runner("trust_threshold", lambda sender: trust[sender])
    runner.add_report(report("threshold", 0, 1))
    assert runner.evidence((2, 2), 0) > 0
    trust[0] = 0.4
    assert runner.evidence((2, 2), 0) == 0
    trust[0] = 0.7
    assert runner.evidence((2, 2), 0) > 0
    assert len(runner.claims_for((2, 2))) == 1


def test_majority_is_one_vote_per_sender_and_discrete():
    runner = build_defense_runner("majority_vote", lambda _: 0.1)
    runner.add_report(report("a", 0, 1))
    runner.add_report(report("b", 0, 1, 1))
    runner.add_report(report("c", 1, 0, 1))
    assert runner.evidence((2, 2), 1) == 0
    assert not runner.is_hard_blocked((2, 2), 1)
    runner.add_report(report("d", 2, 1, 1))
    assert runner.is_hard_blocked((2, 2), 1)
    assert math.isinf(runner.routing_cost((2, 2), 1))


def test_trust_fused_selects_highest_effective_trust_claim_and_hard_blocks():
    trust = {1: 0.80, 2: 0.70}
    runner = build_defense_runner("trust_fused", lambda sender: trust[sender], decay_rate=0.10)
    runner.add_report(report("blocked-old", 1, 1, 0))
    runner.add_report(report("free-new", 2, 0, 9))

    assert runner.occupancy_probability((2, 2), 10) == 0.0
    assert runner.routing_cost((2, 2), 10) == 1.0

    trust[2] = 0.40
    assert runner.occupancy_probability((2, 2), 10) == 1.0
    assert math.isinf(runner.routing_cost((2, 2), 10))
