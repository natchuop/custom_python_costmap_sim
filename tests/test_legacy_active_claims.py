from defense_method_runner import DefenseConfig, DefenseMethodRunner


class Report:
    def __init__(self, timestamp):
        self.sender_id = 0
        self.target_cell = (3, 3)
        self.claim = 1
        self.timestamp = timestamp
        self.confidence = 1.0


def test_legacy_repeated_sender_claim_replaces_instead_of_stacking():
    runner = DefenseMethodRunner(lambda _: 1.0, DefenseConfig(method="full_trust"))
    assert runner.add_report(Report(1))
    assert runner.add_report(Report(2))
    assert len(runner.claims_for((3, 3))) == 1
    assert runner.evidence((3, 3), 2) == 1.0


def test_legacy_majority_is_a_discrete_hard_block():
    runner = DefenseMethodRunner(lambda _: 0.0, DefenseConfig(method="majority_vote"))
    assert runner.add_report(Report(1))
    assert runner.is_hard_blocked((3, 3), 1)
    assert runner.routing_cost((3, 3), 1) == float("inf")
