"""Trust estimators for source-linked modular report fusion."""
from __future__ import annotations

from dataclasses import dataclass

from .models import VerificationOutcome


class TrustModel:
    def score(self, sender_id: int) -> float:
        raise NotImplementedError

    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]:
        raise NotImplementedError


@dataclass
class BayesianTrustModel(TrustModel):
    alpha0: float = 7.0
    beta0: float = 3.0

    def __post_init__(self):
        self.values: dict[int, list[float]] = {}

    def _value(self, sender: int) -> list[float]:
        return self.values.setdefault(sender, [self.alpha0, self.beta0])

    def score(self, sender_id: int) -> float:
        alpha, beta = self._value(sender_id)
        return alpha / (alpha + beta)

    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]:
        before = self.score(sender_id)
        alpha, beta = self._value(sender_id)
        if outcome == VerificationOutcome.CONFIRMED:
            alpha += 1
        elif outcome == VerificationOutcome.CONTRADICTED_FRESH:
            beta += 1
        self.values[sender_id] = [alpha, beta]
        return before, self.score(sender_id)


@dataclass
class ScalarTrustModel(TrustModel):
    initial: float = 0.70
    reward: float = 0.02
    penalty: float = 0.06

    def __post_init__(self):
        self.values: dict[int, float] = {}

    def score(self, sender_id: int) -> float:
        return self.values.get(sender_id, self.initial)

    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]:
        before = self.score(sender_id)
        delta = {
            VerificationOutcome.CONFIRMED: self.reward,
            VerificationOutcome.CONTRADICTED_FRESH: -self.penalty,
        }.get(outcome, 0.0)
        self.values[sender_id] = min(1.0, max(0.0, before + delta))
        return before, self.values[sender_id]


def make_trust_model(name: str, alpha: float = 7.0, beta: float = 3.0) -> TrustModel:
    if name == "bayesian":
        return BayesianTrustModel(alpha, beta)
    return ScalarTrustModel(alpha / (alpha + beta))
