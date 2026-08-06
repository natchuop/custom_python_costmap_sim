"""Trust estimators; they only see verification outcomes, never audit labels."""
from __future__ import annotations
from dataclasses import dataclass
from .models import VerificationOutcome

class TrustModel:
    def score(self, sender_id: int) -> float: raise NotImplementedError
    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]: raise NotImplementedError

@dataclass
class BayesianTrustModel(TrustModel):
    alpha0: float = 7.0
    beta0: float = 3.0
    def __post_init__(self): self.values: dict[int, list[float]] = {}
    def _value(self, sender: int) -> list[float]: return self.values.setdefault(sender, [self.alpha0, self.beta0])
    def score(self, sender_id: int) -> float:
        a, b = self._value(sender_id); return a / (a + b)
    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]:
        before = self.score(sender_id); a, b = self._value(sender_id)
        if outcome == VerificationOutcome.CONFIRMED: a += 1
        elif outcome == VerificationOutcome.CONTRADICTED_FRESH: b += 1
        elif outcome == VerificationOutcome.HONEST_STALE_OR_EXPIRED: a += .05
        self.values[sender_id] = [a, b]
        return before, self.score(sender_id)

@dataclass
class ScalarTrustModel(TrustModel):
    initial: float = .70
    def __post_init__(self): self.values: dict[int, float] = {}
    def score(self, sender_id: int) -> float: return self.values.get(sender_id, self.initial)
    def update(self, sender_id: int, outcome: VerificationOutcome) -> tuple[float, float]:
        before = self.score(sender_id)
        delta = {VerificationOutcome.CONFIRMED: .06, VerificationOutcome.CONTRADICTED_FRESH: -.18, VerificationOutcome.HONEST_STALE_OR_EXPIRED: .01}.get(outcome, 0)
        self.values[sender_id] = min(1., max(0., before + delta))
        return before, self.values[sender_id]

def make_trust_model(name: str, alpha: float = 7., beta: float = 3.) -> TrustModel:
    return BayesianTrustModel(alpha, beta) if name == "bayesian" else ScalarTrustModel(alpha / (alpha + beta))
