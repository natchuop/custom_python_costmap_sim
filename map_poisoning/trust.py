"""Per-recipient source trust models and Source Memory rehabilitation state."""
from __future__ import annotations

from dataclasses import dataclass

from .models import VerificationOutcome


class TrustModel:
    def score(self, sender_id: int) -> float:
        raise NotImplementedError

    def memory_score(self, sender_id: int) -> float:
        return self.score(sender_id)

    def update(
        self,
        sender_id: int,
        outcome: VerificationOutcome,
        *,
        severity: float = 1.0,
    ) -> tuple[float, float]:
        if outcome == VerificationOutcome.CONFIRMED:
            return self.update_batch(sender_id, max(0.0, severity), 0.0)
        if outcome == VerificationOutcome.CONTRADICTED_FRESH:
            return self.update_batch(sender_id, 0.0, max(0.0, severity))
        score = self.score(sender_id)
        return score, score

    def update_batch(
        self,
        sender_id: int,
        confirmed_weight: float,
        contradicted_weight: float,
    ) -> tuple[float, float]:
        raise NotImplementedError


@dataclass
class BayesianTrustModel(TrustModel):
    alpha0: float = 9.0
    beta0: float = 1.0
    evidence_cap: float = 12.0
    confirmation_multiplier: float = 0.25
    contradiction_multiplier: float = 6.0
    memory_recovery_rate: float = 0.05

    def __post_init__(self):
        self.values: dict[int, list[float]] = {}
        self.memory_values: dict[int, float] = {}

    def _value(self, sender: int) -> list[float]:
        return self.values.setdefault(sender, [float(self.alpha0), float(self.beta0)])

    def score(self, sender_id: int) -> float:
        alpha, beta = self._value(sender_id)
        return alpha / (alpha + beta)

    def memory_score(self, sender_id: int) -> float:
        return self.memory_values.setdefault(sender_id, self.score(sender_id))

    def _cap(self, alpha: float, beta: float) -> tuple[float, float]:
        total = alpha + beta
        if total <= self.evidence_cap:
            return alpha, beta
        scale = self.evidence_cap / total
        return alpha * scale, beta * scale

    def _update_memory(self, sender_id: int, new_trust: float) -> None:
        old_memory = self.memory_score(sender_id)
        if new_trust < old_memory:
            memory = new_trust
        else:
            memory = old_memory + self.memory_recovery_rate * (new_trust - old_memory)
            memory = min(new_trust, memory)
        self.memory_values[sender_id] = min(1.0, max(0.0, memory))

    def update_batch(
        self,
        sender_id: int,
        confirmed_weight: float,
        contradicted_weight: float,
    ) -> tuple[float, float]:
        before = self.score(sender_id)
        alpha, beta = self._value(sender_id)
        alpha += max(0.0, confirmed_weight) * self.confirmation_multiplier
        beta += max(0.0, contradicted_weight) * self.contradiction_multiplier
        alpha, beta = self._cap(alpha, beta)
        self.values[sender_id] = [alpha, beta]
        after = self.score(sender_id)
        self._update_memory(sender_id, after)
        return before, after


@dataclass
class ScalarTrustModel(TrustModel):
    initial: float = 0.90
    reward: float = 0.005
    penalty: float = 0.06
    memory_recovery_rate: float = 0.05

    def __post_init__(self):
        self.values: dict[int, float] = {}
        self.memory_values: dict[int, float] = {}

    def score(self, sender_id: int) -> float:
        return self.values.get(sender_id, self.initial)

    def memory_score(self, sender_id: int) -> float:
        return self.memory_values.setdefault(sender_id, self.score(sender_id))

    def update_batch(
        self,
        sender_id: int,
        confirmed_weight: float,
        contradicted_weight: float,
    ) -> tuple[float, float]:
        before = self.score(sender_id)
        delta = self.reward * max(0.0, confirmed_weight) - self.penalty * max(0.0, contradicted_weight)
        after = min(1.0, max(0.0, before + delta))
        self.values[sender_id] = after
        old_memory = self.memory_score(sender_id)
        if after < old_memory:
            memory = after
        else:
            memory = min(after, old_memory + self.memory_recovery_rate * (after - old_memory))
        self.memory_values[sender_id] = memory
        return before, after


def make_trust_model(
    name: str,
    alpha: float = 9.0,
    beta: float = 1.0,
    *,
    evidence_cap: float = 12.0,
    confirmation_multiplier: float = 0.25,
    contradiction_multiplier: float = 6.0,
    memory_recovery_rate: float = 0.05,
) -> TrustModel:
    if name == "bayesian":
        return BayesianTrustModel(
            alpha,
            beta,
            evidence_cap=evidence_cap,
            confirmation_multiplier=confirmation_multiplier,
            contradiction_multiplier=contradiction_multiplier,
            memory_recovery_rate=memory_recovery_rate,
        )
    return ScalarTrustModel(
        alpha / (alpha + beta),
        memory_recovery_rate=memory_recovery_rate,
    )
