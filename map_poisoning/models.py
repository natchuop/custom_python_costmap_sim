"""Low-level data types.  These deliberately contain no simulator imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, TypeAlias

Cell: TypeAlias = tuple[int, int]


class ClaimType(IntEnum):
    FREE = 0
    BLOCKED = 1
    CONGESTED = 2


class SimulationPhase(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    ATTACK = "attack"
    RECOVERY = "recovery"


class AttackType(str, Enum):
    FAKE_OBSTACLE = "fake_obstacle"
    FALSE_CLEARANCE = "false_clearance"
    STALE_REASSERTION = "stale_reassertion"


class VerificationOutcome(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED_FRESH = "contradicted_fresh"
    TEMPORALLY_AMBIGUOUS_OR_EXPIRED = "temporally_ambiguous_or_expired"
    # Compatibility name for old CSV readers.  It no longer implies honesty or
    # rewards trust; audit-only truth labels belong outside operational code.
    HONEST_STALE_OR_EXPIRED = "temporally_ambiguous_or_expired"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ClaimReport:
    report_id: str
    sender_id: int
    target_cell: Cell
    claim: ClaimType
    observation_step: int
    sent_step: int
    received_step: int | None = None
    confidence: float = 1.0
    scenario_event_id: str | None = None


@dataclass(frozen=True)
class ReportAuditLabel:
    report_id: str
    is_malicious: bool
    attack_type: AttackType | None
    obstacle_episode_id: str | None
    actual_state_at_observation: ClaimType
    original_obstacle_appearance_step: int | None = None
    original_obstacle_clearance_step: int | None = None


@dataclass(frozen=True)
class DeliveryTask:
    task_id: str
    pickup: Cell
    dropoff: Cell


@dataclass(frozen=True)
class DirectObservation:
    observer_id: int
    cell: Cell
    claim: ClaimType
    step: int


@dataclass(frozen=True)
class TemporaryObstacleEpisode:
    episode_id: str
    cells: tuple[Cell, ...]
    appearance_step: int
    clearance_step: int


@dataclass(frozen=True)
class AttackEvent:
    event_id: str
    step: int
    attack_type: AttackType
    cells: tuple[Cell, ...]
    claim: ClaimType
    observation_step: int
    sender_id: int
    recipients: tuple[int, ...]
    report_ids: tuple[str, ...]
    obstacle_episode_id: str | None = None


@dataclass(frozen=True)
class SimulationEvent:
    event_id: str
    step: int
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    influence: float
    reason: str


@dataclass(frozen=True)
class TrustUpdate:
    sender_id: int
    old_trust: float
    new_trust: float
    step: int
    outcome: VerificationOutcome
