"""Low-level data types. These deliberately contain no simulator imports."""
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
    HONEST_STALE_OR_EXPIRED = "temporally_ambiguous_or_expired"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ClaimReport:
    """One peer occupancy report.

    Communication is instantaneous in this simulator, so only the time the
    information was observed is retained. ``sensor_confidence`` is the quality
    of that particular observation; source trust is maintained separately.
    """

    report_id: str
    sender_id: int
    target_cell: Cell
    claim: ClaimType
    observation_step: int
    sensor_confidence: float = 1.0
    scenario_event_id: str | None = None
    # Fake-obstacle misinformation has no independent hard expiration. It
    # remains stored until directly cleared/replaced/ignored, while its
    # operational influence still age-decays according to the defense method.
    persistent_until_verified: bool = False


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
    sensor_confidence: float = 1.0


@dataclass(frozen=True)
class TemporaryObstacleEpisode:
    episode_id: str
    cells: tuple[Cell, ...]
    appearance_step: int
    clearance_step: int


@dataclass(frozen=True)
class ReconMissionSnapshot:
    """One immutable clean-recon mission observation.

    These snapshots are deliberately independent of a defense method.  They are
    the only dynamic robot information that attack authoring may use after the
    reconnaissance boundary.
    """

    step: int
    robot_id: int
    mission_id: str | None
    position: Cell
    goal: Cell | None
    planned_path: tuple[Cell, ...] = ()
    visible_cells: tuple[Cell, ...] = ()


@dataclass(frozen=True)
class ReconRouteSnapshot:
    """A sampled route considered by a benign robot during reconnaissance."""

    step: int
    robot_id: int
    mission_id: str | None
    start: Cell
    goal: Cell
    path: tuple[Cell, ...]


@dataclass(frozen=True)
class FrozenReconData:
    """Frozen, defense-independent knowledge available to attack authoring."""

    recon_steps: int
    traffic_heatmap: tuple[tuple[int, ...], ...]
    mission_snapshots: tuple[ReconMissionSnapshot, ...] = ()
    route_snapshots: tuple[ReconRouteSnapshot, ...] = ()
    cell_route_usage: tuple[tuple[Cell, int], ...] = ()
    footprint_route_usage: tuple[tuple[tuple[Cell, ...], int], ...] = ()
    observed_obstacle_events: tuple[tuple[str, int, int, tuple[Cell, ...]], ...] = ()
    visibility_summaries: tuple[tuple[int, int], ...] = ()
    map_hash: str = ""
    scenario_id: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for manifest persistence."""
        from dataclasses import asdict

        return asdict(self)

    def states(self) -> dict[int, dict[int, dict[str, Any]]]:
        """Reconstruct the small legacy state view used by authoring helpers."""
        states: dict[int, dict[int, dict[str, Any]]] = {}
        for snapshot in self.mission_snapshots:
            states.setdefault(snapshot.step, {})[snapshot.robot_id] = {
                "position": snapshot.position,
                "goal": snapshot.goal,
                "path": snapshot.planned_path,
                "visible_cells": snapshot.visible_cells,
                "mission_id": snapshot.mission_id,
            }
        return states

    def route_cells(self) -> tuple[Cell, ...]:
        cells: set[Cell] = set()
        for snapshot in self.route_snapshots:
            cells.update(snapshot.path)
        return tuple(sorted(cells))

    def mission_pairs(self) -> tuple[tuple[Cell, Cell], ...]:
        pairs = {
            (snapshot.position, snapshot.goal)
            for snapshot in self.mission_snapshots
            if snapshot.goal is not None and snapshot.position != snapshot.goal
        }
        return tuple(sorted(pairs))


@dataclass(frozen=True)
class PermanentObstacle:
    obstacle_id: str
    cells: tuple[Cell, ...]


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
