"""Immutable configuration and validation for an experiment run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import AttackType

PRIMARY_METHODS = ("full_trust", "majority_vote", "trust_fused", "source_linked")
ADDITIONAL_METHODS = ("hard_threshold", "soft_probability", "time_decay")
ALL_METHODS = PRIMARY_METHODS + ADDITIONAL_METHODS


@dataclass(frozen=True)
class PhaseConfig:
    recon_steps: int = 450
    attack_steps: int = 1200
    recovery_steps: int = 750
    @property
    def total_steps(self) -> int: return self.recon_steps + self.attack_steps + self.recovery_steps


@dataclass(frozen=True)
class AttackConfig:
    enabled: tuple[str, ...] = tuple(item.value for item in AttackType)
    interval_min: int = 50
    interval_max: int = 50
    candidate_top_k: int = 12
    broadcast: bool = True
    global_awareness: bool = True
    # Rapid interactive runs can contain many actions; this is still a hard
    # cap, while spacing/minimum-unique checks diagnose concentration.
    max_uses_per_footprint: int = 20
    min_center_spacing: int = 3
    min_unique_footprints: int = 3


@dataclass(frozen=True)
class TrustConfig:
    model: str = "scalar"
    prior_alpha: float = 7.0
    prior_beta: float = 3.0
    threshold: float = 0.55


@dataclass(frozen=True)
class FusionConfig:
    method: str = "source_linked"
    admission_policy: str = "accept_all"
    decay_rate: float = 0.006
    # Must dominate the three-unit unknown-cell traversal cost when a highly
    # trusted blocked claim lies on a planned corridor; otherwise trust changes
    # cannot change navigation in an otherwise open controlled scenario.
    cost_scale: float = 40.0
    cost_exponent: float = 1.5
    blocked_probability_threshold: float = 0.70
    max_claim_age: int = 900
    congested_impact: float = 0.50
    duplicate_window_steps: int = 0


@dataclass(frozen=True)
class LoggingConfig:
    output_directory: str = "outputs"
    timeseries_period_steps: int = 5
    generate_plots: bool = True
    plot_format: str = "png"


@dataclass(frozen=True)
class VisualizationConfig:
    animation: bool = False
    fake_influence_min_cost_delta: float = 0.10
    route_impact_min_cost_delta: float = 0.10
    route_impact_eval_period_steps: int = 10


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 15
    phases: PhaseConfig = field(default_factory=PhaseConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    comparison_methods: tuple[str, ...] = PRIMARY_METHODS
    communication_period_steps: int = 4
    temporary_blockage_change_period_steps: int = 400
    map_npy: str | None = None
    map_movingai: str | None = None
    scenario_preset: str | None = None
    manifest_path: str | None = None
    deliveries_per_robot: int = 100
    max_steps: int | None = None
    # Fresh lidar readings override peers.  Older direct memory becomes stale so
    # a later fake obstacle can still change an unused corridor.
    direct_memory_steps: int = 12

    def validate(self) -> None:
        if self.seed < 0: raise ValueError("seed must be nonnegative")
        if self.phases.recon_steps <= 0 or self.phases.attack_steps < 0 or self.phases.recovery_steps <= 0: raise ValueError("phase lengths must be positive (attack may be zero)")
        if self.attacks.interval_min < 1 or self.attacks.interval_max < self.attacks.interval_min: raise ValueError("invalid attack interval")
        if self.attacks.candidate_top_k < 1: raise ValueError("attack candidate_top_k must be positive")
        if self.attacks.max_uses_per_footprint < 1 or self.attacks.min_center_spacing < 0 or self.attacks.min_unique_footprints < 1: raise ValueError("invalid attack diversity settings")
        if self.trust.model not in {"bayesian", "scalar"}: raise ValueError("trust model must be bayesian or scalar")
        if self.trust.prior_alpha <= 0 or self.trust.prior_beta <= 0: raise ValueError("Bayesian priors must be positive")
        if not 0 <= self.trust.threshold <= 1: raise ValueError("trust threshold must be in [0, 1]")
        if self.fusion.method not in ALL_METHODS: raise ValueError(f"unknown defense method: {self.fusion.method}")
        if self.fusion.admission_policy not in {"auto_soft", "accept_all", "hard_reject"}: raise ValueError("unknown admission policy")
        if any(item not in {x.value for x in AttackType} for item in self.attacks.enabled): raise ValueError("unknown attack type")
        if self.map_npy and self.map_movingai: raise ValueError("use one map source")
        if self.scenario_preset is not None:
            from .scenario_presets import PRESETS
            if self.scenario_preset not in PRESETS: raise ValueError(f"unknown scenario preset: {self.scenario_preset}")
        if self.deliveries_per_robot < 1: raise ValueError("deliveries_per_robot must be positive")
        if self.max_steps is not None and self.max_steps < 1: raise ValueError("max_steps must be positive")
        if self.direct_memory_steps < 0: raise ValueError("direct_memory_steps must be nonnegative")
        if self.visualization.fake_influence_min_cost_delta < 0 or self.visualization.route_impact_min_cost_delta < 0:
            raise ValueError("visualization metric thresholds must be non-negative")
        if self.visualization.route_impact_eval_period_steps < 1:
            raise ValueError("route impact evaluation period must be positive")

    @property
    def total_steps(self) -> int: return min(self.phases.total_steps, self.max_steps) if self.max_steps else self.phases.total_steps
    def to_dict(self) -> dict[str, Any]: return asdict(self)
