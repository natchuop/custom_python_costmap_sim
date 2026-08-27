"""Immutable configuration and validation for an experiment run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import AttackType

# Keep this order everywhere user-facing and in comparison output.
PRIMARY_METHODS = ("latest_report", "majority_vote", "full_trust", "trust_fused", "source_memory")
ADDITIONAL_METHODS = ("hard_threshold", "soft_probability", "time_decay", "trust_threshold")
ALL_METHODS = PRIMARY_METHODS + ADDITIONAL_METHODS
MAP_VIEWS = ("combined", "local")


@dataclass(frozen=True)
class PhaseConfig:
    recon_steps: int = 300
    attack_steps: int = 1700
    recovery_steps: int = 500

    @property
    def total_steps(self) -> int:
        return self.recon_steps + self.attack_steps + self.recovery_steps


@dataclass(frozen=True)
class AttackConfig:
    enabled: tuple[str, ...] = tuple(item.value for item in AttackType)
    interval_min: int = 35
    interval_max: int = 40
    candidate_top_k: int = 12
    broadcast: bool = True
    global_awareness: bool = True
    max_uses_per_footprint: int = 20
    min_center_spacing: int = 3
    min_unique_footprints: int = 3
    visibility_delay_min: int = 15
    visibility_delay_max: int = 40


@dataclass(frozen=True)
class TrustConfig:
    model: str = "bayesian"
    prior_alpha: float = 9.0
    prior_beta: float = 1.0
    threshold: float = 0.50
    evidence_cap: float = 12.0
    # Positive evidence is intentionally slower than contradiction evidence so
    # an attacker cannot regain full trust after only a few honest reports.
    confirmation_multiplier: float = 0.25
    contradiction_multiplier: float = 6.0
    source_memory_recovery_rate: float = 0.05


@dataclass(frozen=True)
class FusionConfig:
    method: str = "source_memory"
    admission_policy: str = "accept_all"
    # Retained for the legacy/additional time_decay method. The five primary
    # methods use common linear aging over max_claim_age.
    decay_rate: float = 0.006
    cost_scale: float = 40.0
    cost_exponent: float = 1.5
    blocked_probability_threshold: float = 0.70
    max_claim_age: int = 300
    congested_impact: float = 0.50
    duplicate_window_steps: int = 0
    majority_unknown_cost: float = 3.0


@dataclass(frozen=True)
class LoggingConfig:
    output_directory: str = "outputs"
    timeseries_period_steps: int = 5
    generate_plots: bool = True
    plot_format: str = "png"


@dataclass(frozen=True)
class VisualizationConfig:
    animation: bool = False
    map_view: str = "combined"
    fake_influence_min_cost_delta: float = 0.10
    route_impact_min_cost_delta: float = 0.10
    route_impact_eval_period_steps: int = 25


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
    temporary_blockage_change_period_steps: int = 150
    map_npy: str | None = None
    map_movingai: str | None = None
    scenario_preset: str | None = None
    manifest_path: str | None = None
    deliveries_per_robot: int = 100
    max_steps: int | None = None
    observation_lifetime_steps: int = 300
    lidar_range_cells: int = 5
    confidence_resend_delta: float = 0.10
    periodic_route_check_steps: int = 25
    periodic_route_improvement_epsilon: float = 0.01

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.phases.recon_steps <= 0 or self.phases.attack_steps < 0 or self.phases.recovery_steps <= 0:
            raise ValueError("phase lengths must be positive (attack may be zero)")
        if self.attacks.interval_min < 1 or self.attacks.interval_max < self.attacks.interval_min:
            raise ValueError("invalid attack interval")
        if self.attacks.candidate_top_k < 1:
            raise ValueError("attack candidate_top_k must be positive")
        if self.attacks.max_uses_per_footprint < 1 or self.attacks.min_center_spacing < 0 or self.attacks.min_unique_footprints < 1:
            raise ValueError("invalid attack diversity settings")
        if self.attacks.visibility_delay_min < 1 or self.attacks.visibility_delay_max < self.attacks.visibility_delay_min:
            raise ValueError("invalid attack visibility-delay window")
        if self.trust.model not in {"bayesian", "scalar"}:
            raise ValueError("trust model must be bayesian or scalar")
        if self.trust.prior_alpha <= 0 or self.trust.prior_beta <= 0:
            raise ValueError("Bayesian priors must be positive")
        if not 0 <= self.trust.threshold <= 1:
            raise ValueError("trust threshold must be in [0, 1]")
        if self.trust.evidence_cap <= 0:
            raise ValueError("trust evidence cap must be positive")
        if self.trust.confirmation_multiplier < 0 or self.trust.contradiction_multiplier < 0:
            raise ValueError("trust evidence multipliers must be nonnegative")
        if not 0 < self.trust.source_memory_recovery_rate < 1:
            raise ValueError("source memory recovery rate must be in (0, 1)")
        if self.fusion.method not in ALL_METHODS:
            raise ValueError(f"unknown defense method: {self.fusion.method}")
        if self.fusion.admission_policy not in {"auto_soft", "accept_all", "hard_reject"}:
            raise ValueError("unknown admission policy")
        if self.fusion.max_claim_age < 1:
            raise ValueError("max_claim_age must be positive")
        if self.fusion.majority_unknown_cost < 1:
            raise ValueError("majority unknown cost must be >= 1")
        if any(item not in {x.value for x in AttackType} for item in self.attacks.enabled):
            raise ValueError("unknown attack type")
        if self.map_npy and self.map_movingai:
            raise ValueError("use one map source")
        if self.scenario_preset is not None:
            from .scenario_presets import PRESETS
            if self.scenario_preset not in PRESETS:
                raise ValueError(f"unknown scenario preset: {self.scenario_preset}")
        if self.deliveries_per_robot < 1:
            raise ValueError("deliveries_per_robot must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.observation_lifetime_steps < 1:
            raise ValueError("observation lifetime must be positive")
        if self.lidar_range_cells < 1:
            raise ValueError("lidar range must be positive")
        if not 0 <= self.confidence_resend_delta <= 1:
            raise ValueError("confidence resend delta must be in [0, 1]")
        if self.periodic_route_check_steps < 1:
            raise ValueError("periodic route check must be positive")
        if self.periodic_route_improvement_epsilon < 0:
            raise ValueError("periodic route improvement epsilon must be nonnegative")
        if self.visualization.map_view not in MAP_VIEWS:
            raise ValueError("map_view must be combined or local")
        if self.visualization.fake_influence_min_cost_delta < 0 or self.visualization.route_impact_min_cost_delta < 0:
            raise ValueError("visualization metric thresholds must be non-negative")
        if self.visualization.route_impact_eval_period_steps < 1:
            raise ValueError("route impact evaluation period must be positive")

    @property
    def total_steps(self) -> int:
        return min(self.phases.total_steps, self.max_steps) if self.max_steps else self.phases.total_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
