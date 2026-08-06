"""Headless manifest replay for the modular simulator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SimulationConfig
from .rollout import replay_manifest
from .scenario import ScenarioManifest


@dataclass
class RunResult:
    output_directory: Path
    method: str
    manifest: ScenarioManifest
    summary: dict


def replay(
    config: SimulationConfig,
    manifest: ScenarioManifest,
    method: str,
    output_directory: Path,
) -> RunResult:
    result = replay_manifest(
        config,
        manifest,
        method,
        output_directory,
        show_animation=config.visualization.animation,
    )
    return RunResult(result.output_directory, method, manifest, result.summary)
