"""Application services used by both CLI and Tkinter UI."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .config import SimulationConfig
from .metrics import CsvMetrics
from .scenario import author_manifest, load_manifest, save_manifest
from .simulation import replay
from .map_io import load_movingai, load_npy
from .legacy_adapter import author_legacy_manifest, replay_legacy

def run(config: SimulationConfig, *, comparison: bool = False, manifest_only: bool = False):
    config.validate(); root=Path(config.logging.output_directory)
    grid=load_npy(config.map_npy) if config.map_npy else load_movingai(config.map_movingai) if config.map_movingai else None
    if config.manifest_path:
        manifest=load_manifest(config.manifest_path)
    elif config.engine == "legacy":
        manifest=author_legacy_manifest(config)
    else:
        manifest=author_manifest(config, grid)
    root.mkdir(parents=True,exist_ok=True); save_manifest(manifest,root/"scenario_manifest.json"); CsvMetrics.config(root/"resolved_config.json",config.to_dict())
    if manifest_only: return manifest
    methods=config.comparison_methods if comparison else (config.fusion.method,)
    if config.engine == "legacy":
        return [replay_legacy(config,method,root/method if comparison else root,manifest) for method in methods]
    return [replay(config,manifest,method,root/method if comparison else root) for method in methods]
