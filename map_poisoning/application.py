"""Application services used by both CLI and Tkinter UI."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import platform
import subprocess
import sys
from .config import SimulationConfig
from .metrics import CsvMetrics
from .scenario import author_manifest, load_manifest, save_manifest, scenario_manifest_hash
from .simulation import replay
from .map_io import default_warehouse_map, load_movingai, load_npy
from .audit import audit_manifest

def run(config: SimulationConfig, *, comparison: bool = False, manifest_only: bool = False):
    requested=config; requested.validate(); root=Path(config.logging.output_directory)
    grid=load_npy(config.map_npy) if config.map_npy else load_movingai(config.map_movingai) if config.map_movingai else default_warehouse_map()
    if config.manifest_path:
        manifest=load_manifest(config.manifest_path)
    else:
        manifest=author_manifest(config, grid)
    root.mkdir(parents=True,exist_ok=True); save_manifest(manifest,root/"scenario_manifest.json")
    CsvMetrics.config(root/"requested_config.json",requested.to_dict())
    CsvMetrics.config(root/"effective_config.json",config.to_dict())
    CsvMetrics.config(root/"resolved_config.json",config.to_dict())
    CsvMetrics.config(root/"audit_report.json",audit_manifest(manifest))
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit=None; dirty=None
    CsvMetrics.config(root/"run_metadata.json",{"python_version":sys.version,"platform":platform.platform(),"engine":"modular","settings_source":"modular_cli","scenario_id":manifest.scenario_id,"manifest_hash":manifest.map_hash,"map_hash":manifest.map_hash,"scenario_manifest_hash":scenario_manifest_hash(manifest),"git_commit":commit,"git_dirty":dirty})
    if manifest_only: return manifest
    methods=config.comparison_methods if comparison else (config.fusion.method,)
    results = []
    for index, method in enumerate(methods):
        replay_config = config
        if comparison and config.visualization.animation and index > 0:
            replay_config = replace(config, visualization=replace(config.visualization, animation=False))
        results.append(replay(replay_config, manifest, method, root / method if comparison else root))
    if comparison and config.logging.generate_plots:
        try:
            from .reporting import generate_comparison_report
            generate_comparison_report(root, formats=(config.logging.plot_format,))
        except Exception as exc:
            warning = f"Warning: comparison plot generation failed for {root}: {exc}"
            print(warning)
            (root / "plot_generation_error.txt").write_text(warning + "\n", encoding="utf-8")
    return results
