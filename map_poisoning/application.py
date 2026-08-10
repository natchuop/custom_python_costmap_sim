"""Application services used by both CLI and Tkinter UI."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import sim2
from .config import SimulationConfig
from .metrics import CsvMetrics
from .scenario import author_manifest, author_warehouse_manifest, load_manifest, save_manifest, scenario_manifest_hash
from .simulation import replay
from .map_io import default_warehouse_map, load_movingai, load_npy
from .audit import audit_manifest

def resolve_output_directory(config: SimulationConfig) -> Path:
    """Return the run folder, grouping default results by seed."""
    base = Path(config.logging.output_directory)
    if base.name.lower() == "simulation_results":
        return base / f"seed_{config.seed}"
    return base

def run(config: SimulationConfig, *, comparison: bool = False, manifest_only: bool = False):
    requested=config; requested.validate(); root=resolve_output_directory(config)
    grid=load_npy(config.map_npy) if config.map_npy else load_movingai(config.map_movingai) if config.map_movingai else default_warehouse_map()
    if config.manifest_path:
        manifest=load_manifest(config.manifest_path)
    elif not config.map_npy and not config.map_movingai and not config.scenario_preset:
        manifest=author_warehouse_manifest(config, grid)
    else:
        manifest=author_manifest(config, grid)
    root.mkdir(parents=True,exist_ok=True); save_manifest(manifest,root/"scenario_manifest.json")
    # The modular runner does not mutate the requested configuration while
    # resolving it, so three copies of the same root configuration were
    # needlessly written. Keep one canonical run-level configuration; each
    # method replay writes its own effective_config.json below.
    CsvMetrics.config(root/"run_config.json",config.to_dict())
    CsvMetrics.config(root/"audit_report.json",audit_manifest(manifest))
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit=None; dirty=None
    CsvMetrics.config(root/"run_metadata.json",{"python_version":sys.version,"platform":platform.platform(),"engine":"modular","settings_source":"modular_cli","scenario_id":manifest.scenario_id,"manifest_hash":manifest.map_hash,"scenario_manifest_hash":scenario_manifest_hash(manifest),"git_commit":commit,"git_dirty":dirty,"seed":config.seed,"collected_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"output_directory":str(root)})
    if manifest_only: return manifest
    methods=config.comparison_methods if comparison else (config.fusion.method,)
    results = [replay(config,manifest,method,root/method if comparison else root) for method in methods]
    if comparison and len(results) > 1:
        # All comparison methods use the same authored manifest and therefore
        # share the heatmap used to select attack candidates. Save it once at
        # the comparison root instead of creating one copy per method.
        heatmap = manifest.reconnaissance_heatmap
        first = results[0]
        if heatmap is None and first.log and first.log.get("traffic_heatmap"):
            frame = min(
                int(first.log.get("attack_phase_start_step") or 0),
                len(first.log["traffic_heatmap"]) - 1,
            )
            heatmap = first.log["traffic_heatmap"][frame]
        if heatmap is not None:
            heatmap_array = np.asarray(heatmap, dtype=int)
            np.save(root / "traffic_heatmap.npy", heatmap_array)
            heat_log = dict(first.log or {})
            heat_log["traffic_heatmap"] = [heatmap_array]
            heat_log["attack_phase_start_step"] = 0
            sim2.show_recon_heatmap(
                first.world,
                heat_log,
                output_path=root / "traffic_heatmap.png",
            )
    return results
