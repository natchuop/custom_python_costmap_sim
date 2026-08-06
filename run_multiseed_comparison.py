"""Run fixed-manifest legacy comparison across methods for multiple seeds."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from map_poisoning.application import run
from map_poisoning.config import LoggingConfig, SimulationConfig, VisualizationConfig

METHODS = (
    "full_trust",
    "majority_vote",
    "trust_fused",
    "source_linked",
    "soft_probability",
)
SEEDS = (15, 42)


def main() -> int:
    rows = []
    for seed in SEEDS:
        root = Path(f"outputs/multiseed_{seed}")
        config = SimulationConfig(
            seed=seed,
            attacks=replace(SimulationConfig().attacks, enabled=("fake_obstacle",)),
            logging=LoggingConfig(str(root)),
            visualization=VisualizationConfig(False),
        )
        run(config, manifest_only=True)
        manifest_path = root / "scenario_manifest.json"
        for method in METHODS:
            method_config = replace(
                config,
                fusion=replace(config.fusion, method=method),
                manifest_path=str(manifest_path),
                logging=LoggingConfig(str(root / method)),
            )
            run(method_config, comparison=False)
        audit = subprocess.run(
            [sys.executable, "audit_legacy_methods.py"],
            capture_output=True,
            text=True,
        )
        for method in METHODS:
            summary_path = root / method / "run_summary.csv"
            events_path = root / method / "events.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            mal = 0
            with events_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("kind") == "report_sent" and row.get("is_malicious") == "True":
                        mal += 1
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "malicious_cells": mal,
                    "deliveries": int(summary["benign_total_deliveries_completed"]),
                    "success_rate": float(summary["benign_success_rate"]),
                    "no_path": int(summary["benign_no_path_steps"]),
                    "movement": int(summary["benign_movement_steps"]),
                    "distance": int(summary["benign_total_distance"]),
                    "replans": int(summary["benign_total_replans"]),
                }
            )

    table_path = Path("outputs/multiseed_summary.json")
    table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
