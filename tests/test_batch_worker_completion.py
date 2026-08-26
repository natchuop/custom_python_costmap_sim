from __future__ import annotations

import csv
import json
from dataclasses import replace

from map_poisoning.batch import _valid_resume
from map_poisoning.batch_worker import _write_completion
from map_poisoning.config import LoggingConfig, SimulationConfig


def test_worker_completion_stamp_makes_finished_cell_resumable(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    config = replace(
        SimulationConfig(),
        seed=21,
        fusion=replace(SimulationConfig().fusion, method="source_memory"),
        logging=LoggingConfig(str(output), generate_plots=False),
    )
    (output / "effective_config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    with (output / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "method", "scenario_manifest_hash"])
        writer.writeheader()
        writer.writerow({"seed": 21, "method": "source_memory", "scenario_manifest_hash": "manifest-hash"})

    metadata = {
        "seed": 21,
        "method": "source_memory",
        "scenario_manifest_hash": "manifest-hash",
        "experiment_config_hash": "config-hash",
        "code_revision_hash": "code-hash",
        "git_commit": "unknown",
        "git_dirty": False,
    }
    _write_completion(config, metadata)

    effective = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "batch_cell_complete.json").read_text(encoding="utf-8"))
    assert effective["experiment_config_hash"] == "config-hash"
    assert effective["code_revision_hash"] == "code-hash"
    assert marker == metadata
    assert _valid_resume(output, 21, "source_memory", "manifest-hash", "config-hash", "code-hash")
