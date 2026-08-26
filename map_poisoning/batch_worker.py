"""Isolated worker for one multi-seed method replay.

The parent batch process serializes one immutable ``SimulationConfig`` plus
batch-completion metadata and launches this module in a fresh interpreter.
The worker stamps completion before terminating so interrupted parent batches
can resume without rerunning already-complete 2500-step cells.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from pathlib import Path

from .application import run


def _write_completion(config, metadata: dict) -> None:
    output = Path(config.logging.output_directory)
    effective_path = output / "effective_config.json"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    effective.update(metadata)
    temporary = effective_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(effective_path)

    marker = output / "batch_cell_complete.json"
    marker_tmp = marker.with_suffix(".json.tmp")
    marker_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    marker_tmp.replace(marker)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m map_poisoning.batch_worker <config-pickle>", file=sys.stderr, flush=True)
        return 2
    path = Path(args[0])
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "config" in payload:
        config = payload["config"]
        metadata = dict(payload.get("batch_metadata") or {})
    else:  # Compatibility with any older direct worker invocation.
        config = payload
        metadata = {}
    run(config, comparison=False)
    if metadata:
        _write_completion(config, metadata)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(int(code))
