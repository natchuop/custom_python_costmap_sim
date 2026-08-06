"""Run mechanism tests and a small fixed-manifest audit from repository root."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from map_poisoning.application import run
from map_poisoning.audit import attacker_stream_signature, audit_manifest
from map_poisoning.config import LoggingConfig, PhaseConfig, SimulationConfig, VisualizationConfig


def main() -> int:
    root = Path("verification_outputs")
    root.mkdir(exist_ok=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True, capture_output=True)
    config = SimulationConfig(
        engine="modular", attacks=replace(SimulationConfig().attacks, enabled=("fake_obstacle",)),
        phases=PhaseConfig(20, 40, 20), max_steps=80, deliveries_per_robot=1,
        logging=LoggingConfig(str(root / "replay")), visualization=VisualizationConfig(False),
    )
    manifest = run(config, manifest_only=True)
    manifest_audit = audit_manifest(manifest)
    results = run(replace(config, manifest_path=str(root / "replay" / "scenario_manifest.json")), comparison=True)
    signatures = [attacker_stream_signature(manifest) for _ in results]
    report = {"pytest_returncode": tests.returncode, "pytest_output": tests.stdout, "manifest_audit": manifest_audit, "attacker_streams_identical": len(set(signatures)) == 1, "methods": [item.method for item in results]}
    (root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "audit_summary.md").write_text(f"# Verification summary\n\n- pytest: {'PASS' if tests.returncode == 0 else 'FAIL'}\n- manifest audit: {'PASS' if manifest_audit['passed'] else 'FAIL'}\n- fixed attacker stream: {'PASS' if report['attacker_streams_identical'] else 'FAIL'}\n", encoding="utf-8")
    return 0 if tests.returncode == 0 and manifest_audit["passed"] and report["attacker_streams_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
