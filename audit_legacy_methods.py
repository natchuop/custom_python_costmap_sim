"""Audit legacy defense-method semantics and fixed-manifest replay invariants."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from defense_method_runner import DefenseConfig, DefenseMethodRunner

METHODS = (
    "full_trust",
    "majority_vote",
    "trust_fused",
    "source_linked",
    "soft_probability",
)


def audit_runner_semantics() -> list[str]:
    issues: list[str] = []
    trust = {0: 0.7}
    score = lambda sender: trust.get(sender, 0.7)

    engines = {
        name: DefenseMethodRunner(score, DefenseConfig(method=name, decay_rate=0.006, cost_scale=14, cost_exponent=1.5, max_claim_age=900))
        for name in METHODS
    }

    class Report:
        def __init__(self, sender_id=0, target_cell=(3, 3), claim=1, timestamp=0):
            self.sender_id = sender_id
            self.target_cell = target_cell
            self.claim = claim
            self.timestamp = timestamp
            self.confidence = 1.0

    for engine in engines.values():
        engine.add_report(Report())
        engine.add_report(Report(timestamp=1))
        if len(engine.claims_for((3, 3))) != 1:
            issues.append(f"{engine.method}: active claim replacement failed")

    before = {name: engine.evidence((3, 3), 1) for name, engine in engines.items()}
    trust[0] = 0.1
    after_full = engines["full_trust"].evidence((3, 3), 1)
    after_fused = engines["trust_fused"].evidence((3, 3), 1)
    after_linked = engines["source_linked"].evidence((3, 3), 1)
    after_soft = engines["soft_probability"].evidence((3, 3), 1)

    if after_full != before["full_trust"]:
        issues.append("full_trust evidence changed after trust drop (unexpected)")
    if after_fused != before["trust_fused"]:
        issues.append("trust_fused evidence changed after trust drop (unexpected)")
    if after_linked >= before["source_linked"]:
        issues.append("source_linked evidence did not decrease after trust drop")
    if after_soft != before["soft_probability"]:
        issues.append("soft_probability evidence changed after trust drop (unexpected)")

  # full_trust and soft_probability share confidence-only weighting in legacy runner.
    if before["full_trust"] != before["soft_probability"]:
        issues.append("full_trust and soft_probability evidence differ without trust (implementation mismatch)")

    majority = engines["majority_vote"]
    majority.add_report(Report(sender_id=1, claim=0, timestamp=2))
    if majority.is_hard_blocked((3, 3), 2):
        issues.append("majority_vote blocked on tied votes (expected free/tie)")
    majority.add_report(Report(sender_id=2, claim=1, timestamp=2))
    if not majority.is_hard_blocked((3, 3), 2):
        issues.append("majority_vote failed to hard-block on blocked majority")

    return issues


def audit_replay_directory(root: Path) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    manifest_path = root / "scenario_manifest.json"
    if not manifest_path.exists():
        return {"global": ["missing scenario_manifest.json"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attack_chain = tuple(
        (event["step"], tuple(event["cells"]))
        for event in manifest["attack_events"]
    )
    mal_counts: dict[str, int] = {}

    for method in METHODS:
        method_dir = root / method
        summary_path = method_dir / "run_summary.csv"
        events_path = method_dir / "events.csv"
        method_issues: list[str] = []
        if not summary_path.exists():
            findings[method] = ["missing run_summary.csv"]
            continue
        if not events_path.exists():
            method_issues.append("missing events.csv")
        else:
            with events_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            mal = [
                row
                for row in rows
                if row.get("kind") == "report_sent" and row.get("is_malicious") == "True"
            ]
            mal_counts[method] = len(mal)
            steps = tuple((int(row["step"]), row.get("target_cell")) for row in mal)
            if len(mal) != len(manifest["attack_events"]) and len(mal) < len({e["step"] for e in manifest["attack_events"]}):
                method_issues.append(f"low malicious delivery count: {len(mal)}")

        findings[method] = method_issues

    if mal_counts and len(set(mal_counts.values())) != 1:
        findings["global"] = [f"malicious report counts differ: {mal_counts}"]
    elif mal_counts:
        findings.setdefault("global", []).append(f"malicious report counts match: {next(iter(mal_counts.values()))}")

    findings.setdefault("global", []).append(f"manifest attacks: {len(attack_chain)}")
    return findings


def main() -> int:
    pytest = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    semantic_issues = audit_runner_semantics()
    report = {
        "pytest_returncode": pytest.returncode,
        "pytest_passed": pytest.returncode == 0,
        "semantic_issues": semantic_issues,
        "notes": [
            "full_trust and soft_probability use identical confidence-only weighting in defense_method_runner; identical mission metrics are expected.",
            "time_decay and hard_threshold are not in the primary comparison set.",
        ],
    }
    out = Path("outputs/legacy_method_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if pytest.stdout:
        print(pytest.stdout)
    return 0 if pytest.returncode == 0 and not semantic_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
