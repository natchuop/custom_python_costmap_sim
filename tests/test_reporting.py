import csv
import json
import math

import matplotlib
matplotlib.use("Agg")

from map_poisoning.reporting import (
    RUN_PLOTS,
    COMPARISON_PLOTS,
    generate_comparison_report,
    generate_run_report,
    parse_bool,
    parse_tuple,
    _replan_category,
    REPLAN_REASON_BIN_STEPS,
    RunReportData,
    _valid_attack_metrics,
    _recovery_trust_metrics,
)


SUMMARY_FIELDS = {
    "method": "source_linked",
    "seed": "15",
    "manifest_hash": "same-map",
    "malicious_robot_id": "0",
    "steps_completed": "10",
    "benign_total_deliveries_completed": "2",
    "benign_success_rate": "1.0",
    "benign_deliveries_after_attack": "1",
    "benign_deliveries_after_distrust": "1",
    "benign_no_path_steps": "0",
    "benign_movement_steps": "8",
    "benign_total_distance": "12",
    "benign_total_replans": "3",
    "benign_productive_replans": "2",
    "benign_blocked_world": "1",
    "benign_traffic_wait_steps": "2",
    "vertex_conflicts_detected": "1",
    "head_on_swap_conflicts_detected": "0",
    "reservation_conflicts_detected": "0",
    "traffic_replans": "1",
    "traffic_yield_events": "1",
    "deadlocks_detected": "0",
    "deadlocks_recovered": "0",
    "robot_overlap_violations": "0",
    "time_to_distrust_malicious_robot": "5",
    "malicious_verified_false_reports": "2",
    "final_attacker_trust_mean": "0.3",
}


def _write_run(root, method="source_linked", manifest_hash="same-map"):
    root.mkdir(parents=True, exist_ok=True)
    summary = dict(SUMMARY_FIELDS, method=method, manifest_hash=manifest_hash)
    with (root / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summary))
        writer.writeheader(); writer.writerow(summary)
    fields = [
        "step", "phase", "method", "robot_id", "position", "goal", "deliveries_completed",
        "benign_no_path_steps", "benign_movement_steps", "benign_total_distance", "benign_total_replans",
        "attacker_trust", "attacker_is_trusted", "trust_threshold", "active_fake_claim_count",
        "influential_fake_claim_count", "attacker_attributable_cost_on_route",
        "preferred_route_affected_by_attacker", "route_affected_by_attacker",
    ]
    rows = []
    for step in range(0, 10, 2):
        phase = "RECONNAISSANCE" if step < 4 else "ATTACK" if step < 8 else "RECOVERY"
        for rid in (0, 1, 2):
            rows.append({
                "step": step, "phase": phase, "method": method, "robot_id": rid,
                "position": str((rid + 1, step + 1)), "goal": str((8, 8)),
                "deliveries_completed": step // 6,
                "benign_no_path_steps": 0, "benign_movement_steps": step,
                "benign_total_distance": step, "benign_total_replans": step // 3,
                "attacker_trust": "0.7" if rid == 0 else str(max(0.0, 0.7 - step * 0.1)),
                "attacker_is_trusted": "True" if rid == 0 or step < 4 else "False",
                "trust_threshold": "0.55", "active_fake_claim_count": 4,
                "influential_fake_claim_count": 2 if rid == 1 else 1,
                "attacker_attributable_cost_on_route": "0.2" if rid == 1 else "0.1",
                "preferred_route_affected_by_attacker": "True" if rid == 1 and step >= 4 else "False",
                "route_affected_by_attacker": "True" if rid == 1 and step >= 4 else "False",
            })
    with (root / "robot_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "kind"])
        writer.writeheader(); writer.writerows([
            {"step": 4, "kind": "attacker_distrusted"},
            {"step": 8, "kind": "attacker_retrusted"},
        ])
    (root / "scenario_manifest.json").write_text(json.dumps({"map_hash": manifest_hash, "static_grid": [[0] * 10 for _ in range(10)]}), encoding="utf-8")


def test_parsers_are_explicit_and_safe():
    assert parse_bool("False") is False
    assert parse_bool("true") is True
    assert parse_tuple("(10, 20)") == (10, 20)


def test_individual_report_smoke(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    result = generate_run_report(run)
    required = {name for name in RUN_PLOTS if not name.startswith("04b_") and not name.startswith("04c_") and not name.startswith("06b_")}
    assert required.issubset(result["generated"])
    assert all((run / "plots" / name).exists() and (run / "plots" / name).stat().st_size > 0 for name in result["generated"])
    assert (run / "report_summary.txt").exists()
    assert (run / "plot_manifest.json").exists()


def test_comparison_report_and_derived_metrics(tmp_path):
    root = tmp_path / "comparison"
    for method in ("full_trust", "majority_vote", "trust_fused", "source_linked"):
        _write_run(root / method, method)
    result = generate_comparison_report(root)
    assert set(COMPARISON_PLOTS) == set(result["generated"])
    assert (root / "comparison_summary.csv").exists()
    assert (root / "comparison_report.txt").exists()
    text = (root / "comparison_summary.csv").read_text(encoding="utf-8")
    assert "productive_replan_ratio" in text
    assert "deliveries_per_1000_steps" in text


def test_comparison_rejects_manifest_mismatch(tmp_path):
    root = tmp_path / "comparison"
    _write_run(root / "full_trust", "full_trust", "map-a")
    _write_run(root / "source_linked", "source_linked", "map-b")
    try:
        generate_comparison_report(root)
    except ValueError as exc:
        assert "manifest mismatch" in str(exc)
    else:
        raise AssertionError("manifest mismatch was not rejected")


def test_old_schema_skips_optional_influence_plot(tmp_path):
    run = tmp_path / "old"
    _write_run(run)
    rows = list(csv.DictReader((run / "robot_timeseries.csv").open(newline="", encoding="utf-8")))
    rows = [{key: value for key, value in row.items() if "fake_claim" not in key and "attacker_attributable" not in key and "route_affected" not in key} for row in rows]
    with (run / "robot_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = generate_run_report(run)
    assert "02_fake_claim_influence.png" not in result["generated"]
    assert "01_attacker_trust_over_time.png" in result["generated"]


def test_replan_reason_categories_are_semantically_separate():
    assert _replan_category("malicious_report_on_route") == "malicious report on route"
    assert _replan_category("honest_report_on_route") == "honest report on route"
    assert _replan_category("blocked_world") == "real/world blockage"
    assert _replan_category("path_invalid_or_empty") == "path invalid / empty"


def test_replan_reason_bin_size_is_explicit():
    assert REPLAN_REASON_BIN_STEPS == 100


def test_unsampled_trust_event_is_retained_in_report(tmp_path):
    run = tmp_path / "trust_event"
    _write_run(run)
    with (run / "events.csv").open("a", newline="", encoding="utf-8") as handle:
        handle.write("513,attacker_distrusted\n")
    rows = list(csv.DictReader((run / "events.csv").open(newline="", encoding="utf-8")))
    rows[-1].update({"robot_id": "1", "current_trust": "0.48"})
    with (run / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    result = generate_run_report(run)
    assert "01_attacker_trust_over_time.png" in result["generated"]


def test_multiseed_attack_metrics_use_only_benign_attack_samples():
    rows = []
    for rid, fake, cost, affected in ((0, 99, 9, "True"), (1, 2, 0.4, "True"), (2, 0, 0.2, "False")):
        rows.append({"robot_id": str(rid), "step": "5", "phase": "ATTACK",
                     "influential_fake_claim_count": str(fake),
                     "attacker_attributable_cost_on_route": str(cost),
                     "preferred_route_affected_by_attacker": affected})
    data = RunReportData(None, {"malicious_robot_id": "0"}, rows, [], {}, [])
    metrics = _valid_attack_metrics(data)
    assert metrics["attack_mean_influential_fake_cells"] == 1
    assert metrics["attack_fraction_samples_influenced"] == .5
    assert math.isclose(metrics["attack_mean_attacker_route_cost"], .3)
    assert metrics["attack_fraction_route_affected"] == .5


def test_recovery_trust_gain_is_benign_recovery_delta():
    rows = []
    for rid, start, final in ((1, .2, .7), (2, .4, .8)):
        rows.extend([
            {"robot_id": str(rid), "step": "10", "phase": "RECOVERY", "attacker_trust": str(start)},
            {"robot_id": str(rid), "step": "20", "phase": "RECOVERY", "attacker_trust": str(final)},
        ])
    data = RunReportData(None, {"malicious_robot_id": "0"}, rows, [], {}, [])
    metrics = _recovery_trust_metrics(data)
    assert math.isclose(metrics["recovery_start_attacker_trust_mean"], .3)
    assert math.isclose(metrics["recovery_trust_gain"], .45)
