from types import SimpleNamespace

from map_poisoning.config import SimulationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType
from map_poisoning.reporting import (
    PHYSICAL_AI_FIGURES,
    REFERENCE_DELAY_PUBLICATION_LABEL,
    REFERENCE_DELAY_PUBLICATION_METRIC,
    REFERENCE_FIGURE_METHODS,
    REFERENCE_METHOD_LABELS,
    _aggregate_treatment_rows,
    _plot_paper_a5,
    _plot_reference_delay,
    _reference_filter_runs,
    _reference_manifest_hashes,
    _write_influence_probe,
)
from map_poisoning.reference_experiments import REFERENCE_ATTACK_INTENSITIES, REFERENCE_DELAY_LEVELS
from map_poisoning.trust import make_trust_model


def test_reference_suite_has_exactly_four_methods_and_filters_others():
    runs = [(1, method, SimpleNamespace()) for method in (*REFERENCE_FIGURE_METHODS, "latest_report")]
    assert [method for _, method, _ in _reference_filter_runs(runs)] == list(REFERENCE_FIGURE_METHODS)


def test_new_measurement_controls_preserve_default_behavior():
    config = SimulationConfig()
    assert config.honest_report_delay_steps == 0
    assert config.logging.measure_fusion_runtime is False
    assert config.condition_type == "baseline"


def test_reference_sweep_levels_are_configurable_constants():
    assert len(REFERENCE_ATTACK_INTENSITIES) == 5
    assert REFERENCE_DELAY_LEVELS == (0, 100, 250, 299, 300, 360)
    assert REFERENCE_DELAY_LEVELS[0] == 0
    assert 299 in REFERENCE_DELAY_LEVELS and 300 in REFERENCE_DELAY_LEVELS
    assert REFERENCE_DELAY_LEVELS[-1] > 300


def test_physical_ai_method_order_and_display_labels_are_exact():
    assert REFERENCE_FIGURE_METHODS == ("full_trust", "majority_vote", "trust_fused", "source_memory")
    assert [REFERENCE_METHOD_LABELS[method] for method in REFERENCE_FIGURE_METHODS] == [
        "Full trust", "Majority vote", "Trust-fused", "Proposed",
    ]


def test_physical_ai_folder_inventory_contains_only_eleven_target_figures():
    assert len(PHYSICAL_AI_FIGURES) == 11
    assert len(set(PHYSICAL_AI_FIGURES)) == 11
    assert all(filename.endswith(".png") for filename in PHYSICAL_AI_FIGURES)
    assert "09_legitimate_operational_ignore_vs_delay.png" in PHYSICAL_AI_FIGURES
    assert "09_legitimate_rejection_vs_delay.png" not in PHYSICAL_AI_FIGURES


def test_attack_sweep_aggregation_uses_configured_treatment_not_observed_actions():
    raw = [
        {"method": method, "configured_rate": 12.5, "actual_attack_actions": 999, "map_error_mean": .02}
        for method in REFERENCE_FIGURE_METHODS
    ]
    rows = _aggregate_treatment_rows(raw, "configured_rate", "map_error_mean", transform=lambda value: value * 100)
    assert {row["treatment_level"] for row in rows} == {12.5}
    assert {row["mean"] for row in rows} == {2.0}
    assert {row["actual_attack_actions_mean"] for row in rows} == {999.0}


def test_multilevel_manifest_validation_keeps_each_treatment_and_seed_separate():
    runs = []
    for treatment in ("low", "nominal"):
        for method in REFERENCE_FIGURE_METHODS:
            manifest_hash = f"{treatment}-shared"
            if treatment == "low" and method == "source_memory":
                manifest_hash = "low-mismatch"
            runs.append((1, method, SimpleNamespace(summary={"scenario_manifest_hash": manifest_hash}, treatment=treatment)))

    _, seed_only_mismatches = _reference_manifest_hashes(runs)
    _, treatment_mismatches = _reference_manifest_hashes(
        runs,
        group_by=lambda seed, _method, data: (data.treatment, seed),
    )

    # The legacy seed-only grouping would overwrite low with nominal rows.
    assert seed_only_mismatches == {}
    assert ("low", 1) in treatment_mismatches


def test_a5_blocks_when_any_required_method_has_fewer_than_five_recoveries(tmp_path):
    raw = []
    for method in REFERENCE_FIGURE_METHODS:
        count = 4 if method == "source_memory" else 5
        raw.extend({"method": method, "recovery_status": "recovered", "recovery_time_steps": index + 1} for index in range(count))
        raw.append({"method": method, "recovery_status": "never_affected", "recovery_time_steps": 0})
    assert _plot_paper_a5(raw, tmp_path / "a5.png") is False
    assert not (tmp_path / "a5.png").exists()


def test_influence_probe_uses_all_four_production_methods(tmp_path):
    rows = _write_influence_probe(tmp_path / "probe.csv", decay_rate=.006, max_claim_age=10, threshold=.5)
    assert {row["method"] for row in rows} == set(REFERENCE_FIGURE_METHODS)
    assert all("normalized_influence_percent" in row for row in rows)
    assert {row["production_function_used"] for row in rows} == {
        "FusionEngine.operational_weight -> DefenseMethodRunner.active_claim_weight"
    }
    assert all(row["probe_sensor_confidence"] == 1.0 for row in rows)


def test_influence_probe_values_equal_direct_production_weights(tmp_path):
    max_claim_age = 10
    rows = _write_influence_probe(
        tmp_path / "probe.csv", decay_rate=.006, max_claim_age=max_claim_age, threshold=.5,
    )
    for method in REFERENCE_FIGURE_METHODS:
        trust = make_trust_model(
            "bayesian", 9.0, 1.0, evidence_cap=12.0,
            confirmation_multiplier=.25, contradiction_multiplier=6.0,
            memory_recovery_rate=.05,
        )
        fusion = FusionEngine(
            method, trust.score, trust_memory_score=trust.memory_score,
            decay_rate=.006, max_claim_age=max_claim_age, trust_threshold=.5,
        )
        report = ClaimReport("direct-production", 7, (2, 2), ClaimType.BLOCKED, 0, sensor_confidence=1.0)
        fusion.add(report)
        expected = [fusion.operational_weight(report, age) for age in range(max_claim_age + 1)]
        observed = [
            row["raw_influence"] for row in rows
            if row["method"] == method
        ]
        assert observed == expected


def test_delay_publication_plot_uses_operational_ignore_not_admission_rejection(monkeypatch, tmp_path):
    raw = []
    for method in REFERENCE_FIGURE_METHODS:
        raw.extend([
            {"method": method, "configured_honest_report_delay_steps": 0,
             "honest_operational_ignore_rate": 0.0, "honest_rejection_rate": 0.0},
            {"method": method, "configured_honest_report_delay_steps": 300,
             "honest_operational_ignore_rate": 0.75, "honest_rejection_rate": 0.0},
        ])
    captured = {}

    def capture(fig, _path):
        axis = fig.axes[0]
        captured["ylabel"] = axis.get_ylabel()
        captured["last_values"] = [line.get_ydata()[-1] for line in axis.lines]

    monkeypatch.setattr("map_poisoning.reporting._save", capture)
    assert _plot_reference_delay(raw, tmp_path / "fig9.png") is True
    assert REFERENCE_DELAY_PUBLICATION_METRIC == "honest_operational_ignore_rate"
    assert captured["ylabel"] == REFERENCE_DELAY_PUBLICATION_LABEL
    assert captured["last_values"] == [75.0] * len(REFERENCE_FIGURE_METHODS)
