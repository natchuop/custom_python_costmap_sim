"""Small, resumable orchestration for the four-method reference figures."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .batch import run_multiseed, parse_seed_spec
from .config import AttackConfig, LoggingConfig, SimulationConfig
from .reporting import REFERENCE_FIGURE_METHODS, generate_reference_report

REFERENCE_ATTACK_INTENSITIES = (
    ("very_low", 80, 90), ("low", 55, 65), ("nominal", 35, 40),
    ("high", 25, 30), ("very_high", 18, 22),
)
# Active claims are operationally excluded when age >= max_claim_age. These
# levels cover ordinary, near-cutoff, cutoff, and stale report ages for the
# production default of 300 steps without changing that production rule.
REFERENCE_DELAY_LEVELS = (0, 100, 250, 299, 300, 360)


def _condition_config(config, output, *, condition_type, delay=0, intensity=None, interval=None, measure_runtime=False):
    attacks = config.attacks if interval is None else replace(config.attacks, interval_min=interval[0], interval_max=interval[1])
    rate = None if interval is None else 1000.0 / ((interval[0] + interval[1]) / 2.0)
    return replace(
        config,
        comparison_methods=REFERENCE_FIGURE_METHODS,
        logging=replace(config.logging, output_directory=str(output), generate_plots=False, measure_fusion_runtime=measure_runtime),
        attacks=attacks,
        condition_type=condition_type,
        honest_report_delay_steps=delay,
        attack_intensity_condition=intensity,
        configured_attack_injections_per_1000_steps=rate,
    )


def _write_condition_metadata(config, output, *, condition_type, delay=0, intensity=None, interval=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "condition_type": condition_type,
        "methods": list(REFERENCE_FIGURE_METHODS),
        "enabled_attack_types": list(config.attacks.enabled),
        "configured_honest_report_delay_steps": delay,
        "attack_intensity_condition": intensity,
        "configured_attack_interval_min_steps": interval[0] if interval else None,
        "configured_attack_interval_max_steps": interval[1] if interval else None,
        "configured_attack_injections_per_1000_steps": (1000.0 / ((interval[0] + interval[1]) / 2.0)) if interval else None,
    }
    (output / "reference_condition.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_sweep_metadata(root, *, condition_type, treatments):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {"condition_type": condition_type, "treatments": list(treatments)}
    (root / "reference_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_reference_suite(config: SimulationConfig, seeds, output_directory: str | Path, *, include_sweeps=True,
                        resume=False, measure_runtime=True, only=None, intensity_levels=None,
                        delay_levels=None, generate_report=True):
    """Run a reference suite. Defaults are intentionally suitable for smoke tests."""
    root = Path(output_directory)
    results = {}
    only = only or "all"
    if only in {"all", "normal", "runtime"}:
        base = _condition_config(config, root / "baseline_multiseed", condition_type="baseline", measure_runtime=measure_runtime)
        _write_condition_metadata(base, root / "baseline_multiseed", condition_type="baseline")
        results["baseline"] = run_multiseed(base, tuple(seeds), methods=REFERENCE_FIGURE_METHODS, comparison=True, resume=resume)
    if only in {"all", "normal"}:
        no_attack = _condition_config(config, root / "no_attack", condition_type="no_attack", measure_runtime=measure_runtime)
        no_attack = replace(no_attack, attacks=replace(no_attack.attacks, enabled=()))
        _write_condition_metadata(no_attack, root / "no_attack", condition_type="no_attack")
        results["no_attack"] = run_multiseed(no_attack, tuple(seeds), methods=REFERENCE_FIGURE_METHODS, comparison=True, resume=resume)
    if include_sweeps and only in {"all", "attack_intensity"}:
        intensity_results = []
        selected_intensities = intensity_levels or REFERENCE_ATTACK_INTENSITIES
        intensity_root = root / "attack_intensity_sweep"
        _write_sweep_metadata(intensity_root, condition_type="attack_intensity", treatments=[item[0] if len(item) == 3 else item for item in selected_intensities])
        for item in selected_intensities:
            name, low, high = item if len(item) == 3 else next(level for level in REFERENCE_ATTACK_INTENSITIES if level[0] == item)
            current = _condition_config(config, intensity_root / name, condition_type="attack_intensity", intensity=name, interval=(low, high), measure_runtime=measure_runtime)
            _write_condition_metadata(current, intensity_root / name, condition_type="attack_intensity", intensity=name, interval=(low, high))
            intensity_results.append(run_multiseed(current, tuple(seeds), methods=REFERENCE_FIGURE_METHODS, comparison=True, resume=resume))
        results["attack_intensity"] = intensity_results
    if include_sweeps and only in {"all", "honest_delay"}:
        delay_results = []
        selected_delays = delay_levels or REFERENCE_DELAY_LEVELS
        delay_root = root / "delay_sweep"
        _write_sweep_metadata(delay_root, condition_type="honest_delay", treatments=[f"delay_{delay:04d}" for delay in selected_delays])
        for delay in selected_delays:
            current = _condition_config(config, delay_root / f"delay_{delay:04d}", condition_type="honest_delay", delay=delay, measure_runtime=measure_runtime)
            _write_condition_metadata(current, delay_root / f"delay_{delay:04d}", condition_type="honest_delay", delay=delay)
            delay_results.append(run_multiseed(current, tuple(seeds), methods=REFERENCE_FIGURE_METHODS, comparison=True, resume=resume))
        results["honest_delay"] = delay_results
    baseline_root = root / "baseline_multiseed"
    if generate_report and (baseline_root.exists() or (root / "batch_status.csv").exists()):
        generate_reference_report(root)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the four-method reference figure suite")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seeds", default="1-2", help="small smoke-test default; e.g. 1-10")
    parser.add_argument("--scenario-preset")
    parser.add_argument("--map-npy")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--only", choices=("all", "normal", "attack_intensity", "runtime", "honest_delay"), default="all")
    parser.add_argument("--intensity-levels", help="comma-separated named levels, or omit for all five")
    parser.add_argument("--honest-delay-levels", help="comma-separated step values, or omit for all six")
    parser.add_argument("--no-measure-fusion-runtime", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.render_only:
        generate_reference_report(args.output_directory)
        return 0
    config = SimulationConfig(
        logging=LoggingConfig(output_directory=str(args.output_directory), generate_plots=False),
        scenario_preset=args.scenario_preset, map_npy=args.map_npy, max_steps=args.max_steps,
    )
    levels = None
    if args.intensity_levels:
        known = {level[0]: level for level in REFERENCE_ATTACK_INTENSITIES}
        levels = [known[name.strip()] for name in args.intensity_levels.split(",")]
    delays = None if not args.honest_delay_levels else [int(value) for value in args.honest_delay_levels.split(",")]
    run_reference_suite(config, parse_seed_spec(args.seeds), args.output_directory,
                        include_sweeps=not args.no_sweeps, resume=args.resume,
                        measure_runtime=not args.no_measure_fusion_runtime,
                        only=args.only, intensity_levels=levels, delay_levels=delays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
