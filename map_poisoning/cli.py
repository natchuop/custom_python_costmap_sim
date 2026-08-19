"""CLI parser shared with the GUI launcher."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ALL_METHODS, MAP_VIEWS, PRIMARY_METHODS, AttackConfig, FusionConfig, LoggingConfig, PhaseConfig, SimulationConfig, TrustConfig, VisualizationConfig
from .models import AttackType


def geometry_slug(map_npy: str | None = None, scenario_preset: str | None = None) -> str:
    if scenario_preset:
        return str(scenario_preset)
    if map_npy:
        parent = Path(map_npy).parent.name.strip()
        if parent:
            return parent
        stem = Path(map_npy).stem.strip()
        if stem:
            return stem
    return "default_warehouse"


def attack_slug(enabled) -> str:
    selected = {str(item) for item in (enabled or ())}
    ordered = [kind.value for kind in AttackType if kind.value in selected]
    if not ordered:
        return "no_attacks"
    if ordered == [kind.value for kind in AttackType]:
        return "all_attacks"
    return "+".join(ordered)


def enabled_attacks_from_args(args) -> tuple[str, ...]:
    text = getattr(args, "attacks", None)
    if not text or text == "none":
        return ()
    return tuple(item for item in str(text).split(",") if item)


def suggested_output_directory(
    *,
    method: str = "source_linked",
    seed: int = 15,
    seeds: str | None = None,
    compare: bool = False,
    map_npy: str | None = None,
    scenario_preset: str | None = None,
    enabled_attacks: tuple[str, ...] | None = None,
) -> str:
    """Return a descriptive folder under outputs/ so runs are not dumped into a generic directory."""
    geo = geometry_slug(map_npy, scenario_preset)
    attacks = attack_slug(enabled_attacks if enabled_attacks is not None else tuple(kind.value for kind in AttackType))
    seed_spec = str(seeds).replace(" ", "") if seeds else ""
    if seed_spec:
        name = f"compare_seeds{seed_spec}_{geo}_{attacks}" if compare else f"{method}_seeds{seed_spec}_{geo}_{attacks}"
        return str(Path("outputs") / "multiseed" / name)
    if compare:
        return str(Path("outputs") / "comparisons" / f"seed{seed}_{geo}_{attacks}")
    return str(Path("outputs") / "runs" / f"{method}_seed{seed}_{geo}_{attacks}")


def suggested_output_directory_from_args(args) -> str:
    compare = bool(getattr(args, "compare", False))
    selected = getattr(args, "comparison_methods", None) or ""
    if not compare and "," in str(selected):
        compare = True
    return suggested_output_directory(
        method=getattr(args, "defense_method", "source_linked"),
        seed=int(getattr(args, "seed", 15)),
        seeds=getattr(args, "seeds", None),
        compare=compare,
        map_npy=getattr(args, "map_npy", None),
        scenario_preset=getattr(args, "scenario_preset", None),
        enabled_attacks=enabled_attacks_from_args(args),
    )


def result_location_message(output_directory: str, *, compare: bool = False, multi_seed: bool = False) -> str:
    root = Path(output_directory)
    if multi_seed:
        return f"Created results in {root}\n\nAggregate diagrams:\n{root / 'aggregate' / 'plots'}"
    if compare:
        return (
            f"Created results in {root}\n\n"
            f"Comparison diagrams:\n{root / 'comparison_plots'}\n\n"
            f"Per-method diagrams:\n{root / '<method>' / 'plots'}"
        )
    return f"Created results in {root}\n\nDiagrams:\n{root / 'plots'}"


def parser():
    p=argparse.ArgumentParser(description="Modular multi-robot map-poisoning simulator")
    p.add_argument("--headless",action="store_true",help="Run without importing Tkinter")
    p.add_argument("--compare", action="store_true", help="Replay the same manifest for each comparison method")
    p.add_argument("--comparison-methods",default=None,help="comma-separated methods used with --compare or multi-select GUI runs")
    p.add_argument("--manifest-only",action="store_true",help="Author and save a manifest without replaying it")
    p.add_argument("--manifest",dest="manifest_path")
    p.add_argument("--map-npy"); p.add_argument("--map-movingai")
    p.add_argument("--scenario-preset", choices=("warehouse_002", "warehouse_005", "warehouse_005_rotated"))
    p.add_argument("--output-directory", default=None, help="defaults to a named folder under outputs/runs, outputs/comparisons, or outputs/multiseed")
    p.add_argument("--seed",type=int,default=15); p.add_argument("--defense-method",choices=ALL_METHODS,default="source_linked")
    p.add_argument("--seeds", help="multi-seed specification such as 1-30 or 1,5,10")
    p.add_argument("--methods", help="comma-separated methods for multi-seed mode")
    p.add_argument("--per-run-plots", action="store_true", help="generate individual plots in multi-seed mode")
    p.add_argument("--resume", action="store_true", help="resume matching completed multi-seed cells")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--trust-model",choices=("bayesian","scalar"),default="scalar"); p.add_argument("--admission-policy",choices=("auto_soft","accept_all","hard_reject"),default="accept_all")
    p.add_argument("--trust-threshold",type=float,default=0.55)
    p.add_argument("--attacks",default="fake_obstacle,false_clearance,stale_reassertion",help="comma separated, or 'none'")
    p.add_argument("--recon-steps",type=int,default=450); p.add_argument("--attack-steps",type=int,default=1200); p.add_argument("--recovery-steps",type=int,default=750); p.add_argument("--max-steps",type=int)
    p.add_argument("--deliveries-per-robot",type=int,default=100)
    p.add_argument("--attack-interval-min",type=int,default=50); p.add_argument("--attack-interval-max",type=int,default=50)
    p.add_argument("--map-view", choices=MAP_VIEWS, default="combined", help="belief visualization: combined peer/local or local observations")
    p.add_argument("--temp-obstacle-interval", type=int, default=150, help="steps between temporary-obstacle movements")
    p.add_argument("--no-animation",action="store_true"); p.add_argument("--no-plots",action="store_true",help="Do not generate PNG reports after CSV output"); return p

def config_from_args(args):
    enabled=() if args.attacks == "none" else tuple(x for x in args.attacks.split(",") if x)
    output_directory = args.output_directory or suggested_output_directory_from_args(args)
    raw_methods = getattr(args, "comparison_methods", None)
    comparison_methods = tuple(item.strip() for item in str(raw_methods).split(",") if item.strip()) if raw_methods else (args.defense_method,)
    return SimulationConfig(
        seed=args.seed,
        phases=PhaseConfig(args.recon_steps,args.attack_steps,args.recovery_steps),
        attacks=AttackConfig(enabled=enabled,interval_min=args.attack_interval_min,interval_max=args.attack_interval_max),
        trust=TrustConfig(model=args.trust_model, threshold=float(args.trust_threshold)),
        fusion=FusionConfig(method=args.defense_method,admission_policy=args.admission_policy),
        logging=LoggingConfig(output_directory, generate_plots=not args.no_plots),
        visualization=VisualizationConfig(
            animation=not args.no_animation and not args.headless,
            map_view=getattr(args, "map_view", "combined"),
        ),
        comparison_methods=comparison_methods or (args.defense_method,),
        manifest_path=args.manifest_path,
        map_npy=args.map_npy,
        map_movingai=args.map_movingai,
        scenario_preset=args.scenario_preset,
        max_steps=args.max_steps,
        deliveries_per_robot=args.deliveries_per_robot,
        temporary_blockage_change_period_steps=int(getattr(args, "temp_obstacle_interval", 150)),
    )
