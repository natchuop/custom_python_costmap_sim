"""New simulator entry point.  Use --headless for automated runs."""
from __future__ import annotations
from map_poisoning.application import run
from map_poisoning.cli import config_from_args, parser

def main() -> int:
    args=parser().parse_args(); config=config_from_args(args)
    if args.seeds:
        from map_poisoning.batch import parse_seed_spec, run_multiseed
        seeds = parse_seed_spec(args.seeds)
        methods = tuple(x.strip() for x in args.methods.split(",") if x.strip()) if args.methods else (config.comparison_methods if args.compare else (config.fusion.method,))
        result = run_multiseed(config, seeds, methods=methods, comparison=len(methods) > 1, resume=args.resume, generate_per_run_plots=args.per_run_plots, fail_fast=args.fail_fast)
        print(f"Multi-seed results written to {result.root}")
        return 0
    if not args.headless:
        from map_poisoning.ui import launch
        launch(args); return 0
    results=run(config,comparison=args.compare,manifest_only=args.manifest_only)
    if args.manifest_only: print(f"Manifest written to {config.logging.output_directory}/scenario_manifest.json")
    else:
        for item in results: print(f"{item.method}: {item.output_directory} ({item.summary['attack_actions']} attack actions)")
    return 0

if __name__ == "__main__": raise SystemExit(main())
