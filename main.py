"""New simulator entry point.  Use --headless for automated runs."""
from __future__ import annotations
from map_poisoning.application import resolve_output_directory, run
from map_poisoning.cli import config_from_args, parser

def main() -> int:
    args=parser().parse_args(); config=config_from_args(args)
    if not args.headless:
        from map_poisoning.ui import launch
        launch(args); return 0
    if args.seeds:
        from map_poisoning.batch import parse_seed_spec, run_multiseed
        seeds = parse_seed_spec(args.seeds)
        methods = tuple(item.strip() for item in (args.methods or "trust_threshold,full_trust,majority_vote,trust_fused,source_linked").split(",") if item.strip())
        result = run_multiseed(config, seeds, methods=methods, comparison=True, resume=args.resume, generate_per_run_plots=args.per_run_plots, fail_fast=args.fail_fast)
        print(f"multi-seed results: {result.root} ({len(result.records)} jobs)")
        return 0
    results=run(config,comparison=args.compare,manifest_only=args.manifest_only)
    if args.manifest_only: print(f"Manifest written to {resolve_output_directory(config)}/scenario_manifest.json")
    else:
        for item in results: print(f"{item.method}: {item.output_directory} ({item.summary['attack_actions']} attack actions)")
    return 0

if __name__ == "__main__": raise SystemExit(main())
