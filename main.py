"""New simulator entry point.  Use --headless for automated runs."""
from __future__ import annotations
from map_poisoning.application import resolve_output_directory, run
from map_poisoning.cli import config_from_args, parser

def main() -> int:
    args=parser().parse_args(); config=config_from_args(args)
    if not args.headless:
        from map_poisoning.ui import launch
        launch(args); return 0
    results=run(config,comparison=args.compare,manifest_only=args.manifest_only)
    if args.manifest_only: print(f"Manifest written to {resolve_output_directory(config)}/scenario_manifest.json")
    else:
        for item in results: print(f"{item.method}: {item.output_directory} ({item.summary['attack_actions']} attack actions)")
    return 0

if __name__ == "__main__": raise SystemExit(main())
