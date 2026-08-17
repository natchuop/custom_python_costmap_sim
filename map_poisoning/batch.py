"""Serial, paired multi-seed experiment execution."""
from __future__ import annotations
import csv, hashlib, json, time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from .application import run
from .config import ALL_METHODS, SimulationConfig
from .scenario import load_manifest, scenario_manifest_hash

def parse_seed_spec(text: str) -> tuple[int, ...]:
    if not text or not str(text).strip():
        raise ValueError("seed specification is empty")
    seeds = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            raise ValueError("invalid empty seed item")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"invalid seed range: {token}")
            start, end = (int(part) for part in parts)
            if start < 0 or end < start:
                raise ValueError(f"invalid seed range: {token}")
            seeds.update(range(start, end + 1))
        elif token.isdigit():
            seeds.add(int(token))
        else:
            raise ValueError(f"invalid seed: {token}")
    return tuple(sorted(seeds))

def experiment_config_hash(config: SimulationConfig) -> str:
    value = config.to_dict()
    value.pop("seed", None)
    value["fusion"] = dict(value["fusion"]); value["fusion"].pop("method", None)
    value["logging"] = dict(value["logging"]); value["logging"].pop("output_directory", None); value["logging"].pop("generate_plots", None)
    value["visualization"] = dict(value["visualization"]); value["visualization"].pop("animation", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

@dataclass
class MultiSeedRunRecord:
    seed: int; method: str; status: str; output_directory: str
    scenario_manifest_hash: str = ""; error: str = ""

@dataclass
class MultiSeedResult:
    root: Path
    records: list[MultiSeedRunRecord]

def _now(): return datetime.now(timezone.utc).isoformat()

def _valid_resume(path: Path, seed: int, method: str, manifest_hash: str, config_hash: str) -> bool:
    summary_path = path / "run_summary.csv"
    effective_path = path / "effective_config.json"
    if not summary_path.exists() or not effective_path.exists(): return False
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows: return False
    row = rows[0]
    try: effective = json.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return False
    return (str(row.get("seed")) == str(seed) and row.get("method") == method
            and row.get("scenario_manifest_hash") == manifest_hash
            and effective.get("experiment_config_hash") == config_hash)

def run_multiseed(config: SimulationConfig, seeds: tuple[int, ...], *, methods: tuple[str, ...], comparison: bool = True, resume: bool = False, generate_per_run_plots: bool = False, fail_fast: bool = False) -> MultiSeedResult:
    seeds = tuple(seeds)
    methods = tuple(methods)
    if not seeds: raise ValueError("at least one seed is required")
    if not methods: raise ValueError("at least one defense method is required")
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown: raise ValueError(f"unknown defense method(s): {', '.join(unknown)}")
    if not comparison and len(methods) != 1: raise ValueError("non-comparison batch requires exactly one method")
    root = Path(config.logging.output_directory); root.mkdir(parents=True, exist_ok=True)
    cfg_hash = experiment_config_hash(config)
    records = []; status_rows = []
    for index, seed in enumerate(seeds, 1):
        seed_root = root / f"seed_{seed:04d}"; seed_root.mkdir(parents=True, exist_ok=True)
        seed_cfg = replace(config, seed=seed, logging=replace(config.logging, output_directory=str(seed_root), generate_plots=generate_per_run_plots), visualization=replace(config.visualization, animation=False))
        manifest_path = seed_root / "scenario_manifest.json"
        try:
            if manifest_path.exists(): manifest = load_manifest(manifest_path)
            else: manifest = run(seed_cfg, manifest_only=True)
            manifest_hash = scenario_manifest_hash(manifest)
            print(f"[seed {index:02d}/{len(seeds)}] manifest ready")
            for method in methods:
                output = seed_root / method
                started = time.monotonic(); status = "completed"; error = ""
                method_cfg = replace(seed_cfg, manifest_path=str(manifest_path), fusion=replace(seed_cfg.fusion, method=method), logging=replace(seed_cfg.logging, output_directory=str(output)))
                if resume and _valid_resume(output, seed, method, manifest_hash, cfg_hash):
                    status = "skipped_resume"
                else:
                    try:
                        run(method_cfg, comparison=False)
                        effective = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
                        effective["experiment_config_hash"] = cfg_hash
                        (output / "effective_config.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception as exc:
                        status = "failed"; error = str(exc)
                        if fail_fast: raise
                records.append(MultiSeedRunRecord(seed, method, status, str(output), manifest_hash, error))
                status_rows.append({"seed": seed, "method": method, "status": status, "output_directory": str(output), "scenario_manifest_hash": manifest_hash, "started_at": _now(), "finished_at": _now(), "duration_seconds": round(time.monotonic()-started, 3), "error": error})
                print(f"[seed {seed:02d}/{len(seeds)}] {method} ... {status}")
        except Exception as exc:
            if fail_fast: raise
            for method in methods:
                records.append(MultiSeedRunRecord(seed, method, "failed", str(seed_root / method), "", str(exc)))
                status_rows.append({"seed": seed, "method": method, "status": "failed", "output_directory": str(seed_root / method), "scenario_manifest_hash": "", "started_at": _now(), "finished_at": _now(), "duration_seconds": 0, "error": str(exc)})
    with (root / "batch_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed","method","status","output_directory","scenario_manifest_hash","started_at","finished_at","duration_seconds","error"]); writer.writeheader(); writer.writerows(status_rows)
    (root / "batch_config.json").write_text(json.dumps({"config": config.to_dict(), "seeds": seeds, "methods": methods, "experiment_config_hash": cfg_hash}, indent=2, default=str), encoding="utf-8")
    _write_seed_index(root, records)
    from .reporting import generate_multiseed_report
    generate_multiseed_report(root, formats=(config.logging.plot_format,) if config.logging.generate_plots else ())
    return MultiSeedResult(root, records)

def _write_seed_index(root, records):
    with (root / "seed_manifest_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed","status","scenario_manifest_hash"]); writer.writeheader()
        for seed in sorted({record.seed for record in records}):
            cells = [record for record in records if record.seed == seed]
            writer.writerow({"seed": seed, "status": "completed" if all(cell.status in {"completed", "skipped_resume"} for cell in cells) else "failed", "scenario_manifest_hash": next((cell.scenario_manifest_hash for cell in cells if cell.scenario_manifest_hash), "")})
