"""Serial, paired multi-seed experiment execution."""
from __future__ import annotations
import csv, hashlib, json, subprocess, tempfile, time
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

def code_revision_metadata() -> dict:
    """Return an auditable revision identity for resume safety."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--ignore-submodules=all"], text=True)
        dirty = bool(diff.strip())
        payload = (commit + "\n" + diff.replace("\r\n", "\n")).encode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = False
        payload_parts = []
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            payload_parts.append(path.name.encode() + b"\0" + path.read_bytes())
        payload = b"".join(payload_parts)
    return {"git_commit": commit, "git_dirty": dirty,
            "code_revision_hash": hashlib.sha256(payload).hexdigest()}

@dataclass
class MultiSeedRunRecord:
    seed: int; method: str; status: str; output_directory: str
    scenario_manifest_hash: str = ""; error: str = ""

@dataclass
class MultiSeedResult:
    root: Path
    records: list[MultiSeedRunRecord]

def _now(): return datetime.now(timezone.utc).isoformat()

def _valid_resume(path: Path, seed: int, method: str, manifest_hash: str, config_hash: str, code_hash: str | None = None) -> bool:
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
            and effective.get("experiment_config_hash") == config_hash
            and (code_hash is None or effective.get("code_revision_hash") == code_hash))

def _atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer(handle)
        handle.flush()
    temporary.replace(path)

def _write_batch_status_atomic(root, rows):
    fields = ["seed", "method", "status", "output_directory", "scenario_manifest_hash", "experiment_config_hash", "code_revision_hash", "git_commit", "git_dirty", "started_at", "finished_at", "duration_seconds", "error"]
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    _atomic_write(Path(root) / "batch_status.csv", write)

def _write_batch_config_atomic(root, config):
    _atomic_write(Path(root) / "batch_config.json", lambda handle: handle.write(json.dumps(config, indent=2, default=str)))

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
    revision = code_revision_metadata()
    records = []; status_rows = []
    batch_config = {"config": config.to_dict(), "seeds": list(seeds), "methods": list(methods), "experiment_config_hash": cfg_hash, **revision}
    _write_batch_config_atomic(root, batch_config)
    _write_batch_status_atomic(root, status_rows)
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
                started_at = _now(); started = time.monotonic(); status = "completed"; error = ""
                method_cfg = replace(seed_cfg, manifest_path=str(manifest_path), fusion=replace(seed_cfg.fusion, method=method), logging=replace(seed_cfg.logging, output_directory=str(output)))
                if resume and _valid_resume(output, seed, method, manifest_hash, cfg_hash, revision["code_revision_hash"]):
                    status = "skipped_resume"
                else:
                    try:
                        run(method_cfg, comparison=False)
                        effective = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
                        effective.update({"experiment_config_hash": cfg_hash, **revision})
                        (output / "effective_config.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception as exc:
                        status = "failed"; error = str(exc)
                        if fail_fast: raise
                records.append(MultiSeedRunRecord(seed, method, status, str(output), manifest_hash, error))
                status_rows.append({"seed": seed, "method": method, "status": status, "output_directory": str(output), "scenario_manifest_hash": manifest_hash, "experiment_config_hash": cfg_hash, **revision, "started_at": started_at, "finished_at": _now(), "duration_seconds": round(time.monotonic()-started, 3), "error": error})
                _write_batch_status_atomic(root, status_rows)
                _write_seed_index(root, records, seeds=seeds, methods=methods, config_hash=cfg_hash)
                print(f"[seed {seed:02d}/{len(seeds)}] {method} ... {status}")
        except Exception as exc:
            if fail_fast: raise
            for method in methods:
                records.append(MultiSeedRunRecord(seed, method, "failed", str(seed_root / method), "", str(exc)))
                now = _now(); status_rows.append({"seed": seed, "method": method, "status": "failed", "output_directory": str(seed_root / method), "scenario_manifest_hash": "", "experiment_config_hash": cfg_hash, **revision, "started_at": now, "finished_at": now, "duration_seconds": 0, "error": str(exc)})
                _write_batch_status_atomic(root, status_rows)
                _write_seed_index(root, records, seeds=seeds, methods=methods, config_hash=cfg_hash)
    _write_batch_config_atomic(root, batch_config)
    _write_batch_status_atomic(root, status_rows)
    _write_seed_index(root, records, seeds=seeds, methods=methods, config_hash=cfg_hash)
    from .reporting import generate_multiseed_report
    generate_multiseed_report(root, formats=(config.logging.plot_format,) if config.logging.generate_plots else ())
    try:
        validation = json.loads((root / "aggregate" / "batch_validation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation = {}
    (root / "RESULTS_INDEX.txt").write_text(
        f"Multi-seed results\n\nRequested seeds: {', '.join(map(str, seeds))}\nRequested methods: {', '.join(methods)}\n"
        f"Batch valid: {validation.get('valid', 'unknown')}\n\nRaw runs: {root / 'seed_XXXX' / '<method>'}\n"
        f"Statistics: {root / 'aggregate'}\nAggregate report: {root / 'aggregate' / 'multiseed_report.txt'}\nAggregate plots: {root / 'aggregate' / 'plots'}\n\n"
        "Per-run plots are disabled by default for multi-seed runs; use --per-run-plots to enable them.\n"
        "Individual report example: python -m map_poisoning.reporting outputs/map005_30seed_comparison/seed_0001/source_linked --run\n"
        "Aggregate regeneration: python -m map_poisoning.reporting outputs/map005_30seed_comparison --multiseed\n", encoding="utf-8")
    print(f"Multi-seed results:\n  raw runs:       {root / 'seed_XXXX' / '<method>'}\n  statistics:     {root / 'aggregate'}\n  aggregate plots:{root / 'aggregate' / 'plots'}")
    return MultiSeedResult(root, records)

def _write_seed_index(root, records, *, seeds=None, methods=None, config_hash=""):
    requested = sorted(set(seeds or [record.seed for record in records]))
    methods = tuple(methods or sorted({record.method for record in records}))
    rows = []
    for seed in requested:
        cells = [record for record in records if record.seed == seed]
        complete = len(cells) == len(methods) and all(cell.status in {"completed", "skipped_resume"} for cell in cells)
        rows.append({"seed": seed, "status": "completed" if complete else ("failed" if any(cell.status == "failed" for cell in cells) else "pending"), "scenario_manifest_hash": next((cell.scenario_manifest_hash for cell in cells if cell.scenario_manifest_hash), ""), "experiment_config_hash": config_hash})
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=["seed", "status", "scenario_manifest_hash", "experiment_config_hash"]); writer.writeheader(); writer.writerows(rows)
    _atomic_write(Path(root) / "seed_manifest_index.csv", write)
