"""Versioned scenario manifests and defense-independent attack authoring."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
from .config import SimulationConfig
from .models import AttackEvent, AttackType, ClaimType, TemporaryObstacleEpisode
from .rng import derived_seed, named_rng
from .world import demo_grid, make_episodes

SCHEMA_VERSION = 1

@dataclass(frozen=True)
class ScenarioManifest:
    schema_version: int
    master_seed: int
    derived_seeds: dict[str, int]
    map_hash: str
    map_shape: tuple[int, int]
    static_grid: tuple[tuple[int, ...], ...]
    phase_boundaries: dict[str, int]
    malicious_robot_id: int
    benign_robot_ids: tuple[int, ...]
    obstacle_episodes: tuple[TemporaryObstacleEpisode, ...]
    attack_events: tuple[AttackEvent, ...]
    def to_dict(self): return asdict(self)

def _hash(grid) -> str: return hashlib.sha256(grid.tobytes()).hexdigest()
def _cell_choice(rng, cells): return cells[rng.randrange(min(len(cells), 12))]

def author_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    config.validate(); grid = demo_grid() if grid is None else grid
    phases = config.phases; episodes = make_episodes(grid, config.seed, phases.total_steps, config.temporary_blockage_change_period_steps)
    rng = named_rng(config.seed, "attack_scheduler")
    enabled = [AttackType(x) for x in config.attacks.enabled]
    benign = (1, 2, 3); sender = 0; events=[]; bag=[]; step = phases.recon_steps + rng.randint(config.attacks.interval_min, config.attacks.interval_max); index=0
    free = [(r,c) for r in range(1,grid.shape[0]-1) for c in range(1,grid.shape[1]-1) if not grid[r,c]]
    while step < phases.recon_steps + phases.attack_steps and enabled:
        if not bag:
            bag = enabled.copy(); rng.shuffle(bag)
        feasible = [kind for kind in bag if (kind == AttackType.FAKE_OBSTACLE or (kind == AttackType.FALSE_CLEARANCE and any(e.appearance_step <= step < e.clearance_step for e in episodes)) or (kind == AttackType.STALE_REASSERTION and any(e.clearance_step <= step for e in episodes)))]
        if feasible:
            kind = feasible[0]; bag.remove(kind)
            episode = None
            if kind == AttackType.FAKE_OBSTACLE: cell, claim, observation = _cell_choice(rng, free), ClaimType.BLOCKED, step
            elif kind == AttackType.FALSE_CLEARANCE:
                episode = _cell_choice(rng, [e for e in episodes if e.appearance_step <= step < e.clearance_step]); cell, claim, observation = episode.cells[0], ClaimType.FREE, step
            else:
                episode = _cell_choice(rng, [e for e in episodes if e.clearance_step <= step]); cell, claim, observation = episode.cells[0], ClaimType.BLOCKED, step
            eid=f"attack-{index:04}"; rid=f"report-{index:04}-00"
            events.append(AttackEvent(eid, step, kind, (cell,), claim, observation, sender, benign, (rid,), episode.episode_id if episode else None)); index += 1
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    names=("attack_scheduler", "temporary_obstacles", "robot_routes", "traffic")
    static_grid=tuple(tuple(int(value) for value in row) for row in grid)
    return ScenarioManifest(SCHEMA_VERSION, config.seed, {x:derived_seed(config.seed,x) for x in names}, _hash(grid), tuple(grid.shape), static_grid, {"reconnaissance_end":phases.recon_steps, "attack_end":phases.recon_steps+phases.attack_steps, "total":phases.total_steps}, sender, benign, episodes, tuple(events))

def save_manifest(manifest: ScenarioManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

def load_manifest(path: str | Path) -> ScenarioManifest:
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported scenario manifest schema")
    episodes=tuple(TemporaryObstacleEpisode(x["episode_id"], tuple(map(tuple,x["cells"])), x["appearance_step"],x["clearance_step"]) for x in raw["obstacle_episodes"])
    events=tuple(AttackEvent(x["event_id"],x["step"],AttackType(x["attack_type"]),tuple(map(tuple,x["cells"])),ClaimType(x["claim"]),x["observation_step"],x["sender_id"],tuple(x["recipients"]),tuple(x["report_ids"]),x.get("obstacle_episode_id")) for x in raw["attack_events"])
    return ScenarioManifest(raw["schema_version"],raw["master_seed"],raw["derived_seeds"],raw["map_hash"],tuple(raw["map_shape"]),tuple(tuple(row) for row in raw["static_grid"]),raw["phase_boundaries"],raw["malicious_robot_id"],tuple(raw["benign_robot_ids"]),episodes,events)
