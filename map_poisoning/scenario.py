"""Versioned scenario manifests and defense-independent attack authoring."""
from __future__ import annotations
import hashlib, json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .config import SimulationConfig
from .models import AttackEvent, AttackType, ClaimReport, ClaimType, DeliveryTask, ReportAuditLabel, TemporaryObstacleEpisode
from .rng import derived_seed, named_rng
from .obstacles import (
    FAKE_CENTER_MIN_SPACING,
    FAKE_MIN_REPORT_CELLS,
    author_temporary_obstacle_episodes,
    fake_report_cells,
    footprint_center,
    sample_fake_obstacle_dimensions,
)
from .world import demo_grid
from .planning import astar
from .scenario_presets import preset_for_hash, preset_for_id, validate_fixed_preset

SCHEMA_VERSION = 3

def scenario_manifest_hash(manifest: "ScenarioManifest") -> str:
    """Digest the complete canonical manifest, not only its static map."""
    payload = json.dumps(
        manifest.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

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
    scenario_id: str = ""
    protocol_id: str = "modular_v1"
    robot_starts: dict[int, tuple[int, int]] | None = None
    task_queues: dict[int, tuple[DeliveryTask, ...]] | None = None
    attacker_positions: tuple[tuple[int, int], ...] = ()
    honest_attacker_reports: tuple[ClaimReport, ...] = ()
    report_audit_labels: tuple[ReportAuditLabel, ...] = ()
    candidate_metadata: tuple[dict, ...] = ()
    authoring_warnings: tuple[str, ...] = ()
    reconnaissance_heatmap: tuple[tuple[int, ...], ...] | None = None
    scenario_preset: str | None = None
    def to_dict(self): return asdict(self)

def _hash(grid) -> str: return hashlib.sha256(grid.tobytes()).hexdigest()
def _cell_choice(rng, cells): return cells[rng.randrange(min(len(cells), 12))]

def _episode_is_active(episode, step: int) -> bool:
    return episode.appearance_step <= step < episode.clearance_step


def _feasible_attack_types(enabled: list[AttackType], step: int, episodes, grid) -> list[AttackType]:
    feasible = []
    for kind in enabled:
        if kind == AttackType.FAKE_OBSTACLE:
            feasible.append(kind)
        elif kind == AttackType.FALSE_CLEARANCE and any(_episode_is_active(episode, step) for episode in episodes):
            feasible.append(kind)
        elif kind == AttackType.STALE_REASSERTION and _cleared_reassertable(episodes, step, grid):
            feasible.append(kind)
    return feasible


def _active_temp_cells(episodes, step):
    cells = set()
    for episode in episodes:
        if _episode_is_active(episode, step):
            cells.update(episode.cells)
    return cells


def _center_cell(cells) -> tuple[int, int]:
    row, col = footprint_center(cells)
    return int(round(row)), int(round(col))


def _cleared_reassertable(episodes, step, grid):
    active = _active_temp_cells(episodes, step)
    result = []
    for episode in episodes:
        if episode.clearance_step > step:
            continue
        if any((not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1])) or grid[cell] or cell in active for cell in episode.cells):
            continue
        result.append(episode)
    return result


def _place_fake_cells(rng, route_cells, use_count, selected, config, grid, forbidden, active_cells):
    spacing = max(config.attacks.min_center_spacing, FAKE_CENTER_MIN_SPACING)
    eligible = [
        cell for cell in route_cells
        if use_count.get(cell, 0) < config.attacks.max_uses_per_footprint
        and all(abs(cell[0] - old[0]) + abs(cell[1] - old[1]) >= spacing for old in selected)
        and cell not in active_cells
        and cell not in forbidden
        and not grid[cell]
    ]
    rng.shuffle(eligible)
    candidates = eligible or [
        (row, col)
        for row in range(1, grid.shape[0] - 1)
        for col in range(1, grid.shape[1] - 1)
        if not grid[row, col] and (row, col) not in active_cells and (row, col) not in forbidden
    ]
    rng.shuffle(candidates)
    for center in candidates[:48]:
        for _ in range(8):
            height, width = sample_fake_obstacle_dimensions(rng)
            cells = fake_report_cells(center, height, width, grid, forbidden=forbidden, active_cells=active_cells)
            if len(cells) >= FAKE_MIN_REPORT_CELLS:
                return tuple(cells)
        compact = fake_report_cells(center, 2, 2, grid, forbidden=forbidden, active_cells=active_cells)
        if len(compact) >= FAKE_MIN_REPORT_CELLS:
            return tuple(compact)
    return None


def _instantiate_attack(kind, *, step, rng, episodes, route_cells, use_count, selected, config, grid, forbidden):
    """Place one attack of the requested kind, or return None if it cannot be sited."""
    if kind == AttackType.FAKE_OBSTACLE:
        cells = _place_fake_cells(rng, route_cells, use_count, selected, config, grid, forbidden, _active_temp_cells(episodes, step))
        if not cells:
            return None
        return cells, ClaimType.BLOCKED, step, None, _center_cell(cells)
    if kind == AttackType.FALSE_CLEARANCE:
        active = [episode for episode in episodes if _episode_is_active(episode, step)]
        if not active:
            return None
        episode = _cell_choice(rng, active)
        return tuple(episode.cells), ClaimType.FREE, step, episode, _center_cell(episode.cells)
    cleared = _cleared_reassertable(episodes, step, grid)
    if not cleared:
        return None
    episode = _cell_choice(rng, cleared)
    # Reassert the original blocked observation, not a new observation made at
    # the attack step after the physical obstacle has already cleared.
    return tuple(episode.cells), ClaimType.BLOCKED, max(0, episode.clearance_step - 1), episode, _center_cell(episode.cells)

def _nominal_route_cells(grid, starts, targets) -> list[tuple[int, int]]:
    """Clean-rollout corridor candidates shared by the manifest and robot tasks."""
    rows, cols=grid.shape
    routes=[]
    for index,start in enumerate(starts):
        for offset in (0,1,2):
            goal=targets[(index+offset)%len(targets)]
            route=astar(start,goal,lambda cell: float("inf") if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid[cell] else 1.0)
            if route: routes.extend(route)
    excluded=set(starts)|set(targets)
    return [cell for cell in routes if cell not in excluded]


def _largest_free_component(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return the largest four-connected free component in deterministic order."""
    rows, cols = grid.shape
    remaining = {
        (row, col) for row in range(rows) for col in range(cols) if not grid[row, col]
    }
    largest: list[tuple[int, int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return sorted(largest)


def _spread_free_cells(grid: np.ndarray, count: int) -> tuple[tuple[int, int], ...]:
    """Choose well-separated reachable cells for maps without a fixed preset."""
    component = _largest_free_component(grid)
    if len(component) < count:
        raise ValueError("map does not contain enough mutually reachable free cells")
    chosen = [component[len(component) // 2]]
    while len(chosen) < count:
        def nearest_distance(cell):
            return min(abs(cell[0] - old[0]) + abs(cell[1] - old[1]) for old in chosen)
        candidates = [cell for cell in component if cell not in chosen]
        chosen.append(max(candidates, key=lambda cell: (nearest_distance(cell), -cell[0], -cell[1])))
    return tuple(chosen)


def build_fixed_task_queues(benign_ids, delivery_points, deliveries_per_robot):
    """Use a cyclic, seed-independent pickup/dropoff schedule for each robot."""
    if len(delivery_points) < 2:
        raise ValueError("at least two delivery points are required")
    return {
        robot_id: tuple(
            DeliveryTask(
                f"r{robot_id}-task-{index}",
                delivery_points[(robot_id + index) % len(delivery_points)],
                delivery_points[(robot_id + index + 2) % len(delivery_points)],
            )
            for index in range(deliveries_per_robot)
        )
        for robot_id in benign_ids
    }

def author_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    config.validate(); grid = demo_grid() if grid is None else grid
    preset = preset_for_id(config.scenario_preset) if config.scenario_preset else preset_for_hash(_hash(grid))
    if config.scenario_preset:
        validate_fixed_preset(grid, preset)
    elif preset is not None:
        raise ValueError(f"map matches scenario preset {preset.preset_id}; pass --scenario-preset {preset.preset_id} for fixed experiment geometry")
    phases = config.phases
    rng = named_rng(config.seed, "attack_scheduler")
    type_rng = named_rng(config.seed, "attack_types")
    place_rng = named_rng(config.seed, "attack_placement")
    enabled = [AttackType(x) for x in config.attacks.enabled]
    benign = (1, 2); sender = 0; events=[]; preference_bag=[]; step = phases.recon_steps + rng.randint(config.attacks.interval_min, config.attacks.interval_max); index=0
    free = [(r,c) for r in range(1,grid.shape[0]-1) for c in range(1,grid.shape[1]-1) if not grid[r,c]]
    rows, cols = grid.shape
    if preset:
        starts_tuple = tuple(preset.robot_starts[index] for index in sorted(preset.robot_starts))
        targets = preset.delivery_points
    else:
        layout = _spread_free_cells(grid, 7)
        starts_tuple = layout[:3]
        targets = layout[3:]
    route_cells=_nominal_route_cells(grid, starts_tuple, targets) or free
    forbidden = set(starts_tuple)
    # Temporary obstacles are physical warehouse objects: several multi-cell
    # palettes/carts are active at once and relocate on the configured interval.
    episodes = author_temporary_obstacle_episodes(
        grid,
        named_rng(config.seed, "temporary_obstacles"),
        phases.total_steps,
        config.temporary_blockage_change_period_steps,
        forbidden_cells=forbidden,
    )
    candidate_metadata=[]; use_count: dict[tuple[int,int],int]={}; selected=[]
    while step < phases.recon_steps + phases.attack_steps and enabled:
        if not preference_bag:
            preference_bag = list(AttackType)
            type_rng.shuffle(preference_bag)
        preferred = preference_bag.pop(0)
        feasible = set(_feasible_attack_types(enabled, step, episodes, grid))
        if preferred in feasible:
            ordered = [preferred, *[kind for kind in enabled if kind != preferred and kind in feasible]]
        else:
            ordered = [kind for kind in enabled if kind in feasible]
        for kind in ordered:
            placed = _instantiate_attack(
                kind, step=step, rng=place_rng, episodes=episodes, route_cells=route_cells,
                use_count=use_count, selected=selected, config=config, grid=grid, forbidden=forbidden,
            )
            if placed is None:
                continue
            cells, claim, observation, episode, center = placed
            eid=f"attack-{index:04}"
            rids=tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells)))
            events.append(AttackEvent(eid, step, kind, tuple(cells), claim, observation, sender, benign, rids, episode.episode_id if episode else None)); index += 1
            use_count[center]=use_count.get(center,0)+1; selected.append(center)
            candidate_metadata.append({"candidate_id":f"candidate-{index-1:04}","center":center,"footprint_cells":[list(cell) for cell in cells],"traffic_score":None,"bottleneck_score":None,"estimated_detour_score":None,"rank":None,"selection_weight":None,"prior_use_count":use_count[center]-1})
            break
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    names=("attack_scheduler", "attack_types", "attack_placement", "temporary_obstacles", "robot_routes", "traffic")
    static_grid=tuple(tuple(int(value) for value in row) for row in grid)
    starts = dict(preset.robot_starts) if preset else {0: starts_tuple[0], 1: starts_tuple[1], 2: starts_tuple[2]}
    # Keep the attacker physically active with the same deterministic repeating
    # queue as the other robots; only its reporting behavior is malicious.
    queues=build_fixed_task_queues((sender, *benign), targets, config.deliveries_per_robot)
    if config.deliveries_per_robot < len(targets):
        # Keep the attacker moving through the fixed checkpoint cycle even in
        # short smoke runs where benign queues intentionally contain one task.
        queues[sender] = build_fixed_task_queues(
            (sender,), targets, len(targets)
        )[sender]
    warnings=() if len(set(selected)) >= min(config.attacks.min_unique_footprints,len(selected)) else ("concentrated_attack_manifest",)
    # Script the attacker independently of defense-dependent benign routes.
    attacker_route=tuple(route_cells) or tuple(free)
    positions=tuple(attacker_route[step % len(attacker_route)] for step in range(phases.total_steps))
    def truth(cell, step):
        return ClaimType.BLOCKED if any(cell in episode.cells and episode.appearance_step <= step < episode.clearance_step for episode in episodes) else ClaimType.FREE
    honest=tuple(ClaimReport(
        f"attacker-honest-{step:05}", sender, positions[step],
        truth(positions[step], step), step, sensor_confidence=1.0
    ) for step in range(0, phases.total_steps, config.communication_period_steps))
    labels=tuple(
        ReportAuditLabel(
            report_id,
            True,
            event.attack_type,
            event.obstacle_episode_id,
            ClaimType.BLOCKED
            if event.attack_type in {AttackType.FALSE_CLEARANCE, AttackType.STALE_REASSERTION}
            else ClaimType.FREE,
        )
        for event in events for report_id in event.report_ids
    )
    return ScenarioManifest(SCHEMA_VERSION, config.seed, {x:derived_seed(config.seed,x) for x in names}, _hash(grid), tuple(grid.shape), static_grid, {"reconnaissance_end":phases.recon_steps, "attack_end":phases.recon_steps+phases.attack_steps, "total":phases.total_steps}, sender, benign, episodes, tuple(events), scenario_id=f"scenario-{config.seed}-{_hash(grid)[:12]}", protocol_id="modular_v1", robot_starts=starts, task_queues=queues, attacker_positions=positions, honest_attacker_reports=honest, report_audit_labels=labels, candidate_metadata=tuple(candidate_metadata), authoring_warnings=warnings, reconnaissance_heatmap=None, scenario_preset=config.scenario_preset)

def save_manifest(manifest: ScenarioManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

def load_manifest(path: str | Path) -> ScenarioManifest:
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported scenario manifest schema; author a schema-v3 manifest")
    episodes=tuple(TemporaryObstacleEpisode(x["episode_id"], tuple(map(tuple,x["cells"])), x["appearance_step"],x["clearance_step"]) for x in raw["obstacle_episodes"])
    events=tuple(AttackEvent(x["event_id"],x["step"],AttackType(x["attack_type"]),tuple(map(tuple,x["cells"])),ClaimType(x["claim"]),x["observation_step"],x["sender_id"],tuple(x["recipients"]),tuple(x["report_ids"]),x.get("obstacle_episode_id")) for x in raw["attack_events"])
    starts={int(key):tuple(value) for key,value in (raw.get("robot_starts") or {}).items()}
    queues={int(key):tuple(DeliveryTask(item["task_id"],tuple(item["pickup"]),tuple(item["dropoff"])) for item in value) for key,value in (raw.get("task_queues") or {}).items()}
    reports=tuple(ClaimReport(item["report_id"], item["sender_id"], tuple(item["target_cell"]), ClaimType(item["claim"]), item["observation_step"], item.get("sensor_confidence", 1.0), item.get("scenario_event_id")) for item in raw.get("honest_attacker_reports",()))
    labels=tuple(ReportAuditLabel(item["report_id"],item["is_malicious"],AttackType(item["attack_type"]) if item.get("attack_type") else None,item.get("obstacle_episode_id"),ClaimType(item["actual_state_at_observation"]),item.get("original_obstacle_appearance_step"),item.get("original_obstacle_clearance_step")) for item in raw.get("report_audit_labels",()))
    return ScenarioManifest(raw["schema_version"],raw["master_seed"],raw["derived_seeds"],raw["map_hash"],tuple(raw["map_shape"]),tuple(tuple(row) for row in raw["static_grid"]),raw["phase_boundaries"],raw["malicious_robot_id"],tuple(raw["benign_robot_ids"]),episodes,events,raw.get("scenario_id",""),raw.get("protocol_id","custom"),starts,queues,tuple(map(tuple,raw.get("attacker_positions",()))),reports,labels,tuple(raw.get("candidate_metadata",())),tuple(raw.get("authoring_warnings",())),tuple(tuple(int(value) for value in row) for row in raw["reconnaissance_heatmap"]) if raw.get("reconnaissance_heatmap") else None,raw.get("scenario_preset"))
