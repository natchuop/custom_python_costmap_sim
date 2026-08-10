"""Versioned scenario manifests and defense-independent attack authoring."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import sim2
from .config import SimulationConfig
from .models import AttackEvent, AttackType, ClaimReport, ClaimType, DeliveryTask, ReportAuditLabel, TemporaryObstacleEpisode
from .rng import derived_seed, named_rng
from .temp_obstacles import export_temp_episodes
from .world import demo_grid
from .planning import astar
from .scenario_presets import preset_for_id, validate_fixed_preset

SCHEMA_VERSION = 2

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
    protocol_id: str = "original_legacy_cli"
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

def _temporary_footprint(grid, rng, preferred):
    """Choose one valid 1..5 rectangle with at least four physical cells."""
    rows, cols = grid.shape
    for _ in range(100):
        height, width = sim2.sample_rectangle_dimensions(rng)
        top = (max(1, min(rows - height - 1, preferred[0] - height // 2)),
               max(1, min(cols - width - 1, preferred[1] - width // 2)))
        cells = tuple((r, c) for r in range(top[0], top[0] + height)
                      for c in range(top[1], top[1] + width))
        if len(cells) >= 4 and all(not grid[cell] for cell in cells):
            return cells
    return ((preferred[0], preferred[1]), (preferred[0] + 1, preferred[1]),
            (preferred[0], preferred[1] + 1), (preferred[0] + 1, preferred[1] + 1))

def _nominal_route_cells(grid, starts=None, targets=None) -> list[tuple[int, int]]:
    """Clean-rollout corridor candidates shared by the manifest and robot tasks."""
    rows, cols=grid.shape
    starts=tuple(starts or ((2,2),(rows-3,cols-3),(2,cols-3)))
    targets=tuple(targets or ((rows-3,2),(2,cols-3),(rows-3,cols-3),(2,2)))
    routes=[]
    for index,start in enumerate(starts):
        for offset in (0,1,2):
            goal=targets[(index+offset)%len(targets)]
            route=astar(start,goal,lambda cell: float("inf") if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid[cell] else 1.0)
            if route: routes.extend(route)
    excluded=set(starts)|set(targets)
    return [cell for cell in routes if cell not in excluded]

def author_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    config.validate(); grid = demo_grid() if grid is None else grid
    preset = preset_for_id(config.scenario_preset) if config.scenario_preset else None
    if preset is not None:
        validate_fixed_preset(grid, preset)
    phases = config.phases
    rng = named_rng(config.seed, "attack_scheduler")
    enabled = [AttackType(x) for x in config.attacks.enabled]
    benign = (1, 2); sender = 0; events=[]; step = phases.recon_steps + rng.randint(config.attacks.interval_min, config.attacks.interval_max); index=0
    free = [(r,c) for r in range(1,grid.shape[0]-1) for c in range(1,grid.shape[1]-1) if not grid[r,c]]
    starts = dict(preset.robot_starts) if preset is not None else {0:(2,2),1:(grid.shape[0]-3,grid.shape[1]-3),2:(2,grid.shape[1]-3)}
    targets = tuple(preset.delivery_points) if preset is not None else ((grid.shape[0]-3,2),(2,grid.shape[1]-3),(grid.shape[0]-3,grid.shape[1]-3),(2,2))
    route_cells=_nominal_route_cells(grid, tuple(starts.values()), targets) or free
    # Temporary obstacles are part of the fixed scenario and deliberately sit
    # on nominal traffic corridors so clearance/stale ablations affect behavior.
    episodes=[]
    for episode_index,appearance in enumerate(range(max(1, phases.recon_steps - config.temporary_blockage_change_period_steps//2), phases.total_steps, config.temporary_blockage_change_period_steps)):
        cell=route_cells[(episode_index*7 + rng.randrange(min(12,len(route_cells)))) % len(route_cells)]
        clearance = min(phases.total_steps, phases.recon_steps + max(1, phases.attack_steps // 2))
        episodes.append(TemporaryObstacleEpisode(f"obstacle-{episode_index:03}",_temporary_footprint(grid, rng, cell),appearance,clearance))
    episodes=tuple(episodes)
    candidate_metadata=[]; use_count: dict[tuple[int,int],int]={}; selected=[]
    while step < phases.recon_steps + phases.attack_steps and enabled:
        feasible_types = [kind for kind in enabled if kind == AttackType.FAKE_OBSTACLE or (kind == AttackType.FALSE_CLEARANCE and any(e.appearance_step <= step < e.clearance_step for e in episodes)) or (kind == AttackType.STALE_REASSERTION and any(e.clearance_step <= step for e in episodes))]
        feasible = bool(feasible_types)
        if feasible:
            kind = feasible_types[rng.randrange(len(feasible_types))]
            episode = None
            footprint_height = footprint_width = None
            if kind == AttackType.FAKE_OBSTACLE:
                eligible=[cell for cell in route_cells if use_count.get(cell,0) < config.attacks.max_uses_per_footprint and all(abs(cell[0]-old[0])+abs(cell[1]-old[1]) >= config.attacks.min_center_spacing for old in selected)]
                if not eligible: break
                center = _cell_choice(rng, eligible)
                height, width = sim2.sample_rectangle_dimensions(rng)
                footprint_height, footprint_width = height, width
                # Keep the first legacy-manifest fake large enough for older
                # consumers while subsequent events exercise the full 1..5
                # rectangular sampler.
                if index == 0:
                    while height * width < 15:
                        height, width = sim2.sample_rectangle_dimensions(rng)
                footprint = [(r, c) for r in range(center[0] - height // 2, center[0] - height // 2 + height) for c in range(center[1] - width // 2, center[1] - width // 2 + width) if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not grid[r, c]]
                for _ in range(20):
                    if len(footprint) >= (15 if index == 0 else 4):
                        break
                    height, width = sim2.sample_rectangle_dimensions(rng)
                    footprint_height, footprint_width = height, width
                    footprint = [(r, c) for r in range(center[0] - height // 2, center[0] - height // 2 + height) for c in range(center[1] - width // 2, center[1] - width // 2 + width) if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and not grid[r, c]]
                if len(footprint) < 4:
                    step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
                    continue
                cell, claim, observation = center, ClaimType.BLOCKED, step
            elif kind == AttackType.FALSE_CLEARANCE:
                episode = _cell_choice(rng, [e for e in episodes if e.appearance_step <= step < e.clearance_step]); cell, claim, observation = episode.cells[0], ClaimType.FREE, step
            else:
                episode = _cell_choice(rng, [e for e in episodes if e.clearance_step <= step]); cell, claim, observation = episode.cells[0], ClaimType.BLOCKED, step
            eid=f"attack-{index:04}"; attack_cells = tuple(footprint) if kind == AttackType.FAKE_OBSTACLE else (cell,)
            report_ids = tuple(f"report-{index:04}-{n:02}" for n in range(len(attack_cells)))
            events.append(AttackEvent(eid, step, kind, attack_cells, claim, observation, sender, benign, report_ids, episode.episode_id if episode else None)); index += 1
            use_count[cell]=use_count.get(cell,0)+1; selected.append(cell)
            candidate_metadata.append({"candidate_id":f"candidate-{index-1:04}","center":cell,"footprint_cells":list(attack_cells),"footprint_height":footprint_height,"footprint_width":footprint_width,"traffic_score":None,"bottleneck_score":None,"estimated_detour_score":None,"rank":None,"selection_weight":None,"prior_use_count":use_count[cell]-1})
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    names=("attack_scheduler", "temporary_obstacles", "robot_routes", "traffic")
    static_grid=tuple(tuple(int(value) for value in row) for row in grid)
    queues={rid:tuple(DeliveryTask(f"r{rid}-task-{i}",targets[(rid+i)%4],targets[(rid+i+2)%4]) for i in range(config.deliveries_per_robot)) for rid in (sender,) + benign}
    warnings=() if len(set(selected)) >= min(config.attacks.min_unique_footprints,len(selected)) else ("concentrated_attack_manifest",)
    # Script the attacker independently of defense-dependent benign routes.
    attacker_route=tuple(route_cells) or tuple(free)
    positions=tuple(attacker_route[step % len(attacker_route)] for step in range(phases.total_steps))
    def truth(cell, step):
        return ClaimType.BLOCKED if any(cell in episode.cells and episode.appearance_step <= step < episode.clearance_step for episode in episodes) else ClaimType.FREE
    honest=tuple(ClaimReport(f"attacker-honest-{step:05}",sender,positions[step],truth(positions[step],step),step,step,step) for step in range(0,phases.total_steps,config.communication_period_steps))
    labels=tuple(ReportAuditLabel(report_id,True,event.attack_type,event.obstacle_episode_id,ClaimType.BLOCKED if event.attack_type == AttackType.FALSE_CLEARANCE else ClaimType.FREE) for event in events for report_id in event.report_ids)
    return ScenarioManifest(SCHEMA_VERSION, config.seed, {x:derived_seed(config.seed,x) for x in names}, _hash(grid), tuple(grid.shape), static_grid, {"reconnaissance_end":phases.recon_steps, "attack_end":phases.recon_steps+phases.attack_steps, "total":phases.total_steps}, sender, benign, episodes, tuple(events), scenario_id=f"scenario-{config.seed}-{_hash(grid)[:12]}", protocol_id="original_legacy_cli", robot_starts=starts, task_queues=queues, attacker_positions=positions, honest_attacker_reports=honest, report_audit_labels=labels, candidate_metadata=tuple(candidate_metadata), authoring_warnings=warnings, scenario_preset=config.scenario_preset)

def author_warehouse_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    """Author the default warehouse manifest from a clean recon rollout and heatmap candidates."""
    # Build the exact starts and delivery queues that will be stored in the
    # manifest before authoring reconnaissance. The earlier implementation
    # authored candidates with one automatically generated task stream and
    # then saved a slightly different stream, so attack candidates could miss
    # the victim route during replay.
    layout_world = sim2.GridWorld(np.asarray(grid, dtype=int))
    layout_specs, layout_goals, _ = sim2.build_robot_specs_and_goals(
        layout_world, prior_grid=layout_world.grid
    )
    layout_goals = sim2.filter_reachable_action_points(
        layout_goals, layout_specs, layout_world.grid
    )
    sim2.relocate_starts_for_goals(layout_world, layout_specs, layout_goals, layout_world.grid)
    layout_tasks = sim2.build_delivery_tasks(
        layout_goals,
        num_robots=sim2.DEFAULT_NUM_ROBOTS,
        tasks_per_robot=config.deliveries_per_robot,
    )
    layout_tasks = sim2.repair_delivery_tasks(
        layout_world,
        layout_tasks,
        layout_specs,
        layout_world.grid,
        action_points=layout_goals,
    )
    manifest_robot_starts = {
        spec["robot_id"]: tuple(spec["start"])
        for spec in layout_specs
    }
    manifest_task_queues = {
        robot_id: tuple(tasks)
        for robot_id, tasks in layout_tasks.items()
    }

    old_phase = (sim2.MIN_RECON_STEPS, sim2.MAX_RECON_STEPS)
    sim2.MIN_RECON_STEPS = config.phases.recon_steps
    sim2.MAX_RECON_STEPS = config.phases.recon_steps
    try:
        world, robots, log = sim2.run_simulation(
            grid=grid,
            prior_grid=grid,
            tasks_per_robot=config.deliveries_per_robot,
            max_steps=config.phases.total_steps,
            random_seed=config.seed,
            experiment_mode="clean",
            manifest_robot_starts=manifest_robot_starts,
            manifest_task_queues=manifest_task_queues,
        )
    finally:
        sim2.MIN_RECON_STEPS, sim2.MAX_RECON_STEPS = old_phase
    candidates = sim2.recon_heatmap_attack_candidates(
        world, log["goals"], robots, log["traffic_heatmap"][-1]
    )
    if not candidates:
        old_overlap = sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP
        sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = False
        try:
            candidates = sim2.recon_heatmap_attack_candidates(
                world, log["goals"], robots, log["traffic_heatmap"][-1]
            )
        finally:
            sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = old_overlap
    if not candidates:
        raise RuntimeError("clean warehouse rollout produced no manifest attack candidates")

    def candidates_at_step(step):
        """Score attack candidates against clean traffic/routes at this step."""
        frame = min(max(0, int(step)), len(log["traffic_heatmap"]) - 1)
        for robot in robots:
            rid = robot.robot_id
            robot.path = list(log["robots"][rid]["path"][frame])
            robot.path_index = 0
            robot.position_cell = tuple(log["robots"][rid]["position"][frame])
        old_overlap = sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP
        sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = True
        try:
            return sim2.recon_heatmap_attack_candidates(
                world,
                log["goals"],
                robots,
                log["traffic_heatmap"][frame],
            )
        finally:
            sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = old_overlap
    attacker = log["malicious_robot_id"]
    recipients = tuple(r.robot_id for r in robots if not r.is_malicious)
    rng = named_rng(config.seed, "warehouse_manifest_scheduler")
    enabled_types = [AttackType(x) for x in config.attacks.enabled]
    prior = robots[0].belief_map.initial_prior
    static = np.asarray(prior, dtype=np.uint8)
    episodes = export_temp_episodes(
        static,
        config.seed,
        config.phases.total_steps,
        config.temporary_blockage_change_period_steps,
    )
    events = []
    metadata = []
    warnings = []
    uses = {}
    selected_centers = []
    step = config.phases.recon_steps + rng.randint(
        config.attacks.interval_min, config.attacks.interval_max
    )
    index = 0
    while step < config.phases.recon_steps + config.phases.attack_steps:
        feasible_types = [kind for kind in enabled_types if kind == AttackType.FAKE_OBSTACLE or (kind == AttackType.FALSE_CLEARANCE and any(e.appearance_step <= step < e.clearance_step for e in episodes)) or (kind == AttackType.STALE_REASSERTION and any(e.clearance_step <= step for e in episodes))]
        if not feasible_types:
            break
        selected_attack = feasible_types[rng.randrange(len(feasible_types))]
        if selected_attack != AttackType.FAKE_OBSTACLE:
            eligible = [e for e in episodes if (selected_attack == AttackType.FALSE_CLEARANCE and e.appearance_step <= step < e.clearance_step) or (selected_attack == AttackType.STALE_REASSERTION and e.clearance_step <= step)]
            if not eligible and selected_attack == AttackType.FALSE_CLEARANCE:
                fallback_cells = tuple(tuple(cell) for cell in candidates[0]["report_cells"][: max(4, min(25, len(candidates[0]["report_cells"])) )])
                episode = TemporaryObstacleEpisode(f"attack-obstacle-{index:04}", fallback_cells, max(0, step - 1), step + 1)
                episodes = episodes + (episode,)
                eligible = [episode]
            if not eligible:
                step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
                continue
            episode = eligible[rng.randrange(len(eligible))]
            if selected_attack == AttackType.STALE_REASSERTION and any(
                any(cell in other.cells and other.appearance_step <= step < other.clearance_step for other in episodes)
                for cell in episode.cells
            ):
                fallback_cells = tuple(tuple(cell) for cell in candidates[0]["report_cells"][: max(4, min(25, len(candidates[0]["report_cells"])) )])
                episode = TemporaryObstacleEpisode(f"attack-obstacle-{index:04}", fallback_cells, max(0, step - 2), max(1, step - 1))
                episodes = episodes + (episode,)
            claim = ClaimType.FREE if selected_attack == AttackType.FALSE_CLEARANCE else ClaimType.BLOCKED
            cells = tuple(episode.cells)
            events.append(AttackEvent(f"attack-{index:04}", step, selected_attack, cells, claim, step, attacker, recipients, tuple(f"report-{index:04}-{i:02}" for i in range(len(cells))), episode.episode_id))
            index += 1
            step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
            continue
        step_candidates = candidates_at_step(step)
        pool = (step_candidates or candidates)[:config.attacks.candidate_top_k]
        eligible = [
            candidate
            for candidate in pool
            if uses.get(tuple(candidate["center_cell"]), 0) < config.attacks.max_uses_per_footprint
            and all(
                abs(candidate["center_cell"][0] - old[0])
                + abs(candidate["center_cell"][1] - old[1])
                >= config.attacks.min_center_spacing
                for old in selected_centers
            )
        ]
        if not eligible:
            warnings.append("concentrated_attack_manifest: diversity limits exhausted")
            break
        candidate = eligible[rng.randrange(len(eligible))]
        cells = tuple(tuple(cell) for cell in candidate["report_cells"])
        ids = tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells)))
        events.append(
            AttackEvent(
                f"attack-{index:04}",
                step,
                AttackType.FAKE_OBSTACLE,
                cells,
                ClaimType.BLOCKED,
                step,
                attacker,
                recipients,
                ids,
            )
        )
        center = tuple(candidate["center_cell"])
        metadata.append(
            {
                "candidate_id": f"warehouse-{index:04}",
                "center": center,
                "footprint_cells": cells,
                "footprint_height": candidate.get("footprint_height"),
                "footprint_width": candidate.get("footprint_width"),
                "route_overlap": candidate["path_overlap"],
                "traffic_score": candidate["traffic_score"],
                "bottleneck_score": candidate["bottleneck_score"],
                "estimated_detour_score": candidate["path_proximity_score"],
                "rank": (step_candidates.index(candidate) + 1
                         if candidate in step_candidates
                         else candidates.index(candidate) + 1
                         if candidate in candidates else None),
                "selection_weight": 1 / len(eligible),
                "prior_use_count": uses.get(center, 0),
            }
        )
        uses[center] = uses.get(center, 0) + 1
        selected_centers.append(center)
        index += 1
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    if len(set(selected_centers)) < config.attacks.min_unique_footprints:
        warnings.append("concentrated_attack_manifest: minimum unique footprint count not met")
    layout_world = sim2.GridWorld(np.asarray(grid if grid is not None else prior, dtype=int))
    layout_specs, layout_goals, _ = sim2.build_robot_specs_and_goals(
        layout_world, prior_grid=layout_world.grid
    )
    layout_goals = sim2.filter_reachable_action_points(
        layout_goals, layout_specs, layout_world.grid
    )
    sim2.relocate_starts_for_goals(layout_world, layout_specs, layout_goals, layout_world.grid)
    layout_tasks = sim2.build_delivery_tasks(
        layout_goals,
        num_robots=sim2.DEFAULT_NUM_ROBOTS,
        tasks_per_robot=config.deliveries_per_robot,
    )
    layout_tasks = sim2.repair_delivery_tasks(
        layout_world,
        layout_tasks,
        layout_specs,
        layout_world.grid,
        action_points=layout_goals,
    )
    robot_starts = {spec["robot_id"]: tuple(spec["start"]) for spec in layout_specs}
    task_queues = {
        robot_id: tuple(
            DeliveryTask(f"r{robot_id}-task-{idx}", tuple(task.pickup), tuple(task.dropoff))
            for idx, task in enumerate(layout_tasks[robot_id])
        )
        for robot_id in layout_tasks
    }
    return ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        {"warehouse_manifest_scheduler": config.seed},
        hashlib.sha256(prior.tobytes()).hexdigest(),
        tuple(prior.shape),
        tuple(tuple(int(v) for v in row) for row in prior),
        {
            "reconnaissance_end": config.phases.recon_steps,
            "attack_end": config.phases.recon_steps + config.phases.attack_steps,
            "total": config.phases.total_steps,
        },
        attacker,
        recipients,
        episodes,
        tuple(events),
        scenario_id=f"warehouse-{config.seed}",
        protocol_id="original_legacy_cli",
        robot_starts=robot_starts,
        task_queues=task_queues,
        candidate_metadata=tuple(metadata),
        authoring_warnings=tuple(dict.fromkeys(warnings)),
        reconnaissance_heatmap=tuple(
            tuple(int(value) for value in row)
            for row in log["traffic_heatmap"][
                min(config.phases.recon_steps, len(log["traffic_heatmap"]) - 1)
            ]
        ),
        scenario_preset=config.scenario_preset,
    )


def scenario_manifest_hash(manifest: ScenarioManifest) -> str:
    """Hash the complete authored scenario, not only its static map."""
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def save_manifest(manifest: ScenarioManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

def load_manifest(path: str | Path) -> ScenarioManifest:
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported scenario manifest schema; author a schema-v2 manifest")
    episodes=tuple(TemporaryObstacleEpisode(x["episode_id"], tuple(map(tuple,x["cells"])), x["appearance_step"],x["clearance_step"]) for x in raw["obstacle_episodes"])
    events=tuple(AttackEvent(x["event_id"],x["step"],AttackType(x["attack_type"]),tuple(map(tuple,x["cells"])),ClaimType(x["claim"]),x["observation_step"],x["sender_id"],tuple(x["recipients"]),tuple(x["report_ids"]),x.get("obstacle_episode_id")) for x in raw["attack_events"])
    starts={int(key):tuple(value) for key,value in (raw.get("robot_starts") or {}).items()}
    queues={int(key):tuple(DeliveryTask(item["task_id"],tuple(item["pickup"]),tuple(item["dropoff"])) for item in value) for key,value in (raw.get("task_queues") or {}).items()}
    reports=tuple(ClaimReport(item["report_id"],item["sender_id"],tuple(item["target_cell"]),ClaimType(item["claim"]),item["observation_step"],item["sent_step"],item["received_step"],item.get("confidence",1.),item.get("scenario_event_id")) for item in raw.get("honest_attacker_reports",()))
    labels=tuple(ReportAuditLabel(item["report_id"],item["is_malicious"],AttackType(item["attack_type"]) if item.get("attack_type") else None,item.get("obstacle_episode_id"),ClaimType(item["actual_state_at_observation"]),item.get("original_obstacle_appearance_step"),item.get("original_obstacle_clearance_step")) for item in raw.get("report_audit_labels",()))
    heatmap = raw.get("reconnaissance_heatmap")
    return ScenarioManifest(raw["schema_version"],raw["master_seed"],raw["derived_seeds"],raw["map_hash"],tuple(raw["map_shape"]),tuple(tuple(row) for row in raw["static_grid"]),raw["phase_boundaries"],raw["malicious_robot_id"],tuple(raw["benign_robot_ids"]),episodes,events,raw.get("scenario_id",""),raw.get("protocol_id","custom"),starts,queues,tuple(map(tuple,raw.get("attacker_positions",()))),reports,labels,tuple(raw.get("candidate_metadata",())),tuple(raw.get("authoring_warnings",())),tuple(tuple(int(value) for value in row) for row in heatmap) if heatmap else None,raw.get("scenario_preset"))
