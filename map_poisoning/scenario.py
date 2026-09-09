"""Versioned scenario manifests and defense-independent attack authoring."""
from __future__ import annotations
import hashlib, json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .config import SimulationConfig
from .models import (
    AttackEvent,
    AttackType,
    ClaimReport,
    ClaimType,
    DeliveryTask,
    FrozenReconData,
    PermanentObstacle,
    ReconMissionSnapshot,
    ReconRouteSnapshot,
    ReportAuditLabel,
    TemporaryObstacleEpisode,
)
from .rng import derived_seed, named_rng
from .obstacles import (
    FAKE_CENTER_MIN_SPACING,
    FAKE_MIN_REPORT_CELLS,
    author_permanent_obstacles,
    author_temporary_obstacle_episodes,
    fake_report_cells,
    footprint_center,
    footprint_bottleneck_score,
    footprint_finite_detour_score,
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
    permanent_obstacles: tuple[PermanentObstacle, ...] = ()
    reconnaissance_data: FrozenReconData | None = None

    @property
    def frozen_recon_data(self) -> FrozenReconData | None:
        """Compatibility/readability alias for the persisted recon contract."""
        return self.reconnaissance_data
    def to_dict(self): return asdict(self)

def _hash(grid) -> str: return hashlib.sha256(grid.tobytes()).hexdigest()
def _cell_choice(rng, cells): return cells[rng.randrange(min(len(cells), 12))]

def _episode_is_active(episode, step: int) -> bool:
    return episode.appearance_step <= step < episode.clearance_step


def _feasible_attack_types(enabled: list[AttackType], step: int, episodes, grid, permanent_obstacles=()) -> list[AttackType]:
    feasible = []
    for kind in enabled:
        if kind == AttackType.FAKE_OBSTACLE:
            feasible.append(kind)
        elif kind == AttackType.FALSE_CLEARANCE and (
            permanent_obstacles or any(_episode_is_active(episode, step) for episode in episodes)
        ):
            feasible.append(kind)
        elif kind == AttackType.STALE_REASSERTION and _cleared_reassertable(episodes, step, grid, preferred_age_only=True):
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


STALE_REASSERTION_MIN_AGE = 30
STALE_REASSERTION_MAX_AGE = 100

def _cleared_reassertable(episodes, step, grid, *, preferred_age_only=False):
    active = _active_temp_cells(episodes, step)
    result = []
    for episode in episodes:
        if episode.clearance_step > step:
            continue
        if preferred_age_only:
            age = step - episode.clearance_step
            if not (STALE_REASSERTION_MIN_AGE <= age <= STALE_REASSERTION_MAX_AGE):
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


def _route_corridor_relevance(cells, route_cells, grid) -> tuple[float, float, float]:
    """Static/recon proxy for how attractive a false clearance would be."""
    footprint = {tuple(cell) for cell in cells}
    if not footprint:
        return (0.0, 0.0, 0.0)
    route = {tuple(cell) for cell in route_cells}
    center = _center_cell(cells)
    overlap = float(len(footprint & route))
    min_distance = min(
        (abs(center[0] - cell[0]) + abs(center[1] - cell[1]) for cell in route),
        default=999,
    )
    proximity = 1.0 / (1.0 + float(min_distance))
    detour = float(footprint_finite_detour_score(grid, cells))
    bottleneck = float(footprint_bottleneck_score(grid, cells))
    score = (1.0 + 8.0 * overlap) * (1.0 + 5.0 * proximity)
    score *= (1.0 + 2.0 * detour) * (1.0 + 2.0 * bottleneck)
    return score, detour, proximity


def _mission_reblock_penalty(cells, mission_pairs, grid, permanent_obstacles=()) -> float:
    """Finite route penalty from reblocking a cleared footprint.

    Generic/MovingAI authoring does not inspect any tested defense trajectory.
    Instead it uses the same fixed mission geometry that already seeds the
    nominal route corridor, plus the shared authored permanent obstacles.
    """
    footprint = {tuple(cell) for cell in cells}
    if not footprint or not mission_pairs:
        return 0.0
    rows, cols = grid.shape
    free_grid = np.array(grid, dtype=np.uint8, copy=True)
    for obstacle in permanent_obstacles:
        for cell in obstacle.cells:
            if cell not in footprint:
                free_grid[cell] = 1
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            free_grid[cell] = 0
    blocked_grid = np.array(free_grid, dtype=np.uint8, copy=True)
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            blocked_grid[cell] = 1

    def route_len(grid_value, start, goal):
        path = astar(
            tuple(start), tuple(goal),
            lambda cell: (
                float("inf")
                if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid_value[cell]
                else 1.0
            ),
        )
        return None if path is None else max(0, len(path) - 1)

    best = 0.0
    for start, goal in mission_pairs:
        free_len = route_len(free_grid, start, goal)
        blocked_len = route_len(blocked_grid, start, goal)
        if free_len is None or blocked_len is None:
            continue
        best = max(best, float(blocked_len - free_len))
    return max(0.0, best)


def _instantiate_attack(
    kind,
    *,
    step,
    rng,
    episodes,
    permanent_obstacles,
    route_cells,
    use_count,
    selected,
    config,
    grid,
    forbidden,
    preferred_false_clearance_kind=None,
    stale_target_uses=None,
    false_clearance_target_uses=None,
    mission_pairs=(),
    recon_candidates=None,
    recon_reference_states=None,
    recon_benign_ids=(),
    recon_heatmap=None,
    recon_relevance_cache=None,
):
    """Place one attack of the requested kind, or return None if it cannot be sited."""
    if kind == AttackType.FAKE_OBSTACLE:
        if recon_candidates is not None:
            active_cells = _active_temp_cells(episodes, step)
            pool = [
                candidate
                for candidate in recon_candidates
                if not set(map(tuple, candidate["report_cells"])).intersection(active_cells)
            ][: config.attacks.candidate_top_k]
            used_unique = set(selected)
            require_new_center = len(used_unique) < config.attacks.min_unique_footprints
            eligible = [
                candidate
                for candidate in pool
                if use_count.get(tuple(candidate["center_cell"]), 0)
                < config.attacks.max_uses_per_footprint
                and (
                    not require_new_center
                    or (
                        tuple(candidate["center_cell"]) not in used_unique
                        and all(
                            abs(candidate["center_cell"][0] - old[0])
                            + abs(candidate["center_cell"][1] - old[1])
                            >= config.attacks.min_center_spacing
                            for old in used_unique
                        )
                    )
                )
            ]
            if not eligible:
                eligible = [
                    candidate
                    for candidate in pool
                    if use_count.get(tuple(candidate["center_cell"]), 0)
                    < config.attacks.max_uses_per_footprint
                ]
            if eligible:
                candidate = rng.choices(
                    eligible,
                    weights=list(range(len(eligible), 0, -1)),
                    k=1,
                )[0]
                cells = tuple(tuple(cell) for cell in candidate["report_cells"])
                return cells, ClaimType.BLOCKED, step, None, tuple(candidate["center_cell"])
            return None
        cells = _place_fake_cells(rng, route_cells, use_count, selected, config, grid, forbidden, _active_temp_cells(episodes, step))
        if not cells:
            return None
        return cells, ClaimType.BLOCKED, step, None, _center_cell(cells)
    if recon_reference_states is not None and recon_heatmap is not None:
        from .recon_authoring import _select_episode_attack_target

        selected_target = _select_episode_attack_target(
            episodes,
            permanent_obstacles,
            recon_reference_states,
            step,
            kind,
            recon_benign_ids,
            grid,
            recon_heatmap,
            preferred_physical_kind=preferred_false_clearance_kind,
            relevance_cache=(recon_relevance_cache if recon_relevance_cache is not None else {}),
            target_use_counts=(
                false_clearance_target_uses
                if kind == AttackType.FALSE_CLEARANCE
                else stale_target_uses
                if kind == AttackType.STALE_REASSERTION
                else None
            ),
        )
        if selected_target is None:
            return None
        target, _ = selected_target
        cells = tuple(target.cells)
        claim = ClaimType.FREE if kind == AttackType.FALSE_CLEARANCE else ClaimType.BLOCKED
        observation = (
            step
            if kind == AttackType.FALSE_CLEARANCE
            else max(0, target.clearance_step - 1)
        )
        return cells, claim, observation, target, _center_cell(cells)
    if kind == AttackType.FALSE_CLEARANCE:
        active = [episode for episode in episodes if _episode_is_active(episode, step)]
        permanent = list(permanent_obstacles)
        if preferred_false_clearance_kind == "permanent" and permanent:
            targets = permanent
        elif preferred_false_clearance_kind == "temporary" and active:
            targets = active
        else:
            targets = [*permanent, *active]
        if not targets:
            return None
        # Choose the most route-relevant physical obstacle using only the
        # shared static/nominal route corridor. Deterministic RNG is used only
        # to break exact score ties, so no defense-specific live path leaks in.
        scored = []
        for target in targets:
            score, detour, proximity = _route_corridor_relevance(target.cells, route_cells, grid)
            scored.append((score, detour, proximity, rng.random(), target))
        target = max(scored, key=lambda item: (item[0], item[1], item[2], item[3]))[-1]
        return tuple(target.cells), ClaimType.FREE, step, target, _center_cell(target.cells)
    cleared = _cleared_reassertable(episodes, step, grid, preferred_age_only=True)
    if not cleared:
        return None
    # Reasserting the exact same historical observation repeatedly is not a new
    # piece of evidence under the one-active-report-per-(sender, cell) rule.
    # Prefer an episode that has not already been used by Stale Reassertion so
    # each authored event can actually enter fusion rather than being rejected
    # as a duplicate/out-of-order historical claim.
    stale_target_uses = stale_target_uses or {}
    unused = [episode for episode in cleared if stale_target_uses.get(episode.episode_id, 0) == 0]
    if not unused:
        return None
    scored = []
    for episode in unused:
        score, detour, proximity = _route_corridor_relevance(episode.cells, route_cells, grid)
        reblock_penalty = _mission_reblock_penalty(
            episode.cells, mission_pairs, grid, permanent_obstacles
        )
        age = max(0, step - episode.clearance_step)
        freshness = 1.0 + 2.0 * max(0.0, 1.0 - (age - STALE_REASSERTION_MIN_AGE) / max(1.0, STALE_REASSERTION_MAX_AGE - STALE_REASSERTION_MIN_AGE))
        score *= (1.0 + 4.0 * reblock_penalty) * freshness
        # Positive finite reblocking impact dominates a zero-penalty target;
        # the existing route relevance and freshness break ties within class.
        scored.append((int(reblock_penalty > 0.0), score, reblock_penalty, detour, proximity, -age, rng.random(), episode))
    # Do not spend a stale event on a footprint for which the frozen
    # reconnaissance says reblocking cannot lengthen any observed mission.
    # This keeps stale authoring focused on interventions with a demonstrated
    # navigation consequence while retaining the historical/physical checks
    # above.
    positive = [item for item in scored if item[0] > 0]
    if positive:
        scored = positive
    else:
        return None
    episode = max(scored, key=lambda item: item[:-1])[-1]
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
    # Import lazily to keep the low-level scenario model independent of the
    # rollout module during normal package import.
    from .recon_authoring import (
        _historical_visibility_delay,
        _recon_reblock_features,
        _select_episode_attack_target,
        recon_heatmap_attack_candidates,
        run_clean_reference_rollout,
        run_reconnaissance,
    )
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
    bootstrap_route_cells = _nominal_route_cells(grid, starts_tuple, targets) or free
    nominal_mission_pairs = tuple(
        (tuple(start), tuple(targets[(robot_index + offset) % len(targets)]))
        for robot_index, start in enumerate(starts_tuple)
        for offset in (0, 1, 2)
    )
    protected = set(starts_tuple) | set(targets)
    queues = build_fixed_task_queues((sender, *benign), targets, config.deliveries_per_robot)
    if config.deliveries_per_robot < len(targets):
        queues[sender] = build_fixed_task_queues((sender,), targets, len(targets))[sender]
    phase_boundaries = {
        "reconnaissance_end": phases.recon_steps,
        "attack_end": phases.recon_steps + phases.attack_steps,
        "total": phases.total_steps,
    }
    static_grid = tuple(tuple(int(value) for value in row) for row in grid)
    # Bootstrap reconnaissance is attack-free and physical-obstacle-free. It
    # supplies the same seed-stable traffic prior used by the warehouse author,
    # so physical obstacle placement is not driven by a defense or by a
    # geometry-only nominal route.
    bootstrap = ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        {},
        _hash(grid),
        tuple(grid.shape),
        static_grid,
        phase_boundaries,
        sender,
        benign,
        (),
        (),
        scenario_id=f"scenario-{config.seed}-{_hash(grid)[:12]}-bootstrap",
        protocol_id="modular_v1",
        robot_starts={0: starts_tuple[0], 1: starts_tuple[1], 2: starts_tuple[2]},
        task_queues=queues,
    )
    traffic_heatmap, _, _ = run_clean_reference_rollout(config, bootstrap)
    traffic_heatmap = np.asarray(traffic_heatmap, dtype=np.int32)

    required_anchors = tuple(dict.fromkeys((*starts_tuple, *targets)))
    permanent_obstacles = author_permanent_obstacles(
        grid,
        named_rng(config.seed, "permanent_obstacles"),
        forbidden_cells=protected,
        required_anchors=required_anchors,
        traffic_heatmap=traffic_heatmap,
    )
    permanent_cells = {cell for obstacle in permanent_obstacles for cell in obstacle.cells}
    physical_base_grid = np.array(grid, dtype=np.uint8, copy=True)
    for cell in permanent_cells:
        physical_base_grid[cell] = 1
    forbidden = protected | permanent_cells
    # Temporary obstacles are physical objects: six concurrent footprints by
    # default, authored against the same permanent world for every defense.
    episodes = author_temporary_obstacle_episodes(
        physical_base_grid,
        named_rng(config.seed, "temporary_obstacles"),
        phases.total_steps,
        config.temporary_blockage_change_period_steps,
        forbidden_cells=forbidden,
        required_anchors=required_anchors,
        traffic_heatmap=traffic_heatmap,
    )
    reference_manifest = ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        {},
        _hash(grid),
        tuple(grid.shape),
        static_grid,
        phase_boundaries,
        sender,
        benign,
        episodes,
        (),
        scenario_id=f"scenario-{config.seed}-{_hash(grid)[:12]}-recon-reference",
        protocol_id="modular_v1",
        robot_starts={0: starts_tuple[0], 1: starts_tuple[1], 2: starts_tuple[2]},
        task_queues=queues,
        permanent_obstacles=permanent_obstacles,
    )
    frozen_recon = run_reconnaissance(config, reference_manifest)
    route_cells = list(frozen_recon.route_cells()) or bootstrap_route_cells
    mission_pairs = frozen_recon.mission_pairs() or nominal_mission_pairs
    traffic_heatmap = np.asarray(frozen_recon.traffic_heatmap, dtype=np.int32)
    reference_states = frozen_recon.states()
    recon_robots = []
    for victim_id in benign:
        trace = [
            tuple(reference_states[step][victim_id]["position"])
            for step in sorted(reference_states)
            if victim_id in reference_states[step]
        ]
        if len(trace) < 2:
            continue
        recon_robots.append(SimpleNamespace(
            robot_id=int(victim_id),
            position=trace[0],
            goal=trace[-1],
            path=trace,
        ))

    def recon_visibility_delay(victim_id, report_cells):
        return _historical_visibility_delay(
            reference_states,
            victim_id,
            report_cells,
            config.attacks.visibility_delay_min,
            config.attacks.visibility_delay_max,
        )

    base_candidates = recon_heatmap_attack_candidates(
        physical_base_grid,
        targets,
        recon_robots,
        traffic_heatmap,
        rng=place_rng,
        require_route_overlap=True,
        forbidden_cells=forbidden,
        active_temp_cells=(),
        visible_cells_by_robot={},
        future_visibility_delay_fn=recon_visibility_delay,
    )
    if AttackType.FAKE_OBSTACLE in set(enabled) and not base_candidates:
        base_candidates = recon_heatmap_attack_candidates(
            physical_base_grid,
            targets,
            recon_robots,
            traffic_heatmap,
            rng=named_rng(config.seed, "attack_placement_visibility_fallback"),
            require_route_overlap=True,
            forbidden_cells=forbidden,
            active_temp_cells=(),
            visible_cells_by_robot={},
            future_visibility_delay_fn=None,
        )
    candidate_metadata=[]; use_count: dict[tuple[int,int],int]={}; selected=[]
    authoring_warnings = []
    false_clearance_kind_uses = {"permanent": 0, "temporary": 0}
    false_clearance_target_uses = {}
    stale_reassertion_target_uses = {}
    physical_target_relevance_cache = {}
    while step < phases.recon_steps + phases.attack_steps and enabled:
        if not preference_bag:
            preference_bag = list(AttackType)
            type_rng.shuffle(preference_bag)
        preferred = preference_bag.pop(0)
        feasible = set(_feasible_attack_types(enabled, step, episodes, grid, permanent_obstacles))
        used_types = {event.attack_type for event in events}
        unseen_feasible = [kind for kind in enabled if kind in feasible and kind not in used_types]
        if unseen_feasible:
            # Coverage is a scenario-authoring invariant, not a defense
            # advantage: when a requested attack type has a valid target, give
            # it one opportunity before repeatedly sampling types already
            # represented in the manifest.
            ordered = [*unseen_feasible, *[kind for kind in enabled if kind in feasible and kind not in unseen_feasible]]
        elif preferred in feasible:
            ordered = [preferred, *[kind for kind in enabled if kind != preferred and kind in feasible]]
        else:
            ordered = [kind for kind in enabled if kind in feasible]
        for kind in ordered:
            preferred_false_clearance_kind = None
            if kind == AttackType.FALSE_CLEARANCE:
                # Prefer both permanent and temporary targets when feasible,
                # but let frozen-recon consequence remain the primary ranking
                # signal rather than forcing an artificial class split.
                unused_kinds = [
                    name for name, count in false_clearance_kind_uses.items() if count == 0
                ]
                preferred_false_clearance_kind = min(unused_kinds) if unused_kinds else None
            placed = _instantiate_attack(
                kind, step=step, rng=place_rng, episodes=episodes, permanent_obstacles=permanent_obstacles, route_cells=route_cells,
                use_count=use_count, selected=selected, config=config, grid=grid, forbidden=forbidden,
                preferred_false_clearance_kind=preferred_false_clearance_kind,
                stale_target_uses=stale_reassertion_target_uses,
                false_clearance_target_uses=false_clearance_target_uses,
                mission_pairs=mission_pairs,
                recon_candidates=base_candidates,
                recon_reference_states=reference_states,
                recon_benign_ids=benign,
                recon_heatmap=traffic_heatmap,
                recon_relevance_cache=physical_target_relevance_cache,
            )
            if placed is None:
                continue
            cells, claim, observation, episode, center = placed
            eid=f"attack-{index:04}"
            rids=tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells)))
            target_id = (
                getattr(episode, "episode_id", None)
                or getattr(episode, "obstacle_id", None)
            ) if episode else None
            prior_target_use_count = (
                false_clearance_target_uses.get(str(target_id), 0)
                if kind == AttackType.FALSE_CLEARANCE and target_id is not None
                else 0
            )
            events.append(AttackEvent(eid, step, kind, tuple(cells), claim, observation, sender, benign, rids, target_id)); index += 1
            use_count[center]=use_count.get(center,0)+1; selected.append(center)
            metadata = {
                "candidate_id": f"candidate-{index-1:04}",
                "event_id": eid,
                "attack_type": kind.value,
                "center": center,
                "footprint_cells": [list(cell) for cell in cells],
                "traffic_score": float(sum(traffic_heatmap[cell] for cell in cells)),
                "bottleneck_score": float(footprint_bottleneck_score(physical_base_grid, cells)),
                "estimated_detour_score": float(footprint_finite_detour_score(physical_base_grid, cells)),
                "rank": None,
                "selection_weight": None,
                "prior_use_count": use_count[center] - 1,
                "target_prior_use_count": prior_target_use_count,
                "target_reused": bool(prior_target_use_count),
                "reference_step": phases.recon_steps - 1,
                "heatmap_reference_steps": phases.recon_steps,
                "selection_basis": (
                    "frozen_reconnaissance_traffic_and_static_geometry"
                    if kind == AttackType.FAKE_OBSTACLE
                    else "frozen_reconnaissance_traffic_plus_static_geometry_and_authored_physical_schedule"
                ),
            }
            if kind == AttackType.STALE_REASSERTION:
                metadata.update(_recon_reblock_features(
                    frozen_recon,
                    benign,
                    physical_base_grid,
                    cells,
                    clearance_step=getattr(episode, "clearance_step", None),
                ))
            elif kind == AttackType.FALSE_CLEARANCE:
                metadata["recon_false_clearance_shortcut_steps"] = _mission_reblock_penalty(
                    cells,
                    mission_pairs,
                    physical_base_grid,
                    permanent_obstacles,
                )
            candidate_metadata.append(metadata)
            if kind == AttackType.STALE_REASSERTION and episode is not None:
                stale_reassertion_target_uses[episode.episode_id] = stale_reassertion_target_uses.get(episode.episode_id, 0) + 1
            if kind == AttackType.FALSE_CLEARANCE and episode is not None:
                target_kind = "permanent" if hasattr(episode, "obstacle_id") else "temporary"
                false_clearance_kind_uses[target_kind] += 1
                false_clearance_target_uses[str(target_id)] = prior_target_use_count + 1
                if prior_target_use_count:
                    authoring_warnings.append("false_clearance_target_reused_after_positive_candidates_exhausted")
            break
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    names=("attack_scheduler", "attack_types", "attack_placement", "permanent_obstacles", "temporary_obstacles", "robot_routes", "traffic")
    starts = dict(preset.robot_starts) if preset else {0: starts_tuple[0], 1: starts_tuple[1], 2: starts_tuple[2]}
    # Keep the attacker physically active with the same deterministic repeating
    # queue as the other robots; only its reporting behavior is malicious.
    warnings = list(authoring_warnings)
    if len(set(selected)) < min(config.attacks.min_unique_footprints, len(selected)):
        warnings.append("concentrated_attack_manifest")
    # Script the attacker independently of defense-dependent benign routes.
    attacker_route=tuple(route_cells) or tuple(free)
    positions=tuple(attacker_route[step % len(attacker_route)] for step in range(phases.total_steps))
    def truth(cell, step):
        return ClaimType.BLOCKED if (
            cell in permanent_cells
            or any(cell in episode.cells and episode.appearance_step <= step < episode.clearance_step for episode in episodes)
        ) else ClaimType.FREE
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
    return ScenarioManifest(SCHEMA_VERSION, config.seed, {x:derived_seed(config.seed,x) for x in names}, _hash(grid), tuple(grid.shape), static_grid, phase_boundaries, sender, benign, episodes, tuple(events), scenario_id=f"scenario-{config.seed}-{_hash(grid)[:12]}", protocol_id="modular_v1", robot_starts=starts, task_queues=queues, attacker_positions=positions, honest_attacker_reports=honest, report_audit_labels=labels, candidate_metadata=tuple(candidate_metadata), authoring_warnings=tuple(dict.fromkeys(warnings)), reconnaissance_heatmap=tuple(tuple(int(value) for value in row) for row in frozen_recon.traffic_heatmap), scenario_preset=config.scenario_preset, permanent_obstacles=permanent_obstacles, reconnaissance_data=frozen_recon)

def save_manifest(manifest: ScenarioManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

def load_manifest(path: str | Path) -> ScenarioManifest:
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported scenario manifest schema; author a schema-v3 manifest")
    episodes=tuple(TemporaryObstacleEpisode(x["episode_id"], tuple(map(tuple,x["cells"])), x["appearance_step"],x["clearance_step"]) for x in raw["obstacle_episodes"])
    events=tuple(AttackEvent(x["event_id"],x["step"],AttackType(x["attack_type"]),tuple(map(tuple,x["cells"])),ClaimType(x["claim"]),x["observation_step"],x["sender_id"],tuple(x["recipients"]),tuple(x["report_ids"]),x.get("obstacle_episode_id")) for x in raw["attack_events"])
    permanents=tuple(PermanentObstacle(x["obstacle_id"], tuple(map(tuple, x["cells"]))) for x in raw.get("permanent_obstacles", ()))
    starts={int(key):tuple(value) for key,value in (raw.get("robot_starts") or {}).items()}
    queues={int(key):tuple(DeliveryTask(item["task_id"],tuple(item["pickup"]),tuple(item["dropoff"])) for item in value) for key,value in (raw.get("task_queues") or {}).items()}
    reports=tuple(ClaimReport(item["report_id"], item["sender_id"], tuple(item["target_cell"]), ClaimType(item["claim"]), item["observation_step"], item.get("sensor_confidence", 1.0), item.get("scenario_event_id"), bool(item.get("persistent_until_verified", False))) for item in raw.get("honest_attacker_reports",()))
    labels=tuple(ReportAuditLabel(item["report_id"],item["is_malicious"],AttackType(item["attack_type"]) if item.get("attack_type") else None,item.get("obstacle_episode_id"),ClaimType(item["actual_state_at_observation"]),item.get("original_obstacle_appearance_step"),item.get("original_obstacle_clearance_step")) for item in raw.get("report_audit_labels",()))
    recon_raw = raw.get("reconnaissance_data")
    recon = None
    if recon_raw:
        mission_snapshots = tuple(
            ReconMissionSnapshot(
                int(item["step"]),
                int(item["robot_id"]),
                item.get("mission_id"),
                tuple(item["position"]),
                tuple(item["goal"]) if item.get("goal") is not None else None,
                tuple(tuple(cell) for cell in item.get("planned_path", ())),
                tuple(tuple(cell) for cell in item.get("visible_cells", ())),
            )
            for item in recon_raw.get("mission_snapshots", ())
        )
        route_snapshots = tuple(
            ReconRouteSnapshot(
                int(item["step"]),
                int(item["robot_id"]),
                item.get("mission_id"),
                tuple(item["start"]),
                tuple(item["goal"]),
                tuple(tuple(cell) for cell in item.get("path", ())),
            )
            for item in recon_raw.get("route_snapshots", ())
        )
        recon = FrozenReconData(
            int(recon_raw["recon_steps"]),
            tuple(tuple(int(value) for value in row) for row in recon_raw["traffic_heatmap"]),
            mission_snapshots,
            route_snapshots,
            tuple((tuple(item[0]), int(item[1])) for item in recon_raw.get("cell_route_usage", ())),
            tuple((tuple(tuple(cell) for cell in item[0]), int(item[1])) for item in recon_raw.get("footprint_route_usage", ())),
            tuple((str(item[0]), int(item[1]), int(item[2]), tuple(tuple(cell) for cell in item[3])) for item in recon_raw.get("observed_obstacle_events", ())),
            tuple((int(item[0]), int(item[1])) for item in recon_raw.get("visibility_summaries", ())),
            recon_raw.get("map_hash", ""),
            recon_raw.get("scenario_id", ""),
            int(recon_raw.get("seed", raw["master_seed"])),
        )
    return ScenarioManifest(raw["schema_version"],raw["master_seed"],raw["derived_seeds"],raw["map_hash"],tuple(raw["map_shape"]),tuple(tuple(row) for row in raw["static_grid"]),raw["phase_boundaries"],raw["malicious_robot_id"],tuple(raw["benign_robot_ids"]),episodes,events,raw.get("scenario_id",""),raw.get("protocol_id","custom"),starts,queues,tuple(map(tuple,raw.get("attacker_positions",()))),reports,labels,tuple(raw.get("candidate_metadata",())),tuple(raw.get("authoring_warnings",())),tuple(tuple(int(value) for value in row) for row in raw["reconnaissance_heatmap"]) if raw.get("reconnaissance_heatmap") else None,raw.get("scenario_preset"),permanents,recon)
