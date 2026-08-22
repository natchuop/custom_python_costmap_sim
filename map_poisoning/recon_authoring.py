"""Reconnaissance heatmap authoring for default-warehouse manifests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .models import AttackEvent, AttackType, ClaimReport, ClaimType, DeliveryTask, ReportAuditLabel, TemporaryObstacleEpisode
from .map_io import default_warehouse_map
from .obstacles import (
    FAKE_MIN_REPORT_CELLS,
    author_temporary_obstacle_episodes,
    fake_report_cells,
    sample_fake_obstacle_dimensions,
)
from .rng import derived_seed, named_rng
from .rollout import run_manifest_rollout
from .scenario import ScenarioManifest, _hash
from .warehouse_layout import DEFAULT_NUM_ROBOTS, build_warehouse_layout, manhattan
from .planning import astar

ATTACK_CANDIDATE_LIMIT = 24
ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = False
ATTACK_MIN_DISTANCE_FROM_GOAL = 2
ATTACK_MIN_DISTANCE_FROM_ANY_BENIGN_ROBOT = 2
MALICIOUS_FAKE_OBJECT_CENTER_MIN_SPACING = 3
SCHEMA_VERSION = 2


def _build_attacker_route_cells(
    grid: np.ndarray,
    start: tuple[int, int],
    tasks: tuple[DeliveryTask, ...],
) -> list[tuple[int, int]]:
    """Deterministic static route for scripted honest attacker reports."""
    rows, cols = grid.shape
    cost = lambda cell: float("inf") if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid[cell] else 1.0
    route: list[tuple[int, int]] = [tuple(start)]
    current = tuple(start)
    for task in tasks:
        for goal in (tuple(task.pickup), tuple(task.dropoff)):
            segment = astar(current, goal, cost) or [current]
            extension = [tuple(cell) for cell in segment]
            if extension and route and extension[0] == route[-1]:
                extension = extension[1:]
            route.extend(extension)
            current = goal
    return route or [tuple(start)]


def _positive_traffic_values(heatmap: np.ndarray) -> np.ndarray:
    values = heatmap[heatmap > 0]
    return values if values.size else np.array([0.0])


def _average_traffic_bounds(heatmap: np.ndarray) -> tuple[float, float]:
    values = _positive_traffic_values(heatmap)
    return float(values.min()), float(values.max())


def _is_average_traffic_cell(cell: tuple[int, int], heatmap: np.ndarray) -> bool:
    row, col = cell
    if not (0 <= row < heatmap.shape[0] and 0 <= col < heatmap.shape[1]):
        return False
    minimum, maximum = _average_traffic_bounds(heatmap)
    if maximum <= minimum:
        return heatmap[row, col] > 0
    score = heatmap[row, col]
    return minimum <= score <= maximum


def _traffic_score(cell: tuple[int, int], heatmap: np.ndarray) -> float:
    row, col = cell
    return float(heatmap[row, col]) if 0 <= row < heatmap.shape[0] and 0 <= col < heatmap.shape[1] else 0.0


def _fake_average_traffic_score(center, heatmap, height, width) -> float:
    from .obstacles import footprint_from_center

    cells = footprint_from_center(center, height, width)
    scores = [
        _traffic_score(cell, heatmap)
        for cell in cells
        if 0 <= cell[0] < heatmap.shape[0] and 0 <= cell[1] < heatmap.shape[1]
    ]
    return float(np.mean(scores)) if scores else 0.0


def _footprint_bottleneck_score(grid, cells) -> float:
    if not cells:
        return 0.0
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return float(height * width) / max(1, len(cells))


def _is_valid_recon_attack_cell(
    cell,
    grid,
    goals,
    robots,
    heatmap,
    *,
    forbidden_cells=(),
    active_temp_cells=(),
):
    if not _is_average_traffic_cell(cell, heatmap):
        return False
    if cell in set(forbidden_cells):
        return False
    if any(manhattan(cell, tuple(goal)) < ATTACK_MIN_DISTANCE_FROM_GOAL for goal in goals):
        return False
    if any(
        manhattan(cell, robot.position) < ATTACK_MIN_DISTANCE_FROM_ANY_BENIGN_ROBOT
        for robot in robots
        if robot.robot_id != 0
    ):
        return False
    report_cells = fake_report_cells(
        cell,
        3,
        3,
        grid,
        forbidden=forbidden_cells,
        active_cells=active_temp_cells,
    )
    return len(report_cells) >= FAKE_MIN_REPORT_CELLS


def recon_heatmap_attack_candidates(
    grid,
    goals,
    robots,
    heatmap: np.ndarray,
    *,
    placed_centers=(),
    rng=None,
    require_route_overlap: bool = ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP,
    forbidden_cells=(),
    active_temp_cells=(),
):
    candidates = []
    rows, cols = heatmap.shape
    rng = rng or named_rng(0, "warehouse_candidates")
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if any(manhattan(cell, center) < MALICIOUS_FAKE_OBJECT_CENTER_MIN_SPACING for center in placed_centers):
                continue
            if not _is_valid_recon_attack_cell(
                cell,
                grid,
                goals,
                robots,
                heatmap,
                forbidden_cells=forbidden_cells,
                active_temp_cells=active_temp_cells,
            ):
                continue
            height, width = sample_fake_obstacle_dimensions(rng)
            report_cells = fake_report_cells(
                cell,
                height,
                width,
                grid,
                forbidden=forbidden_cells,
                active_cells=active_temp_cells,
            )
            if len(report_cells) < FAKE_MIN_REPORT_CELLS:
                continue
            path_overlap = 0
            affected_victims = 0
            path_proximity_score = 0.0
            for victim in robots:
                if victim.robot_id == 0:
                    continue
                remaining = list(victim.path or ())
                if not remaining:
                    continue
                overlap = len(set(remaining).intersection(report_cells))
                path_overlap += overlap
                if overlap > 0:
                    affected_victims += 1
                    path_proximity_score += 10.0 + overlap
                    continue
                min_distance = min(
                    manhattan(report_cell, path_cell)
                    for report_cell in report_cells
                    for path_cell in remaining
                )
                if min_distance <= 2:
                    affected_victims += 1
                path_proximity_score += 1.0 / (1.0 + min_distance)
            if require_route_overlap and path_overlap <= 0:
                continue
            candidates.append(
                {
                    "center_cell": cell,
                    "report_cells": report_cells,
                    "traffic_score": _fake_average_traffic_score(cell, heatmap, height, width),
                    "footprint_height": height,
                    "footprint_width": width,
                    "report_cell_count": len(report_cells),
                    "path_overlap": path_overlap,
                    "path_proximity_score": path_proximity_score,
                    "affected_victims": affected_victims,
                    "bottleneck_score": _footprint_bottleneck_score(grid, report_cells),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["affected_victims"],
            item["path_overlap"],
            item["path_proximity_score"],
            item["report_cell_count"],
            item["traffic_score"],
        ),
        reverse=True,
    )
    return candidates[:ATTACK_CANDIDATE_LIMIT]


def run_clean_recon_rollout(config: SimulationConfig, manifest: ScenarioManifest):
    """Replay a no-attack manifest through reconnaissance and return the traffic heatmap."""
    recon_config = replace(config, max_steps=config.phases.recon_steps, visualization=replace(config.visualization, animation=False))
    _, robots, log = run_manifest_rollout(recon_config, manifest, "full_trust", show_progress=False)
    heatmap = np.zeros(manifest.map_shape, dtype=np.int32)
    benign = set(manifest.benign_robot_ids)
    for event in log["events"]:
        if event.get("kind") != "robot_action" or event.get("phase") != "RECONNAISSANCE":
            continue
        if event.get("robot_id") not in benign:
            continue
        position = tuple(event["position"])
        heatmap[position] += 1
    if not heatmap.any():
        for sample in log["timeseries"]:
            if sample["step"] >= config.phases.recon_steps:
                continue
            if sample["robot_id"] in benign:
                heatmap[tuple(sample["position"])] += 1
    return heatmap, robots, log


def save_traffic_heatmap_artifacts(root: Path, heatmap: np.ndarray, *, title: str = "Reconnaissance traffic heatmap") -> None:
    root.mkdir(parents=True, exist_ok=True)
    array = np.asarray(heatmap, dtype=np.int32)
    np.save(root / "traffic_heatmap.npy", array)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 6))
        masked = array.astype(float)
        masked[masked <= 0] = np.nan
        valid = masked[np.isfinite(masked)]
        vmax = float(np.nanpercentile(valid, 99.0)) if valid.size else None
        if vmax is not None and vmax <= 0:
            vmax = None
        image = axis.imshow(masked, origin="upper", cmap="hot", vmin=0, vmax=vmax)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.tight_layout()
        figure.savefig(root / "traffic_heatmap.png", dpi=160)
        plt.close(figure)
    except Exception:
        pass


def author_warehouse_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    from .map_io import (
        WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS,
        WAREHOUSE_NARROW_CORRIDOR_CELLS,
    )
    from .scenario import _active_temp_cells

    config.validate()
    grid = np.asarray(default_warehouse_map() if grid is None else grid, dtype=np.uint8)
    starts, goals, task_queues = build_warehouse_layout(grid, config.deliveries_per_robot)
    sender = 0
    benign = tuple(robot_id for robot_id in starts if robot_id != sender)
    forbidden = set(starts.values()) | set(WAREHOUSE_NARROW_CORRIDOR_CELLS)
    episodes = author_temporary_obstacle_episodes(
        grid,
        named_rng(config.seed, "temporary_obstacles"),
        config.phases.total_steps,
        config.temporary_blockage_change_period_steps,
        forbidden_cells=forbidden,
        required_anchors=WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS,
    )
    skeleton = ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        {name: derived_seed(config.seed, name) for name in ("attack_scheduler", "attack_types", "attack_placement", "temporary_obstacles", "robot_routes", "traffic")},
        _hash(grid),
        tuple(grid.shape),
        tuple(tuple(int(value) for value in row) for row in grid),
        {
            "reconnaissance_end": config.phases.recon_steps,
            "attack_end": config.phases.recon_steps + config.phases.attack_steps,
            "total": config.phases.total_steps,
        },
        sender,
        benign,
        episodes,
        (),
        scenario_id=f"warehouse-{config.seed}",
        protocol_id="modular_v1",
        robot_starts=starts,
        task_queues=task_queues,
    )
    heatmap, robots, recon_log = run_clean_recon_rollout(config, skeleton)
    place_rng = named_rng(config.seed, "attack_placement")
    task_cells = {
        tuple(task.pickup)
        for queue in task_queues.values()
        for task in queue
    } | {
        tuple(task.dropoff)
        for queue in task_queues.values()
        for task in queue
    }
    protected_cells = set(starts.values()) | set(goals) | task_cells
    candidates = recon_heatmap_attack_candidates(
        grid,
        goals,
        robots,
        heatmap,
        rng=place_rng,
        forbidden_cells=protected_cells,
    )
    if not candidates:
        candidates = recon_heatmap_attack_candidates(
            grid,
            goals,
            robots,
            heatmap,
            rng=place_rng,
            require_route_overlap=False,
            forbidden_cells=protected_cells,
        )
    if not candidates:
        raise RuntimeError("clean warehouse rollout produced no manifest attack candidates")

    def candidates_at_step(step: int):
        frame = min(max(0, step), config.phases.recon_steps - 1)
        for robot in robots:
            samples = [row for row in recon_log["timeseries"] if row["robot_id"] == robot.robot_id and row["step"] == frame]
            if samples:
                robot.position = tuple(samples[-1]["position"])
        dynamic_forbidden = protected_cells | {tuple(robot.position) for robot in robots}
        return recon_heatmap_attack_candidates(
            grid,
            goals,
            robots,
            heatmap,
            rng=place_rng,
            require_route_overlap=True,
            forbidden_cells=dynamic_forbidden,
            active_temp_cells=_active_temp_cells(episodes, step),
        )

    rng = named_rng(config.seed, "warehouse_manifest_scheduler")
    enabled_types = [AttackType(value) for value in config.attacks.enabled]
    events = []
    metadata = []
    warnings = []
    uses: dict[tuple[int, int], int] = {}
    selected_centers = []
    step = config.phases.recon_steps + rng.randint(config.attacks.interval_min, config.attacks.interval_max)
    index = 0
    while step < config.phases.recon_steps + config.phases.attack_steps and enabled_types:
        feasible = [
            kind
            for kind in enabled_types
            if kind == AttackType.FAKE_OBSTACLE
            or (
                kind == AttackType.FALSE_CLEARANCE
                and any(episode.appearance_step <= step < episode.clearance_step for episode in episodes)
            )
            or (
                kind == AttackType.STALE_REASSERTION
                and any(episode.clearance_step <= step for episode in episodes)
            )
        ]
        if not feasible:
            break
        selected_attack = feasible[rng.randrange(len(feasible))]
        if selected_attack != AttackType.FAKE_OBSTACLE:
            eligible = [
                episode
                for episode in episodes
                if (
                    selected_attack == AttackType.FALSE_CLEARANCE
                    and episode.appearance_step <= step < episode.clearance_step
                )
                or (selected_attack == AttackType.STALE_REASSERTION and episode.clearance_step <= step)
            ]
            if not eligible:
                step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
                continue
            episode = eligible[rng.randrange(len(eligible))]
            claim = ClaimType.FREE if selected_attack == AttackType.FALSE_CLEARANCE else ClaimType.BLOCKED
            cells = tuple(episode.cells)
            events.append(
                AttackEvent(
                    f"attack-{index:04}",
                    step,
                    selected_attack,
                    cells,
                    claim,
                    step,
                    sender,
                    benign,
                    tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells))),
                    episode.episode_id,
                )
            )
            index += 1
            step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
            continue
        step_candidates = candidates_at_step(step)
        if not step_candidates:
            frame = min(max(0, step), config.phases.recon_steps - 1)
            for robot in robots:
                samples = [row for row in recon_log["timeseries"] if row["robot_id"] == robot.robot_id and row["step"] == frame]
                if samples:
                    robot.position = tuple(samples[-1]["position"])
            dynamic_forbidden = protected_cells | {tuple(robot.position) for robot in robots}
            step_candidates = recon_heatmap_attack_candidates(
                grid,
                goals,
                robots,
                heatmap,
                rng=place_rng,
                require_route_overlap=False,
                forbidden_cells=dynamic_forbidden,
                active_temp_cells=_active_temp_cells(episodes, step),
            )
        if step_candidates:
            pool = step_candidates[: config.attacks.candidate_top_k]
        else:
            active_now = _active_temp_cells(episodes, step)
            blocked_now = protected_cells | {tuple(robot.position) for robot in robots} | set(active_now)
            fallback = [
                candidate
                for candidate in candidates
                if not any(tuple(cell) in blocked_now for cell in candidate["report_cells"])
            ]
            pool = fallback[: config.attacks.candidate_top_k]
        eligible = [
            candidate
            for candidate in pool
            if uses.get(tuple(candidate["center_cell"]), 0) < config.attacks.max_uses_per_footprint
            and all(
                abs(candidate["center_cell"][0] - old[0]) + abs(candidate["center_cell"][1] - old[1])
                >= config.attacks.min_center_spacing
                for old in selected_centers
            )
        ]
        if not eligible:
            warnings.append("concentrated_attack_manifest: diversity limits exhausted")
            break
        candidate = eligible[rng.randrange(len(eligible))]
        cells = tuple(tuple(cell) for cell in candidate["report_cells"])
        center = tuple(candidate["center_cell"])
        events.append(
            AttackEvent(
                f"attack-{index:04}",
                step,
                AttackType.FAKE_OBSTACLE,
                cells,
                ClaimType.BLOCKED,
                step,
                sender,
                benign,
                tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells))),
            )
        )
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
                "rank": (step_candidates.index(candidate) + 1 if candidate in step_candidates else candidates.index(candidate) + 1),
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
    labels = tuple(
        ReportAuditLabel(
            report_id,
            True,
            event.attack_type,
            event.obstacle_episode_id,
            ClaimType.BLOCKED if event.attack_type == AttackType.FALSE_CLEARANCE else ClaimType.FREE,
        )
        for event in events
        for report_id in event.report_ids
    )
    attacker_route = _build_attacker_route_cells(
        grid,
        starts[sender],
        tuple(task_queues.get(sender, ())),
    )
    attacker_positions = tuple(
        attacker_route[step % len(attacker_route)]
        for step in range(config.phases.total_steps)
    )
    honest_attacker_reports = tuple(
        ClaimReport(
            f"attacker-honest-{step:05}",
            sender,
            attacker_positions[step],
            ClaimType.BLOCKED
            if any(
                episode.appearance_step <= step < episode.clearance_step
                and attacker_positions[step] in episode.cells
                for episode in episodes
            )
            else ClaimType.FREE,
            step,
            step,
            step,
        )
        for step in range(0, config.phases.total_steps, config.communication_period_steps)
    )
    return ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        {name: derived_seed(config.seed, name) for name in ("attack_scheduler", "attack_types", "attack_placement", "temporary_obstacles", "robot_routes", "traffic", "warehouse_manifest_scheduler")},
        _hash(grid),
        tuple(grid.shape),
        tuple(tuple(int(value) for value in row) for row in grid),
        {
            "reconnaissance_end": config.phases.recon_steps,
            "attack_end": config.phases.recon_steps + config.phases.attack_steps,
            "total": config.phases.total_steps,
        },
        sender,
        benign,
        episodes,
        tuple(events),
        scenario_id=f"warehouse-{config.seed}",
        protocol_id="modular_v1",
        robot_starts=starts,
        task_queues=task_queues,
        attacker_positions=attacker_positions,
        honest_attacker_reports=honest_attacker_reports,
        report_audit_labels=labels,
        candidate_metadata=tuple(metadata),
        authoring_warnings=tuple(dict.fromkeys(warnings)),
        reconnaissance_heatmap=tuple(tuple(int(value) for value in row) for row in heatmap.tolist()),
    )
