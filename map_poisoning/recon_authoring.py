"""Reconnaissance heatmap authoring for default-warehouse manifests."""
from __future__ import annotations

from dataclasses import replace
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
    ReconMissionSnapshot,
    ReconRouteSnapshot,
    ReportAuditLabel,
)
from .map_io import default_warehouse_map
from .obstacles import (
    FAKE_MIN_REPORT_CELLS,
    author_permanent_obstacles,
    author_temporary_obstacle_episodes,
    fake_report_cells,
    footprint_center,
    footprint_finite_detour_score,
    sample_fake_obstacle_dimensions,
)
from .rng import derived_seed, named_rng
from .rollout import run_manifest_rollout
from .scenario import ScenarioManifest, _hash
from .warehouse_layout import build_warehouse_layout, manhattan
from .planning import astar

ATTACK_CANDIDATE_LIMIT = 24
ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = True
ATTACK_MIN_DISTANCE_FROM_GOAL = 2
ATTACK_MIN_DISTANCE_FROM_ANY_BENIGN_ROBOT = 2
MALICIOUS_FAKE_OBJECT_CENTER_MIN_SPACING = 3
ATTACK_NEAR_FUTURE_ROUTE_STEPS = 80
ATTACK_MAX_ROUTE_DISTANCE = 3
FALSE_CLEARANCE_MIN_REMAINING_STEPS = 20
STALE_REASSERTION_PREFERRED_AGE = (30, 100)
SCHEMA_VERSION = 3


def _center_cell(cells) -> tuple[int, int]:
    center = footprint_center(cells)
    return int(round(center[0])), int(round(center[1]))


def _nearby_recon_traffic_score(heatmap: np.ndarray, cells, radius: int = 2) -> float:
    """Recon traffic near a blocked footprint, not on the blocked cells themselves.

    Physical obstacles cannot accumulate position heat on their own cells, so
    False Clearance relevance must look at the surrounding traffic corridor.
    This uses only the frozen reconnaissance heatmap.
    """
    footprint = {tuple(cell) for cell in cells}
    rows, cols = heatmap.shape
    nearby = set()
    for row, col in footprint:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                cell = (row + dr, col + dc)
                if 0 <= cell[0] < rows and 0 <= cell[1] < cols and cell not in footprint:
                    nearby.add(cell)
    return float(sum(float(heatmap[cell]) for cell in nearby))


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
    footprint = {tuple(cell) for cell in cells}
    rows, cols = grid.shape
    blocked_neighbors = 0
    boundary_neighbors = 0
    for row, col in footprint:
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in footprint:
                continue
            boundary_neighbors += 1
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols) or grid[neighbor]:
                blocked_neighbors += 1
    return blocked_neighbors / max(1, boundary_neighbors)


def _reference_detour_score(grid, victim, report_cells, active_temp_cells=()) -> float | None:
    """Return clean-reference path-length increase caused by a fake block."""
    if victim.goal is None or tuple(victim.position) == tuple(victim.goal):
        return 0.0
    rows, cols = grid.shape
    physical = {tuple(cell) for cell in active_temp_cells}
    fake = {tuple(cell) for cell in report_cells}

    def cost(cell, extra=()):
        return (
            float("inf")
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols)
            or bool(grid[cell])
            or cell in physical
            or cell in extra
            else 1.0
        )

    baseline = astar(tuple(victim.position), tuple(victim.goal), lambda cell: cost(cell))
    if baseline is None:
        return 0.0
    attacked = astar(tuple(victim.position), tuple(victim.goal), lambda cell: cost(cell, fake))
    if attacked is None:
        # The stress experiment targets meaningful detours, not fabricated
        # total disconnections.  No-path behavior remains measurable for real
        # runtime interactions, but it is not deliberately authored here.
        return None
    return float(max(0, len(attacked) - len(baseline)))


def _episode_route_candidate(episode, victim, step, visible_cells, *, stale=False):
    """Describe how strongly an episode can affect a clean-reference victim."""
    remaining = list(victim.path or ())[:ATTACK_NEAR_FUTURE_ROUTE_STEPS]
    if not remaining:
        return None
    footprint = {tuple(cell) for cell in episode.cells}
    visible = {tuple(cell) for cell in visible_cells}
    if footprint.intersection(visible):
        return None
    overlap = len(set(remaining).intersection(footprint))
    min_distance = min(
        manhattan(path_cell, footprint_cell)
        for path_cell in remaining
        for footprint_cell in footprint
    )
    nearest_index = min(
        range(len(remaining)),
        key=lambda index: min(manhattan(remaining[index], cell) for cell in footprint),
    )
    age = max(0, step - episode.clearance_step) if stale else None
    remaining_lifetime = max(0, episode.clearance_step - step) if not stale else None
    # Direct overlap is strongest; nearby traffic is still useful when a
    # moving robot is about to enter the footprint.
    relevance = (100.0 + overlap * 10.0) if overlap else (20.0 / (1.0 + min_distance))
    center = footprint_center(episode.cells)
    center_cell = (int(round(center[0])), int(round(center[1])))
    return {
        "victim_id": int(victim.robot_id),
        "route_overlap": int(overlap),
        "victim_distance": int(manhattan(tuple(victim.position), center_cell)),
        "route_distance_steps": int(nearest_index),
        "min_route_distance": int(min_distance),
        "remaining_obstacle_lifetime": remaining_lifetime,
        "age_since_clearance": age,
        "target_visible_to_victim": False,
        "relevance_score": relevance,
    }


def _historical_visibility_delay(reference_states, victim_id, report_cells, delay_min, delay_max):
    """Estimate verification delay using only observed reconnaissance history."""
    reference_states = _reference_states(reference_states)
    footprint = {tuple(cell) for cell in report_cells}
    steps = sorted(step for step in reference_states if victim_id in reference_states[step])
    state_by_step = {step: reference_states[step][victim_id] for step in steps}
    for step in steps:
        current_visible = {tuple(cell) for cell in state_by_step[step].get("visible_cells", ())}
        if footprint.intersection(current_visible):
            continue
        for delay in range(delay_min, delay_max + 1):
            future = state_by_step.get(step + delay)
            if future is None:
                continue
            future_visible = {tuple(cell) for cell in future.get("visible_cells", ())}
            if footprint.intersection(future_visible):
                return int(delay)
    return None


def _recon_false_clearance_shortcut_score(reference_states, benign_ids, grid, cells) -> float:
    """Best finite shortcut exposed by falsely clearing a physical obstacle.

    Only position/goal pairs that were actually present in the frozen
    reconnaissance reference are sampled. This deliberately avoids all
    post-recon routes while directly measuring whether the blocked footprint
    forces an observed mission to take a longer path.
    """
    reference_states = _reference_states(reference_states)
    footprint = {tuple(cell) for cell in cells}
    if not footprint:
        return 0.0
    rows, cols = grid.shape
    cleared_grid = np.array(grid, dtype=np.uint8, copy=True)
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            cleared_grid[cell] = 0
    blocked_grid = np.array(cleared_grid, dtype=np.uint8, copy=True)
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            blocked_grid[cell] = 1

    available_steps = sorted(reference_states)
    if not available_steps:
        return 0.0
    # At most twelve reconnaissance snapshots are sufficient to cover the
    # observed mission geometry without making manifest authoring expensive.
    if len(available_steps) > 12:
        indexes = np.linspace(0, len(available_steps) - 1, 12, dtype=int)
        sample_steps = [available_steps[int(index)] for index in indexes]
    else:
        sample_steps = available_steps

    def route_len(grid_value, start, goal):
        path = astar(
            start,
            goal,
            lambda cell: (
                float("inf")
                if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid_value[cell]
                else 1.0
            ),
        )
        return None if path is None else max(0, len(path) - 1)

    best = 0.0
    for step in sample_steps:
        states = reference_states.get(step, {})
        for victim_id in benign_ids:
            state = states.get(victim_id)
            if not state or state.get("goal") is None:
                continue
            start = tuple(state["position"])
            goal = tuple(state["goal"])
            blocked_len = route_len(blocked_grid, start, goal)
            cleared_len = route_len(cleared_grid, start, goal)
            if blocked_len is None or cleared_len is None:
                continue
            best = max(best, float(blocked_len - cleared_len))
    return max(0.0, best)


def _recon_reblock_features(reference_states, benign_ids, grid, cells, *, clearance_step=None):
    """Measure stale-candidate consequences across frozen recon missions.

    The feature set is intentionally explicit so selection cannot let generic
    traffic overwhelm demonstrated mission consequence.  A mission is counted
    once by its stable mission id (or by its robot/start/goal tuple for legacy
    reference fixtures that predate mission ids).
    """
    reference_states = _reference_states(reference_states)
    footprint = {tuple(cell) for cell in cells}
    rows, cols = grid.shape
    snapshots = []
    for step in sorted(reference_states):
        states = reference_states[step]
        for robot_id in benign_ids:
            state = states.get(robot_id, states.get(str(robot_id)))
            if state is None or state.get("goal") is None:
                continue
            snapshots.append((int(step), int(robot_id), state))
    if len(snapshots) > 24:
        # Preserve the first observation of each mission and fill the remainder
        # uniformly.  This bounds authoring cost without losing recurrence.
        selected = []
        seen_missions = set()
        for step, robot_id, state in snapshots:
            mission_key = state.get("mission_id") or (
                robot_id,
                tuple(state["position"]),
                tuple(state["goal"]),
            )
            if mission_key not in seen_missions:
                selected.append((step, robot_id, state))
                seen_missions.add(mission_key)
        selected_keys = {(step, robot_id) for step, robot_id, _ in selected}
        remainder = [item for item in snapshots if (item[0], item[1]) not in selected_keys]
        if len(selected) < 24 and remainder:
            indexes = np.linspace(0, len(remainder) - 1, 24 - len(selected), dtype=int)
            selected.extend(remainder[int(index)] for index in indexes)
        snapshots = selected[:24]

    def route(grid_value, start, goal):
        path = astar(
            tuple(start), tuple(goal),
            lambda cell: (
                float("inf")
                if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid_value[cell]
                else 1.0
            ),
        )
        return path

    free_grid = np.array(grid, dtype=np.uint8, copy=True)
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            free_grid[cell] = 0
    blocked_grid = np.array(free_grid, dtype=np.uint8, copy=True)
    for cell in footprint:
        if 0 <= cell[0] < rows and 0 <= cell[1] < cols:
            blocked_grid[cell] = 1

    positive = []
    missions = set()
    robots = set()
    route_capture = 0
    near_capture = 0
    post_clear_usage = 0
    for step, robot_id, state in snapshots:
        start = tuple(state["position"])
        goal = tuple(state["goal"])
        free_path = route(free_grid, start, goal)
        blocked_path = route(blocked_grid, start, goal)
        if free_path is None or blocked_path is None:
            continue
        penalty = max(0, len(blocked_path) - len(free_path))
        if penalty <= 0:
            continue
        positive.append(float(penalty))
        mission_key = state.get("mission_id") or (robot_id, start, goal)
        missions.add(mission_key)
        robots.add(robot_id)
        intersects = bool(footprint.intersection(free_path))
        if intersects:
            route_capture += 1
            if clearance_step is not None and step >= int(clearance_step):
                post_clear_usage += 1
        elif free_path and min(
            min(abs(path_cell[0] - cell[0]) + abs(path_cell[1] - cell[1]) for cell in footprint)
            for path_cell in free_path
        ) <= 1:
            near_capture += 1

    return {
        "recon_stale_reblock_penalty_steps": max(positive, default=0.0),
        "mean_positive_reblock_penalty": float(np.mean(positive)) if positive else 0.0,
        "affected_route_snapshot_count": len(positive),
        "affected_mission_count": len(missions),
        "distinct_robot_count": len(robots),
        "route_capture_count": route_capture,
        "route_near_capture_count": near_capture,
        "post_clear_usage_count": post_clear_usage,
        "reopen_benefit_count": int(bool(post_clear_usage and positive)),
    }


def _recon_victim_relevance(reference_states, benign_ids, cells):
    reference_states = _reference_states(reference_states)
    footprint = {tuple(cell) for cell in cells}
    center = _center_cell(cells)
    options = []
    for victim_id in benign_ids:
        trace = [
            tuple(reference_states[step][victim_id]["position"])
            for step in sorted(reference_states)
            if victim_id in reference_states[step]
        ]
        if not trace:
            continue
        overlap = sum(1 for cell in trace if cell in footprint)
        min_distance = min(manhattan(cell, center) for cell in trace)
        options.append({
            "victim_id": int(victim_id),
            "route_overlap": int(overlap),
            "victim_distance": int(min_distance),
            "route_distance_steps": int(min(range(len(trace)), key=lambda i: manhattan(trace[i], center))),
            "min_route_distance": int(min_distance),
        })
    if not options:
        return None
    options.sort(key=lambda item: (item["route_overlap"], -item["min_route_distance"]), reverse=True)
    return options[0]


def _select_episode_attack_target(
    episodes,
    permanent_obstacles,
    reference_states,
    step,
    attack_type,
    benign_ids,
    grid,
    heatmap,
    preferred_physical_kind=None,
    relevance_cache=None,
    target_use_counts=None,
):
    """Choose a physical target using recon/static knowledge only.

    The attack step is used solely to determine which authored temporary
    obstacle episode is physically active/cleared. Robot state after recon is
    never consulted. Permanent obstacles are eligible only for False Clearance.
    """
    permanent_cells = {cell for obstacle in permanent_obstacles for cell in obstacle.cells}
    active_temp_cells = {
        tuple(cell)
        for episode in episodes
        if episode.appearance_step <= step < episode.clearance_step
        for cell in episode.cells
    }
    targets = []
    if attack_type == AttackType.FALSE_CLEARANCE:
        for obstacle in permanent_obstacles:
            targets.append((obstacle, "permanent", None, None))
        for episode in episodes:
            if not (episode.appearance_step <= step < episode.clearance_step):
                continue
            remaining = episode.clearance_step - step
            if remaining < FALSE_CLEARANCE_MIN_REMAINING_STEPS:
                continue
            targets.append((episode, "temporary", remaining, None))
    else:
        for episode in episodes:
            age = step - episode.clearance_step
            if not (STALE_REASSERTION_PREFERRED_AGE[0] <= age <= STALE_REASSERTION_PREFERRED_AGE[1]):
                continue
            if set(episode.cells).intersection(active_temp_cells | permanent_cells):
                continue
            targets.append((episode, "temporary", None, age))

    candidates = []
    cache = relevance_cache if relevance_cache is not None else {}
    for target, kind, remaining, age in targets:
        cells = tuple(target.cells)
        target_id = (
            getattr(target, "episode_id", None)
            or getattr(target, "obstacle_id", None)
            or repr(cells)
        )
        if (
            attack_type == AttackType.STALE_REASSERTION
            and target_use_counts is not None
            and target_use_counts.get(str(target_id), 0) > 0
        ):
            continue
        # False Clearance and Stale Reassertion use different mission-level
        # relevance semantics for the same temporary episode. Keep their
        # cached scores separate so an episode scored while physically active
        # cannot later leak that attack-specific score into stale selection.
        cache_key = (attack_type.value, kind, str(target_id), cells)
        static_relevance = cache.get(cache_key, ...)
        if static_relevance is ...:
            victim = _recon_victim_relevance(reference_states, benign_ids, cells)
            stale_features = (
                _recon_reblock_features(
                    reference_states,
                    benign_ids,
                    grid,
                    cells,
                    clearance_step=getattr(target, "clearance_step", None),
                )
                if attack_type == AttackType.STALE_REASSERTION
                else {}
            )
            if victim is None:
                if attack_type != AttackType.STALE_REASSERTION or not stale_features.get("affected_mission_count"):
                    cache[cache_key] = None
                    continue
                first = next(
                    (
                        state
                        for state_step in sorted(_reference_states(reference_states))
                        for state in _reference_states(reference_states)[state_step].values()
                        if state.get("goal") is not None
                    ),
                    None,
                )
                if first is None:
                    cache[cache_key] = None
                    continue
                victim = {
                    "victim_id": int(next(iter(benign_ids), 0)),
                    "route_overlap": 0,
                    "victim_distance": 0,
                    "route_distance_steps": 0,
                    "min_route_distance": 0,
                }
            traffic = sum(
                float(heatmap[cell])
                for cell in cells
                if 0 <= cell[0] < heatmap.shape[0] and 0 <= cell[1] < heatmap.shape[1]
            )
            nearby_traffic = _nearby_recon_traffic_score(heatmap, cells, radius=2)
            detour_grid = np.array(grid, dtype=np.uint8, copy=True)
            # Other permanent obstacles remain physical while the target under
            # evaluation is treated as the candidate block itself.
            for obstacle in permanent_obstacles:
                if obstacle is target:
                    continue
                for cell in obstacle.cells:
                    detour_grid[cell] = 1
            if attack_type == AttackType.STALE_REASSERTION:
                stale_features = _recon_reblock_features(
                    reference_states,
                    benign_ids,
                    detour_grid,
                    cells,
                    clearance_step=getattr(target, "clearance_step", None),
                )
            detour = footprint_finite_detour_score(detour_grid, cells)
            if detour <= 0.0:
                cache[cache_key] = None
                continue
            bottleneck = _footprint_bottleneck_score(detour_grid, cells)
            shortcut = _recon_false_clearance_shortcut_score(
                reference_states, benign_ids, detour_grid, cells
            ) if attack_type == AttackType.FALSE_CLEARANCE else 0.0
            if attack_type == AttackType.FALSE_CLEARANCE and shortcut <= 0.0:
                # A False Clearance event is only eligible when frozen
                # reconnaissance shows that clearing this footprint shortens
                # at least one observed mission.  Traffic/proximity alone is
                # not enough to call the event navigation-relevant.
                cache[cache_key] = None
                continue
            reblock_penalty = (
                float(stale_features.get("recon_stale_reblock_penalty_steps", 0.0))
                if attack_type == AttackType.STALE_REASSERTION
                else 0.0
            )
            visibility_delay = _historical_visibility_delay(
                reference_states, victim["victim_id"], cells, 1, 40
            )
            # False Clearance is useful only when the obstacle sits beside a
            # route robots actually used during reconnaissance. Because the
            # footprint itself is physically blocked, on-footprint traffic is
            # usually zero; surrounding corridor traffic and route proximity
            # are therefore the primary relevance signals.
            proximity = 1.0 / (1.0 + float(victim["min_route_distance"]))
            route_overlap = float(victim["route_overlap"])
            score = (nearby_traffic + traffic + 1.0)
            # Direct overlap with a reconnaissance-derived route is the
            # strongest fair signal that falsely clearing this obstacle can
            # expose an attractive shortcut. Proximity is a softer fallback
            # for physical footprints just beside the observed corridor.
            score *= 1.0 + 2.0 * route_overlap
            score *= 1.0 + 3.0 * bottleneck
            score *= 1.0 + 2.0 * detour
            score *= 1.0 + 4.0 * proximity
            # A positive observed-mission gain is the strongest fair signal.
            # False Clearance prefers blocks whose removal exposes a shortcut;
            # Stale Reassertion prefers cleared footprints whose reblocking
            # would make the same recon-observed missions longer again.
            mission_gain = shortcut if attack_type == AttackType.FALSE_CLEARANCE else reblock_penalty
            score *= 1.0 + 4.0 * mission_gain
            static_relevance = {
                **victim,
                "target_visible_to_victim": None,
                "recon_visibility_delay_estimate": visibility_delay,
                "traffic_score": traffic,
                "nearby_traffic_score": nearby_traffic,
                "route_proximity_score": proximity,
                "bottleneck_score": bottleneck,
                "reference_detour_score": detour,
                "recon_false_clearance_shortcut_steps": shortcut,
                "recon_stale_reblock_penalty_steps": reblock_penalty,
                "physical_obstacle_kind": kind,
                "score": score,
            }
            if stale_features:
                static_relevance.update(stale_features)
            cache[cache_key] = static_relevance
        if static_relevance is None:
            continue
        relevance = dict(static_relevance)
        score = float(relevance.pop("score"))
        if attack_type == AttackType.STALE_REASSERTION and age is not None:
            # Recently cleared obstacles retain more credible historical weight
            # and are more likely to sit on a route that has just reopened.
            lo, hi = STALE_REASSERTION_PREFERRED_AGE
            freshness = 1.0 + 2.0 * max(0.0, 1.0 - (age - lo) / max(1.0, hi - lo))
            score *= freshness
        relevance["remaining_obstacle_lifetime"] = remaining
        relevance["age_since_clearance"] = age
        if attack_type == AttackType.STALE_REASSERTION:
            positive = float(relevance.get("recon_stale_reblock_penalty_steps", 0.0)) > 0.0
            if relevance.get("reopen_benefit_count", 0) and positive:
                tier = 4
            elif relevance.get("route_capture_count", 0) and positive:
                tier = 3
            elif positive:
                tier = 2
            else:
                tier = 1
            rank_key = (
                tier,
                int(relevance.get("affected_mission_count", 0)),
                int(relevance.get("route_capture_count", 0)),
                int(relevance.get("distinct_robot_count", 0)),
                float(relevance.get("mean_positive_reblock_penalty", 0.0)),
                float(relevance.get("recon_stale_reblock_penalty_steps", 0.0)),
                int(relevance.get("post_clear_usage_count", 0)),
                score,
            )
        else:
            # Consequence is the primary ranking signal.  Physical-class
            # diversity is only a soft tie-break, so an unused permanent or
            # temporary class cannot outrank a materially larger shortcut.
            rank_key = (
                float(relevance.get("recon_false_clearance_shortcut_steps", 0.0)),
                int(
                    preferred_physical_kind is not None
                    and relevance.get("physical_obstacle_kind") == preferred_physical_kind
                ),
                score,
            )
        candidates.append((rank_key, target, relevance))
    if not candidates:
        return None
    # For stale targets, a finite positive recon-derived reblocking penalty is
    # categorically more useful than a zero-penalty footprint. Do not consume
    # an event on a zero-consequence footprint while a positive candidate is
    # available; if none exists, leave the slot unused rather than authoring a
    # stale report that cannot affect any recon-observed mission.
    if attack_type == AttackType.FALSE_CLEARANCE:
        positive_candidates = [
            item for item in candidates
            if float(item[2].get("recon_false_clearance_shortcut_steps", 0.0)) > 0.0
        ]
        if not positive_candidates:
            return None
        # Prefer a distinct physical target when one exists, but allow reuse
        # after the positive-consequence candidate pool is exhausted. This
        # preserves a fixed attack schedule without silently spending events
        # on targets that have no frozen-recon navigation consequence.
        if target_use_counts is not None:
            unused_candidates = [
                item for item in positive_candidates
                if target_use_counts.get(
                    str(getattr(item[1], "episode_id", None)
                        or getattr(item[1], "obstacle_id", None)),
                    0,
                ) == 0
            ]
            candidates = unused_candidates or positive_candidates
        else:
            candidates = positive_candidates
    elif attack_type == AttackType.STALE_REASSERTION:
        positive_candidates = [
            item for item in candidates
            if float(item[2].get("recon_stale_reblock_penalty_steps", 0.0)) > 0.0
        ]
        if not positive_candidates:
            return None
        candidates = positive_candidates
    # Candidate targets are dataclasses without an ordering relation.  Use
    # their stable authored ID as the deterministic tie-breaker instead of
    # asking Python to compare TemporaryObstacleEpisode instances.
    candidates.sort(
        key=lambda item: (
            item[0],
            getattr(item[1], "episode_id", getattr(item[1], "obstacle_id", "")),
        ),
        reverse=True,
    )
    _, target, relevance = candidates[0]
    return target, relevance


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
    visible_cells_by_robot=None,
    future_visibility_delay_fn=None,
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
            victim_options = []
            visible_cells_by_robot = visible_cells_by_robot or {}
            for victim in robots:
                if victim.robot_id == 0:
                    continue
                visible = {tuple(item) for item in visible_cells_by_robot.get(victim.robot_id, ())}
                if visible.intersection(report_cells):
                    continue
                visibility_delay = None
                if future_visibility_delay_fn is not None:
                    visibility_delay = future_visibility_delay_fn(victim.robot_id, report_cells)
                    if visibility_delay is None:
                        continue
                remaining = list(victim.path or ())
                if not remaining:
                    continue
                overlap = len(set(remaining).intersection(report_cells))
                nearest_index = min(
                    range(len(remaining)),
                    key=lambda index: min(manhattan(remaining[index], cell) for cell in report_cells),
                )
                min_distance = min(
                    manhattan(report_cell, path_cell)
                    for report_cell in report_cells
                    for path_cell in remaining
                )
                reference_detour = _reference_detour_score(grid, victim, report_cells, active_temp_cells)
                local_detour = footprint_finite_detour_score(grid, report_cells)
                detour = max(local_detour, 0.0 if reference_detour is None else reference_detour)
                if detour <= 0:
                    continue
                proximity = (10.0 + overlap) if overlap else 1.0 / (1.0 + min_distance)
                victim_options.append({
                    "victim_id": victim.robot_id,
                    "path_overlap": overlap,
                    "victim_distance": manhattan(tuple(victim.position), cell),
                    "path_proximity_score": proximity,
                    "route_distance_steps": nearest_index,
                    "reference_detour_score": detour,
                    "first_visibility_delay": visibility_delay,
                })
            if not victim_options:
                continue
            victim_options.sort(
                key=lambda item: (
                    item["reference_detour_score"],
                    item["path_overlap"],
                    item["path_proximity_score"],
                    -item["route_distance_steps"],
                ),
                reverse=True,
            )
            victim = victim_options[0]
            if require_route_overlap and victim["path_overlap"] <= 0:
                continue
            candidates.append(
                {
                    "center_cell": cell,
                    "report_cells": report_cells,
                    "traffic_score": _fake_average_traffic_score(cell, heatmap, height, width),
                    "footprint_height": height,
                    "footprint_width": width,
                    "report_cell_count": len(report_cells),
                    "path_overlap": victim["path_overlap"],
                    "path_proximity_score": victim["path_proximity_score"],
                    "affected_victims": len(victim_options),
                    "victim_id": victim["victim_id"],
                    "victim_distance": victim["victim_distance"],
                    "route_distance_steps": victim["route_distance_steps"],
                    "reference_detour_score": victim["reference_detour_score"],
                    "target_visible_to_victim": False,
                    "first_visibility_delay": victim["first_visibility_delay"],
                    "bottleneck_score": _footprint_bottleneck_score(grid, report_cells),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["reference_detour_score"],
            item["path_overlap"],
            item["bottleneck_score"],
            item["path_proximity_score"],
            -item["route_distance_steps"],
            item["report_cell_count"],
            item["traffic_score"],
        ),
        reverse=True,
    )
    return candidates[:ATTACK_CANDIDATE_LIMIT]


def _reference_states(reference) -> dict:
    """Return the legacy state mapping from either a frozen dataset or a mapping."""
    if isinstance(reference, FrozenReconData):
        return reference.states()
    return reference or {}


def freeze_recon_data(config: SimulationConfig, manifest: ScenarioManifest, log: dict) -> FrozenReconData:
    """Convert one clean rollout prefix into immutable authoring data.

    The returned object contains no live robot/world references.  In particular,
    nothing after ``recon_steps`` can be appended to it by a replay.
    """
    recon_steps = int(manifest.phase_boundaries.get("reconnaissance_end", config.phases.recon_steps))
    states = log.get("reference_states") or {}
    heatmap = np.zeros(manifest.map_shape, dtype=np.int32)
    benign = set(manifest.benign_robot_ids)
    mission_snapshots = []
    route_snapshots = []
    cell_usage: dict[tuple[int, int], int] = {}
    visibility_counts: dict[int, int] = {}
    last_paths: dict[int, tuple[tuple[int, int], ...]] = {}

    for step in sorted(int(value) for value in states if int(value) < recon_steps):
        for robot_id in sorted(benign):
            state = states.get(step, states.get(str(step), {})).get(robot_id)
            if state is None:
                state = states.get(step, states.get(str(step), {})).get(str(robot_id))
            if state is None:
                continue
            position = tuple(state["position"])
            goal = tuple(state["goal"]) if state.get("goal") is not None else None
            path = tuple(tuple(cell) for cell in state.get("path", ()))
            visible = tuple(sorted(tuple(cell) for cell in state.get("visible_cells", ())))
            mission_id = state.get("mission_id")
            heatmap[position] += 1
            visibility_counts[robot_id] = visibility_counts.get(robot_id, 0) + len(visible)
            mission_snapshots.append(
                ReconMissionSnapshot(step, robot_id, mission_id, position, goal, path, visible)
            )
            if goal is not None and path:
                # Sampling every five steps keeps serialized manifests compact;
                # route changes are always retained even between samples.
                if step % 5 == 0 or path != last_paths.get(robot_id):
                    route_snapshots.append(
                        ReconRouteSnapshot(step, robot_id, mission_id, position, goal, path)
                    )
                    for cell in path:
                        cell_usage[cell] = cell_usage.get(cell, 0) + 1
                last_paths[robot_id] = path

    observed_obstacles = tuple(
        (
            episode.episode_id,
            int(episode.appearance_step),
            int(episode.clearance_step),
            tuple(tuple(cell) for cell in episode.cells),
        )
        for episode in manifest.obstacle_episodes
        if episode.appearance_step < recon_steps or episode.clearance_step <= recon_steps
    )
    return FrozenReconData(
        recon_steps,
        tuple(tuple(int(value) for value in row) for row in heatmap.tolist()),
        tuple(mission_snapshots),
        tuple(route_snapshots),
        tuple(sorted((cell, count) for cell, count in cell_usage.items())),
        (),
        observed_obstacles,
        tuple(sorted(visibility_counts.items())),
        manifest.map_hash,
        f"recon-{manifest.map_hash[:12]}-seed{manifest.master_seed}",
        int(manifest.master_seed),
    )


def _run_reconnaissance_rollout(config: SimulationConfig, manifest: ScenarioManifest):
    """Run the shared prefix while retaining live objects for compatibility."""
    reference_config = replace(
        config,
        max_steps=config.phases.recon_steps,
        visualization=replace(config.visualization, animation=False),
    )
    # A caller may hand us a full manifest; recon is nevertheless always clean.
    reference_manifest = replace(manifest, attack_events=(), report_audit_labels=())
    # The internal method name is retained for compatibility with the native
    # planner, but this is a neutral authoring pass: peer sharing is disabled
    # and the malicious robot is virtual, so no defense-specific trust/fusion
    # behavior or attacker traffic can affect frozen candidate placement.
    _, robots, log = run_manifest_rollout(
        reference_config,
        reference_manifest,
        "full_trust",
        show_progress=False,
        capture_reference_state=True,
        neutral_recon=True,
        virtual_attacker_recon=True,
    )
    frozen = freeze_recon_data(config, reference_manifest, log)
    return frozen, robots, log


def run_reconnaissance(config: SimulationConfig, manifest: ScenarioManifest) -> FrozenReconData:
    """Run and freeze the one clean reconnaissance process for any map."""
    frozen, _, _ = _run_reconnaissance_rollout(config, manifest)
    return frozen


def run_clean_reference_rollout(config: SimulationConfig, manifest: ScenarioManifest):
    """Compatibility wrapper returning the historical heatmap/robots/log tuple."""
    frozen, robots, log = _run_reconnaissance_rollout(config, manifest)
    return np.asarray(frozen.traffic_heatmap, dtype=np.int32), robots, log


# Compatibility name retained for callers; behavior is reconnaissance-only.
run_clean_recon_rollout = run_clean_reference_rollout


def save_traffic_heatmap_artifacts(root: Path, heatmap: np.ndarray, *, title: str = "Attack-free reference traffic heatmap") -> None:
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
    """Author the default warehouse scenario from frozen reconnaissance only."""
    from .map_io import (
        WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS,
        WAREHOUSE_NARROW_CORRIDOR_CELLS,
    )
    from .scenario import _active_temp_cells

    config.validate()
    grid = np.asarray(default_warehouse_map() if grid is None else grid, dtype=np.uint8)
    starts, goals, task_queues = build_warehouse_layout(
        grid,
        config.deliveries_per_robot,
        seed=config.seed,
    )
    sender = 0
    benign = tuple(robot_id for robot_id in starts if robot_id != sender)
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
    placement_forbidden = protected_cells | set(WAREHOUSE_NARROW_CORRIDOR_CELLS)
    # The warehouse intentionally contains multiple operating regions, so all
    # task points are not globally connected.  Preserve the narrow attacker-bay
    # corridor as one explicit group, and preserve each robot's own mission
    # graph (start + every pickup/dropoff) independently.  This prevents a
    # physical footprint from cutting a robot off from one of its required task
    # regions without incorrectly requiring unrelated warehouse regions to be
    # mutually reachable.
    required_anchors = tuple(WAREHOUSE_CORRIDOR_CONNECTIVITY_ANCHORS)
    required_anchor_groups = tuple(
        tuple(sorted(
            {tuple(starts[robot_id])}
            | {
                tuple(cell)
                for task in task_queues.get(robot_id, ())
                for cell in (task.pickup, task.dropoff)
            }
        ))
        for robot_id in sorted(starts)
    )

    phase_boundaries = {
        "reconnaissance_end": config.phases.recon_steps,
        "attack_end": config.phases.recon_steps + config.phases.attack_steps,
        "total": config.phases.total_steps,
    }
    seed_names = (
        "attack_scheduler", "attack_types", "attack_placement",
        "permanent_obstacles", "temporary_obstacles", "robot_routes",
        "traffic", "warehouse_manifest_scheduler",
    )
    derived = {name: derived_seed(config.seed, name) for name in seed_names}
    static_grid = tuple(tuple(int(value) for value in row) for row in grid)

    # Bootstrap reconnaissance is attack-free and physical-obstacle-free. It
    # supplies only a seed-stable traffic prior for placing the physical world.
    # A second reconnaissance-only rollout against that authored world becomes
    # the final frozen attack-authoring reference.
    bootstrap = ScenarioManifest(
        SCHEMA_VERSION, config.seed, derived, _hash(grid), tuple(grid.shape),
        static_grid, phase_boundaries, sender, benign, (), (),
        scenario_id=f"warehouse-{config.seed}-bootstrap",
        protocol_id="modular_v1",
        robot_starts=starts,
        task_queues=task_queues,
    )
    bootstrap_heatmap, _, _ = run_clean_reference_rollout(config, bootstrap)

    permanent_obstacles = author_permanent_obstacles(
        grid,
        named_rng(config.seed, "permanent_obstacles"),
        count=3,
        forbidden_cells=placement_forbidden,
        required_anchors=required_anchors,
        required_anchor_groups=required_anchor_groups,
        traffic_heatmap=bootstrap_heatmap,
    )
    if len(permanent_obstacles) != 3:
        raise ValueError(
            f"default warehouse authoring could place only {len(permanent_obstacles)} of 3 permanent obstacles"
        )
    permanent_cells = {
        tuple(cell) for obstacle in permanent_obstacles for cell in obstacle.cells
    }
    physical_base_grid = np.array(grid, dtype=np.uint8, copy=True)
    for cell in permanent_cells:
        physical_base_grid[cell] = 1

    episodes = author_temporary_obstacle_episodes(
        physical_base_grid,
        named_rng(config.seed, "temporary_obstacles"),
        config.phases.total_steps,
        config.temporary_blockage_change_period_steps,
        forbidden_cells=placement_forbidden | permanent_cells,
        active_count=6,
        required_anchors=required_anchors,
        required_anchor_groups=required_anchor_groups,
        traffic_heatmap=bootstrap_heatmap,
    )

    reference_manifest = ScenarioManifest(
        SCHEMA_VERSION, config.seed, derived, _hash(grid), tuple(grid.shape),
        static_grid, phase_boundaries, sender, benign, episodes, (),
        scenario_id=f"warehouse-{config.seed}-recon-reference",
        protocol_id="modular_v1",
        robot_starts=starts,
        task_queues=task_queues,
        permanent_obstacles=permanent_obstacles,
    )
    frozen_recon = run_reconnaissance(config, reference_manifest)
    heatmap = np.asarray(frozen_recon.traffic_heatmap, dtype=np.int32)
    reference_states = frozen_recon.states()
    if reference_states and max(reference_states) >= config.phases.recon_steps:
        raise AssertionError("post-recon reference state leaked into manifest authoring")

    # Build fixed victim proxies from positions actually observed during recon.
    # No path/goal/visibility from step >= recon_end exists in this structure.
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

    candidate_forbidden = placement_forbidden | permanent_cells
    place_rng = named_rng(config.seed, "attack_placement")
    base_candidates = recon_heatmap_attack_candidates(
        physical_base_grid,
        goals,
        recon_robots,
        heatmap,
        rng=place_rng,
        require_route_overlap=True,
        forbidden_cells=candidate_forbidden,
        active_temp_cells=(),
        visible_cells_by_robot={},
        future_visibility_delay_fn=recon_visibility_delay,
    )
    warnings = []
    if AttackType.FAKE_OBSTACLE in {AttackType(value) for value in config.attacks.enabled} and not base_candidates:
        # Keep route/traffic/detour relevance even when no 15..40-step historical
        # verification pattern exists in the finite reconnaissance trace.
        warnings.append("recon_visibility_window_relaxed_no_candidate")
        base_candidates = recon_heatmap_attack_candidates(
            physical_base_grid,
            goals,
            recon_robots,
            heatmap,
            rng=named_rng(config.seed, "attack_placement_visibility_fallback"),
            require_route_overlap=True,
            forbidden_cells=candidate_forbidden,
            active_temp_cells=(),
            visible_cells_by_robot={},
            future_visibility_delay_fn=None,
        )

    enabled_types = [AttackType(value) for value in config.attacks.enabled]
    rng = named_rng(config.seed, "warehouse_manifest_scheduler")
    events = []
    metadata = []
    uses: dict[tuple[int, int], int] = {}
    selected_centers = []
    step = config.phases.recon_steps + rng.randint(
        config.attacks.interval_min, config.attacks.interval_max
    )
    index = 0
    attack_end = config.phases.recon_steps + config.phases.attack_steps
    false_clearance_kind_uses = {"permanent": 0, "temporary": 0}
    false_clearance_target_uses = {}
    stale_reassertion_target_uses = {}
    physical_target_relevance_cache = {}

    while step < attack_end and enabled_types:
        # False Clearance is intentionally diverse across the two physical
        # truth sources added for this experiment. Prefer the currently
        # underused kind when it has an eligible recon/static candidate; this
        # is seed-stable and defense-independent. Stale Reassertion remains
        # tied only to cleared temporary episodes.
        # Prefer coverage of both physical target classes when feasible, but
        # let frozen-recon consequence remain the primary target-ranking signal.
        # This keeps False Clearance consequential without using
        # defense-specific live routes.
        unused_false_clearance_kinds = [
            kind for kind, count in false_clearance_kind_uses.items() if count == 0
        ]
        preferred_false_clearance_kind = (
            min(unused_false_clearance_kinds) if unused_false_clearance_kinds else None
        )
        episode_targets = {
            kind: _select_episode_attack_target(
                episodes,
                permanent_obstacles,
                reference_states,
                step,
                kind,
                benign,
                grid,
                heatmap,
                preferred_physical_kind=(
                    preferred_false_clearance_kind
                    if kind == AttackType.FALSE_CLEARANCE
                    else None
                ),
                relevance_cache=physical_target_relevance_cache,
                target_use_counts=(
                    false_clearance_target_uses
                    if kind == AttackType.FALSE_CLEARANCE
                    else stale_reassertion_target_uses
                    if kind == AttackType.STALE_REASSERTION
                    else None
                ),
            )
            for kind in (AttackType.FALSE_CLEARANCE, AttackType.STALE_REASSERTION)
        }
        active_temp = _active_temp_cells(episodes, step)
        step_candidates = [
            candidate for candidate in base_candidates
            if not set(map(tuple, candidate["report_cells"])).intersection(active_temp)
        ]
        feasible = [
            kind for kind in enabled_types
            if (kind == AttackType.FAKE_OBSTACLE and bool(step_candidates))
            or (kind != AttackType.FAKE_OBSTACLE and episode_targets.get(kind) is not None)
        ]
        if not feasible:
            step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
            continue
        selected_attack = feasible[rng.randrange(len(feasible))]

        if selected_attack != AttackType.FAKE_OBSTACLE:
            target = episode_targets[selected_attack]
            if target is None:
                step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
                continue
            physical_target, relevance = target
            cells = tuple(physical_target.cells)
            target_id = (
                getattr(physical_target, "episode_id", None)
                or getattr(physical_target, "obstacle_id", None)
            )
            claim = (
                ClaimType.FREE
                if selected_attack == AttackType.FALSE_CLEARANCE
                else ClaimType.BLOCKED
            )
            observation_step = (
                step
                if selected_attack == AttackType.FALSE_CLEARANCE
                else max(0, physical_target.clearance_step - 1)
            )
            event_id = f"attack-{index:04}"
            events.append(AttackEvent(
                event_id, step, selected_attack, cells, claim, observation_step,
                sender, benign,
                tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells))),
                target_id,
            ))
            if selected_attack == AttackType.FALSE_CLEARANCE:
                false_clearance_kind_uses[relevance["physical_obstacle_kind"]] += 1
                target_key = str(target_id)
                prior_use_count = false_clearance_target_uses.get(target_key, 0)
                false_clearance_target_uses[target_key] = prior_use_count + 1
                if prior_use_count:
                    warnings.append("false_clearance_target_reused_after_positive_candidates_exhausted")
            elif selected_attack == AttackType.STALE_REASSERTION:
                stale_key = str(target_id)
                stale_reassertion_target_uses[stale_key] = stale_reassertion_target_uses.get(stale_key, 0) + 1
            metadata.append({
                "candidate_id": event_id,
                "event_id": event_id,
                "attack_type": selected_attack.value,
                "center": _center_cell(cells),
                "footprint_cells": cells,
                "physical_obstacle_kind": relevance["physical_obstacle_kind"],
                "intended_victim_id": relevance["victim_id"],
                "reference_step": config.phases.recon_steps - 1,
                "observation_step": observation_step,
                "route_overlap": relevance["route_overlap"],
                "victim_distance": relevance["victim_distance"],
                "reference_route_distance_steps": relevance["route_distance_steps"],
                "min_route_distance": relevance["min_route_distance"],
                "remaining_obstacle_lifetime": relevance["remaining_obstacle_lifetime"],
                "age_since_clearance": relevance["age_since_clearance"],
                "target_visible_to_victim": None,
                "recon_visibility_delay_estimate": relevance["recon_visibility_delay_estimate"],
                "traffic_score": relevance["traffic_score"],
                "nearby_traffic_score": relevance.get("nearby_traffic_score"),
                "route_proximity_score": relevance.get("route_proximity_score"),
                "bottleneck_score": relevance["bottleneck_score"],
                "reference_detour_score": relevance["reference_detour_score"],
                "recon_false_clearance_shortcut_steps": relevance.get("recon_false_clearance_shortcut_steps"),
                "recon_stale_reblock_penalty_steps": relevance.get("recon_stale_reblock_penalty_steps"),
                "prior_use_count": prior_use_count if selected_attack == AttackType.FALSE_CLEARANCE else None,
                "target_reused": bool(prior_use_count) if selected_attack == AttackType.FALSE_CLEARANCE else False,
                "heatmap_reference_steps": config.phases.recon_steps,
                "selection_basis": "frozen_reconnaissance_traffic_plus_static_geometry_and_authored_physical_schedule",
            })
            index += 1
            step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
            continue

        pool = step_candidates[: config.attacks.candidate_top_k]
        used_unique = set(uses)
        require_new_center = len(used_unique) < config.attacks.min_unique_footprints
        eligible = [
            candidate
            for candidate in pool
            if uses.get(tuple(candidate["center_cell"]), 0) < config.attacks.max_uses_per_footprint
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
                candidate for candidate in pool
                if uses.get(tuple(candidate["center_cell"]), 0) < config.attacks.max_uses_per_footprint
            ]
        if not eligible:
            warnings.append("recon_attack_candidate_unavailable")
            step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)
            continue
        weights = list(range(len(eligible), 0, -1))
        candidate = rng.choices(eligible, weights=weights, k=1)[0]
        cells = tuple(tuple(cell) for cell in candidate["report_cells"])
        center = tuple(candidate["center_cell"])
        event_id = f"attack-{index:04}"
        events.append(AttackEvent(
            event_id, step, AttackType.FAKE_OBSTACLE, cells, ClaimType.BLOCKED,
            step, sender, benign,
            tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells))),
        ))
        metadata.append({
            "candidate_id": f"warehouse-{index:04}",
            "event_id": event_id,
            "attack_type": AttackType.FAKE_OBSTACLE.value,
            "center": center,
            "footprint_cells": cells,
            "footprint_height": candidate.get("footprint_height"),
            "footprint_width": candidate.get("footprint_width"),
            "route_overlap": candidate["path_overlap"],
            "victim_distance": candidate.get("victim_distance"),
            "intended_victim_id": candidate.get("victim_id"),
            "reference_step": config.phases.recon_steps - 1,
            "reference_route_distance_steps": candidate.get("route_distance_steps"),
            "reference_detour_score": candidate.get("reference_detour_score"),
            "target_visible_to_victim": False if candidate.get("first_visibility_delay") is not None else None,
            "remaining_obstacle_lifetime": None,
            "age_since_clearance": None,
            "first_visibility_delay": candidate.get("first_visibility_delay"),
            "visibility_delay_window": [
                config.attacks.visibility_delay_min,
                config.attacks.visibility_delay_max,
            ],
            "heatmap_reference_steps": config.phases.recon_steps,
            "traffic_score": candidate["traffic_score"],
            "bottleneck_score": candidate["bottleneck_score"],
            "estimated_detour_score": candidate["path_proximity_score"],
            "rank": step_candidates.index(candidate) + 1,
            "selection_weight": 1 / len(eligible),
            "prior_use_count": uses.get(center, 0),
            "selection_basis": "frozen_reconnaissance_traffic_and_static_geometry",
        })
        uses[center] = uses.get(center, 0) + 1
        selected_centers.append(center)
        index += 1
        step += rng.randint(config.attacks.interval_min, config.attacks.interval_max)

    if AttackType.FAKE_OBSTACLE in enabled_types and len(set(selected_centers)) < min(
        config.attacks.min_unique_footprints, len(selected_centers)
    ):
        warnings.append("concentrated_attack_manifest: minimum unique footprint count not met")

    labels = tuple(
        ReportAuditLabel(
            report_id,
            True,
            event.attack_type,
            event.obstacle_episode_id,
            ClaimType.BLOCKED
            if event.attack_type in {AttackType.FALSE_CLEARANCE, AttackType.STALE_REASSERTION}
            else ClaimType.FREE,
        )
        for event in events
        for report_id in event.report_ids
    )
    attacker_route = _build_attacker_route_cells(
        physical_base_grid,
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
            if (
                attacker_positions[step] in permanent_cells
                or any(
                    episode.appearance_step <= step < episode.clearance_step
                    and attacker_positions[step] in episode.cells
                    for episode in episodes
                )
            )
            else ClaimType.FREE,
            step,
            sensor_confidence=1.0,
        )
        for step in range(0, config.phases.total_steps, config.communication_period_steps)
    )
    return ScenarioManifest(
        SCHEMA_VERSION,
        config.seed,
        derived,
        _hash(grid),
        tuple(grid.shape),
        static_grid,
        phase_boundaries,
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
        reconnaissance_heatmap=tuple(
            tuple(int(value) for value in row) for row in heatmap.tolist()
        ),
        permanent_obstacles=permanent_obstacles,
        reconnaissance_data=frozen_recon,
    )
