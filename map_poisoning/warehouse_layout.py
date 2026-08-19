"""Warehouse robot layout and delivery-task generation (sim2 parity)."""
from __future__ import annotations

import math

from .map_io import (
    WAREHOUSE_ATTACKER_START,
    WAREHOUSE_CORRIDOR_DELIVERY,
    WAREHOUSE_EXCLUDED_ACTION_POINTS,
    WAREHOUSE_NARROW_CORRIDOR_CELLS,
)
from .models import DeliveryTask
from .planning import astar

DEFAULT_NUM_ROBOTS = 3
DEFAULT_NUM_ACTION_POINTS = 14
ACTION_POINT_EDGE_WEIGHT = 2.0
ACTION_POINT_CORNER_WEIGHT = 3.0
ACTION_POINT_OBSTACLE_PROXIMITY_WEIGHT = 1.4
ACTION_POINT_SPREAD_WEIGHT = 2.5
ACTION_POINT_MIN_SPACING_RATIO = 0.18
ACTION_POINT_OBSTACLE_RADIUS = 4
ACTION_POINT_CORNER_RADIUS = 2
ACTION_POINT_EDGE_MARGIN_CELLS = 1


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    return [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]


def _in_bounds(grid, cell: tuple[int, int]) -> bool:
    row, col = cell
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def _is_free(grid, cell: tuple[int, int]) -> bool:
    return _in_bounds(grid, cell) and not grid[cell]


def find_free_cells(grid) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    return [(row, col) for row in range(rows) for col in range(cols) if not grid[row, col]]


def route_exists(grid, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    return astar(start, goal, lambda cell: float("inf") if not _is_free(grid, cell) else 1.0) is not None


def nearest_enterable_cell(grid, preferred: tuple[int, int], forbidden=None) -> tuple[int, int]:
    forbidden = set(forbidden or ())
    if _is_free(grid, preferred) and preferred not in forbidden:
        return preferred
    rows, cols = grid.shape
    frontier = [preferred]
    seen = {preferred}
    while frontier:
        cell = frontier.pop(0)
        for neighbor in _neighbors(cell):
            if neighbor in seen or neighbor in forbidden:
                continue
            seen.add(neighbor)
            if _is_free(grid, neighbor):
                return neighbor
            frontier.append(neighbor)
    raise ValueError(f"no enterable cell near {preferred}")


def nearest_safe_start_cell(grid, preferred: tuple[int, int], forbidden=None) -> tuple[int, int]:
    return nearest_enterable_cell(grid, preferred, forbidden)


def edge_score(grid, cell: tuple[int, int]) -> float:
    """Higher near map edges, lower in the open center. Matches main sim2."""
    row, col = cell
    rows, cols = grid.shape
    distance_to_edge = min(row, col, rows - 1 - row, cols - 1 - col)
    max_possible = max(1, min(rows, cols) // 2)
    return 1.0 - min(1.0, distance_to_edge / max_possible)


def count_blocked_nearby(grid, cell: tuple[int, int], radius: int) -> int:
    row, col = cell
    total = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) > radius or (dr == 0 and dc == 0):
                continue
            probe = (row + dr, col + dc)
            if not _in_bounds(grid, probe) or grid[probe]:
                total += 1
    return total


def corner_pressure_score(grid, cell: tuple[int, int], radius: int = ACTION_POINT_CORNER_RADIUS) -> float:
    """Shelf-end / alcove score: blocked structure in both row and column."""
    row, col = cell
    vertical = 0
    horizontal = 0
    for offset in range(1, radius + 1):
        up, down = (row - offset, col), (row + offset, col)
        left, right = (row, col - offset), (row, col + offset)
        if not _in_bounds(grid, up) or grid[up]:
            vertical += 1
        if not _in_bounds(grid, down) or grid[down]:
            vertical += 1
        if not _in_bounds(grid, left) or grid[left]:
            horizontal += 1
        if not _in_bounds(grid, right) or grid[right]:
            horizontal += 1
    return float(min(vertical, horizontal))


def is_good_action_point_candidate(grid, cell: tuple[int, int], forbidden=None) -> bool:
    forbidden = set(forbidden or ())
    if cell in forbidden or not _is_free(grid, cell):
        return False
    margin = ACTION_POINT_EDGE_MARGIN_CELLS
    row, col = cell
    if row < margin or col < margin or row >= grid.shape[0] - margin or col >= grid.shape[1] - margin:
        return False
    return True


def choose_strategic_action_points(grid, count: int, forbidden=None) -> list[tuple[int, int]]:
    forbidden = set(forbidden or ())
    candidates = []
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if not is_good_action_point_candidate(grid, cell, forbidden):
                continue
            score = (
                ACTION_POINT_EDGE_WEIGHT * edge_score(grid, cell)
                + ACTION_POINT_CORNER_WEIGHT * corner_pressure_score(grid, cell)
                + ACTION_POINT_OBSTACLE_PROXIMITY_WEIGHT
                * count_blocked_nearby(grid, cell, ACTION_POINT_OBSTACLE_RADIUS)
            )
            candidates.append((score, cell))
    if not candidates:
        raise ValueError("No valid strategic action-point candidates found.")
    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    used = set(forbidden)
    min_spacing = max(2, int(min(rows, cols) * ACTION_POINT_MIN_SPACING_RATIO))
    while candidates and len(selected) < count:
        best_index = None
        best_total = -math.inf
        for index, (base_score, cell) in enumerate(candidates):
            if cell in used:
                continue
            nearest = min((manhattan(cell, other) for other in selected), default=min(rows, cols))
            if nearest < min_spacing:
                continue
            total = base_score + ACTION_POINT_SPREAD_WEIGHT * nearest
            if total > best_total:
                best_total = total
                best_index = index
        if best_index is None:
            min_spacing = max(1, min_spacing - 1)
            if min_spacing <= 1:
                break
            continue
        _, chosen = candidates.pop(best_index)
        selected.append(chosen)
        used.add(chosen)
    for _, cell in candidates:
        if len(selected) >= count:
            break
        if cell not in used:
            selected.append(cell)
            used.add(cell)
    if len(selected) < count:
        raise ValueError(f"Only found {len(selected)} strategic action points, requested {count}.")
    return selected


def build_delivery_tasks(action_points, num_robots=DEFAULT_NUM_ROBOTS, tasks_per_robot=100):
    if len(action_points) < 2:
        raise ValueError("Need at least two action points to build delivery tasks.")
    offset = max(1, len(action_points) // 2)
    tasks_by_robot = {}
    for robot_id in range(num_robots):
        tasks = []
        for task_idx in range(tasks_per_robot):
            pickup_index = (robot_id + task_idx) % len(action_points)
            dropoff_index = (pickup_index + offset) % len(action_points)
            pickup = action_points[pickup_index]
            dropoff = action_points[dropoff_index]
            if pickup == dropoff:
                dropoff = action_points[(dropoff_index + 1) % len(action_points)]
            tasks.append(DeliveryTask(f"r{robot_id}-task-{task_idx}", pickup, dropoff))
        tasks_by_robot[robot_id] = tuple(tasks)
    return tasks_by_robot


def repair_delivery_tasks(grid, tasks_by_robot, starts_by_robot, action_points):
    repaired = {}
    candidate_points = []
    for point in action_points:
        point = nearest_enterable_cell(grid, tuple(point))
        if point not in candidate_points:
            candidate_points.append(point)
    fallback_pairs = []
    for robot_id, start in starts_by_robot.items():
        for pickup in candidate_points:
            if not route_exists(grid, start, pickup):
                continue
            for dropoff in candidate_points:
                if pickup == dropoff:
                    continue
                if route_exists(grid, pickup, dropoff):
                    fallback_pairs.append((pickup, dropoff))
        if not fallback_pairs:
            raise RuntimeError(f"Robot {robot_id} has no reachable delivery tasks from start {start}")
    for robot_id, tasks in tasks_by_robot.items():
        start = starts_by_robot[robot_id]
        fixed = []
        for index, task in enumerate(tasks):
            pickup = nearest_enterable_cell(grid, task.pickup)
            dropoff = nearest_enterable_cell(grid, task.dropoff)
            if not route_exists(grid, start, pickup) or not route_exists(grid, pickup, dropoff):
                pickup, dropoff = fallback_pairs[index % len(fallback_pairs)]
            fixed.append(DeliveryTask(task.task_id, pickup, dropoff))
        repaired[robot_id] = tuple(fixed)
    return repaired


def _largest_free_component(grid) -> list[tuple[int, int]]:
    remaining = set(find_free_cells(grid))
    largest: list[tuple[int, int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        component = [start]
        while queue:
            row, col = queue.pop(0)
            for neighbor in _neighbors((row, col)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _map_region_anchors(grid) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    return [
        (rows // 5, cols // 5),
        (rows // 5, cols // 2),
        (rows // 5, 4 * cols // 5),
        (rows // 2, cols // 5),
        (rows // 2, 4 * cols // 5),
        (4 * rows // 5, cols // 5),
        (4 * rows // 5, cols // 2),
        (4 * rows // 5, 4 * cols // 5),
    ]


def _too_close_to_corridor(cell: tuple[int, int], clearance: int = 4) -> bool:
    return any(manhattan(cell, corridor) < clearance for corridor in WAREHOUSE_NARROW_CORRIDOR_CELLS)


def choose_spread_out_starts(grid, count: int, forbidden=None) -> list[tuple[int, int]]:
    """Attacker starts in the bay; the other robots spawn from map-region anchors."""
    blocked = set(forbidden or ())
    attacker = WAREHOUSE_ATTACKER_START if _is_free(grid, WAREHOUSE_ATTACKER_START) else nearest_enterable_cell(
        grid, WAREHOUSE_ATTACKER_START, blocked
    )
    chosen = [attacker]
    blocked = blocked | {attacker} | set(WAREHOUSE_NARROW_CORRIDOR_CELLS)
    while len(chosen) < count:
        best = None
        best_distance = -1
        for anchor in _map_region_anchors(grid):
            try:
                cell = nearest_enterable_cell(grid, anchor, forbidden=blocked)
            except ValueError:
                continue
            if cell in blocked or _too_close_to_corridor(cell):
                continue
            distance = min(manhattan(cell, old) for old in chosen)
            if distance > best_distance:
                best = cell
                best_distance = distance
        if best is None:
            break
        chosen.append(best)
        blocked.add(best)
    floor = [
        cell
        for cell in _largest_free_component(grid)
        if cell not in blocked and not _too_close_to_corridor(cell)
    ]
    while len(chosen) < count:
        remaining = [cell for cell in floor if cell not in chosen]
        if not remaining:
            remaining = [
                cell
                for cell in _largest_free_component(grid)
                if cell not in blocked and cell not in chosen
            ]
        if not remaining:
            raise ValueError("warehouse map does not contain enough mutually reachable start cells")
        def nearest_distance(cell):
            return min(manhattan(cell, old) for old in chosen)
        chosen.append(max(remaining, key=lambda cell: (nearest_distance(cell), -cell[0], -cell[1])))
        blocked.add(chosen[-1])
    return chosen


def refine_action_points(grid, action_points, forbidden, target_count, excluded) -> list[tuple[int, int]]:
    """Drop excluded cells and refill to the target count with strategic picks."""
    excluded_set = frozenset(tuple(cell) for cell in excluded)
    points = [tuple(cell) for cell in action_points if tuple(cell) not in excluded_set]
    blocked = set(forbidden).union(points).union(excluded_set)
    while len(points) < target_count:
        extra = choose_strategic_action_points(grid, target_count - len(points), forbidden=blocked)
        added = False
        for point in extra:
            cell = tuple(point)
            if cell in excluded_set or cell in points:
                continue
            points.append(cell)
            blocked.add(cell)
            added = True
            if len(points) >= target_count:
                break
        if not added:
            break
    return points


def build_warehouse_layout(grid, deliveries_per_robot: int):
    """Return starts, goals, and task queues for the default warehouse map."""
    start_cells = choose_spread_out_starts(grid, DEFAULT_NUM_ROBOTS)
    starts = {robot_id: start_cells[robot_id] for robot_id in range(DEFAULT_NUM_ROBOTS)}
    used_starts = set(start_cells)
    action_points = choose_strategic_action_points(
        grid,
        DEFAULT_NUM_ACTION_POINTS,
        forbidden=used_starts | set(WAREHOUSE_EXCLUDED_ACTION_POINTS),
    )
    action_points = refine_action_points(
        grid,
        action_points,
        used_starts,
        DEFAULT_NUM_ACTION_POINTS,
        WAREHOUSE_EXCLUDED_ACTION_POINTS,
    )
    if _is_free(grid, WAREHOUSE_CORRIDOR_DELIVERY) and WAREHOUSE_CORRIDOR_DELIVERY not in action_points:
        mouth = (13, 8)
        replace_at = min(
            range(len(action_points)),
            key=lambda index: manhattan(action_points[index], mouth),
        ) if action_points else None
        if replace_at is not None and len(action_points) >= DEFAULT_NUM_ACTION_POINTS:
            action_points[replace_at] = WAREHOUSE_CORRIDOR_DELIVERY
        else:
            action_points.append(WAREHOUSE_CORRIDOR_DELIVERY)
    goals = list(action_points)
    tasks = build_delivery_tasks(action_points, DEFAULT_NUM_ROBOTS, deliveries_per_robot)
    tasks = repair_delivery_tasks(grid, tasks, starts, action_points)
    return starts, goals, tasks
