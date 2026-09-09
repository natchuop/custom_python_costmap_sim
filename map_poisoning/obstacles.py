"""Physical temporary-obstacle and fake-obstacle footprints.

These helpers follow the main-branch warehouse objects: palettes and carts are
multi-cell rectangles, and fake obstacles report a rectangle of free cells
rather than a single pixel.
"""
from __future__ import annotations

import random
from collections import deque

from .models import PermanentObstacle, TemporaryObstacleEpisode


TEMP_ACTIVE_COUNT = 6
PERMANENT_OBSTACLE_COUNT = 3
TEMP_MIN_SPACING = 5
TEMP_EDGE_MARGIN_RATIO = 0.12
TEMP_MAX_SIDE = 5
TEMP_MIN_AREA = 4
TEMP_PLACEMENT_ATTEMPTS = 500
TEMP_MOVE_TELEPORT_ATTEMPTS = 80
FAKE_MAX_SIDE = 7
FAKE_MIN_AREA = 4
FAKE_MIN_REPORT_CELLS = 4
FAKE_CENTER_MIN_SPACING = 6


def footprint_cells(top_left, height: int, width: int) -> list[tuple[int, int]]:
    row0, col0 = top_left
    return [(row, col) for row in range(row0, row0 + height) for col in range(col0, col0 + width)]


def footprint_from_center(center, height: int, width: int) -> list[tuple[int, int]]:
    row, col = center
    return footprint_cells((row - height // 2, col - width // 2), height, width)


def footprint_center(cells) -> tuple[float, float]:
    return sum(cell[0] for cell in cells) / len(cells), sum(cell[1] for cell in cells) / len(cells)


def sample_rectangle_dimensions(rng: random.Random, min_side: int = 1, max_side: int = TEMP_MAX_SIDE, min_area: int = TEMP_MIN_AREA) -> tuple[int, int]:
    for _ in range(100):
        height = rng.randint(min_side, max_side)
        width = rng.randint(min_side, max_side)
        if height * width >= min_area:
            return height, width
    return min_side, max(min_side, (min_area + min_side - 1) // min_side)


def sample_fake_obstacle_dimensions(rng: random.Random) -> tuple[int, int]:
    """Sample a valid fake footprint with a moderate-size bias.

    The absolute 7x7 bound is unchanged.  Weighting side lengths toward 3--6
    makes consequential footprints more common without eliminating compact
    bottleneck attacks or making the maximum rectangle the default.
    """
    sides = tuple(range(1, FAKE_MAX_SIDE + 1))
    weights = (1, 2, 5, 7, 7, 5, 3)
    for _ in range(100):
        height = rng.choices(sides, weights=weights, k=1)[0]
        width = rng.choices(sides, weights=weights, k=1)[0]
        if height * width >= FAKE_MIN_AREA:
            return height, width
    return 3, 3


def _in_bounds(grid, cell) -> bool:
    return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    return [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]


def _anchors_remain_connected_with_blocked(grid, blocked, anchors) -> bool:
    """Connectivity check using a pre-normalized blocked-cell set.

    Stop as soon as every required anchor has been reached instead of walking
    the entire free-space component. Manifest authoring calls this helper many
    times while testing obstacle moves, so the early exit materially reduces
    runtime without changing placement semantics.
    """
    live = []
    for anchor in anchors or ():
        cell = tuple(anchor)
        if cell in blocked or not _in_bounds(grid, cell) or grid[cell]:
            return False
        live.append(cell)
    if len(live) < 2:
        return True

    start = live[0]
    remaining = set(live[1:])
    remaining.discard(start)
    if not remaining:
        return True

    seen = {start}
    queue = [start]
    while queue:
        cell = queue.pop()
        for neighbor in _neighbors(cell):
            if neighbor in seen or neighbor in blocked or not _in_bounds(grid, neighbor) or grid[neighbor]:
                continue
            if neighbor in remaining:
                remaining.remove(neighbor)
                if not remaining:
                    return True
            seen.add(neighbor)
            queue.append(neighbor)
    return False


def anchors_remain_connected(grid, extra_blocked, anchors) -> bool:
    """True if every required anchor stays in one free component after extra blocks."""
    blocked = {tuple(cell) for cell in extra_blocked}
    return _anchors_remain_connected_with_blocked(grid, blocked, anchors)


def anchor_groups_remain_connected(grid, extra_blocked, anchor_groups) -> bool:
    """Require connectivity within each mission group independently.

    Some maps intentionally contain multiple disconnected operating regions.
    Flattening every robot's task points into one anchor set would therefore
    reject every valid obstacle. This helper preserves each robot's own
    start/pickup/dropoff connectivity without requiring unrelated regions to
    be mutually reachable.
    """
    blocked = {tuple(cell) for cell in extra_blocked}
    return all(
        _anchors_remain_connected_with_blocked(grid, blocked, group)
        for group in (anchor_groups or ())
    )


def can_place_temporary_footprint(
    grid,
    cells,
    forbidden_cells=None,
    *,
    occupied_footprints=(),
    required_anchors=(),
    required_anchor_groups=(),
) -> bool:
    forbidden = set(forbidden_cells or ())
    occupied = {tuple(cell) for footprint in occupied_footprints for cell in footprint}
    for cell in cells:
        if cell in forbidden or cell in occupied or not _in_bounds(grid, cell) or grid[cell]:
            return False
    if required_anchors:
        extra = occupied | {tuple(cell) for cell in cells}
        if not anchors_remain_connected(grid, extra, required_anchors):
            return False
    if required_anchor_groups:
        extra = occupied | {tuple(cell) for cell in cells}
        if not anchor_groups_remain_connected(grid, extra, required_anchor_groups):
            return False
    return True


def far_enough_from_footprints(cells, selected_footprints, min_spacing: int) -> bool:
    center_r, center_c = footprint_center(cells)
    for other in selected_footprints:
        other_r, other_c = footprint_center(other)
        if abs(center_r - other_r) + abs(center_c - other_c) < min_spacing:
            return False
    return True


def candidate_temporary_regions(rows: int, cols: int):
    row_margin = max(2, int(rows * TEMP_EDGE_MARGIN_RATIO))
    col_margin = max(2, int(cols * TEMP_EDGE_MARGIN_RATIO))
    r_min, r_max = row_margin, rows - row_margin
    c_min, c_max = col_margin, cols - col_margin
    r_mid = (r_min + r_max) // 2
    c_mid = (c_min + c_max) // 2
    return [
        (r_min, r_mid, c_min, c_mid),
        (r_min, r_mid, c_mid, c_max),
        (r_mid, r_max, c_min, c_mid),
        (r_mid, r_max, c_mid, c_max),
    ]


def choose_temporary_object_footprints(
    grid,
    rng: random.Random,
    blocked_count: int = TEMP_ACTIVE_COUNT,
    forbidden_cells=None,
    required_anchors=(),
    required_anchor_groups=(),
    traffic_heatmap=None,
) -> list[list[tuple[int, int]]]:
    """Return up to ``blocked_count`` relevant, well-spaced physical footprints.

    When reconnaissance traffic is available, candidate generation is driven by
    the frozen traffic cells rather than repeated whole-map random sampling.
    This is both more relevant to navigation and substantially cheaper on large
    maps. A random fallback remains for maps/scenarios without traffic data.
    """
    rows, cols = grid.shape
    selected: list[list[tuple[int, int]]] = []
    spacing = min(TEMP_MIN_SPACING, max(2, min(rows, cols) // 6))
    traffic_available = (
        traffic_heatmap is not None
        and getattr(traffic_heatmap, "shape", None) == grid.shape
        and bool(getattr(traffic_heatmap, "any", lambda: False)())
    )

    if traffic_available:
        centers = [
            (row, col)
            for row in range(1, rows - 1)
            for col in range(1, cols - 1)
            if not grid[row, col]
            and (row, col) not in set(forbidden_cells or ())
            and traffic_heatmap[row, col] > 0
        ]
        rng.shuffle(centers)
        centers.sort(key=lambda cell: float(traffic_heatmap[cell]), reverse=True)
        # Recon traces generally contain far fewer than 120 unique cells. The
        # cap prevents long-route maps from turning authoring into a grid scan.
        centers = centers[:120]
        shapes = ((2, 2), (2, 3), (3, 2), (1, 4), (4, 1))
        scored = []
        for center in centers:
            for height, width in shapes:
                cells = footprint_from_center(center, height, width)
                if len(cells) < TEMP_MIN_AREA:
                    continue
                if not can_place_temporary_footprint(
                    grid, cells, forbidden_cells, required_anchors=()
                ):
                    continue
                traffic = _traffic_value(traffic_heatmap, cells)
                if traffic <= 0.0:
                    continue
                detour = footprint_finite_detour_score(grid, cells)
                if detour <= 0.0:
                    continue
                bottleneck = footprint_bottleneck_score(grid, cells)
                score = traffic * (1.0 + 2.0 * bottleneck) + detour * (1.0 + bottleneck)
                scored.append((score, traffic, detour, bottleneck, tuple(cells)))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        for _, _, _, _, cells in scored:
            if len(selected) >= blocked_count:
                break
            if not far_enough_from_footprints(cells, selected, spacing):
                continue
            if not can_place_temporary_footprint(
                grid,
                cells,
                forbidden_cells,
                occupied_footprints=selected,
                required_anchors=required_anchors,
                required_anchor_groups=required_anchor_groups,
            ):
                continue
            selected.append(list(cells))
        if len(selected) >= blocked_count:
            return selected[:blocked_count]

    # Fallback for sparse/no-traffic scenarios. Keep the existing random
    # rectangle behavior, but do cheap geometry checks before connectivity.
    regions = candidate_temporary_regions(rows, cols)
    for index in range(max(blocked_count, 1) * 4):
        if len(selected) >= blocked_count:
            break
        placed = False
        region_order = list(range(len(regions)))
        rng.shuffle(region_order)
        preferred = index % len(regions)
        region_order = [preferred, *[item for item in region_order if item != preferred]]
        for region_idx in region_order:
            r_min, r_max, c_min, c_max = regions[region_idx]
            for _ in range(TEMP_PLACEMENT_ATTEMPTS):
                height, width = sample_rectangle_dimensions(rng)
                if rng.random() < 0.4:
                    if rng.random() < 0.5:
                        height = 1
                    else:
                        width = 1
                if height * width < TEMP_MIN_AREA:
                    continue
                if r_max - r_min <= height + 2 or c_max - c_min <= width + 2:
                    continue
                row = rng.randrange(r_min, max(r_min + 1, r_max - height))
                col = rng.randrange(c_min, max(c_min + 1, c_max - width))
                cells = footprint_cells((row, col), height, width)
                if not can_place_temporary_footprint(
                    grid, cells, forbidden_cells, occupied_footprints=selected, required_anchors=()
                ):
                    continue
                if not far_enough_from_footprints(cells, selected, spacing):
                    continue
                if required_anchors and not can_place_temporary_footprint(
                    grid, cells, forbidden_cells, occupied_footprints=selected,
                    required_anchors=required_anchors,
                    required_anchor_groups=required_anchor_groups,
                ):
                    continue
                if required_anchor_groups and not can_place_temporary_footprint(
                    grid, cells, forbidden_cells, occupied_footprints=selected,
                    required_anchors=required_anchors,
                    required_anchor_groups=required_anchor_groups,
                ):
                    continue
                selected.append(cells)
                placed = True
                break
            if placed:
                break
    return selected[:blocked_count]


def _try_shift_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=(), required_anchor_groups=()):
    distance = rng.randint(1, 3)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rng.shuffle(directions)
    for drow, dcol in directions:
        candidate = [(row + drow * distance, col + dcol * distance) for row, col in cells]
        if can_place_temporary_footprint(
            grid,
            candidate,
            forbidden_cells,
            occupied_footprints=other_footprints,
            required_anchors=required_anchors,
            required_anchor_groups=required_anchor_groups,
        ):
            return candidate
    return None


def _try_teleport_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=(), required_anchor_groups=()):
    height = max(row for row, _ in cells) - min(row for row, _ in cells) + 1
    width = max(col for _, col in cells) - min(col for _, col in cells) + 1
    old = footprint_center(cells)
    rows, cols = grid.shape
    for _ in range(TEMP_MOVE_TELEPORT_ATTEMPTS):
        row = rng.randrange(1, max(2, rows - height))
        col = rng.randrange(1, max(2, cols - width))
        candidate = footprint_cells((row, col), height, width)
        center = footprint_center(candidate)
        if abs(center[0] - old[0]) + abs(center[1] - old[1]) < 3:
            continue
        if can_place_temporary_footprint(
            grid,
            candidate,
            forbidden_cells,
            occupied_footprints=other_footprints,
            required_anchors=required_anchors,
            required_anchor_groups=required_anchor_groups,
        ):
            return candidate
    return None


def move_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=(), required_anchor_groups=()):
    """Shift or teleport one physical obstacle, matching main's 50/50 choice."""
    original = {tuple(cell) for cell in cells}
    methods = [_try_shift_footprint, _try_teleport_footprint]
    if rng.random() < 0.5:
        methods.reverse()
    for method in methods:
        moved = method(grid, cells, rng, forbidden_cells, other_footprints, required_anchors, required_anchor_groups)
        if moved is not None and {tuple(cell) for cell in moved} != original:
            return moved
    # If no bounded shift/teleport is safe, keep the obstacle in place for this
    # movement window. This preserves deterministic physical occupancy and
    # connectivity while avoiding an expensive whole-map replacement search in
    # crowded 3-permanent/6-temporary scenarios.
    return list(cells)


def fake_report_cells(center, height: int, width: int, grid, *, forbidden=None, active_cells=None) -> list[tuple[int, int]]:
    """Free cells inside a fake-obstacle rectangle. Walls may be overlapped visually but are not reported."""
    blocked = set(forbidden or ()) | set(active_cells or ())
    reportable = []
    for cell in footprint_from_center(center, height, width):
        if not _in_bounds(grid, cell) or grid[cell] or cell in blocked:
            continue
        reportable.append(cell)
    return reportable


def author_temporary_obstacle_episodes(
    grid,
    rng: random.Random,
    total_steps: int,
    period: int,
    forbidden_cells=None,
    active_count: int = TEMP_ACTIVE_COUNT,
    required_anchors=(),
    required_anchor_groups=(),
    traffic_heatmap=None,
) -> tuple[TemporaryObstacleEpisode, ...]:
    """Precompute main-style concurrent, moving temporary obstacles into the manifest."""
    period = max(1, int(period))
    footprints = choose_temporary_object_footprints(
        grid,
        rng,
        blocked_count=active_count,
        forbidden_cells=forbidden_cells,
        required_anchors=required_anchors,
        required_anchor_groups=required_anchor_groups,
        traffic_heatmap=traffic_heatmap,
    )
    if not footprints:
        return ()
    episodes = []
    current = [list(item) for item in footprints]
    for window, start in enumerate(range(0, total_steps, period)):
        end = min(total_steps, start + period)
        if window > 0:
            moved = []
            for index, cells in enumerate(current):
                others = moved + current[index + 1:]
                moved.append(
                    move_footprint(
                        grid,
                        cells,
                        rng,
                        forbidden_cells,
                        others,
                        required_anchors,
                        required_anchor_groups,
                    )
                )
            current = moved
        for index, cells in enumerate(current):
            episodes.append(
                TemporaryObstacleEpisode(
                    f"obstacle-{window:03}-{index:02}",
                    tuple(tuple(cell) for cell in cells),
                    start,
                    end,
                )
            )
    return tuple(episodes)


def footprint_bottleneck_score(grid, cells) -> float:
    """Fraction of footprint boundary edges adjacent to walls/map edge."""
    footprint = {tuple(cell) for cell in cells}
    if not footprint:
        return 0.0
    blocked = 0
    boundary = 0
    for cell in footprint:
        for neighbor in _neighbors(cell):
            if neighbor in footprint:
                continue
            boundary += 1
            if not _in_bounds(grid, neighbor) or grid[neighbor]:
                blocked += 1
    return blocked / max(1, boundary)


def footprint_finite_detour_score(grid, cells) -> float:
    """Return a positive finite *local* detour score for a footprint.

    A bounded breadth-first search is intentionally used here instead of a
    whole-map A* query. Candidate scoring is called many times during manifest
    authoring, and the only question at this stage is whether nearby traffic
    can get around the footprint with a finite extra cost. Global mission
    connectivity is checked separately before a permanent obstacle is selected.
    """
    footprint = {tuple(cell) for cell in cells}
    if not footprint:
        return 0.0
    rows, cols = grid.shape
    min_r = min(row for row, _ in footprint)
    max_r = max(row for row, _ in footprint)
    min_c = min(col for _, col in footprint)
    max_c = max(col for _, col in footprint)

    margin = 8
    box_r0 = max(0, min_r - margin)
    box_r1 = min(rows - 1, max_r + margin)
    box_c0 = max(0, min_c - margin)
    box_c1 = min(cols - 1, max_c + margin)

    def free(cell, blocked):
        row, col = cell
        return (
            box_r0 <= row <= box_r1
            and box_c0 <= col <= box_c1
            and not grid[row, col]
            and (not blocked or cell not in footprint)
        )

    def local_distance(start_cell, goal, blocked):
        if not free(start_cell, blocked) or not free(goal, blocked):
            return None
        queue = deque([(start_cell, 0)])
        seen = {start_cell}
        while queue:
            cell, distance = queue.popleft()
            if cell == goal:
                return distance
            for neighbor in _neighbors(cell):
                if neighbor in seen or not free(neighbor, blocked):
                    continue
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    pairs = []
    for row in range(min_r, max_r + 1):
        left, right = (row, min_c - 2), (row, max_c + 2)
        if free(left, False) and free(right, False):
            pairs.append((left, right))
    for col in range(min_c, max_c + 1):
        up, down = (min_r - 2, col), (max_r + 2, col)
        if free(up, False) and free(down, False):
            pairs.append((up, down))
    best = 0.0
    for start_cell, goal in pairs[:6]:
        baseline = local_distance(start_cell, goal, False)
        attacked = local_distance(start_cell, goal, True)
        if baseline is None or attacked is None:
            continue
        best = max(best, float(attacked - baseline))
    return max(0.0, best)


def _traffic_value(traffic_heatmap, cells) -> float:
    if traffic_heatmap is None:
        return 0.0
    total = 0.0
    for row, col in cells:
        if 0 <= row < traffic_heatmap.shape[0] and 0 <= col < traffic_heatmap.shape[1]:
            total += float(traffic_heatmap[row, col])
    return total


def author_permanent_obstacles(
    grid,
    rng: random.Random,
    *,
    count: int = PERMANENT_OBSTACLE_COUNT,
    forbidden_cells=None,
    required_anchors=(),
    required_anchor_groups=(),
    traffic_heatmap=None,
) -> tuple[PermanentObstacle, ...]:
    """Author deterministic, connectivity-safe runtime permanent obstacles.

    Candidates prefer reconnaissance traffic and chokepoints, must create a
    positive finite detour, and may shrink to 1x2/2x1 footprints on narrow maps.
    """
    forbidden = set(forbidden_cells or ())
    selected: list[list[tuple[int, int]]] = []
    shapes = ((2, 2), (1, 3), (3, 1), (1, 2), (2, 1))
    all_centers = [
        (row, col)
        for row in range(1, grid.shape[0] - 1)
        for col in range(1, grid.shape[1] - 1)
        if not grid[row, col] and (row, col) not in forbidden
    ]
    if traffic_heatmap is not None and getattr(traffic_heatmap, "shape", None) == grid.shape:
        hot = [cell for cell in all_centers if traffic_heatmap[cell] > 0]
        cold = [cell for cell in all_centers if traffic_heatmap[cell] <= 0]
        rng.shuffle(hot)
        rng.shuffle(cold)
        hot.sort(key=lambda cell: float(traffic_heatmap[cell]), reverse=True)
        # Prefer recon-observed traffic first, but keep a bounded geometry-only
        # fallback pool. Very short smoke-test reconnaissance can otherwise
        # expose too few cells to place the requested third permanent obstacle.
        centers = hot[:120]
        if len(centers) < 180:
            centers.extend(cold[: 180 - len(centers)])
    else:
        rng.shuffle(all_centers)
        centers = all_centers[:180]
    scored = []
    for center in centers:
        for height, width in shapes:
            cells = footprint_from_center(center, height, width)
            # Candidate scoring only needs geometric validity. The much more
            # expensive whole-map connectivity check is deferred until a top
            # candidate is actually considered for selection.
            if not can_place_temporary_footprint(
                grid, cells, forbidden, occupied_footprints=selected, required_anchors=()
            ):
                continue
            detour = footprint_finite_detour_score(grid, cells)
            if detour <= 0.0:
                continue
            traffic = _traffic_value(traffic_heatmap, cells)
            bottleneck = footprint_bottleneck_score(grid, cells)
            score = traffic * (1.0 + 2.0 * bottleneck) + detour * (1.0 + bottleneck)
            scored.append((score, traffic, detour, bottleneck, tuple(cells)))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    for _, _, _, _, cells in scored:
        if len(selected) >= count:
            break
        if not can_place_temporary_footprint(
            grid, cells, forbidden, occupied_footprints=selected,
            required_anchors=required_anchors,
            required_anchor_groups=required_anchor_groups,
        ):
            continue
        if not far_enough_from_footprints(cells, selected, max(2, min(TEMP_MIN_SPACING, min(grid.shape)//8))):
            continue
        selected.append(list(cells))
    return tuple(
        PermanentObstacle(f"permanent-{index:02}", tuple(tuple(cell) for cell in cells))
        for index, cells in enumerate(selected[:count])
    )
