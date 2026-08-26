"""Physical temporary-obstacle and fake-obstacle footprints.

These helpers follow the main-branch warehouse objects: palettes and carts are
multi-cell rectangles, and fake obstacles report a rectangle of free cells
rather than a single pixel.
"""
from __future__ import annotations

import random

from .models import TemporaryObstacleEpisode


TEMP_ACTIVE_COUNT = 6
TEMP_MIN_SPACING = 5
TEMP_EDGE_MARGIN_RATIO = 0.12
TEMP_MAX_SIDE = 5
TEMP_MIN_AREA = 4
TEMP_PLACEMENT_ATTEMPTS = 500
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


def anchors_remain_connected(grid, extra_blocked, anchors) -> bool:
    """True if every required anchor stays in one free component after extra blocks."""
    blocked = {tuple(cell) for cell in extra_blocked}
    live = []
    for anchor in anchors or ():
        cell = tuple(anchor)
        if cell in blocked or not _in_bounds(grid, cell) or grid[cell]:
            return False
        live.append(cell)
    if len(live) < 2:
        return True
    start = live[0]
    seen = {start}
    queue = [start]
    while queue:
        cell = queue.pop()
        for neighbor in _neighbors(cell):
            if neighbor in seen or neighbor in blocked or not _in_bounds(grid, neighbor) or grid[neighbor]:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return all(anchor in seen for anchor in live)


def can_place_temporary_footprint(
    grid,
    cells,
    forbidden_cells=None,
    *,
    occupied_footprints=(),
    required_anchors=(),
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
) -> list[list[tuple[int, int]]]:
    """Return up to ``blocked_count`` well-spaced multi-cell physical footprints."""
    rows, cols = grid.shape
    regions = candidate_temporary_regions(rows, cols)
    selected: list[list[tuple[int, int]]] = []
    spacing = min(TEMP_MIN_SPACING, max(2, min(rows, cols) // 6))
    for index in range(max(blocked_count, 1) * 4):
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
                    grid,
                    cells,
                    forbidden_cells,
                    occupied_footprints=selected,
                    required_anchors=required_anchors,
                ):
                    continue
                if not far_enough_from_footprints(cells, selected, spacing):
                    continue
                selected.append(cells)
                placed = True
                break
            if placed:
                break
        if len(selected) >= blocked_count:
            break
    return selected[:blocked_count]


def _try_shift_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=()):
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
        ):
            return candidate
    return None


def _try_teleport_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=()):
    height = max(row for row, _ in cells) - min(row for row, _ in cells) + 1
    width = max(col for _, col in cells) - min(col for _, col in cells) + 1
    old = footprint_center(cells)
    rows, cols = grid.shape
    for _ in range(TEMP_PLACEMENT_ATTEMPTS):
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
        ):
            return candidate
    return None


def move_footprint(grid, cells, rng: random.Random, forbidden_cells=None, other_footprints=(), required_anchors=()):
    """Shift or teleport one physical obstacle, matching main's 50/50 choice."""
    original = {tuple(cell) for cell in cells}
    methods = [_try_shift_footprint, _try_teleport_footprint]
    if rng.random() < 0.5:
        methods.reverse()
    for method in methods:
        moved = method(grid, cells, rng, forbidden_cells, other_footprints, required_anchors)
        if moved is not None and {tuple(cell) for cell in moved} != original:
            return moved
    replacements = choose_temporary_object_footprints(
        grid,
        rng,
        blocked_count=1,
        forbidden_cells=forbidden_cells,
        required_anchors=required_anchors,
    )
    for candidate in replacements:
        if {tuple(cell) for cell in candidate} == original:
            continue
        if not can_place_temporary_footprint(
            grid,
            candidate,
            forbidden_cells,
            occupied_footprints=other_footprints,
            required_anchors=required_anchors,
        ):
            continue
        return candidate
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
) -> tuple[TemporaryObstacleEpisode, ...]:
    """Precompute main-style concurrent, moving temporary obstacles into the manifest."""
    period = max(1, int(period))
    footprints = choose_temporary_object_footprints(
        grid,
        rng,
        blocked_count=active_count,
        forbidden_cells=forbidden_cells,
        required_anchors=required_anchors,
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
