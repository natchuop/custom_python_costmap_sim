import math
import heapq
import copy
import argparse
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from defense_method_runner import DEFENSE_METHODS, build_defense_runner



# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 12

GRID_ROWS = 24
GRID_COLS = 36

MAX_STEPS = 2400

ENABLE_FALLBACK_EXPLORATION = True
FALLBACK_MIN_WAYPOINT_DISTANCE = 8

COMMUNICATION_PERIOD_STEPS = 4

EXPERIMENT_MODE = "attack"  # "clean" or "attack"

ENABLE_MALICIOUS_REPORTS = EXPERIMENT_MODE == "attack"

TRUST_ACCEPT_THRESHOLD = 0.55
TRUST_INITIAL_VALUE = 0.90
TRUST_REWARD = 0.02
TRUST_PENALTY = 0.06
TRUST_MODEL_NAME = "scalar"
TRUST_BAYES_PRIOR_ALPHA = 9.0
TRUST_BAYES_PRIOR_BETA = 1.0

SENSOR_RADIUS = 4

# Lidar-style local perception.
# The robot knows the map boundary and static prior, but live disruptions
# are discovered through ray-casted sensing.
LIDAR_RANGE_CELLS = 6.0
LIDAR_NUM_RAYS = 24
LIDAR_STEP_CELLS = 0.25
SHOW_LIDAR_RAYS = True

TEMP_ACTIVE_OBJECT_COUNT_BLOCKED = 6
TEMP_OBJECT_POOL_MULTIPLIER = 4
TEMP_OBJECT_MIN_SPACING = 5
TEMP_OBJECT_EDGE_MARGIN_RATIO = 0.12

TEMP_BLOCKED_OBJECT_SIZE_RANGE = (1, 5)
RECTANGLE_MIN_AREA = 4
RECTANGLE_MAX_SIDE = 5
TEMP_OBJECT_PLACEMENT_ATTEMPTS = 500

ENABLE_DYNAMIC_TEMP_BLOCKAGES = True
TEMP_BLOCKAGE_CHANGE_PERIOD_STEPS = 150

ENABLE_AUTO_TEMP_OBJECTS_FOR_LOADED_MAPS = True

SHOW_ANIMATION = True
ANIMATION_INTERVAL_MS = 1

CELL_SIZE = 1.0
ROBOT_SPEED_CELLS_PER_STEP = 1.0

ROBOT_FOOTPRINT_ROWS = 1
ROBOT_FOOTPRINT_COLS = 1
ROBOT_VISUAL_SCALE = 1.0

SPAWN_COLLISION_GRACE_STEPS = 100

START_CLEARANCE_CELLS = 1
START_SEARCH_MAX_RADIUS = 8

CONFIDENCE_DECAY_PER_STEP = 0.995
CONFIDENCE_UNKNOWN_THRESHOLD = 0.20

SELF_BLOCK_MEMORY_STEPS = 250
PEER_BLOCK_MEMORY_STEPS = 120
FREE_MEMORY_STEPS = 40

DEFAULT_NUM_ROBOTS = 3
DEFAULT_NUM_ACTION_POINTS = 14
TASKS_PER_ROBOT = 100

# Strategic synthetic action-point placement.
# These are used when a map does not already define PICKUP/DROPOFF/CHARGING cells.
ACTION_POINT_EDGE_WEIGHT = 2.0
ACTION_POINT_CORNER_WEIGHT = 3.0
ACTION_POINT_OBSTACLE_PROXIMITY_WEIGHT = 1.4
ACTION_POINT_SPREAD_WEIGHT = 2.5

ACTION_POINT_MIN_SPACING_RATIO = 0.18
ACTION_POINT_OBSTACLE_RADIUS = 4
ACTION_POINT_CORNER_RADIUS = 2

# Do not place goals directly on the boundary. The robot has a 1x1 footprint,
# because apparently occupying physical space matters.
ACTION_POINT_EDGE_MARGIN_CELLS = max(
    ROBOT_FOOTPRINT_ROWS,
    ROBOT_FOOTPRINT_COLS,
)

ATTACK_MODE = "recon_heatmap"

# Occupancy-claim defense policy. Available values are defined in
# defense_method_runner.py and can also be selected with --defense-method.
DEFENSE_METHOD = "trust_threshold"

# Phase 1: attacker observes benign movement and builds a traffic heatmap.
MIN_RECON_STEPS = 300
MAX_RECON_STEPS = 450
RECON_MIN_GOAL_VISITS = 1
RECON_MIN_GOAL_COVERAGE_RATIO = 0.70

# Phase 2: attacker injects fake blocked-object reports at learned medium-traffic corridors.
ATTACK_CANDIDATE_LIMIT = 24

# Instead of attacking the hottest corridors, attack average-traveled corridors.
# These are common enough to matter, but not constantly visited and instantly verified.
ATTACK_TRAFFIC_LOW_PERCENTILE = 35
ATTACK_TRAFFIC_HIGH_PERCENTILE = 95

# Fake object footprint. It may visually overlap walls, but only free/action cells
# will receive malicious BLOCKED reports.
MALICIOUS_FAKE_OBJECT_ROWS = 5
MALICIOUS_FAKE_OBJECT_COLS = 5

# Prefer fake objects that block several usable cells, not one sad pixel of deception.
MALICIOUS_FAKE_OBJECT_MIN_REPORT_CELLS = 4

# Add a new malicious fake object periodically during the attack phase.
MALICIOUS_FAKE_OBJECT_INJECTION_PERIOD_STEPS = 20

# Keep fake object centers separated so the attacker does not spam the same area.
MALICIOUS_FAKE_OBJECT_CENTER_MIN_SPACING = 6

# Do not place fake objects near goals. Robots must eventually visit goals,
# so fake blocks there are easy to disprove and weakly disruptive.
ATTACK_MIN_DISTANCE_FROM_GOAL = int(math.ceil(LIDAR_RANGE_CELLS)) + 1

# Do not place fake objects near any benign robot. The lie should not be
# immediately verifiable by current lidar.
ATTACK_MIN_DISTANCE_FROM_ANY_BENIGN_ROBOT = int(math.ceil(LIDAR_RANGE_CELLS)) + 2

# When evaluating whether a fake blocked cell would affect a victim,
# keep it outside immediate lidar range but not so far away that it is irrelevant.
ATTACK_MIN_DISTANCE_FROM_VICTIM = int(math.ceil(LIDAR_RANGE_CELLS)) + 2
ATTACK_MAX_DISTANCE_FROM_VICTIM = 45

# Keep red fake-object display visible briefly, but do not accumulate forever.
MALICIOUS_FAKE_OBJECT_DISPLAY_TTL = 50

# Topology-aware diagnostics help place legitimate temporary objects in useful
# warehouse regions. Attack candidate ranking remains route/traffic driven.
ENABLE_TOPOLOGY_AWARE_BLOCKAGES = True
ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP = False

# Finite poisoning window followed by recovery. Old malicious claims remain in
# each defense method, but no new lies are added after the burst. This directly
# exposes whether current source trust can retroactively release stale evidence.
ATTACK_BURST_DURATION_STEPS = 1200
STOP_ATTACK_AFTER_ALL_VICTIMS_DISTRUST = False

def cell_to_xy(cell):
    """
    Converts the robot's anchor cell to the center of its physical footprint.

    For a 1x1 robot, position_cell is the top-left anchor cell.
    The visual marker and lidar origin are placed at the footprint center.
    """
    row, col = cell

    x = (col + ROBOT_FOOTPRINT_COLS / 2.0) * CELL_SIZE
    y = (row + ROBOT_FOOTPRINT_ROWS / 2.0) * CELL_SIZE

    return np.array([x, y], dtype=float)


def xy_to_cell(position_xy):
    x, y = position_xy
    col = int(x // CELL_SIZE)
    row = int(y // CELL_SIZE)
    return row, col

def robot_footprint_cells(anchor_cell):
    """
    Returns all grid cells covered by the robot footprint.

    The robot position is treated as the top-left anchor cell of the footprint.

    Example:
        ROBOT_FOOTPRINT_ROWS = 1
        ROBOT_FOOTPRINT_COLS = 1

        anchor_cell = (10, 20)

        covered cells:
            (10, 20), (10, 21),
            (11, 20), (11, 21)
    """
    ar, ac = anchor_cell
    cells = []

    for dr in range(ROBOT_FOOTPRINT_ROWS):
        for dc in range(ROBOT_FOOTPRINT_COLS):
            cells.append((ar + dr, ac + dc))

    return cells

# If True, the malicious robot repeatedly lies about cells on the victim's planned path.
MALICIOUS_REPORT_PATH_BLOCKS = True

# If True, honest robots share directly observed blocked/free cells.
HONEST_ROBOTS_SHARE_OBSERVATIONS = True
HONEST_REPORT_REFRESH_STEPS = 80

# Performance controls. These reduce repeated full-grid bookkeeping without
# changing the underlying trust or routing policy.
CONFIDENCE_DECAY_UPDATE_PERIOD_STEPS = 5
DEFENSE_PRUNE_PERIOD_STEPS = 20

# Source-linked route adaptation controls. A trust update triggers planning only
# when it materially releases malicious risk on the near-term route. This keeps
# retroactive reweighting distinctive without recalculating the same route after
# every separately verified cell in one fake object.
SOURCE_LINKED_REPLAN_COOLDOWN_STEPS = 25
SOURCE_LINKED_MIN_TRUST_DELTA = 0.10
SOURCE_LINKED_MIN_ROUTE_RISK_DROP = 0.20
SOURCE_LINKED_ROUTE_LOOKAHEAD_ANCHORS = 40
MALICIOUS_ROUTE_PROXIMITY_CELLS = 2

# Shared replan churn controls. These apply to every defense method so fallback
# stalls and empty-path retries do not dominate runtime or outcomes.
PATH_INVALID_REPLAN_COOLDOWN_STEPS = 8
FALLBACK_GOAL_RETRY_COOLDOWN_STEPS = 20

# Physical traffic coordination is independent from trust and belief fusion.
TRAFFIC_REPLAN_WAIT_THRESHOLD = 3
TRAFFIC_REPLAN_COOLDOWN_STEPS = 5
TRAFFIC_LOOKAHEAD_CELLS = 6
TRAFFIC_CELL_PENALTY = 4.0
TRAFFIC_DEADLOCK_WAIT_THRESHOLD = 10
TRAFFIC_JOINT_REPEAT_THRESHOLD = 5
TRAFFIC_YIELD_SEARCH_RADIUS = 20


# ============================================================
# Cell states
# ============================================================

class CellState(IntEnum):
    FREE = 0
    OCCUPIED_STATIC = 1
    OCCUPIED_DYNAMIC = 2
    UNKNOWN = 3
    TEMPORARILY_BLOCKED = 4
    CONGESTED = 5
    PICKUP = 6
    DROPOFF = 7
    CHARGING = 8


class ClaimType(IntEnum):
    FREE = 0
    BLOCKED = 1
    CONGESTED = 2


CELL_LABELS = {
    CellState.FREE: "free",
    CellState.OCCUPIED_STATIC: "occupied_static",
    CellState.OCCUPIED_DYNAMIC: "occupied_dynamic",
    CellState.UNKNOWN: "unknown",
    CellState.TEMPORARILY_BLOCKED: "temporarily_blocked",
    CellState.CONGESTED: "congested",
    CellState.PICKUP: "pickup",
    CellState.DROPOFF: "dropoff",
    CellState.CHARGING: "charging",
}

def load_grid_from_movingai_map(path):
    """Load a MovingAI ``.map`` grid into simulator cell states.

    Traversable symbols: '.', 'G', 'S'.
    Blocked symbols: '@', 'O', 'T', 'W'.
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    try:
        height_line = next(line for line in lines if line.lower().startswith("height"))
        width_line = next(line for line in lines if line.lower().startswith("width"))
        map_index = next(i for i, line in enumerate(lines) if line.lower() == "map")
    except StopIteration as exc:
        raise ValueError(f"{path} is not a valid MovingAI map") from exc

    height = int(height_line.split()[1])
    width = int(width_line.split()[1])
    rows = lines[map_index + 1: map_index + 1 + height]

    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"MovingAI map dimensions do not match header in {path}")

    traversable = {".", "G", "S"}
    grid = np.full((height, width), int(CellState.OCCUPIED_STATIC), dtype=int)
    for r, row in enumerate(rows):
        for c, symbol in enumerate(row):
            if symbol in traversable:
                grid[r, c] = int(CellState.FREE)
    return grid


def load_grid_from_npy(path):
    grid = np.load(path)

    if grid.ndim != 2:
        raise ValueError(
            f"{path} must contain a 2D array, got shape {grid.shape}"
        )

    return grid.astype(int)

def state_name(value):
    try:
        return CELL_LABELS[CellState(int(value))]
    except Exception:
        return str(value)

def is_blocking_state(state):
    state = CellState(int(state))
    return state in (
        CellState.OCCUPIED_STATIC,
        CellState.OCCUPIED_DYNAMIC,
        CellState.TEMPORARILY_BLOCKED,
    )


def is_action_state(state):
    state = CellState(int(state))
    return state in (
        CellState.PICKUP,
        CellState.DROPOFF,
        CellState.CHARGING,
    )


def find_cells_with_state(grid, wanted_states):
    wanted_states = {int(s) for s in wanted_states}
    cells = []

    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            if int(grid[r, c]) in wanted_states:
                cells.append((r, c))

    return cells


def find_free_cells(grid):
    cells = []

    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            if not is_blocking_state(grid[r, c]):
                cells.append((r, c))

    return cells

def can_place_temporary_object(grid, cell):
    """
    Temporary objects should only be placed on normal free cells.

    Do not place them on walls, shelves, pickup/dropoff/charging zones,
    or existing blocked/congested cells.
    """
    r, c = cell

    if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
        return False

    return CellState(int(grid[r, c])) == CellState.FREE


def place_temporary_objects(dynamic_grid, temporary_objects):
    """
    Adds truth-only runtime disruptions to the dynamic map.

    These are legitimate warehouse disruptions:
    pallets, carts, stopped forklifts, blocked staging areas, etc.
    Robots should not know these from the static prior.
    """
    for cell, state in temporary_objects:
        if not can_place_temporary_object(dynamic_grid, cell):
            continue

        r, c = cell
        dynamic_grid[r, c] = int(state)

    return dynamic_grid


def footprint_cells(top_left, height, width):
    r0, c0 = top_left
    return [
        (r, c)
        for r in range(r0, r0 + height)
        for c in range(c0, c0 + width)
    ]


def sample_rectangle_dimensions(rng, min_side=1, max_side=RECTANGLE_MAX_SIDE,
                                min_area=RECTANGLE_MIN_AREA):
    """Sample a bounded rectangular footprint from the supplied seeded RNG."""
    for _ in range(100):
        if hasattr(rng, "integers"):
            height = int(rng.integers(min_side, max_side + 1))
            width = int(rng.integers(min_side, max_side + 1))
        else:
            height = int(rng.randint(min_side, max_side))
            width = int(rng.randint(min_side, max_side))
        if height * width >= min_area:
            return height, width
    return min_side, max(min_side, (min_area + min_side - 1) // min_side)


def can_place_temporary_footprint(grid, cells, forbidden_cells=None):
    """
    A temporary object footprint is valid only if every cell is normal FREE.

    This prevents pallets/carts from being spawned inside walls, shelves,
    pickup/dropoff zones, charging zones, or already blocked areas.
    """
    forbidden_cells = forbidden_cells or set()
    for cell in cells:
        if cell in forbidden_cells:
            return False
        if not can_place_temporary_object(grid, cell):
            return False

    return True


def footprint_center(cells):
    avg_r = sum(cell[0] for cell in cells) / len(cells)
    avg_c = sum(cell[1] for cell in cells) / len(cells)
    return avg_r, avg_c


def far_enough_from_footprints(cells, selected_footprints, min_spacing):
    center_r, center_c = footprint_center(cells)

    for other_cells, _ in selected_footprints:
        other_r, other_c = footprint_center(other_cells)
        distance = abs(center_r - other_r) + abs(center_c - other_c)

        if distance < min_spacing:
            return False

    return True


def candidate_temporary_regions(rows, cols):
    """
    Returns broad placement regions that avoid the extreme map edges while still
    spreading objects across the usable warehouse area.

    Regions are arranged as a 2x2 middle-safe layout:
    upper-left, upper-right, lower-left, lower-right.
    """
    row_margin = max(2, int(rows * TEMP_OBJECT_EDGE_MARGIN_RATIO))
    col_margin = max(2, int(cols * TEMP_OBJECT_EDGE_MARGIN_RATIO))

    r_min = row_margin
    r_max = rows - row_margin
    c_min = col_margin
    c_max = cols - col_margin

    r_mid = (r_min + r_max) // 2
    c_mid = (c_min + c_max) // 2

    return [
        (r_min, r_mid, c_min, c_mid),
        (r_min, r_mid, c_mid, c_max),
        (r_mid, r_max, c_min, c_mid),
        (r_mid, r_max, c_mid, c_max),
    ]


def anchor_enterable_on_static_grid(grid, anchor_cell):
    """Return whether the full robot footprint fits on static traversable cells."""
    for r, c in robot_footprint_cells(anchor_cell):
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return False
        if is_blocking_state(grid[r, c]):
            return False
    return True


def topology_bottleneck_score(grid, cell, radius=3):
    """Cheap articulation proxy for a 1x1-footprint planning graph.

    High scores indicate narrow, low-branching regions with obstacles pressing
    from opposing sides. Blocking these areas is operationally meaningful while
    avoiding an expensive all-pairs articulation analysis every attack cycle.
    """
    r, c = cell
    if not anchor_enterable_on_static_grid(grid, cell):
        return 0.0

    neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
    degree = sum(anchor_enterable_on_static_grid(grid, n) for n in neighbors)
    low_branching = max(0.0, 4.0 - float(degree))

    horizontal_pressure = 0
    vertical_pressure = 0
    for d in range(1, radius + 1):
        if not anchor_enterable_on_static_grid(grid, (r, c-d)):
            horizontal_pressure += 1
        if not anchor_enterable_on_static_grid(grid, (r, c+d)):
            horizontal_pressure += 1
        if not anchor_enterable_on_static_grid(grid, (r-d, c)):
            vertical_pressure += 1
        if not anchor_enterable_on_static_grid(grid, (r+d, c)):
            vertical_pressure += 1

    opposing_pressure = min(horizontal_pressure, vertical_pressure)
    return low_branching + 0.75 * opposing_pressure


def footprint_bottleneck_score(grid, cells):
    valid = [topology_bottleneck_score(grid, tuple(cell)) for cell in cells]
    return max(valid) if valid else 0.0


def choose_temporary_object_footprints(
    static_grid,
    blocked_count=TEMP_ACTIVE_OBJECT_COUNT_BLOCKED,
    min_spacing=TEMP_OBJECT_MIN_SPACING,
    rng=None,
):
    """
    Selects larger temporary blockage footprints.

    We keep a larger pool of possible blockage locations, then activate a subset.
    No congestion. Just legitimate temporary obstacles: pallets, carts, forklifts,
    blocked staging areas, and the usual warehouse nonsense.
    """
    rows, cols = static_grid.shape

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    selected = []
    regions = candidate_temporary_regions(rows, cols)

    pool_blocked_count = blocked_count * TEMP_OBJECT_POOL_MULTIPLIER

    object_specs = [
        (
            CellState.TEMPORARILY_BLOCKED,
            TEMP_BLOCKED_OBJECT_SIZE_RANGE,
        )
        for _ in range(pool_blocked_count)
    ]

    for index, (state, size_range) in enumerate(object_specs):
        placed = False

        region_order = list(range(len(regions)))
        rng.shuffle(region_order)

        preferred_region = index % len(regions)
        region_order = [preferred_region] + [
            region_idx for region_idx in region_order
            if region_idx != preferred_region
        ]

        for region_idx in region_order:
            r_min, r_max, c_min, c_max = regions[region_idx]

            for _ in range(TEMP_OBJECT_PLACEMENT_ATTEMPTS):
                height, width = sample_rectangle_dimensions(
                    rng, size_range[0], size_range[1]
                )

                # Sometimes make blockages cart-like instead of square.
                if rng.random() < 0.4:
                    if rng.random() < 0.5:
                        height = 1
                    else:
                        width = 1

                if height * width < RECTANGLE_MIN_AREA:
                    continue

                if r_max - r_min <= height + 2 or c_max - c_min <= width + 2:
                    continue

                r = int(rng.integers(r_min, r_max - height))
                c = int(rng.integers(c_min, c_max - width))

                cells = footprint_cells((r, c), height, width)

                if not can_place_temporary_footprint(static_grid, cells):
                    continue

                if not far_enough_from_footprints(cells, selected, min_spacing):
                    continue

                selected.append((cells, state))
                placed = True
                break

            if placed:
                break

        if not placed:
            print(f"Warning: could not place temporary blockage candidate {index}")

    if ENABLE_TOPOLOGY_AWARE_BLOCKAGES:
        selected.sort(
            key=lambda item: footprint_bottleneck_score(
                static_grid,
                item[0],
            ),
            reverse=True,
        )
    else:
        rng.shuffle(selected)

    return selected[:blocked_count]


def robot_occupied_cells(robots):
    """Cells currently occupied by any robot footprint."""
    occupied = set()
    for robot in robots or []:
        occupied.update(robot.occupied_cells)
    return occupied


def cell_occupied_by_benign_robot(cell, robots):
    """True when a benign robot's footprint currently covers this cell."""
    for robot in robots:
        if robot.is_malicious:
            continue
        if cell in robot.occupied_cells:
            return True
    return False


def apply_temporary_obstacle_episodes(grid, episodes, step, forbidden_cells=None):
    """Apply manifest temporary-obstacle episodes, skipping cells occupied by robots."""
    forbidden_cells = forbidden_cells or set()
    truth = grid.copy()
    for episode in episodes:
        if episode.appearance_step <= step < episode.clearance_step:
            for cell in episode.cells:
                if cell not in forbidden_cells:
                    truth[cell] = CellState.TEMPORARILY_BLOCKED
    return truth


def apply_temporary_footprints(dynamic_grid, temporary_footprints):
    for cells, state in temporary_footprints:
        for r, c in cells:
            dynamic_grid[r, c] = int(state)

    return dynamic_grid

class TemporaryBlockageManager:
    """
    Maintains runtime temporary blockages that change during a long simulation.

    The static prior stays unchanged.
    The world truth grid is rebuilt from the static prior whenever blockages change.
    Robots must rediscover cleared/new blockages through lidar.
    """

    def __init__(
        self,
        static_grid,
        active_count=TEMP_ACTIVE_OBJECT_COUNT_BLOCKED,
        change_period=TEMP_BLOCKAGE_CHANGE_PERIOD_STEPS,
        seed=RANDOM_SEED,
    ):
        self.static_grid = np.array(static_grid, dtype=int).copy()
        self.active_count = int(active_count)
        self.change_period = int(change_period)
        self.rng = np.random.default_rng(seed)

        self.pool = choose_temporary_object_footprints(
            self.static_grid,
            blocked_count=self.active_count,
            rng=self.rng,
        )

        if len(self.pool) < self.active_count:
            print(
                f"Warning: only generated {len(self.pool)} temporary blockage "
                f"candidates for requested active count {self.active_count}"
            )

        self.active_indices = set()
        self.current_footprints = {}
        self.movement_decisions = {}
        self.refresh_active_blockages(force=True)

    def refresh_active_blockages(self, force=False, forbidden_cells=None):
        forbidden_cells = forbidden_cells or set()
        if not self.pool:
            self.active_indices = set()
            return

        # Existing objects persist and use a seeded 50/50 shift/teleport choice.
        if not force:
            for idx in tuple(self.active_indices):
                cells, state = self.current_footprints.get(idx, self.pool[idx])
                moved, movement = self._move_footprint(cells, forbidden_cells)
                self.movement_decisions[idx] = movement
                if moved is not None:
                    self.current_footprints[idx] = (moved, state)
        else:
            self.movement_decisions = {idx: "unchanged" for idx in self.active_indices}
        candidate_indices = [idx for idx in range(len(self.pool)) if idx not in self.active_indices]
        self.rng.shuffle(candidate_indices)

        eligible = []
        occupied_by_kept = {
            cell
            for idx in self.active_indices
            if idx in self.current_footprints
            for cell in self.current_footprints[idx][0]
        }
        for idx in candidate_indices:
            cells, _ = self.pool[idx]
            if any(cell in forbidden_cells for cell in cells):
                continue
            if any(cell in occupied_by_kept for cell in cells):
                continue
            eligible.append(idx)
            occupied_by_kept.update(cells)
            if len(eligible) >= self.active_count:
                break

        if forbidden_cells and len(eligible) < self.active_count:
            print(
                f"Warning: only {len(eligible)} temporary blockages avoid "
                f"current robot positions (requested {self.active_count})"
            )

        active_count = min(self.active_count, len(eligible))
        if active_count == 0 and not force and self.active_indices:
            kept = [
                idx
                for idx in self.active_indices
                if not any(cell in forbidden_cells for cell in self.pool[idx][0])
            ]
            if kept:
                self.active_indices = set(kept[:min(self.active_count, len(kept))])
                print("Active temporary blockages (kept prior set avoiding robots):")
                for idx in sorted(self.active_indices):
                    cells, state = self.pool[idx]
                    center_r, center_c = footprint_center(cells)
                    print(
                        f"  candidate {idx}: {state_name(state)}, "
                        f"cells={len(cells)}, center=({center_r:.1f}, {center_c:.1f})"
                    )
                return

        kept = [idx for idx in self.active_indices if idx in self.current_footprints]
        self.active_indices = set((kept + eligible)[:active_count])
        for idx in self.active_indices:
            self.current_footprints.setdefault(idx, self.pool[idx])

    def _try_shift_footprint(self, cells, forbidden_cells):
        if not cells:
            return None
        distance = int(self.rng.integers(1, 4))
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.rng.shuffle(directions)
        for dr, dc in directions:
            candidate = [(r + dr * distance, c + dc * distance) for r, c in cells]
            if can_place_temporary_footprint(self.static_grid, candidate, forbidden_cells):
                return candidate
        return None

    def _try_teleport_footprint(self, cells, forbidden_cells, other_footprints=()):
        if not cells:
            return None
        height = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
        width = max(c for _, c in cells) - min(c for _, c in cells) + 1
        old_center = footprint_center(cells)
        for _ in range(TEMP_OBJECT_PLACEMENT_ATTEMPTS):
            r = int(self.rng.integers(1, max(2, self.static_grid.shape[0] - height)))
            c = int(self.rng.integers(1, max(2, self.static_grid.shape[1] - width)))
            candidate = footprint_cells((r, c), height, width)
            center = footprint_center(candidate)
            if abs(center[0] - old_center[0]) + abs(center[1] - old_center[1]) < 3:
                continue
            if any(set(candidate) & set(other) for other in other_footprints):
                continue
            if can_place_temporary_footprint(self.static_grid, candidate, forbidden_cells):
                return candidate
        return None

    def _move_footprint(self, cells, forbidden_cells):
        other = [self.current_footprints[idx][0] for idx in self.active_indices
                 if self.current_footprints.get(idx, (None,))[0] is not cells]
        preferred_shift = bool(self.rng.random() >= 0.5)
        methods = (self._try_shift_footprint, self._try_teleport_footprint) if preferred_shift else (self._try_teleport_footprint, self._try_shift_footprint)
        for method in methods:
            candidate = (method(cells, forbidden_cells, other)
                         if method.__name__ == "_try_teleport_footprint"
                         else method(cells, forbidden_cells))
            if candidate is not None:
                return candidate, "shift" if method.__name__ == "_try_shift_footprint" else "teleport"
        return cells, "unchanged"

    def should_update(self, step):
        if not ENABLE_DYNAMIC_TEMP_BLOCKAGES:
            return False

        if step <= 0:
            return False

        return step % self.change_period == 0

    def build_truth_grid(self, forbidden_cells=None):
        forbidden_cells = forbidden_cells or set()
        dynamic = self.static_grid.copy()

        active_footprints = [
            self.current_footprints.get(idx, self.pool[idx])
            for idx in sorted(self.active_indices)
        ]

        if forbidden_cells:
            filtered = []
            for cells, state in active_footprints:
                kept_cells = [cell for cell in cells if cell not in forbidden_cells]
                if kept_cells:
                    filtered.append((kept_cells, state))
            active_footprints = filtered

        return apply_temporary_footprints(dynamic, active_footprints)

    def update_world_if_needed(self, world, step, robots=None):
        if not self.should_update(step):
            return False

        forbidden_cells = robot_occupied_cells(robots)
        print(f"Changing temporary blockages at step {step}")
        self.refresh_active_blockages(forbidden_cells=forbidden_cells)
        world.grid = self.build_truth_grid(forbidden_cells=forbidden_cells)
        return True

def make_dynamic_grid_with_auto_temporary_objects(static_grid):
    dynamic = np.array(static_grid, dtype=int).copy()

    temporary_footprints = choose_temporary_object_footprints(dynamic)
    dynamic = apply_temporary_footprints(dynamic, temporary_footprints)

    print("Initial temporary runtime blockages added:")
    for idx, (cells, state) in enumerate(temporary_footprints):
        center_r, center_c = footprint_center(cells)
        print(
            f"  object {idx}: {state_name(state)}, "
            f"cells={len(cells)}, center=({center_r:.1f}, {center_c:.1f})"
        )

    return dynamic


def nearest_free_cell(world, preferred_cell, forbidden=None):
    if forbidden is None:
        forbidden = set()

    preferred_cell = tuple(preferred_cell)

    if world.can_enter(preferred_cell) and preferred_cell not in forbidden:
        return preferred_cell

    frontier = []
    heapq.heappush(frontier, (0, preferred_cell))
    visited = {preferred_cell}

    while frontier:
        _, cell = heapq.heappop(frontier)
        r, c = cell

        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            neighbor = (nr, nc)

            if neighbor in visited:
                continue

            visited.add(neighbor)

            if not world.in_bounds(neighbor):
                continue

            if world.can_enter(neighbor) and neighbor not in forbidden:
                return neighbor

            distance = abs(neighbor[0] - preferred_cell[0]) + abs(neighbor[1] - preferred_cell[1])
            heapq.heappush(frontier, (distance, neighbor))

    raise ValueError(f"No free cell found near {preferred_cell}")

def nearest_enterable_cell(world, preferred_cell, forbidden=None):
    """
    Finds the nearest cell where the robot footprint can legally fit.

    This is stricter than nearest_free_cell because the robot has a physical
    footprint, not a magical point with delusions of passing through pallets.
    """
    if forbidden is None:
        forbidden = set()

    preferred_cell = tuple(preferred_cell)

    if world.can_enter(preferred_cell) and preferred_cell not in forbidden:
        return preferred_cell

    frontier = []
    heapq.heappush(frontier, (0, preferred_cell))
    visited = {preferred_cell}

    while frontier:
        _, cell = heapq.heappop(frontier)
        r, c = cell

        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            neighbor = (nr, nc)

            if neighbor in visited:
                continue

            visited.add(neighbor)

            if not world.in_bounds(neighbor):
                continue

            if world.can_enter(neighbor) and neighbor not in forbidden:
                return neighbor

            distance = abs(neighbor[0] - preferred_cell[0]) + abs(neighbor[1] - preferred_cell[1])
            heapq.heappush(frontier, (distance, neighbor))

    raise ValueError(f"No enterable cell found near {preferred_cell}")

def has_start_clearance(world, anchor_cell, clearance=START_CLEARANCE_CELLS):
    """
    Checks whether a robot start has extra free space around its footprint.

    The anchor cell is the top-left cell of the robot footprint.
    """
    ar, ac = anchor_cell

    r_min = ar - clearance
    r_max = ar + ROBOT_FOOTPRINT_ROWS + clearance
    c_min = ac - clearance
    c_max = ac + ROBOT_FOOTPRINT_COLS + clearance

    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            cell = (r, c)

            if not world.in_bounds(cell):
                return False

            if world.is_truth_blocked(cell):
                return False

    return True


def nearest_safe_start_cell(world, preferred_cell, forbidden=None):
    """
    Finds a start cell where the robot footprint fits and has a little clearance.

    Falls back to nearest_enterable_cell if the map is too cramped.
    """
    if forbidden is None:
        forbidden = set()

    preferred_cell = tuple(preferred_cell)

    best_fallback = None

    for radius in range(START_SEARCH_MAX_RADIUS + 1):
        candidates = []

        pr, pc = preferred_cell

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) != radius:
                    continue

                cell = (pr + dr, pc + dc)

                if cell in forbidden:
                    continue

                if not world.in_bounds(cell):
                    continue

                if not world.can_enter(cell):
                    continue

                if best_fallback is None:
                    best_fallback = cell

                if has_start_clearance(world, cell):
                    candidates.append(cell)

        if candidates:
            # Pick the closest candidate by Manhattan distance.
            candidates.sort(
                key=lambda cell: (
                    abs(cell[0] - preferred_cell[0]) + abs(cell[1] - preferred_cell[1]),
                    cell[0],
                    cell[1],
                )
            )
            return candidates[0]

    if best_fallback is not None:
        return best_fallback

    return nearest_enterable_cell(world, preferred_cell, forbidden=forbidden)

def choose_spread_out_free_cells(world, count, forbidden=None):
    """
    Choose free cells spread across the map.

    This creates synthetic action points when the map does not already contain
    PICKUP/DROPOFF/CHARGING semantic cells.
    """
    if forbidden is None:
        forbidden = set()

    free_cells = [
        cell
        for cell in find_free_cells(world.grid)
        if cell not in forbidden
    ]

    if not free_cells:
        raise ValueError("No free cells available for action points.")

    anchors = [
        (world.rows // 5, world.cols // 5),
        (world.rows // 5, world.cols // 2),
        (world.rows // 5, 4 * world.cols // 5),
        (world.rows // 2, world.cols // 5),
        (world.rows // 2, 4 * world.cols // 5),
        (4 * world.rows // 5, world.cols // 5),
        (4 * world.rows // 5, world.cols // 2),
        (4 * world.rows // 5, 4 * world.cols // 5),
    ]

    selected = []
    used = set(forbidden)

    for anchor in anchors:
        if len(selected) >= count:
            break

        try:
            cell = nearest_enterable_cell(world, anchor, forbidden=used)
        except ValueError:
            continue

        selected.append(cell)
        used.add(cell)

    # If anchors were not enough, fill from remaining free cells.
    for cell in free_cells:
        if not world.can_enter(cell):
            continue
        if len(selected) >= count:
            break

        if cell not in used:
            selected.append(cell)
            used.add(cell)

    return selected

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def count_blocked_nearby(world, cell, radius):
    """
    Counts blocked cells near a candidate action point.

    More nearby obstacles means the point is closer to shelves, corners,
    narrow aisles, or awkward warehouse geometry. Humanity calls this
    "strategic placement" after inventing congestion and then acting surprised.
    """
    cr, cc = cell
    count = 0

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) > radius:
                continue

            neighbor = (cr + dr, cc + dc)

            if not world.in_bounds(neighbor):
                count += 1
                continue

            if world.is_truth_blocked(neighbor):
                count += 1

    return count


def corner_pressure_score(world, cell, radius=ACTION_POINT_CORNER_RADIUS):
    """
    Scores whether a cell is near an obstacle corner or shelf end.

    A good 'corner-ish' goal is not inside a wall, but it has blocked structure
    in both row and column directions nearby. This tends to select shelf ends,
    alcoves, and edge-adjacent access points instead of boring open floors.
    """
    r, c = cell

    vertical_blocked = 0
    horizontal_blocked = 0

    for offset in range(1, radius + 1):
        up = (r - offset, c)
        down = (r + offset, c)
        left = (r, c - offset)
        right = (r, c + offset)

        if not world.in_bounds(up) or world.is_truth_blocked(up):
            vertical_blocked += 1
        if not world.in_bounds(down) or world.is_truth_blocked(down):
            vertical_blocked += 1
        if not world.in_bounds(left) or world.is_truth_blocked(left):
            horizontal_blocked += 1
        if not world.in_bounds(right) or world.is_truth_blocked(right):
            horizontal_blocked += 1

    return min(vertical_blocked, horizontal_blocked)


def edge_score(world, cell):
    """
    Higher score near map edges, lower score near the center.
    """
    r, c = cell

    distance_to_edge = min(
        r,
        c,
        world.rows - 1 - r,
        world.cols - 1 - c,
    )

    max_possible = max(1, min(world.rows, world.cols) // 2)

    return 1.0 - min(1.0, distance_to_edge / max_possible)


def is_good_action_point_candidate(world, cell, forbidden):
    """
    Candidate must be enterable by the robot footprint and not already reserved.
    Also avoids placing goals too close to the hard boundary.
    """
    if cell in forbidden:
        return False

    r, c = cell
    margin = ACTION_POINT_EDGE_MARGIN_CELLS

    if r < margin or r >= world.rows - margin:
        return False

    if c < margin or c >= world.cols - margin:
        return False

    if not world.can_enter(cell):
        return False

    return True


def choose_strategic_action_points(world, count, forbidden=None):
    """
    Chooses synthetic action points that are:
    - near map edges,
    - near obstacle corners / shelf ends,
    - spread across the map,
    - still reachable by the robot footprint.

    This replaces naive open-space goal placement with map-aware placement.
    It generalizes to any occupancy grid because it scores geometry instead
    of hardcoding coordinates.
    """
    if forbidden is None:
        forbidden = set()

    candidates = []

    for r in range(world.rows):
        for c in range(world.cols):
            cell = (r, c)

            if not is_good_action_point_candidate(world, cell, forbidden):
                continue

            obstacle_score = count_blocked_nearby(
                world,
                cell,
                ACTION_POINT_OBSTACLE_RADIUS,
            )

            score = (
                ACTION_POINT_EDGE_WEIGHT * edge_score(world, cell)
                + ACTION_POINT_CORNER_WEIGHT * corner_pressure_score(world, cell)
                + ACTION_POINT_OBSTACLE_PROXIMITY_WEIGHT * obstacle_score
            )

            candidates.append((score, cell))

    if not candidates:
        raise ValueError("No valid strategic action-point candidates found.")

    candidates.sort(reverse=True)

    selected = []
    used = set(forbidden)

    min_spacing = max(
        2,
        int(min(world.rows, world.cols) * ACTION_POINT_MIN_SPACING_RATIO),
    )

    while candidates and len(selected) < count:
        best_index = None
        best_total_score = -float("inf")

        for idx, (base_score, cell) in enumerate(candidates):
            if cell in used:
                continue

            if selected:
                nearest_selected_distance = min(
                    manhattan(cell, other)
                    for other in selected
                )
            else:
                nearest_selected_distance = min(world.rows, world.cols)

            if nearest_selected_distance < min_spacing:
                continue

            spread_score = ACTION_POINT_SPREAD_WEIGHT * nearest_selected_distance

            total_score = base_score + spread_score

            if total_score > best_total_score:
                best_total_score = total_score
                best_index = idx

        if best_index is None:
            # Relax spacing if the map is cramped. Warehouses are apparently
            # designed by people who hate graph search.
            min_spacing = max(1, min_spacing - 1)

            if min_spacing <= 1:
                break

            continue

        _, chosen = candidates.pop(best_index)
        selected.append(chosen)
        used.add(chosen)

    if len(selected) < count:
        for _, cell in candidates:
            if len(selected) >= count:
                break

            if cell in used:
                continue

            selected.append(cell)
            used.add(cell)

    if len(selected) < count:
        raise ValueError(
            f"Only found {len(selected)} strategic action points, requested {count}."
        )

    return selected

# ============================================================
# Message model
# ============================================================

@dataclass
class PeerReport:
    sender_id: int
    target_cell: tuple
    claim: ClaimType
    timestamp: int
    confidence: float = 1.0
    is_malicious: bool = False


@dataclass
class PendingClaim:
    sender_id: int
    target_cell: tuple
    claim: ClaimType
    timestamp: int
    is_malicious: bool = False

@dataclass
class DeliveryTask:
    pickup: tuple
    dropoff: tuple


class TrustModel:
    """
    Base interface for robot-to-robot trust.

    Swap subclasses to change trust behavior without rewriting GridRobot.
    """

    def score(self, sender_id):
        raise NotImplementedError

    def should_accept(self, report):
        raise NotImplementedError

    def observe_claim(self, report):
        """
        Called when a report is accepted and stored for later verification.
        """
        pass

    def verify_claim(self, claim, truth_matches):
        """
        Called when the robot later directly senses the reported cell.
        """
        raise NotImplementedError

    def snapshot(self):
        """
        Returns loggable trust state.
        """
        return {}


class ScalarTrustModel(TrustModel):
    """
    Current trust behavior:
    - every unknown sender starts at initial trust
    - true reports increase trust
    - false reports decrease trust
    - reports below threshold are rejected
    """

    def __init__(
        self,
        self_id,
        initial_value=TRUST_INITIAL_VALUE,
        accept_threshold=TRUST_ACCEPT_THRESHOLD,
        reward=TRUST_REWARD,
        penalty=TRUST_PENALTY,
    ):
        self.self_id = int(self_id)
        self.initial_value = float(initial_value)
        self.accept_threshold = float(accept_threshold)
        self.reward = float(reward)
        self.penalty = float(penalty)
        self.values = {}

    def score(self, sender_id):
        sender_id = int(sender_id)

        if sender_id == self.self_id:
            return 1.0

        if sender_id not in self.values:
            self.values[sender_id] = self.initial_value

        return self.values[sender_id]

    def should_accept(self, report):
        return self.score(report.sender_id) >= self.accept_threshold

    def verify_claim(self, claim, truth_matches):
        sender_id = int(claim.sender_id)

        if sender_id == self.self_id:
            return

        current = self.score(sender_id)

        if truth_matches:
            self.values[sender_id] = min(1.0, current + self.reward)
        else:
            self.values[sender_id] = max(0.0, current - self.penalty)

    def snapshot(self):
        return copy.deepcopy(self.values)
    

class BayesianTrustModel(TrustModel):
    """
    Beta reputation model:
    trust = alpha / (alpha + beta)

    True verified reports increase alpha.
    False verified reports increase beta.
    """

    def __init__(
        self,
        self_id,
        prior_alpha=9.0,
        prior_beta=1.0,
        accept_threshold=TRUST_ACCEPT_THRESHOLD,
    ):
        self.self_id = int(self_id)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.accept_threshold = float(accept_threshold)
        self.params = {}

    def _ensure(self, sender_id):
        sender_id = int(sender_id)

        if sender_id not in self.params:
            self.params[sender_id] = [
                self.prior_alpha,
                self.prior_beta,
            ]

        return self.params[sender_id]

    def score(self, sender_id):
        sender_id = int(sender_id)

        if sender_id == self.self_id:
            return 1.0

        alpha, beta = self._ensure(sender_id)
        return alpha / (alpha + beta)

    def should_accept(self, report):
        return self.score(report.sender_id) >= self.accept_threshold

    def verify_claim(self, claim, truth_matches):
        sender_id = int(claim.sender_id)

        if sender_id == self.self_id:
            return

        alpha, beta = self._ensure(sender_id)

        if truth_matches:
            alpha += 1.0
        else:
            beta += 1.0

        self.params[sender_id] = [alpha, beta]

    def snapshot(self):
        return {
            sender_id: {
                "alpha": alpha,
                "beta": beta,
                "score": alpha / (alpha + beta),
            }
            for sender_id, (alpha, beta) in self.params.items()
        }
    
def make_trust_model(robot_id):
    if TRUST_MODEL_NAME == "scalar":
        return ScalarTrustModel(self_id=robot_id)

    if TRUST_MODEL_NAME == "bayesian":
        return BayesianTrustModel(
            self_id=robot_id,
            prior_alpha=TRUST_BAYES_PRIOR_ALPHA,
            prior_beta=TRUST_BAYES_PRIOR_BETA,
        )

    raise ValueError(f"Unknown TRUST_MODEL_NAME: {TRUST_MODEL_NAME}")

# ============================================================
# Grid world
# ============================================================

class GridWorld:
    """
    Ground-truth grid world.

    The simulator knows this full truth.
    Robots do not automatically know this full truth.
    Humanity calls this "partial observability" because "robots are not psychic"
    sounded too emotionally honest.
    """

    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)

        if self.grid.ndim != 2:
            raise ValueError("grid must be a 2D array")

        self.rows, self.cols = self.grid.shape

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_truth_blocked(self, cell):
        if not self.in_bounds(cell):
            return True

        r, c = cell
        state = CellState(int(self.grid[r, c]))

        return state in (
            CellState.OCCUPIED_STATIC,
            CellState.OCCUPIED_DYNAMIC,
            CellState.TEMPORARILY_BLOCKED,
        )

    def truth_state(self, cell):
        """
        Returns the ground-truth cell state from the single dynamic map.
        """
        if not self.in_bounds(cell):
            return CellState.OCCUPIED_DYNAMIC

        r, c = cell
        return CellState(int(self.grid[r, c]))

    def can_enter(self, cell, robot_positions=None):
        """
        Checks whether a robot footprint centered at cell can enter.

        The robot is not a point anymore. Revolutionary. Distressing that this
        needed saying, but useful.
        """
        footprint = robot_footprint_cells(cell)

        for footprint_cell in footprint:
            if not self.in_bounds(footprint_cell):
                return False

            if self.is_truth_blocked(footprint_cell):
                return False

            if robot_positions is not None and footprint_cell in robot_positions:
                return False

        return True

    def set_dynamic_state(self, cell, state):
        if not self.in_bounds(cell):
            return

        r, c = cell
        self.grid[r, c] = int(state)

    def observe_cells(self, center_cell, radius, robot_positions=None):
        """
        Grid-level sensor model.

        Robots observe nearby map cells by radius.
        They also receive global positions of all other robots, so agents know
        robot occupancy at all times.
        """
        observations = {}

        cr, cc = center_cell

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue

                cell = (cr + dr, cc + dc)

                if not self.in_bounds(cell):
                    continue

                observations[cell] = self.truth_state(cell)

        if robot_positions is not None:
            for cell in robot_positions:
                if not self.in_bounds(cell):
                    continue

                if cell == center_cell:
                    continue

                observations[cell] = CellState.OCCUPIED_DYNAMIC

        return observations

    def observe_cells_lidar(
        self,
        position_xy,
        max_range_cells=LIDAR_RANGE_CELLS,
        num_rays=LIDAR_NUM_RAYS,
        step_cells=LIDAR_STEP_CELLS,
        robot_positions=None,
    ):
        """
        Ray-casted 2D lidar sensor.

        Each ray starts at the robot's continuous position and travels outward.
        Cells along the ray are observed as free/known until the ray hits a
        blocking cell or reaches max range.

        This creates line-of-sight sensing with occlusion. The robot does not
        magically observe cells behind obstacles. A tragic loss for omniscience,
        but a win for realism.
        """
        observations = {}
        rays = []

        origin = np.array(position_xy, dtype=float)
        max_range_world = float(max_range_cells) * CELL_SIZE
        step_world = float(step_cells) * CELL_SIZE

        for ray_idx in range(num_rays):
            angle = 2.0 * math.pi * ray_idx / num_rays
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)

            ray_points = []
            visited_cells = set()
            distance = 0.0

            while distance <= max_range_world:
                point = origin + direction * distance
                cell = xy_to_cell(point)

                ray_points.append(tuple(point.tolist()))

                # Map edge is known, but outside the map is not a real cell
                # to insert into the belief grid.
                if not self.in_bounds(cell):
                    break

                if cell not in visited_cells:
                    visited_cells.add(cell)

                    state = self.truth_state(cell)
                    observations[cell] = state

                    if self.is_truth_blocked(cell):
                        break

                distance += step_world

            rays.append(ray_points)

        if robot_positions is not None:
            for cell in robot_positions:
                if not self.in_bounds(cell):
                    continue

                # Other robots are only added if they are inside lidar range.
                other_xy = cell_to_xy(cell)
                if np.linalg.norm(other_xy - origin) <= max_range_world:
                    observations[cell] = CellState.OCCUPIED_DYNAMIC

        return observations, rays

# ============================================================
# Belief map
# ============================================================

class RobotBeliefMap:
    """
    Per-robot grid-level belief.

    This is the important conceptual shift.

    The world has ground truth.
    Each robot has its own belief about that truth.
    Peer reports modify belief only if trust allows it.
    """

    def __init__(self, initial_grid):
        initial_grid = np.array(initial_grid, dtype=int)

        self.rows, self.cols = initial_grid.shape

        self.initial_prior = initial_grid.copy()

        self.belief = np.full(
            (self.rows, self.cols),
            CellState.UNKNOWN,
            dtype=int,
        )

        self.confidence = np.zeros((self.rows, self.cols), dtype=float)
        self.source = np.full((self.rows, self.cols), "unknown", dtype=object)
        self.last_updated = np.full((self.rows, self.cols), -1, dtype=int)

        self.last_decayed = np.full((self.rows, self.cols), -1, dtype=int)

        # Set by GridRobot after its trust model is constructed. The defense
        # runner owns peer-claim evidence; this belief map retains direct sensor
        # observations and the static prior.
        self.defense_runner = None
        self.current_timestamp = 0

        self._initialize_from_static_prior()

    def _initialize_from_static_prior(self):
        for r in range(self.rows):
            for c in range(self.cols):
                state = CellState(int(self.initial_prior[r, c]))

                if state in (
                    CellState.OCCUPIED_STATIC,
                    CellState.OCCUPIED_DYNAMIC,
                    CellState.TEMPORARILY_BLOCKED,
                ):
                    self.belief[r, c] = state
                    self.confidence[r, c] = 1.0
                    self.source[r, c] = "initial_map"

                elif state in (
                    CellState.PICKUP,
                    CellState.DROPOFF,
                    CellState.CHARGING,
                ):
                    self.belief[r, c] = state
                    self.confidence[r, c] = 0.95
                    self.source[r, c] = "initial_map"

                else:
                    self.belief[r, c] = CellState.FREE
                    self.confidence[r, c] = 0.65
                    self.source[r, c] = "initial_map"

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def update_from_sensor(self, observations, timestamp):
        """
        Direct sensing has high confidence.

        A robot should believe its own local sensor more than robot gossip.
        """
        changed = []

        for cell, observed_state in observations.items():
            if not self.in_bounds(cell):
                continue

            r, c = cell
            observed_state = CellState(int(observed_state))

            old_state = CellState(int(self.belief[r, c]))

            self.belief[r, c] = observed_state
            self.confidence[r, c] = 1.0
            self.source[r, c] = "self_sensor"
            self.last_updated[r, c] = timestamp
            self.last_decayed[r, c] = timestamp

            if old_state != observed_state:
                changed.append((cell, old_state, observed_state))

        return changed

    def apply_peer_report(self, report, trust_score):
        """
        Trust-weighted belief update.

        High-trust reports can mark cells blocked/free.
        Reports remain available to the defense runner for audit and trust
        verification even when their current operational influence is zero.
        """
        cell = report.target_cell

        if not self.in_bounds(cell):
            return False

        r, c = cell

        old_conf = float(self.confidence[r, c])

        if report.claim == ClaimType.BLOCKED:
            claimed_state = CellState.TEMPORARILY_BLOCKED
        elif report.claim == ClaimType.CONGESTED:
            claimed_state = CellState.CONGESTED
        else:
            claimed_state = CellState.FREE

        # Known map obstacles cannot be overwritten by peer gossip.
        if CellState(int(self.initial_prior[r, c])) in (
            CellState.OCCUPIED_STATIC,
            CellState.OCCUPIED_DYNAMIC,
            CellState.TEMPORARILY_BLOCKED,
        ):
            return False

        # Fresh direct sensing remains authoritative. Operational route
        # arbitration is handled by the defense runner and belief map.
        if self.source[r, c] == "self_sensor" and trust_score < 0.95:
            return False

        new_conf = max(old_conf * 0.90, min(0.95, trust_score))

        self.belief[r, c] = claimed_state
        self.confidence[r, c] = new_conf
        self.source[r, c] = f"peer_{report.sender_id}"
        self.last_updated[r, c] = report.timestamp
        self.last_decayed[r, c] = report.timestamp

        return True

    def apply_confidence_decay(self, timestamp):
        """
        Confidence decays for non-static, non-semantic information.

        Important distinction:
        - last_updated = when the belief was last observed/reported
        - last_decayed = when confidence was last numerically decayed

        This prevents repeated over-decay using the full age every step.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                prior_state = CellState(int(self.initial_prior[r, c]))

                if prior_state in (
                    CellState.OCCUPIED_STATIC,
                    CellState.PICKUP,
                    CellState.DROPOFF,
                    CellState.CHARGING,
                ):
                    continue

                last_seen = int(self.last_updated[r, c])

                if last_seen < 0:
                    continue

                current_state = CellState(int(self.belief[r, c]))
                current_source = str(self.source[r, c])

                observation_age = max(0, timestamp - last_seen)

                last_decayed = int(self.last_decayed[r, c])
                if last_decayed < 0:
                    last_decayed = last_seen

                decay_steps = max(0, timestamp - last_decayed)

                if decay_steps <= 0:
                    continue

                # Directly observed blockages should persist for a while.
                if current_source == "self_sensor" and current_state in (
                    CellState.TEMPORARILY_BLOCKED,
                    CellState.OCCUPIED_DYNAMIC,
                ):
                    if observation_age <= SELF_BLOCK_MEMORY_STEPS:
                        self.last_decayed[r, c] = timestamp
                        continue

                # Peer-reported blockages persist, but less than self-observed ones.
                if current_source.startswith("peer_") and current_state in (
                    CellState.TEMPORARILY_BLOCKED,
                    CellState.OCCUPIED_DYNAMIC,
                ):
                    if observation_age <= PEER_BLOCK_MEMORY_STEPS:
                        self.last_decayed[r, c] = timestamp
                        continue

                # Free-space memory should expire faster because the world can change.
                if current_state in (
                    CellState.FREE,
                    CellState.UNKNOWN,
                    CellState.CONGESTED,
                ):
                    if observation_age <= FREE_MEMORY_STEPS:
                        self.last_decayed[r, c] = timestamp
                        continue

                old_conf = float(self.confidence[r, c])
                new_conf = old_conf * (CONFIDENCE_DECAY_PER_STEP ** decay_steps)

                self.confidence[r, c] = new_conf
                self.last_decayed[r, c] = timestamp

                if new_conf < CONFIDENCE_UNKNOWN_THRESHOLD:
                    self.belief[r, c] = CellState.UNKNOWN
                    self.source[r, c] = "decayed_unknown"

    def attach_defense_runner(self, defense_runner):
        self.defense_runner = defense_runner

    def set_planning_time(self, timestamp):
        self.current_timestamp = int(timestamp)
        if self.defense_runner is not None:
            self.defense_runner.set_time(timestamp)

    def _direct_free_observation(self, cell):
        r, c = tuple(cell)
        return (
            self.source[r, c] == "self_sensor"
            and CellState(int(self.belief[r, c])) in (
                CellState.FREE,
                CellState.PICKUP,
                CellState.DROPOFF,
                CellState.CHARGING,
            )
        )

    def direct_free_strength(self, cell, timestamp=None):
        if not self._direct_free_observation(cell):
            return 0.0
        r, c = tuple(cell)
        now = self.current_timestamp if timestamp is None else int(timestamp)
        age = max(0, now - int(self.last_updated[r, c]))
        return 1.25 * float(self.confidence[r, c]) * math.exp(-0.01 * age)

    def footprint_is_peer_hard_blocked(self, footprint, timestamp=None):
        """Peer hard blocks apply only where direct sensing has not verified free."""
        if self.defense_runner is None:
            return False

        now = self.current_timestamp if timestamp is None else int(timestamp)
        for footprint_cell in footprint:
            if self._direct_free_observation(footprint_cell) and self.defense_runner.method != "trust_threshold":
                continue
            if self._direct_free_observation(footprint_cell) and self.defense_runner.blocked_support(footprint_cell, now) <= self.direct_free_strength(footprint_cell, now):
                continue
            if self.defense_runner.is_hard_blocked(footprint_cell, now):
                return True
        return False

    def is_blocked_for_planning(self, cell):
        footprint = robot_footprint_cells(cell)

        for footprint_cell in footprint:
            if not self.in_bounds(footprint_cell):
                return True

            r, c = footprint_cell
            state = CellState(int(self.belief[r, c]))

            # Static prior and directly observed physical obstacles remain hard
            # constraints. Peer reports are handled separately by the selected
            # defense method instead of being written into this grid.
            if state == CellState.OCCUPIED_STATIC:
                return True

            if state in (
                CellState.OCCUPIED_DYNAMIC,
                CellState.TEMPORARILY_BLOCKED,
            ) and self.source[r, c] in ("initial_map", "self_sensor"):
                return True

        if self.footprint_is_peer_hard_blocked(footprint):
            return True

        return False

    def traversal_cost(self, cell):
        """
        Cost used by A*.

        Unknown/congested cells are not impossible, just undesirable.
        Blocked cells are impossible.
        """
        if not self.in_bounds(cell):
            return float("inf")

        if self.is_blocked_for_planning(cell):
            return float("inf")

        max_cost = 1.0

        for footprint_cell in robot_footprint_cells(cell):
            if not self.in_bounds(footprint_cell):
                return float("inf")

            r, c = footprint_cell
            state = CellState(int(self.belief[r, c]))

            if state == CellState.UNKNOWN:
                max_cost = max(max_cost, 3.0)

        if self.defense_runner is not None:
            # A recipient's fresh local FREE observation is authoritative for
            # that cell.  Peer evidence remains stored for audit/trust, but it
            # must not keep adding a routing penalty after direct verification.
            # Directly observed blocks were already returned as hard blocks by
            # ``is_blocked_for_planning`` above.
            peer_cost = 1.0
            for footprint_cell in robot_footprint_cells(cell):
                if self._direct_free_observation(footprint_cell) and self.defense_runner.method != "trust_threshold":
                    continue
                contribution = self.defense_runner.routing_cost(footprint_cell, self.current_timestamp)
                if self._direct_free_observation(footprint_cell):
                    ratio = min(1.0, self.defense_runner.blocked_support(footprint_cell, self.current_timestamp) / max(self.direct_free_strength(footprint_cell, self.current_timestamp), 1e-9))
                    contribution = 1.0 + (contribution - 1.0) * ratio
                peer_cost = max(peer_cost, contribution)
            if math.isinf(peer_cost):
                return float("inf")
            max_cost = max(max_cost, peer_cost)

        return max_cost

    def display_grid(self):
        """
        Converts belief states into a display-friendly grid.
        """
        return self.belief.copy()


# ============================================================
# A* planner, 4-direction only
# ============================================================

class AStarPlanner4:
    """
    A* over a grid-level belief map.

    Movement is only 4-direction:
        up, down, left, right

    No diagonal movement. The robot is not a bishop in a chess set.
    """

    @staticmethod
    def neighbors_4(cell):
        r, c = cell
        return [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]

    @staticmethod
    def heuristic(a, b):
        # Manhattan distance is appropriate for 4-direction movement.
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, belief_map, start, goal):
        if not belief_map.in_bounds(start):
            raise RuntimeError(f"start out of bounds: {start}")

        if not belief_map.in_bounds(goal):
            raise RuntimeError(f"goal out of bounds: {goal}")

        if belief_map.is_blocked_for_planning(start):
            raise RuntimeError(f"start is blocked in belief map: {start}")

        if belief_map.is_blocked_for_planning(goal):
            raise RuntimeError(f"goal is blocked in belief map: {goal}")

        frontier = []
        heapq.heappush(frontier, (0.0, start))

        came_from = {start: None}
        cost_so_far = {start: 0.0}

        expanded_nodes = 0

        while frontier:
            _, current = heapq.heappop(frontier)
            expanded_nodes += 1

            if current == goal:
                break

            for neighbor in self.neighbors_4(current):
                step_cost = belief_map.traversal_cost(neighbor)

                if math.isinf(step_cost):
                    continue

                new_cost = cost_so_far[current] + step_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            raise RuntimeError("A* failed to find a route")

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()

        stats = {
            "expanded_nodes": expanded_nodes,
            "visited_cells": len(cost_so_far),
            "path_cells": len(path),
            "path_cost": cost_so_far[goal],
        }

        return path, stats

def route_exists_for_prior(prior_grid, start, goal):
    """
    Returns True only if A* can route from start to goal using the static prior.

    This checks real connectivity for the robot footprint, not just whether
    the endpoint cell itself is enterable.
    """
    belief_map = RobotBeliefMap(prior_grid)
    planner = AStarPlanner4()

    try:
        planner.plan(belief_map, tuple(start), tuple(goal))
        return True
    except RuntimeError:
        return False

def filter_reachable_action_points(action_points, robot_specs, prior_grid):
    """
    Removes action points that no robot can reach from its starting region.

    A goal being enterable is not enough. If no robot can route to it with
    the current robot-footprint planner, it should not be used as a pickup/dropoff
    goal.
    """
    filtered = []

    for point in action_points:
        point = tuple(point)

        reachable_by = []

        for spec in robot_specs:
            robot_id = int(spec["robot_id"])
            start = tuple(spec["start"])

            if route_exists_for_prior(prior_grid, start, point):
                reachable_by.append(robot_id)

        if reachable_by:
            filtered.append(point)
        else:
            print(
                f"Removing unreachable action point {point}: "
                f"no robot start can reach it"
            )

    if len(filtered) < 2:
        raise RuntimeError(
            f"Only {len(filtered)} reachable action points remain. "
            f"Need at least 2 to build pickup/dropoff tasks."
        )

    return filtered


def relocate_starts_for_goals(world, robot_specs, goals, prior_grid):
    """Ensure every robot can reach at least two of the retained goals."""
    used_starts = {tuple(spec["start"]) for spec in robot_specs}
    forbidden_goals = set(goals)
    free_cells = find_free_cells(prior_grid)

    for spec in robot_specs:
        start = tuple(spec["start"])
        reachable_count = sum(
            route_exists_for_prior(prior_grid, start, goal)
            for goal in goals
        )
        if reachable_count >= 2:
            continue

        used_starts.discard(start)
        replacement = None
        candidates = sorted(
            free_cells,
            key=lambda cell: manhattan(tuple(cell), start),
        )

        for candidate in candidates:
            try:
                safe_candidate = nearest_safe_start_cell(
                    world,
                    candidate,
                    forbidden=used_starts.union(forbidden_goals),
                )
            except ValueError:
                continue

            if safe_candidate in used_starts:
                continue

            candidate_reachability = sum(
                route_exists_for_prior(prior_grid, safe_candidate, goal)
                for goal in goals
            )
            if candidate_reachability >= 2:
                replacement = safe_candidate
                break

        if replacement is None:
            raise RuntimeError(
                f"Robot {spec['robot_id']} has no start with access to "
                "at least two retained goals"
            )

        spec["start"] = replacement
        used_starts.add(replacement)
        print(
            f"Robot {spec['robot_id']} start relocated for retained goals: "
            f"{start} -> {replacement}"
        )

def plan_to_reachable_fallback(
    belief_map,
    start,
    goal,
    min_waypoint_distance=FALLBACK_MIN_WAYPOINT_DISTANCE,
):
    """
    Finds a reachable fallback waypoint when the real goal is not reachable.

    The waypoint is selected from the robot's currently reachable region.
    Preference:
    1. reachable cells closest to the real goal,
    2. not too close to the current position,
    3. otherwise, any reachable cell that causes movement.

    This keeps the robot moving and sensing instead of freezing forever.
    """
    if not belief_map.in_bounds(start):
        raise RuntimeError(f"fallback start out of bounds: {start}")

    if belief_map.is_blocked_for_planning(start):
        raise RuntimeError(f"fallback start is blocked: {start}")

    frontier = []
    heapq.heappush(frontier, (0.0, start))

    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)

        for neighbor in AStarPlanner4.neighbors_4(current):
            step_cost = belief_map.traversal_cost(neighbor)

            if math.isinf(step_cost):
                continue

            new_cost = cost_so_far[current] + step_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

    reachable_cells = [
        cell for cell in came_from
        if cell != start
    ]

    if not reachable_cells:
        raise RuntimeError("no reachable fallback cells")

    useful_cells = [
        cell for cell in reachable_cells
        if manhattan(cell, start) >= min_waypoint_distance
    ]

    if not useful_cells:
        useful_cells = reachable_cells

    # Prefer reachable cells that get us closer to the real goal.
    # Tie-break by choosing cells farther from the current position so the
    # robot actually moves instead of twitching in place like cursed machinery.
    waypoint = min(
        useful_cells,
        key=lambda cell: (
            manhattan(cell, goal),
            -manhattan(cell, start),
            cost_so_far[cell],
        ),
    )

    path = []
    current = waypoint

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()

    stats = {
        "fallback": True,
        "reachable_cells": len(reachable_cells),
        "path_cells": len(path),
        "waypoint": waypoint,
        "distance_to_real_goal": manhattan(waypoint, goal),
    }

    return path, stats

def is_good_distance_for_fake_block(victim_robot, cell):
    """
    A useful fake blockage should be outside immediate lidar verification range,
    but not so far away that it stops mattering.

    This creates a practical attack window:
    - too close: victim verifies it immediately
    - too far: route may change before the victim reaches it
    - middle distance: believable and disruptive
    """
    distance = manhattan(victim_robot.position, cell)

    return (
        ATTACK_MIN_DISTANCE_FROM_VICTIM
        <= distance
        <= ATTACK_MAX_DISTANCE_FROM_VICTIM
    )


def is_known_static_or_blocked_prior(victim_robot, cell):
    """
    Do not waste a malicious report on something the victim already knows is blocked.
    """
    if not victim_robot.belief_map.in_bounds(cell):
        return True

    r, c = cell
    prior_state = CellState(int(victim_robot.belief_map.initial_prior[r, c]))

    return prior_state in (
        CellState.OCCUPIED_STATIC,
        CellState.OCCUPIED_DYNAMIC,
        CellState.TEMPORARILY_BLOCKED,
    )


def unique_cells_in_order(cells):
    seen = set()
    result = []

    for cell in cells:
        cell = tuple(cell)

        if cell in seen:
            continue

        seen.add(cell)
        result.append(cell)

    return result


def traffic_heatmap_score(cell, traffic_heatmap):
    r, c = cell

    if r < 0 or r >= traffic_heatmap.shape[0]:
        return 0

    if c < 0 or c >= traffic_heatmap.shape[1]:
        return 0

    return int(traffic_heatmap[r, c])

def positive_traffic_values(traffic_heatmap):
    values = traffic_heatmap[traffic_heatmap > 0]

    if values.size == 0:
        return values

    return values.astype(float)


def average_traffic_bounds(traffic_heatmap):
    """
    Returns the low/high traffic range for average-traveled attack targets.

    We intentionally avoid:
    - zero traffic cells, because nobody uses them,
    - extreme hotspot cells, because they are likely to be verified quickly.
    """
    values = positive_traffic_values(traffic_heatmap)

    if values.size == 0:
        return None, None

    low = np.percentile(values, ATTACK_TRAFFIC_LOW_PERCENTILE)
    high = np.percentile(values, ATTACK_TRAFFIC_HIGH_PERCENTILE)

    return low, high


def is_average_traffic_cell(cell, traffic_heatmap):
    low, high = average_traffic_bounds(traffic_heatmap)

    if low is None or high is None:
        return False

    score = traffic_heatmap_score(cell, traffic_heatmap)

    return low <= score <= high


def fake_object_footprint_cells(center_cell, height=MALICIOUS_FAKE_OBJECT_ROWS,
                                width=MALICIOUS_FAKE_OBJECT_COLS):
    """
    Builds a rectangular fake object footprint around a center cell.

    The footprint is allowed to overlap walls visually. That is okay because
    this is a malicious belief-map artifact, not a physical truth object.
    """
    center_r, center_c = center_cell

    row_start = center_r - height // 2
    col_start = center_c - width // 2

    cells = []

    for dr in range(height):
        for dc in range(width):
            cells.append((row_start + dr, col_start + dc))

    return cells


def can_report_fake_block_cell(cell, world, goals, robots):
    """
    Returns True if this cell can be part of the malicious reported blockage.

    We allow the fake object's visual footprint to overlap walls, but we only
    send reports for cells that can actually change a victim's belief.
    """
    if not world.in_bounds(cell):
        return False

    # Do not send reports for known static/dynamic blocked truth cells.
    # Victims already know many of these from the static prior, and peer reports
    # on those cells will be rejected by apply_peer_report.
    if world.is_truth_blocked(cell):
        return False

    if is_near_any_goal(cell, goals):
        return False

    if cell_occupied_by_benign_robot(cell, robots):
        return False

    return True


def fake_object_report_cells(center_cell, world, goals, robots, height=MALICIOUS_FAKE_OBJECT_ROWS,
                             width=MALICIOUS_FAKE_OBJECT_COLS):
    """
    Returns the subset of the fake object footprint that should actually be
    reported as BLOCKED.

    The full rectangle may overlap walls for visualization, but reports only go
    to usable cells where a victim can update its belief.
    """
    report_cells = []

    for cell in fake_object_footprint_cells(center_cell, height, width):
        if can_report_fake_block_cell(cell, world, goals, robots):
            report_cells.append(cell)

    return report_cells


def fake_object_average_traffic_score(center_cell, traffic_heatmap, height=MALICIOUS_FAKE_OBJECT_ROWS,
                                      width=MALICIOUS_FAKE_OBJECT_COLS):
    """
    Scores the fake object by average traffic over reportable footprint cells.
    """
    cells = fake_object_footprint_cells(center_cell, height, width)

    scores = [
        traffic_heatmap_score(cell, traffic_heatmap)
        for cell in cells
        if (
            0 <= cell[0] < traffic_heatmap.shape[0]
            and 0 <= cell[1] < traffic_heatmap.shape[1]
        )
    ]

    if not scores:
        return 0.0

    return float(np.mean(scores))

def is_near_any_goal(cell, goals):
    for goal in goals:
        if manhattan(cell, tuple(goal)) < ATTACK_MIN_DISTANCE_FROM_GOAL:
            return True

    return False


def is_near_any_benign_robot(cell, robots):
    for robot in robots:
        if robot.is_malicious:
            continue

        if manhattan(cell, robot.position) < ATTACK_MIN_DISTANCE_FROM_ANY_BENIGN_ROBOT:
            return True

    return False

def is_near_previous_fake_object_center(cell, placed_centers):
    for center in placed_centers:
        if manhattan(cell, center) < MALICIOUS_FAKE_OBJECT_CENTER_MIN_SPACING:
            return True

    return False

def recon_coverage_satisfied(recon_goal_visit_counts):
    if not recon_goal_visit_counts:
        return True

    visited_count = sum(
        1
        for count in recon_goal_visit_counts.values()
        if count >= RECON_MIN_GOAL_VISITS
    )

    coverage_ratio = visited_count / len(recon_goal_visit_counts)

    return coverage_ratio >= RECON_MIN_GOAL_COVERAGE_RATIO

def is_valid_recon_attack_cell(cell, world, goals, robots, traffic_heatmap):
    """
    A fake object center should be in an average-traveled region.

    It does not need to be a perfect free-space cell because the fake object
    footprint may overlap walls visually. But it must generate enough reportable
    free/action cells to affect robot planning.
    """
    if not world.in_bounds(cell):
        return False

    if not is_average_traffic_cell(cell, traffic_heatmap):
        return False

    if is_near_any_goal(cell, goals):
        return False

    if cell_occupied_by_benign_robot(cell, robots):
        return False

    report_cells = fake_object_report_cells(
        cell,
        world,
        goals,
        robots,
    )

    if len(report_cells) < MALICIOUS_FAKE_OBJECT_MIN_REPORT_CELLS:
        return False

    return True


def recon_heatmap_attack_candidates(
    world,
    goals,
    robots,
    traffic_heatmap,
    placed_fake_object_centers=None,
    rng=None,
):
    candidates = []

    if placed_fake_object_centers is None:
        placed_fake_object_centers = []

    rows, cols = traffic_heatmap.shape
    rng = rng or np.random.default_rng(RANDOM_SEED)

    for r in range(rows):
        for c in range(cols):
            cell = (r, c)

            if is_near_previous_fake_object_center(
                    cell,
                    placed_fake_object_centers,
                ):
                continue

            if not is_valid_recon_attack_cell(
                cell,
                world,
                goals,
                robots,
                traffic_heatmap,
            ):
                continue

            height, width = sample_rectangle_dimensions(rng)
            report_cells = fake_object_report_cells(cell, world, goals, robots, height, width)
            if len(report_cells) < MALICIOUS_FAKE_OBJECT_MIN_REPORT_CELLS:
                continue

            benign_victims = [robot for robot in robots if not robot.is_malicious]
            path_overlap = 0
            affected_victims = 0
            path_proximity_score = 0.0
            bottleneck_score = footprint_bottleneck_score(
                world.grid,
                report_cells,
            )

            # Fast attack scoring: use overlap with the victims' current routes
            # and distance to those routes. The earlier version deep-copied a
            # belief map and ran A* once per report cell, per victim, per
            # candidate. That dominated runtime while adding little useful
            # discrimination because path overlap is already the strongest
            # indicator that a fake object will force a replan.
            for victim in benign_victims:
                remaining_path = (
                    victim.path[victim.path_index:]
                    if victim.path
                    else []
                )
                if not remaining_path:
                    continue

                remaining_path_set = set(remaining_path)
                overlap = len(remaining_path_set.intersection(report_cells))
                path_overlap += overlap

                if overlap > 0:
                    affected_victims += 1
                    path_proximity_score += 10.0 + overlap
                    continue

                # For non-overlapping candidates, reward objects close to a
                # current route without launching another planner.
                min_distance = min(
                    manhattan(report_cell, path_cell)
                    for report_cell in report_cells
                    for path_cell in remaining_path
                )
                if min_distance <= 2:
                    affected_victims += 1
                path_proximity_score += 1.0 / (1.0 + min_distance)

            if ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP and path_overlap <= 0:
                continue

            candidates.append(
                {
                    "center_cell": cell,
                    "report_cells": report_cells,
                    "traffic_score": fake_object_average_traffic_score(cell, traffic_heatmap, height, width),
                    "footprint_height": height,
                    "footprint_width": width,
                    "report_cell_count": len(report_cells),
                    "path_overlap": path_overlap,
                    "path_proximity_score": path_proximity_score,
                    "affected_victims": affected_victims,
                    "bottleneck_score": bottleneck_score,
                }
            )

    # We are not choosing the absolute hottest path anymore.
    # Prefer medium traffic plus enough cells to create a meaningful fake blockage.
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

def evaluate_fake_block_impact(victim_robot, candidate_cell):
    """
    Measures how much damage a fake blocked-cell report would do to a victim.

    Returns:
    - impact_score
    - causes_no_path
    - original_path_length
    - attacked_path_length

    Higher impact_score means a better attack target.
    """
    original_path = list(victim_robot.path)

    if not original_path:
        return None

    original_remaining = original_path[victim_robot.path_index:]

    if len(original_remaining) <= 1:
        return None

    original_path_length = len(original_remaining)

    test_belief = copy.deepcopy(victim_robot.belief_map)

    r, c = candidate_cell

    if not test_belief.in_bounds(candidate_cell):
        return None

    # Do not fake-block the victim's current position or a cell it can verify immediately.
    if candidate_cell == victim_robot.position:
        return None
    
    if candidate_cell == victim_robot.goal:
        return None

    if not is_good_distance_for_fake_block(victim_robot, candidate_cell):
        return None

    if is_known_static_or_blocked_prior(victim_robot, candidate_cell):
        return None

    test_belief.belief[r, c] = CellState.TEMPORARILY_BLOCKED
    test_belief.confidence[r, c] = 1.0
    test_belief.source[r, c] = "simulated_attack"

    planner = AStarPlanner4()

    try:
        attacked_path, _ = planner.plan(
            test_belief,
            victim_robot.position,
            victim_robot.goal,
        )

        attacked_path_length = len(attacked_path)
        delay = attacked_path_length - original_path_length

        impact_score = delay

        return {
            "impact_score": impact_score,
            "causes_no_path": False,
            "original_path_length": original_path_length,
            "attacked_path_length": attacked_path_length,
            "candidate_cell": candidate_cell,
        }

    except RuntimeError:
        return {
            "impact_score": 10_000,
            "causes_no_path": True,
            "original_path_length": original_path_length,
            "attacked_path_length": None,
            "candidate_cell": candidate_cell,
        }
    
# ============================================================
# Robot agent
# ============================================================

class GridRobot:
    """
    Autonomous grid robot.

    The robot is given a grid-level map at construction time.
    That is the "known map in its mind."

    It does not own the simulator's truth map.
    It owns a belief map initialized from the input grid.
    """

    def __init__(
        self,
        robot_id,
        initial_grid,
        start_cell,
        goal_cell=None,
        task_queue=None,
        sensor_radius=4,
        is_malicious=False,
        trust_initial_value=TRUST_INITIAL_VALUE,
        defense_method=DEFENSE_METHOD,
        defense_config=None,
    ):
        self.robot_id = int(robot_id)
        self.position_cell = tuple(start_cell)
        self.position_xy = cell_to_xy(start_cell)
        self.position_history = deque([self.position_cell], maxlen=50)

        self.motion_target_cell = None
        self.motion_target_xy = None

        self.task_queue = list(task_queue) if task_queue is not None else []
        self.task_index = 0
        self.carrying_item = False
        self.completed_tasks = 0

        if self.task_queue:
            self.goal = tuple(self.task_queue[0].pickup)
        elif goal_cell is not None:
            self.goal = tuple(goal_cell)
        else:
            raise ValueError(f"Robot {self.robot_id} needs either a goal_cell or task_queue.")
        self.sensor_radius = int(sensor_radius)
        self.is_malicious = bool(is_malicious)

        self.belief_map = RobotBeliefMap(initial_grid)

        if self.belief_map.is_blocked_for_planning(self.position_cell):
            raise ValueError(
                f"Robot {self.robot_id} start is blocked in initial map: {self.position_cell}"
            )

        if self.belief_map.is_blocked_for_planning(self.goal):
            raise ValueError(
                f"Robot {self.robot_id} goal is blocked in initial map: {self.goal}"
            )

        self.planner = AStarPlanner4()

        self.path = []
        self.path_index = 0
        self.last_plan_stats = {}
        self.using_fallback_path = False

        self.inbox = []
        self.pending_claims = []

        self.trust_model = make_trust_model(self.robot_id)
        self.defense_runner = build_defense_runner(
            method=defense_method,
            trust_score=self.trust_model.score,
            **(defense_config or {}),
        )
        self.belief_map.attach_defense_runner(self.defense_runner)
        self.defense_method = defense_method

        self.replan_count = 0
        self.accepted_reports = 0
        self.rejected_reports = 0
        self.verified_true_reports = 0
        self.verified_false_reports = 0
        self.last_trust_events = []
        self.last_shared_claim = {}
        self.last_shared_step = {}
        self.pending_outbound = {}
        self.defense_replan_needed = False
        self.last_source_linked_replan_step = -10**9
        self.source_linked_replan_context = None
        self.source_linked_replan_suppressed = {
            "no_trust_change": 0,
            "small_trust_change": 0,
            "no_route_influence": 0,
            "small_route_risk_drop": 0,
            "cooldown": 0,
        }
        self.last_path_invalid_replan_step = -10**9
        self.last_fallback_goal_retry_step = -10**9
        self.replanned_this_step = False

        # Physical traffic state is deliberately separate from trust,
        # attacker state, and belief-map state.
        self.consecutive_traffic_waits = 0
        self.total_traffic_waits = 0
        self.last_traffic_move_step = -1
        self.traffic_replan_count = 0
        self.traffic_wait_steps = []
        self.traffic_deadlock_active = False
        self.active_deadlock_id = None
        self.traffic_mode = "NORMAL"
        self.traffic_blocked_by = None
        self.active_yield_target = None
        self.saved_original_goal = None
        self.saved_original_path = None
        self.saved_original_path_index = 0
        self.yield_blocked_cell = None
        self.yield_conflict_cells = set()
        self.idle_relocated = False
        self.last_traffic_signature = None
        self.traffic_replans_suppressed = 0
        self.intent_commit_mismatches = 0
        self.last_traffic_replan_step = -10**9

        # Detailed instrumentation for separating useful route adaptation from
        # repeated planning churn.
        self.replan_events = []
        self.delivery_completion_steps = []
        self.current_step = -1
        self.current_phase = "INITIALIZATION"

        self.completed = False

    @property
    def position(self):
        return self.position_cell

    @property
    def occupied_cells(self):
        cells = set(robot_footprint_cells(self.position_cell))

        if self.motion_target_cell is not None:
            cells.update(robot_footprint_cells(tuple(self.motion_target_cell)))

        return cells

    @position.setter
    def position(self, cell):
        self.position_cell = tuple(cell)
        self.position_xy = cell_to_xy(cell)
        self.motion_target_cell = None
        self.motion_target_xy = None

    def trust_for(self, sender_id):
        return self.trust_model.score(sender_id)

    def reward_sender(self, sender_id):
        dummy_claim = PendingClaim(
            sender_id=sender_id,
            target_cell=self.position_cell,
            claim=ClaimType.FREE,
            timestamp=0,
        )
        self.trust_model.verify_claim(dummy_claim, truth_matches=True)

    def penalize_sender(self, sender_id):
        dummy_claim = PendingClaim(
            sender_id=sender_id,
            target_cell=self.position_cell,
            claim=ClaimType.FREE,
            timestamp=0,
        )
        self.trust_model.verify_claim(dummy_claim, truth_matches=False)

    def receive_report(self, report):
        self.inbox.append(report)

    def current_planned_next_cell(self):
        if not self.path:
            return None

        if self.path_index + 1 >= len(self.path):
            return None

        return self.path[self.path_index + 1]

    def propose_move_intent(self):
        """Freeze the exact target approved by centralized traffic control."""
        target = self.motion_target_cell
        if target is None and not self.completed and self.traffic_mode != "YIELDING_PARKED":
            target = self.current_planned_next_cell()
        return {
            "robot_id": self.robot_id,
            "current_cell": tuple(self.position),
            "target_cell": tuple(target) if target is not None else None,
            "current_cells": set(robot_footprint_cells(self.position)),
            "target_cells": set(robot_footprint_cells(target)) if target is not None else set(),
            "motion_target": self.motion_target_cell is not None,
        }

    def commit_move_intent(self, intent, world, approved):
        """Commit exactly the previously coordinated intent."""
        target = intent.get("target_cell")
        if not approved:
            self.consecutive_traffic_waits += 1
            self.total_traffic_waits += 1
            self.traffic_wait_steps.append(self.current_step)
            return False, "traffic_wait"
        if target is None:
            return False, "already_completed" if self.completed else "no_next_cell"
        if self.motion_target_cell is not None and tuple(self.motion_target_cell) != tuple(target):
            self.intent_commit_mismatches += 1
            return False, "intent_commit_mismatch"
        if not world.can_enter(target, None):
            return False, "blocked_world"
        if self.motion_target_cell is None:
            self.motion_target_cell = tuple(target)
            self.motion_target_xy = cell_to_xy(target)
        moved, event = self.advance_continuous_motion()
        if moved:
            self.consecutive_traffic_waits = 0
            self.last_traffic_move_step = self.current_step
            self.position_history.append(tuple(self.position))
        return moved, event

    def _cell_has_direct_free_observation(self, cell):
        r, c = tuple(cell)
        return (
            self.belief_map.source[r, c] == "self_sensor"
            and CellState(int(self.belief_map.belief[r, c])) in (
                CellState.FREE,
                CellState.PICKUP,
                CellState.DROPOFF,
                CellState.CHARGING,
            )
        )

    def should_replan_for_path_state(self, timestamp=None):
        """Return True when the current path is unusable and a replan is due."""
        step = self.current_step if timestamp is None else int(timestamp)

        if not self.path_invalid_or_empty():
            return False

        if step - self.last_path_invalid_replan_step < PATH_INVALID_REPLAN_COOLDOWN_STEPS:
            return False

        if (
            self.using_fallback_path
            and self.path
            and self.path_index + 1 >= len(self.path)
            and step - self.last_fallback_goal_retry_step < FALLBACK_GOAL_RETRY_COOLDOWN_STEPS
        ):
            return False

        return True

    def plan_path(self, reason="unspecified", timestamp=None, phase=None):
        old_remaining = list(self.path[self.path_index:]) if self.path else []
        old_next_five = old_remaining[:5]
        old_goal = tuple(self.goal)
        event_step = self.current_step if timestamp is None else int(timestamp)

        tried_real_goal = False
        try:
            self.path, self.last_plan_stats = self.planner.plan(
                self.belief_map,
                self.position,
                self.goal,
            )
            self.using_fallback_path = False
            tried_real_goal = True

        except RuntimeError:
            if not ENABLE_FALLBACK_EXPLORATION:
                raise

            tried_real_goal = True
            self.last_fallback_goal_retry_step = event_step
            self.path, self.last_plan_stats = plan_to_reachable_fallback(
                self.belief_map,
                self.position,
                self.goal,
            )
            self.using_fallback_path = True

        self.path_index = 0
        self.replan_count += 1
        self.replanned_this_step = True

        reason_text = str(reason)
        if (
            "path_invalid" in reason_text
            or reason_text == "path_invalid_or_empty"
        ):
            self.last_path_invalid_replan_step = event_step
        if tried_real_goal and not self.using_fallback_path:
            self.last_fallback_goal_retry_step = event_step

        new_path = list(self.path)
        new_next_five = new_path[:5]
        event_phase = self.current_phase if phase is None else str(phase)

        replan_context = self.source_linked_replan_context or {}
        self.replan_events.append({
            "step": event_step,
            "phase": event_phase,
            "reason": str(reason),
            "goal": old_goal,
            "old_path_length": len(old_remaining),
            "new_path_length": len(new_path),
            "path_length_delta": len(new_path) - len(old_remaining),
            "identical_path": old_remaining == new_path,
            "next_five_changed": old_next_five != new_next_five,
            "used_fallback": bool(self.using_fallback_path),
            "expanded_nodes": self.last_plan_stats.get("expanded_nodes"),
            "path_cost": self.last_plan_stats.get("path_cost"),
            "source_linked_sender_id": replan_context.get("sender_id"),
            "source_linked_old_trust": replan_context.get("old_trust"),
            "source_linked_new_trust": replan_context.get("new_trust"),
            "source_linked_trust_delta": replan_context.get("trust_delta"),
            "source_linked_route_risk_before": replan_context.get("route_risk_before"),
            "source_linked_route_risk_after": replan_context.get("route_risk_after"),
            "source_linked_route_risk_drop": replan_context.get("route_risk_drop"),
        })

    def path_invalid_or_empty(self):
        if not self.path:
            return True

        if self.path_index >= len(self.path):
            return True

        # If we are at the end of a fallback path, force another planning attempt.
        # This gives the robot repeated chances to switch back to its real goal
        # if a temporary blockage clears later.
        if self.path_index + 1 >= len(self.path):
            if self.using_fallback_path:
                return True

            if self.position_cell != self.goal:
                return True

        for cell in self.path[self.path_index:]:
            if self.belief_map.is_blocked_for_planning(cell):
                return True

        return False

    def update_from_sensor(self, world, timestamp, all_robot_positions=None):
        self.belief_map.set_planning_time(timestamp)
        if timestamp % DEFENSE_PRUNE_PERIOD_STEPS == 0:
            self.defense_runner.prune(timestamp)

        if timestamp % CONFIDENCE_DECAY_UPDATE_PERIOD_STEPS == 0:
            self.belief_map.apply_confidence_decay(timestamp)

        observations, lidar_rays = world.observe_cells_lidar(
            self.position_xy,
            max_range_cells=LIDAR_RANGE_CELLS,
            num_rays=LIDAR_NUM_RAYS,
            step_cells=LIDAR_STEP_CELLS,
            robot_positions=all_robot_positions,
        )

        changed = self.belief_map.update_from_sensor(observations, timestamp)

        self.verify_pending_claims(observations, timestamp)

        return changed, observations, lidar_rays

    def _source_linked_route_cells(self):
        """Return unique footprint cells along the near-term remaining route."""
        if not self.path:
            return []

        anchors = self.path[
            self.path_index:self.path_index + SOURCE_LINKED_ROUTE_LOOKAHEAD_ANCHORS
        ]
        cells = []
        seen = set()
        for anchor in anchors:
            for cell in robot_footprint_cells(anchor):
                cell = tuple(cell)
                r, c = cell
                # Match the planner: a direct local FREE observation suppresses
                # peer risk, so it cannot justify a source-linked replan.
                if (
                    self.belief_map.source[r, c] == "self_sensor"
                    and CellState(int(self.belief_map.belief[r, c])) in (
                        CellState.FREE,
                        CellState.PICKUP,
                        CellState.DROPOFF,
                        CellState.CHARGING,
                    )
                ):
                    continue
                if cell not in seen:
                    seen.add(cell)
                    cells.append(cell)
        return cells

    def verify_pending_claims(self, observations, timestamp):
        """Verify peer claims and schedule only material source-linked replans.

        Trust is still updated for every verified claim. Replanning, however, is
        decided once per sender per sensor update using the aggregate trust change
        and the resulting reduction in that sender's influence on the near-term
        route. This preserves retroactive source-linked correction while avoiding
        hundreds of identical-path replans.
        """
        still_pending = []
        verified_by_sender = {}
        old_trust_by_sender = {}

        for claim in self.pending_claims:
            cell = claim.target_cell

            if cell not in observations:
                still_pending.append(claim)
                continue

            truth_state = CellState(int(observations[cell]))

            if claim.claim == ClaimType.BLOCKED:
                truth_matches = truth_state in (
                    CellState.OCCUPIED_STATIC,
                    CellState.OCCUPIED_DYNAMIC,
                    CellState.TEMPORARILY_BLOCKED,
                )
            elif claim.claim == ClaimType.CONGESTED:
                truth_matches = truth_state == CellState.CONGESTED
            else:
                truth_matches = truth_state not in (
                    CellState.OCCUPIED_STATIC,
                    CellState.OCCUPIED_DYNAMIC,
                    CellState.TEMPORARILY_BLOCKED,
                )

            sender_id = int(claim.sender_id)
            if sender_id not in old_trust_by_sender:
                old_trust_by_sender[sender_id] = self.trust_model.score(sender_id)

            self.trust_model.verify_claim(claim, truth_matches)
            verified_by_sender.setdefault(sender_id, []).append((claim, truth_matches))

            if truth_matches:
                self.verified_true_reports += 1
            else:
                self.verified_false_reports += 1

            self.last_trust_events.append({
                "step": timestamp,
                "observer_id": self.robot_id,
                "sender_id": claim.sender_id,
                "target_cell": claim.target_cell,
                "claim": int(claim.claim),
                "truth_matches": truth_matches,
                "new_trust": self.trust_model.score(claim.sender_id),
                "is_malicious": claim.is_malicious,
            })

        self.pending_claims = still_pending

        if self.defense_method == "trust_threshold" and verified_by_sender:
            threshold = float(self.defense_runner.config.trust_threshold)
            route_cells = self._source_linked_route_cells()
            for sender_id in verified_by_sender:
                old_trust = float(old_trust_by_sender[sender_id])
                new_trust = float(self.trust_model.score(sender_id))
                crossed = (old_trust >= threshold) != (new_trust >= threshold)
                if not crossed or not route_cells:
                    continue
                risk_before = self.defense_runner.sender_route_risk(
                    sender_id, route_cells, timestamp=timestamp, trust_override=old_trust
                )
                risk_after = self.defense_runner.sender_route_risk(
                    sender_id, route_cells, timestamp=timestamp, trust_override=new_trust
                )
                if abs(risk_before - risk_after) <= 1e-9:
                    continue
                self.defense_replan_needed = True
                self.source_linked_replan_context = {
                    "sender_id": sender_id,
                    "old_trust": old_trust,
                    "new_trust": new_trust,
                    "threshold": threshold,
                    "crossing": "activated" if new_trust >= threshold else "deactivated",
                    "route_risk_before": risk_before,
                    "route_risk_after": risk_after,
                    "route_risk_drop": max(0.0, risk_before - risk_after),
                }
                break
            return

        if self.defense_method != "source_linked" or not verified_by_sender:
            return

        route_cells = self._source_linked_route_cells()

        for sender_id, results in verified_by_sender.items():
            old_trust = float(old_trust_by_sender[sender_id])
            new_trust = float(self.trust_model.score(sender_id))
            trust_delta = old_trust - new_trust

            if trust_delta <= 0.0:
                self.source_linked_replan_suppressed["no_trust_change"] += 1
                continue
            if trust_delta < SOURCE_LINKED_MIN_TRUST_DELTA:
                self.source_linked_replan_suppressed["small_trust_change"] += 1
                continue
            if not route_cells:
                self.source_linked_replan_suppressed["no_route_influence"] += 1
                continue

            risk_before = self.defense_runner.sender_route_risk(
                sender_id,
                route_cells,
                timestamp=timestamp,
                trust_override=old_trust,
            )
            risk_after = self.defense_runner.sender_route_risk(
                sender_id,
                route_cells,
                timestamp=timestamp,
                trust_override=new_trust,
            )
            risk_drop = max(0.0, risk_before - risk_after)

            if risk_before <= 0.0:
                self.source_linked_replan_suppressed["no_route_influence"] += 1
                continue
            if risk_drop < SOURCE_LINKED_MIN_ROUTE_RISK_DROP:
                self.source_linked_replan_suppressed["small_route_risk_drop"] += 1
                continue
            if timestamp - self.last_source_linked_replan_step < SOURCE_LINKED_REPLAN_COOLDOWN_STEPS:
                self.source_linked_replan_suppressed["cooldown"] += 1
                continue

            self.defense_replan_needed = True
            self.last_source_linked_replan_step = int(timestamp)
            self.source_linked_replan_context = {
                "sender_id": sender_id,
                "old_trust": old_trust,
                "new_trust": new_trust,
                "trust_delta": trust_delta,
                "route_risk_before": risk_before,
                "route_risk_after": risk_after,
                "route_risk_drop": risk_drop,
                "verified_claims": len(results),
                "verified_false_claims": sum(not match for _, match in results),
            }
            break

    def reports_affect_remaining_route(self, reports):
        if not reports or not self.path:
            return False

        remaining = list(self.path[self.path_index:])
        for report in reports:
            target = tuple(report.target_cell)

            if self._cell_has_direct_free_observation(target):
                continue

            if (
                report.claim == ClaimType.FREE
                and not report.is_malicious
                and self.defense_method != "majority_vote"
            ):
                continue

            for anchor in remaining:
                if target in robot_footprint_cells(anchor):
                    return True

            # A fake footprint can be adjacent to the current route without
            # covering its center anchor. Treat that as close enough to
            # replan, allowing the robot to detour before it reaches the
            # claimed obstacle and still recover if later direct sensing says
            # the cell is actually free.
            if (
                report.is_malicious
                and report.claim == ClaimType.BLOCKED
                and any(
                    manhattan(target, anchor) <= MALICIOUS_ROUTE_PROXIMITY_CELLS
                    for anchor in remaining
                )
            ):
                return True

        return False

    def process_inbox(self):
        accepted = []
        rejected = []

        for report in self.inbox:
            # Every defense receives the same valid claim stream. Methods that do
            # not use trust simply ignore it; source-linked methods reweight the
            # stored claim using current trust during planning.
            if not self.belief_map.in_bounds(report.target_cell):
                self.rejected_reports += 1
                rejected.append(report)
                continue

            applied = self.defense_runner.add_report(report)

            if applied:
                self.accepted_reports += 1
                accepted.append(report)

                self.pending_claims.append(
                    PendingClaim(
                        sender_id=report.sender_id,
                        target_cell=report.target_cell,
                        claim=report.claim,
                        timestamp=report.timestamp,
                        is_malicious=report.is_malicious,
                    )
                )
                self.trust_model.observe_claim(report)
            else:
                self.rejected_reports += 1
                rejected.append(report)

        self.inbox = []

        return accepted, rejected

    def update_delivery_state(self):
        """
        Advances the robot through pickup/dropoff tasks.

        If the robot reaches a pickup, it starts carrying.
        If it reaches a dropoff, it completes the task and moves to the next one.
        """
        if not self.task_queue:
            if self.position_cell == self.goal:
                self.completed = True
            return

        while self.task_index < len(self.task_queue):
            task = self.task_queue[self.task_index]

            if not self.carrying_item and self.position_cell == tuple(task.pickup):
                self.carrying_item = True
                self.goal = tuple(task.dropoff)
                self.path = []
                self.path_index = 0
                return

            if self.carrying_item and self.position_cell == tuple(task.dropoff):
                self.carrying_item = False
                self.completed_tasks += 1
                self.delivery_completion_steps.append(int(self.current_step))
                self.task_index += 1
                self.path = []
                self.path_index = 0

                if self.task_index >= len(self.task_queue):
                    self.completed = True
                    self.goal = tuple(task.dropoff)
                    return

                next_task = self.task_queue[self.task_index]
                self.goal = tuple(next_task.pickup)
                return

            return

    def advance_continuous_motion(self):
        """
        Moves continuously toward self.motion_target_cell.

        The robot's continuous position changes every step.
        The robot's grid cell changes only when it reaches the center
        of the target cell.
        """
        if self.motion_target_cell is None or self.motion_target_xy is None:
            return False, "no_motion_target"

        delta = self.motion_target_xy - self.position_xy
        distance = np.linalg.norm(delta)

        max_step = ROBOT_SPEED_CELLS_PER_STEP * CELL_SIZE

        if distance <= max_step:
            self.position_xy = self.motion_target_xy.copy()
            self.position_cell = tuple(self.motion_target_cell)

            self.motion_target_cell = None
            self.motion_target_xy = None

            self.path_index += 1
            self.update_delivery_state()

            return True, "moved_cell"

        direction = delta / distance
        self.position_xy = self.position_xy + direction * max_step

        # Important:
        # Do NOT update self.position_cell here.
        # The agent still considers itself inside the previous grid cell
        # until it reaches the center of the next one.
        return True, "moved_continuous"

    def move_one_cell(self, world, occupied_by_other_robots, ignore_robot_collisions=False):
        if self.completed:
            return False, "already_completed"

        self.update_delivery_state()

        if self.completed:
            return False, "completed"

        # If already moving toward a cell, continue that movement.
        if self.motion_target_cell is not None:
            return self.advance_continuous_motion()

        if (
            not self.replanned_this_step
            and self.should_replan_for_path_state()
        ):
            try:
                self.plan_path(
                    reason="path_invalid_during_move",
                    timestamp=self.current_step,
                    phase=self.current_phase,
                )
            except RuntimeError:
                return False, "no_path"

        next_cell = self.current_planned_next_cell()

        if next_cell is None:
            self.update_delivery_state()

            if self.completed:
                return False, "completed"

            return False, "no_next_cell"

        # Physical traffic safety applies from step 0. A traffic wait is not
        # environmental evidence and must never be written into the belief map.
        if set(robot_footprint_cells(next_cell)) & set(occupied_by_other_robots or ()):
            self.consecutive_traffic_waits += 1
            self.total_traffic_waits += 1
            self.traffic_wait_steps.append(self.current_step)
            return False, "traffic_wait"

        if not world.can_enter(next_cell, None):
            r, c = next_cell
            self.belief_map.belief[r, c] = CellState.OCCUPIED_DYNAMIC
            self.belief_map.confidence[r, c] = 1.0
            self.belief_map.source[r, c] = "blocked_move"

            self.path = []
            self.path_index = 0
            self.motion_target_cell = None
            self.motion_target_xy = None

            return False, "blocked_move"

        self.motion_target_cell = tuple(next_cell)
        self.motion_target_xy = cell_to_xy(next_cell)

        return self.advance_continuous_motion()

    def should_share_observation(self, cell, claim, timestamp):
        """Return whether a fresh observation should enter the outbound queue."""
        cell = tuple(cell)
        claim_value = int(claim)
        previous = self.last_shared_claim.get(cell)
        last_step = self.last_shared_step.get(cell, -10**9)
        pending = self.pending_outbound.get(cell)

        if previous == claim_value and timestamp - last_step < HONEST_REPORT_REFRESH_STEPS:
            return False
        if pending is not None and int(pending.claim) == claim_value:
            return False
        return True

    def queue_outbound_reports(self, reports):
        for report in reports:
            self.pending_outbound[tuple(report.target_cell)] = report

    def drain_outbound_reports(self, sent_step):
        reports = list(self.pending_outbound.values())
        self.pending_outbound.clear()
        for report in reports:
            cell = tuple(report.target_cell)
            self.last_shared_claim[cell] = int(report.claim)
            self.last_shared_step[cell] = int(sent_step)
        return reports

    def make_observation_reports(
        self,
        observations,
        timestamp,
        attack_phase_active=False,
    ):
        """
        Honest robots report what they directly observe.

        To keep the first version manageable, they only report blocked and congested cells.
        """
        reports = []

        if self.is_malicious and attack_phase_active:
            for cell, observed_state in observations.items():
                observed_state = CellState(int(observed_state))

                if (
                    observed_state == CellState.TEMPORARILY_BLOCKED
                    and self.should_share_observation(cell, ClaimType.BLOCKED, timestamp)
                ):
                    reports.append(
                        PeerReport(
                            sender_id=self.robot_id,
                            target_cell=cell,
                            claim=ClaimType.BLOCKED,
                            timestamp=timestamp,
                            is_malicious=False,
                        )
                    )

            return reports  

        for cell, observed_state in observations.items():
            observed_state = CellState(int(observed_state))

            if observed_state in (
                CellState.OCCUPIED_STATIC,
                CellState.OCCUPIED_DYNAMIC,
                CellState.TEMPORARILY_BLOCKED,
            ) and self.should_share_observation(cell, ClaimType.BLOCKED, timestamp):
                reports.append(
                    PeerReport(
                        sender_id=self.robot_id,
                        target_cell=cell,
                        claim=ClaimType.BLOCKED,
                        timestamp=timestamp,
                        is_malicious=False,
                    )
                )

            elif observed_state in (
                CellState.FREE,
                CellState.PICKUP,
                CellState.DROPOFF,
                CellState.CHARGING,
            ) and self.should_share_observation(cell, ClaimType.FREE, timestamp):
                reports.append(
                    PeerReport(
                        sender_id=self.robot_id,
                        target_cell=cell,
                        claim=ClaimType.FREE,
                        timestamp=timestamp,
                        is_malicious=False,
                    )
                )

        return reports

    def choose_malicious_report(
        self,
        timestamp,
        world,
        robots,
        goals,
        traffic_heatmap,
        placed_fake_object_centers=None,
    ):
        """
        Reconnaissance-based false obstacle attack.

        The malicious robot selects a medium-traffic fake object footprint and
        reports several usable cells in that footprint as BLOCKED.

        The fake object may visually overlap walls, but only non-blocked cells are
        sent as reports because static wall cells cannot usefully poison belief.
        """
        if not self.is_malicious:
            return []

        if not ENABLE_MALICIOUS_REPORTS:
            return []

        candidate_objects = recon_heatmap_attack_candidates(
            world,
            goals,
            robots,
            traffic_heatmap,
            placed_fake_object_centers=placed_fake_object_centers,
        )

        if not candidate_objects:
            return []

        chosen = candidate_objects[0]

        reports = []

        for cell in chosen["report_cells"]:
            reports.append(
                PeerReport(
                    sender_id=self.robot_id,
                    target_cell=cell,
                    claim=ClaimType.BLOCKED,
                    timestamp=timestamp,
                    is_malicious=True,
                )
            )

        return reports

# ============================================================
# Demo map creation
# ============================================================

def make_demo_static_grid(rows=GRID_ROWS, cols=GRID_COLS):
    """
    Creates a warehouse-ish grid.

    This function is only the demo input source.
    The robot does not define the grid internally.
    It receives this grid from outside.
    """
    grid = np.full((rows, cols), CellState.FREE, dtype=int)

    # Boundaries
    grid[0, :] = CellState.OCCUPIED_STATIC
    grid[rows - 1, :] = CellState.OCCUPIED_STATIC
    grid[:, 0] = CellState.OCCUPIED_STATIC
    grid[:, cols - 1] = CellState.OCCUPIED_STATIC

    # Shelf-like blocks
    shelf_rects = [
        (3, 4, 14, 2),
        (3, 10, 14, 2),
        (3, 16, 14, 2),

        (20, 4, 13, 2),
        (20, 10, 13, 2),
        (20, 16, 13, 2),

        (11, 22, 2, 8),
        (17, 22, 2, 8),
    ]

    for r, c, h, w in shelf_rects:
        grid[r:r + h, c:c + w] = CellState.OCCUPIED_STATIC

    # Semantic zones
    grid[2, 2] = CellState.CHARGING
    grid[rows - 3, 2] = CellState.PICKUP
    grid[rows - 3, cols - 3] = CellState.DROPOFF

    return grid


def make_demo_dynamic_grid(static_grid):
    """
    Creates the ground-truth runtime map.

    The static grid is the warehouse prior:
    walls, shelves, charging, pickup, and dropoff zones.

    This dynamic grid adds legitimate temporary runtime disruptions that
    robots do not know initially. They must discover them with lidar or
    receive peer reports.
    """
    dynamic = np.array(static_grid, dtype=int).copy()

    temporary_objects = [
        # Pallet/cart blocking a narrow aisle near the middle shelves.
        ((12, 18), CellState.TEMPORARILY_BLOCKED),
        ((13, 18), CellState.TEMPORARILY_BLOCKED),

        # Staging cart near a shelf end.
        ((6, 21), CellState.TEMPORARILY_BLOCKED),
        ((7, 21), CellState.TEMPORARILY_BLOCKED),

        # Temporary loading obstruction near the lower aisle.
        ((18, 13), CellState.TEMPORARILY_BLOCKED),

        # Congestion near a high-traffic region.
        ((8, 24), CellState.CONGESTED),
        ((8, 25), CellState.CONGESTED),
        ((9, 24), CellState.CONGESTED),

        # Mild congestion near the lower-right routing corridor.
        ((15, 30), CellState.CONGESTED),
        ((16, 30), CellState.CONGESTED),
    ]

    return place_temporary_objects(dynamic, temporary_objects)


# ============================================================
# Simulation
# ============================================================

def choose_malicious_robot_id(robot_specs, goals, prior_grid):
    """Return the fixed attacker identity used by every experiment.

    Robot identities are part of the experiment protocol: R0 is malicious and
    R1/R2 are benign victims. Choosing the nearest reachable robot made the
    role depend on the map and seed, which could silently turn R1 into the
    attacker and invalidate comparisons and playback labels.
    """
    robot_ids = {int(spec["robot_id"]) for spec in robot_specs}
    if 0 not in robot_ids:
        raise RuntimeError("The experiment team must include malicious Robot 0.")
    return 0

def validate_start_and_goal(world, robot_specs, goal):
    if not world.in_bounds(goal):
        raise ValueError(f"Goal out of bounds: {goal}")

    if not world.can_enter(goal):
        raise ValueError(
            f"Goal is blocked: {goal}, state={state_name(world.truth_state(goal))}"
        )

    seen_starts = set()

    for spec in robot_specs:
        rid = spec["robot_id"]
        start = tuple(spec["start"])

        if not world.in_bounds(start):
            raise ValueError(f"Robot {rid} start out of bounds: {start}")

        if not world.can_enter(start):
            raise ValueError(
                f"Robot {rid} starts in blocked cell {start}, "
                f"state={state_name(world.truth_state(start))}"
            )

        if start in seen_starts:
            raise ValueError(f"Multiple robots start in the same cell: {start}")

        seen_starts.add(start)

def repair_delivery_tasks(
    world,
    tasks_by_robot,
    robot_specs,
    prior_grid,
    action_points=None,
):
    """
    Keeps only tasks that are reachable for each robot.

    A task is valid for a robot only if:
    1. robot start -> pickup is reachable
    2. pickup -> dropoff is reachable

    This prevents assigning a robot a valid-looking but unreachable goal.
    """
    repaired = {}

    starts_by_robot = {
        int(spec["robot_id"]): tuple(spec["start"])
        for spec in robot_specs
    }

    for robot_id, tasks in tasks_by_robot.items():
        start = starts_by_robot[int(robot_id)]
        repaired_tasks = []

        # With smaller footprints, connectivity can differ between robots and
        # the round-robin task pairing may assign a valid point to the wrong
        # robot. Keep the task count stable by building reachable replacements
        # from the same action-point set when that happens.
        candidate_points = []
        source_points = action_points
        if source_points is None:
            source_points = [
                point
                for task in tasks
                for point in (task.pickup, task.dropoff)
            ]

        for point in source_points:
            point = nearest_enterable_cell(world, point)
            if point not in candidate_points:
                candidate_points.append(point)

        fallback_pairs = []
        for pickup in candidate_points:
            if not route_exists_for_prior(prior_grid, start, pickup):
                continue
            for dropoff in candidate_points:
                if pickup == dropoff:
                    continue
                if route_exists_for_prior(prior_grid, pickup, dropoff):
                    fallback_pairs.append((pickup, dropoff))

        if not fallback_pairs:
            raise RuntimeError(
                f"Robot {robot_id} has no reachable delivery tasks from start {start}"
            )

        for task in tasks:
            pickup = nearest_enterable_cell(world, task.pickup)
            dropoff = nearest_enterable_cell(world, task.dropoff)

            can_reach_pickup = route_exists_for_prior(
                prior_grid,
                start,
                pickup,
            )

            can_reach_dropoff = route_exists_for_prior(
                prior_grid,
                pickup,
                dropoff,
            )

            if not can_reach_pickup or not can_reach_dropoff:
                print(
                    f"Replacing unreachable task for Robot {robot_id}: "
                    f"start={start}, pickup={pickup}, dropoff={dropoff}, "
                    f"start_to_pickup={can_reach_pickup}, "
                    f"pickup_to_dropoff={can_reach_dropoff}"
                )
                pickup, dropoff = fallback_pairs[len(repaired_tasks) % len(fallback_pairs)]

            repaired_tasks.append(
                DeliveryTask(
                    pickup=pickup,
                    dropoff=dropoff,
                )
            )

        repaired[robot_id] = repaired_tasks

    return repaired


def refine_action_points(world, action_points, forbidden, target_count, excluded):
    """Drop excluded cells and refill to the target count with strategic picks."""
    excluded_set = frozenset(tuple(cell) for cell in excluded)
    points = [tuple(cell) for cell in action_points if tuple(cell) not in excluded_set]
    blocked = set(forbidden).union(points).union(excluded_set)

    while len(points) < target_count:
        extra = choose_strategic_action_points(
            world,
            target_count - len(points),
            forbidden=blocked,
        )

        for point in extra:
            cell = tuple(point)
            if cell in excluded_set or cell in points:
                continue
            points.append(cell)
            blocked.add(cell)

    return points


def build_robot_specs_and_goals(
    world,
    num_robots=DEFAULT_NUM_ROBOTS,
    prior_grid=None,
):
    """
    Builds valid robot starts and action points from the actual map.

    Action points are the places robots route between:
    pickup -> dropoff -> pickup -> dropoff.

    If the map has semantic PICKUP/DROPOFF/CHARGING cells, use them.
    If not, create synthetic action points from spread-out free cells.
    """
    grid = world.grid

    pickup_cells = find_cells_with_state(grid, [CellState.PICKUP])
    dropoff_cells = find_cells_with_state(grid, [CellState.DROPOFF])
    charging_cells = find_cells_with_state(grid, [CellState.CHARGING])
    free_cells = find_free_cells(grid)

    if not free_cells:
        raise ValueError("Map has no valid free cells for robots.")

    semantic_action_points = []

    for cell in pickup_cells + dropoff_cells + charging_cells:
        try:
            action_cell = nearest_enterable_cell(world, cell, forbidden=set())

            if action_cell not in semantic_action_points:
                semantic_action_points.append(action_cell)
        except ValueError:
            continue

    used_starts = set()
    robot_specs = []

    preferred_starts = charging_cells + pickup_cells + free_cells

    for robot_id in range(num_robots):
        preferred_start = preferred_starts[min(robot_id, len(preferred_starts) - 1)]

        start = nearest_safe_start_cell(
            world,
            preferred_start,
            forbidden=used_starts,
        )

        used_starts.add(start)

        robot_specs.append(
            {
                "robot_id": robot_id,
                "start": start,
            }
        )

        print(
            f"Robot {robot_id} start: preferred={preferred_start}, "
            f"chosen={start}, footprint={robot_footprint_cells(start)}"
        )

    if len(semantic_action_points) >= 2:
        action_points = semantic_action_points.copy()

        if len(action_points) < DEFAULT_NUM_ACTION_POINTS:
            extra_points = choose_strategic_action_points(
                world,
                count=DEFAULT_NUM_ACTION_POINTS - len(action_points),
                forbidden=used_starts.union(set(action_points)),
            )

            action_points.extend(extra_points)
    else:
        action_points = choose_strategic_action_points(
            world,
            count=DEFAULT_NUM_ACTION_POINTS,
            forbidden=used_starts,
        )

    try:
        from map_poisoning.map_io import WAREHOUSE_EXCLUDED_ACTION_POINTS

        excluded = WAREHOUSE_EXCLUDED_ACTION_POINTS
    except ImportError:
        excluded = frozenset()

    if excluded:
        action_points = refine_action_points(
            world,
            action_points,
            used_starts,
            DEFAULT_NUM_ACTION_POINTS,
            excluded,
        )

    goals = action_points.copy()
    display_goals = action_points.copy()

    # A smaller footprint can expose connectivity that was hidden by the old
    # 2x2 start placement. Move any isolated robot to the nearest safe cell
    # that can reach at least two action points, preserving the robot count.
    routing_grid = world.grid if prior_grid is None else prior_grid
    action_point_set = set(action_points)

    for spec in robot_specs:
        start = tuple(spec["start"])
        reachable_count = sum(
            route_exists_for_prior(routing_grid, start, point)
            for point in action_points
        )

        if reachable_count >= 2:
            continue

        old_start = start
        used_starts.discard(old_start)
        replacement = None

        candidates = sorted(
            free_cells,
            key=lambda cell: manhattan(tuple(cell), old_start),
        )

        for candidate in candidates:
            try:
                safe_candidate = nearest_safe_start_cell(
                    world,
                    candidate,
                    forbidden=used_starts.union(action_point_set),
                )
            except ValueError:
                continue

            if safe_candidate in used_starts:
                continue

            candidate_reachability = sum(
                route_exists_for_prior(routing_grid, safe_candidate, point)
                for point in action_points
            )
            if candidate_reachability >= 2:
                replacement = safe_candidate
                break

        if replacement is None:
            raise RuntimeError(
                f"Robot {spec['robot_id']} has no start with access to "
                "at least two action points"
            )

        spec["start"] = replacement
        used_starts.add(replacement)
        print(
            f"Robot {spec['robot_id']} start relocated for footprint: "
            f"{old_start} -> {replacement}"
        )

    return robot_specs, goals, display_goals

def build_delivery_tasks(action_points, num_robots, tasks_per_robot=TASKS_PER_ROBOT):
    """
    Builds pickup/dropoff task queues.

    Example:
    - Robot 0: point 0 -> point 5, then point 1 -> point 6
    - Robot 1: point 1 -> point 6, then point 2 -> point 7

    The offset pairing creates movement across the map instead of tiny local hops.
    """
    if len(action_points) < 2:
        raise ValueError("Need at least two action points to build delivery tasks.")

    tasks_by_robot = {}

    offset = max(1, len(action_points) // 2)

    for robot_id in range(num_robots):
        tasks = []

        for task_idx in range(tasks_per_robot):
            pickup_index = (robot_id + task_idx) % len(action_points)
            dropoff_index = (pickup_index + offset) % len(action_points)

            pickup = action_points[pickup_index]
            dropoff = action_points[dropoff_index]

            if pickup == dropoff:
                dropoff = action_points[(dropoff_index + 1) % len(action_points)]

            tasks.append(
                DeliveryTask(
                    pickup=pickup,
                    dropoff=dropoff,
                )
            )

        tasks_by_robot[robot_id] = tasks

    return tasks_by_robot

def broadcast_reports(robots, reports_by_sender):
    """
    Simple all-to-all communication.

    Later, replace this with range-limited communication.
    For now, the research variable is trust, not wireless propagation drama.
    """
    for sender_id, reports in reports_by_sender.items():
        for report in reports:
            for robot in robots:
                if robot.robot_id == sender_id:
                    continue

                robot.receive_report(report)


def _traffic_yield_target(robot, robots, world):
    """Find a deterministic nearby passing/parking cell without teleporting."""
    occupied = set().union(*(other.occupied_cells for other in robots if other is not robot))
    forbidden_goals = {tuple(other.goal) for other in robots}
    candidates = []
    for distance, previous in enumerate(reversed(robot.position_history), start=1):
        if tuple(previous) != tuple(robot.position):
            candidates.append((tuple(previous), distance))
    frontier = [(tuple(robot.position), 0)]
    seen = {tuple(robot.position)}
    while frontier:
        cell, distance = frontier.pop(0)
        if distance >= TRAFFIC_YIELD_SEARCH_RADIUS:
            continue
        for neighbor in AStarPlanner4.neighbors_4(cell):
            neighbor = tuple(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append((neighbor, distance + 1))
            candidates.append((neighbor, distance + 1))
    valid = []
    for candidate, distance in candidates:
        if candidate == tuple(robot.position) or candidate in occupied or candidate in forbidden_goals:
            continue
        if not world.can_enter(candidate, occupied):
            continue
        degree = sum(world.can_enter(neighbor, occupied) for neighbor in AStarPlanner4.neighbors_4(candidate))
        clearance = sum(
            world.can_enter((candidate[0] + dr, candidate[1] + dc), occupied)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        )
        valid.append((candidate, distance, degree, clearance))
    if not valid:
        return None
    passing = [item for item in valid if item[2] >= 3]
    pool = passing or valid
    return max(pool, key=lambda item: (item[2], item[3], -item[1]))[0]


def build_narrow_corridor_topology(grid):
    """Return one-cell corridor segments between static junctions/rooms."""
    rows, cols = np.asarray(grid).shape
    free = {
        (r, c) for r in range(rows) for c in range(cols)
        if int(grid[r, c]) not in {
            int(CellState.OCCUPIED_STATIC), int(CellState.OCCUPIED_DYNAMIC),
            int(CellState.TEMPORARILY_BLOCKED),
        }
    }
    neighbors = lambda cell: [candidate for candidate in AStarPlanner4.neighbors_4(cell) if candidate in free]
    degree = {cell: len(neighbors(cell)) for cell in free}
    endpoints = {cell for cell in free if degree[cell] != 2}
    segments = {}
    corridor_by_cell = {}
    visited_edges = set()
    corridor_index = 0
    for endpoint in endpoints:
        for neighbor in neighbors(endpoint):
            edge = frozenset((endpoint, neighbor))
            if edge in visited_edges:
                continue
            chain = [endpoint]
            previous, current = endpoint, neighbor
            visited_edges.add(edge)
            while True:
                chain.append(current)
                next_cells = [cell for cell in neighbors(current) if cell != previous]
                if degree[current] != 2 or not next_cells:
                    break
                previous, current = current, next_cells[0]
                visited_edges.add(frozenset((previous, current)))
            if len(chain) < 3:
                continue
            corridor_id = f"C{corridor_index}"
            corridor_index += 1
            segments[corridor_id] = {
                "cells": tuple(chain), "endpoint_a": chain[0],
                "endpoint_b": chain[-1], "length": len(chain),
                "owner_robot_id": None,
            }
            for cell in chain:
                corridor_by_cell[cell] = corridor_id
    return corridor_by_cell, segments


def _start_robot_yield(robot, blocker_id, blocked_cell, robots, world, step):
    target = _traffic_yield_target(robot, robots, world)
    if target is None:
        return None
    robot.traffic_mode = "YIELDING"
    robot.traffic_blocked_by = blocker_id
    robot.active_yield_target = tuple(target)
    robot.yield_blocked_cell = tuple(blocked_cell) if blocked_cell else None
    robot.yield_conflict_cells = set(robot_footprint_cells(blocked_cell)) if blocked_cell is not None else set()
    robot.saved_original_goal = tuple(robot.goal)
    robot.saved_original_path = list(robot.path)
    robot.saved_original_path_index = robot.path_index
    robot.goal = tuple(target)
    robot.path = []
    robot.path_index = 0
    try:
        robot.plan_path(reason="traffic_yield_started", timestamp=step, phase=robot.current_phase)
    except RuntimeError:
        robot.traffic_mode = "NORMAL"
        robot.goal = robot.saved_original_goal
        robot.path = robot.saved_original_path or []
        robot.path_index = robot.saved_original_path_index
        robot.active_yield_target = None
        return None
    return {
        "step": step, "event_type": "traffic_yield_started", "robot_id": robot.robot_id,
        "other_robot_ids": (blocker_id,) if blocker_id is not None else (),
        "current_cell": tuple(robot.position), "requested_cell": blocked_cell,
        "goal": tuple(robot.saved_original_goal), "wait_age": robot.consecutive_traffic_waits,
        "traffic_mode": robot.traffic_mode, "yield_target": tuple(target),
        "deadlock_id": robot.active_deadlock_id,
    }


def _restore_robot_goal_after_yield(robot, step):
    original_goal = robot.saved_original_goal
    if original_goal is None:
        return None
    deadlock_id = robot.active_deadlock_id
    robot.traffic_deadlock_active = False
    robot.active_deadlock_id = None
    robot.traffic_mode = "NORMAL"
    robot.traffic_blocked_by = None
    robot.active_yield_target = None
    robot.yield_blocked_cell = None
    robot.yield_conflict_cells = set()
    robot.goal = tuple(original_goal)
    robot.path = []
    robot.path_index = 0
    robot.consecutive_traffic_waits = 0
    robot.last_traffic_signature = None
    robot.plan_path(reason="traffic_deadlock_recovered", timestamp=step, phase=robot.current_phase)
    robot.saved_original_goal = None
    robot.saved_original_path = None
    return {
        "step": step, "event_type": "traffic_deadlock_recovered", "robot_id": robot.robot_id,
        "other_robot_ids": (), "current_cell": tuple(robot.position),
        "requested_cell": None, "wait_age": 0, "traffic_mode": robot.traffic_mode,
        "yield_target": None, "deadlock_id": deadlock_id,
    }


def _start_idle_parking(robot, blocker_id, blocked_cell, robots, world, step):
    robot.idle_relocated = True
    robot.completed = False
    event = _start_robot_yield(robot, blocker_id, blocked_cell, robots, world, step)
    if event:
        event["reason"] = "completed_robot_parking"
        robot.saved_original_goal = None
    return event


def coordinate_robot_intents(robots, world, step, traffic_state=None):
    """Approve frozen movement intents before any robot commits motion."""
    if traffic_state is None:
        traffic_state = getattr(coordinate_robot_intents, "_default_state", None)
        if traffic_state is None or step == 0:
            traffic_state = coordinate_robot_intents._default_state = {}
    traffic_state.setdefault("next_deadlock_id", 1)
    traffic_state.setdefault("last_joint_positions", None)
    traffic_state.setdefault("same_joint_state_streak", 0)
    events = []

    for idle in robots:
        if not idle.completed or idle.traffic_mode not in ("NORMAL", "IDLE_PARKED"):
            continue
        for active in robots:
            if active is idle or active.completed:
                continue
            requested = active.motion_target_cell or active.current_planned_next_cell()
            if requested is not None and idle.occupied_cells & set(robot_footprint_cells(requested)):
                parked = _start_idle_parking(idle, active.robot_id, requested, robots, world, step)
                if parked:
                    events.append(parked)
                break

    for robot in robots:
        if robot.traffic_mode == "YIELDING" and robot.active_yield_target is not None and tuple(robot.position) == tuple(robot.active_yield_target):
            robot.traffic_mode = "YIELDING_PARKED"
            robot.path = []
            robot.path_index = 0
            robot.motion_target_cell = None
            robot.motion_target_xy = None

    # A parked yield is held for one coordinated phase; this preserves a
    # stable deadlock episode while the priority robot clears the conflict.
    for robot in robots:
        if robot.traffic_mode != "YIELDING_PARKED" or robot.saved_original_goal is None:
            continue
        blocked = any(robot.yield_conflict_cells & other.occupied_cells for other in robots if other is not robot)
        if not blocked:
            try:
                recovered = _restore_robot_goal_after_yield(robot, step)
            except RuntimeError:
                recovered = None
            if recovered:
                events.append(recovered)

    intents = {}
    for robot in robots:
        frozen = robot.propose_move_intent()
        intents[robot.robot_id] = {
            "robot": robot, "frozen": frozen,
            "current": set(frozen["current_cells"]),
            "target": set(frozen["target_cells"]),
            "target_anchor": frozen["target_cell"], "approved": False,
        }
        robot._traffic_intent = frozen

    joint = tuple(sorted((robot.robot_id, tuple(robot.position)) for robot in robots))
    if traffic_state["last_joint_positions"] == joint:
        traffic_state["same_joint_state_streak"] += 1
    else:
        traffic_state["same_joint_state_streak"] = 1
    traffic_state["last_joint_positions"] = joint
    repeated_joint = traffic_state["same_joint_state_streak"] >= TRAFFIC_JOINT_REPEAT_THRESHOLD

    ordered = sorted(intents.values(), key=lambda item: (-item["robot"].consecutive_traffic_waits, item["robot"].robot_id))
    approved = {}
    swap_pairs = set()
    for left in intents.values():
        for right in intents.values():
            if left is right:
                continue
            if left["target"] & right["current"] and right["target"] & left["current"]:
                swap_pairs.add(frozenset((left["robot"].robot_id, right["robot"].robot_id)))

    for item in ordered:
        robot = item["robot"]
        target = item["target"]
        if not target:
            approved[robot.robot_id] = True
            item["approved"] = True
            continue
        conflict_kind = None
        blockers = []
        for other in intents.values():
            if other is item:
                continue
            other_robot = other["robot"]
            if other_robot.idle_relocated and other_robot.traffic_mode in ("YIELDING", "YIELDING_PARKED"):
                continue
            pair = frozenset((robot.robot_id, other_robot.robot_id))
            if pair in swap_pairs:
                conflict_kind = "traffic_swap_conflict"
                blockers.append(other_robot.robot_id)
                break
            if target & other["target"] and other["approved"]:
                conflict_kind = "traffic_vertex_conflict"
                blockers.append(other_robot.robot_id)
                break
            if target & other["current"] and not other["approved"]:
                conflict_kind = "traffic_reservation_conflict"
                blockers.append(other_robot.robot_id)
                break
        if conflict_kind:
            approved[robot.robot_id] = False
            item["approved"] = False
            robot.consecutive_traffic_waits += 1
            robot.total_traffic_waits += 1
            robot.traffic_wait_steps.append(step)
            robot.traffic_blocked_by = blockers[0] if blockers else None
            events.append({
                "step": step, "event_type": conflict_kind, "robot_id": robot.robot_id,
                "other_robot_ids": tuple(blockers), "requested_cell": item["target_anchor"],
                "wait_age": robot.consecutive_traffic_waits,
            })
            if (robot.consecutive_traffic_waits >= TRAFFIC_DEADLOCK_WAIT_THRESHOLD or repeated_joint) and not robot.traffic_deadlock_active:
                robot.traffic_deadlock_active = True
                number = int(traffic_state["next_deadlock_id"])
                traffic_state["next_deadlock_id"] = number + 1
                robot.active_deadlock_id = f"deadlock-{number:06d}"
                events.append({
                    "step": step, "event_type": "traffic_deadlock_detected", "robot_id": robot.robot_id,
                    "other_robot_ids": tuple(blockers), "requested_cell": item["target_anchor"],
                    "wait_age": robot.consecutive_traffic_waits, "deadlock_id": robot.active_deadlock_id,
                })
        else:
            approved[robot.robot_id] = True
            item["approved"] = True

    deadlocked = [robot for robot in robots if robot.traffic_deadlock_active and robot.traffic_mode == "NORMAL"]
    if deadlocked and (repeated_joint or any(robot.consecutive_traffic_waits >= TRAFFIC_DEADLOCK_WAIT_THRESHOLD for robot in deadlocked)):
        yielding = min(deadlocked, key=lambda robot: (robot.consecutive_traffic_waits, robot.robot_id))
        requested = next((event.get("requested_cell") for event in reversed(events) if event.get("robot_id") == yielding.robot_id and event.get("event_type", "").startswith("traffic_")), None)
        yielded = _start_robot_yield(yielding, yielding.traffic_blocked_by, requested, robots, world, step)
        if yielded:
            events.append(yielded)

    return approved, events


def assert_no_robot_overlap(robots, log=None, step=None):
    occupied = {}
    for robot in robots:
        for cell in robot_footprint_cells(robot.position):
            occupied.setdefault(tuple(cell), []).append(robot.robot_id)
    violations = [{"step": step, "robot_ids": ids, "cell": cell} for cell, ids in occupied.items() if len(ids) > 1]
    if violations:
        if log is not None:
            log.setdefault("traffic_events", []).extend({"event_type": "traffic_overlap_violation", **item} for item in violations)
            log["robot_overlap_violations"] = int(log.get("robot_overlap_violations", 0)) + len(violations)
        raise RuntimeError(f"physical robot overlap at step {step}: {violations}")


def run_simulation(
    grid=None,
    prior_grid=None,
    defense_method=DEFENSE_METHOD,
    defense_config=None,
    tasks_per_robot=TASKS_PER_ROBOT,
    max_steps=MAX_STEPS,
    random_seed=RANDOM_SEED,
    experiment_mode=EXPERIMENT_MODE,
    attack_events=None,
    obstacle_episodes=None,
    manifest_robot_starts=None,
    manifest_task_queues=None,
    manifest_malicious_robot_id=None,
    map_view="combined",
):
    np.random.seed(random_seed)

    if grid is None:
        prior_grid = make_demo_static_grid()
        grid = make_demo_dynamic_grid(prior_grid)
    elif prior_grid is None:
        # External maps are treated as both truth and prior unless the caller
        # provides a separate prior. This preserves backward compatibility.
        prior_grid = np.array(grid, dtype=int).copy()

    world = GridWorld(grid)

    temp_blockage_manager = None
    fixed_obstacle_episodes = tuple(obstacle_episodes or ())

    if ENABLE_DYNAMIC_TEMP_BLOCKAGES and prior_grid is not None and not fixed_obstacle_episodes:
        temp_blockage_manager = TemporaryBlockageManager(
            prior_grid,
            active_count=TEMP_ACTIVE_OBJECT_COUNT_BLOCKED,
            change_period=TEMP_BLOCKAGE_CHANGE_PERIOD_STEPS,
            seed=random_seed,
        )
        world.grid = temp_blockage_manager.build_truth_grid()
    elif fixed_obstacle_episodes:
        world.grid = prior_grid.copy()
        for episode in fixed_obstacle_episodes:
            if episode.appearance_step <= 0 < episode.clearance_step:
                for cell in episode.cells:
                    world.grid[cell] = CellState.TEMPORARILY_BLOCKED

    robot_specs, goals, display_goals = build_robot_specs_and_goals(
        world,
        num_robots=DEFAULT_NUM_ROBOTS,
        prior_grid=prior_grid,
    )

    if manifest_robot_starts and manifest_task_queues:
        robot_specs = [
            {"robot_id": robot_id, "start": tuple(manifest_robot_starts[robot_id])}
            for robot_id in range(DEFAULT_NUM_ROBOTS)
        ]
        tasks_by_robot = {
            robot_id: [
                DeliveryTask(pickup=tuple(task.pickup), dropoff=tuple(task.dropoff))
                for task in manifest_task_queues[robot_id]
            ]
            for robot_id in manifest_task_queues
        }
        goals = display_goals.copy()
    else:
        goals = filter_reachable_action_points(
            goals,
            robot_specs,
            prior_grid,
        )

        relocate_starts_for_goals(world, robot_specs, goals, prior_grid)

        display_goals = goals.copy()

        tasks_by_robot = build_delivery_tasks(
            goals,
            num_robots=DEFAULT_NUM_ROBOTS,
            tasks_per_robot=tasks_per_robot,
        )

        tasks_by_robot = repair_delivery_tasks(
            world,
            tasks_by_robot,
            robot_specs,
            prior_grid,
            action_points=goals,
        )

    goal = goals[0]

    # Manifest tasks and starts are authored against the static prior grid; do not
    # validate them on a step-0 truth grid that already includes temp blockages.
    validation_world = (
        GridWorld(prior_grid)
        if manifest_robot_starts and manifest_task_queues
        else world
    )

    for robot_id, tasks in tasks_by_robot.items():
        for task in tasks:
            validate_start_and_goal(
                validation_world,
                [{"robot_id": robot_id, "start": task.pickup}],
                task.dropoff,
            )

    validate_start_and_goal(validation_world, robot_specs, goal)

    malicious_robot_id = (
        int(manifest_malicious_robot_id)
        if manifest_malicious_robot_id is not None
        else choose_malicious_robot_id(
            robot_specs,
            goals,
            prior_grid,
        )
    )

    robots = []

    for spec in robot_specs:
        robot_id = spec["robot_id"]

        robot = GridRobot(
            robot_id=robot_id,
            initial_grid=prior_grid,
            start_cell=spec["start"],
            goal_cell=None,
            task_queue=tasks_by_robot[robot_id],
            sensor_radius=SENSOR_RADIUS,
            is_malicious=(robot_id == malicious_robot_id),
            defense_method=defense_method,
            defense_config=defense_config,
        )
        robots.append(robot)

    # DEBUG: check initial planner validity for every robot
    print("\n--- INITIAL PLANNING DEBUG ---")

    for robot in robots:
        print(
            f"Robot {robot.robot_id}: "
            f"start={robot.position}, "
            f"goal={robot.goal}, "
            f"malicious={robot.is_malicious}, "
            f"start_blocked={robot.belief_map.is_blocked_for_planning(robot.position)}, "
            f"goal_blocked={robot.belief_map.is_blocked_for_planning(robot.goal)}"
        )

        try:
            path, stats = robot.planner.plan(
                robot.belief_map,
                robot.position,
                robot.goal,
            )
            print(f"  initial path length={len(path)}, stats={stats}")
        except RuntimeError as e:
            print(f"  initial planning failed: {e}")

    print("--- END INITIAL PLANNING DEBUG ---\n")

    log = {
        "truth_grid": [],
        "truth_dynamic": [],
        "malicious_fake_objects": [],
        "attack_overlays": [],
        "temporary_movement": [],
        "attack_events": [
            {"event_id": getattr(event, "event_id", ""), "step": int(event.step),
             "attack_type": getattr(getattr(event, "attack_type", None), "value", str(getattr(event, "attack_type", "")))}
            for event in (attack_events or ())
        ],
        "map_view": map_view,
        "traffic_heatmap": [],
        "traffic_events": [],
        "robot_overlap_violations": 0,
        "phase": [],
        "robots": {
            robot.robot_id: {
                "position": [],
                "position_xy": [],
                "path": [],
                "belief": [],
                "local_belief": [],
                "combined_belief": [],
                "effective_peer_cells": [],
                "peer_provenance": [],
                "direct_blocked_cells": [],
                "trust": [],
                "events": [],
                "accepted_reports": [],
                "rejected_reports": [],
                "replan_count": [],
                "completed": [],

                "carrying_item": [],
                "completed_tasks": [],
                "current_goal": [],
                "lidar_rays": [],
                "malicious_claim_cells_on_route": [],
                "traffic_waits": [],
                "traffic_deadlock_active": [],
                "active_deadlock_id": [],
                "traffic_mode": [],
                "traffic_replans": [],
            }
            for robot in robots
        },
        "reports": [],
        "trust_events": [],
        "malicious_robot_id": malicious_robot_id,
        "goal": goal,
        "goals": goals,
        "display_goals": display_goals,
        "attack_phase_start_step": None,
        "defense_method": defense_method,
        "defense_config": dict(defense_config or {}),
        "tasks_per_robot": int(tasks_per_robot),
        "max_steps": int(max_steps),
        "random_seed": int(random_seed),
        "experiment_mode": experiment_mode,
    }

    # Initial planning
    for robot in robots:
        try:
            robot.plan_path(reason="initial_plan", timestamp=-1, phase="INITIALIZATION")
        except RuntimeError:
            pass

    traffic_heatmap = np.zeros_like(world.grid, dtype=int)
    traffic_state = {}
    traffic_state["corridor_by_cell"], traffic_state["corridors"] = build_narrow_corridor_topology(prior_grid)

    recon_goal_visit_counts = {
        tuple(goal): 0
        for goal in goals
    }

    attack_phase_started = False
    attack_phase_start_step = None
    attack_injection_stop_step = None

    active_malicious_fake_objects = {}
    active_attack_overlays = {}

    placed_malicious_fake_object_centers = []
    last_malicious_fake_object_step = None

    enable_malicious_reports = experiment_mode == "attack"

    for step in range(max_steps):
        in_recon_phase = not attack_phase_started
        if in_recon_phase:
            current_phase = "RECONNAISSANCE"
        elif attack_injection_stop_step is None:
            current_phase = "ATTACK"
        else:
            current_phase = "RECOVERY"
        for robot in robots:
            robot.current_step = step
            robot.current_phase = current_phase
            robot.replanned_this_step = False

        active_malicious_fake_objects = {
            cell: created_step
            for cell, created_step in active_malicious_fake_objects.items()
            if step - created_step <= MALICIOUS_FAKE_OBJECT_DISPLAY_TTL
        }
        active_attack_overlays = {
            key: value
            for key, value in active_attack_overlays.items()
            if step - value[0] <= MALICIOUS_FAKE_OBJECT_DISPLAY_TTL
        }
        if temp_blockage_manager is not None:
            changed_temp_blockages = temp_blockage_manager.update_world_if_needed(
                world,
                step,
                robots=robots,
            )

            if changed_temp_blockages:
                for robot in robots:
                    robot.path = []
                    robot.path_index = 0
                    robot.motion_target_cell = None
                    robot.motion_target_xy = None
        elif fixed_obstacle_episodes:
            rebuilt = apply_temporary_obstacle_episodes(
                prior_grid,
                fixed_obstacle_episodes,
                step,
                forbidden_cells=robot_occupied_cells(robots),
            )
            if not np.array_equal(rebuilt, world.grid):
                world.grid = rebuilt
                for robot in robots:
                    robot.path = []
                    robot.path_index = 0
                    robot.motion_target_cell = None
                    robot.motion_target_xy = None

        log["temporary_movement"].append(
            dict(temp_blockage_manager.movement_decisions)
            if temp_blockage_manager is not None and changed_temp_blockages else {}
        )

        reports_by_sender = {robot.robot_id: [] for robot in robots}

        # 1. Robots sense locally.
        observations_by_robot = {}
        lidar_rays_by_robot = {}

        all_robot_positions = set()

        for robot in robots:
            all_robot_positions.update(robot.occupied_cells)

        for robot in robots:
            if step < SPAWN_COLLISION_GRACE_STEPS:
                other_robot_positions = set()
            else:
                other_robot_positions = all_robot_positions - robot.occupied_cells

            changed, observations, lidar_rays = robot.update_from_sensor(
                world,
                step,
                all_robot_positions=other_robot_positions,
            )
            observations_by_robot[robot.robot_id] = observations
            lidar_rays_by_robot[robot.robot_id] = lidar_rays

            if robot.last_trust_events:
                log["trust_events"].extend(robot.last_trust_events)
                robot.last_trust_events = []

            if HONEST_ROBOTS_SHARE_OBSERVATIONS:
                robot.queue_outbound_reports(
                    robot.make_observation_reports(
                        observations,
                        step,
                        attack_phase_active=(current_phase == "ATTACK"),
                    )
                )

        # 2. Malicious robot creates a sustained fixed-duration campaign of reinforced bottleneck lies.
        malicious_ids = [r.robot_id for r in robots if r.is_malicious]
        all_victims_distrust = bool(malicious_ids) and all(
            all(victim.trust_for(attacker_id) < TRUST_ACCEPT_THRESHOLD for attacker_id in malicious_ids)
            for victim in robots
            if not victim.is_malicious
        )
        burst_complete = (
            attack_phase_start_step is not None
            and step - attack_phase_start_step >= ATTACK_BURST_DURATION_STEPS
        )
        should_stop_injection = (
            attack_phase_started
            and attack_injection_stop_step is None
            and (
                burst_complete
            )
        )
        if should_stop_injection:
            attack_injection_stop_step = step
            log["attack_injection_stop_step"] = step
            print(f"\n--- ATTACK INJECTION STOPPED AT STEP {step}; RECOVERY PHASE STARTING ---")

        should_inject_fake_object = (
            attack_events is None
            and
            enable_malicious_reports
            and attack_phase_started
            and attack_injection_stop_step is None
            and step % COMMUNICATION_PERIOD_STEPS == 0
            and (
                last_malicious_fake_object_step is None
                or step - last_malicious_fake_object_step >= MALICIOUS_FAKE_OBJECT_INJECTION_PERIOD_STEPS
            )
        )

        if should_inject_fake_object:
            malicious_robots = [r for r in robots if r.is_malicious]

            for attacker in malicious_robots:
                fake_reports = attacker.choose_malicious_report(
                    timestamp=step,
                    world=world,
                    robots=robots,
                    goals=goals,
                    traffic_heatmap=traffic_heatmap,
                    placed_fake_object_centers=placed_malicious_fake_object_centers,
                )

                if fake_reports:
                    reports_by_sender[attacker.robot_id].extend(fake_reports)

                    # Store the approximate fake object center so the next attack
                    # does not choose the same location again.
                    report_rows = [report.target_cell[0] for report in fake_reports]
                    report_cols = [report.target_cell[1] for report in fake_reports]

                    center = (
                        int(round(sum(report_rows) / len(report_rows))),
                        int(round(sum(report_cols) / len(report_cols))),
                    )

                    placed_malicious_fake_object_centers.append(center)
                    last_malicious_fake_object_step = step
                    for report in fake_reports:
                        active_attack_overlays[("fake_obstacle", tuple(report.target_cell))] = (step, "fake_obstacle")

        fixed_attack_injected = False
        if attack_events is not None:
            for event in attack_events:
                if int(event.step) != step:
                    continue
                fixed_attack_injected = True
                for cell in event.cells:
                    active_attack_overlays[(event.attack_type.value, tuple(cell))] = (step, event.attack_type.value)
                for cell in event.cells:
                    reports_by_sender[event.sender_id].append(
                        PeerReport(
                            sender_id=event.sender_id,
                            target_cell=tuple(cell),
                            claim=ClaimType(int(event.claim)),
                            timestamp=int(event.observation_step),
                            is_malicious=True,
                        )
                    )

        # 3. Broadcast reports periodically.
        # Fixed-manifest attacks have an explicit delivery step.  They must not
        # disappear merely because that step falls between normal periodic
        # broadcasts (for example step 470 with a four-step cadence).
        if COMMUNICATION_PERIOD_STEPS > 0 and (step % COMMUNICATION_PERIOD_STEPS == 0 or fixed_attack_injected):
            for robot in robots:
                reports_by_sender[robot.robot_id].extend(
                    robot.drain_outbound_reports(step)
                )
            broadcast_reports(robots, reports_by_sender)

            for sender_id, reports in reports_by_sender.items():
                for report in reports:
                    log["reports"].append(
                        {
                            "step": step,
                            "observation_step": report.timestamp,
                            "sent_step": step,
                            "sender_id": sender_id,
                            "target_cell": report.target_cell,
                            "claim": int(report.claim),
                            "is_malicious": report.is_malicious,
                        }
                    )

        # 4. Robots process messages and replan if needed.
        for robot in robots:
            old_path = list(robot.path)

            accepted, rejected = robot.process_inbox()

            route_affected = robot.reports_affect_remaining_route(accepted)
            path_invalid = robot.should_replan_for_path_state(step)
            trust_reweight = bool(robot.defense_replan_needed)
            should_replan = route_affected or trust_reweight or path_invalid
            accepted_malicious = [
                report
                for report in accepted
                if report.is_malicious
            ]

            if should_replan:
                reasons = []
                if route_affected:
                    if any(report.is_malicious for report in accepted):
                        reasons.append("malicious_report_on_route")
                    else:
                        reasons.append("honest_report_on_route")
                if trust_reweight:
                    reasons.append("source_linked_trust_reweight")
                if path_invalid:
                    reasons.append("path_invalid_or_empty")
                if accepted_malicious:
                    reasons.append("malicious_report_accepted")

                try:
                    robot.plan_path(
                        reason="+".join(reasons),
                        timestamp=step,
                        phase=current_phase,
                    )
                except RuntimeError:
                    robot.path = []
                    robot.path_index = 0
                finally:
                    robot.defense_replan_needed = False
                    robot.source_linked_replan_context = None

            if accepted_malicious and old_path != list(robot.path):
                print(
                    f"Step {step}: R{robot.robot_id} replanned after accepting malicious report. "
                    f"old_len={len(old_path)} new_len={len(robot.path)}"
                )

        # 5. Freeze and coordinate all physical movement before committing it.
        approved, traffic_events = coordinate_robot_intents(robots, world, step, traffic_state)
        log["traffic_events"].extend(traffic_events)
        for robot in robots:
            intent = getattr(robot, "_traffic_intent", robot.propose_move_intent())
            moved, event = robot.commit_move_intent(intent, world, approved.get(robot.robot_id, False))
            if event in {"blocked_world", "blocked_move"}:
                try:
                    robot.plan_path(reason="blocked_move", timestamp=step, phase=current_phase)
                except RuntimeError:
                    pass
            log["robots"][robot.robot_id]["events"].append(event)

        # Physical overlap is an invariant, independent of sensor grace.
        assert_no_robot_overlap(robots, log, step)

        # Record benign traffic after movement so the heatmap matches the logged animation state.
        if in_recon_phase:
            for robot in robots:
                if robot.is_malicious:
                    continue

                occupied_cells = set(robot_footprint_cells(robot.position))

                # Count actual occupied footprint.
                for cell in occupied_cells:
                    r, c = cell

                    if 0 <= r < traffic_heatmap.shape[0] and 0 <= c < traffic_heatmap.shape[1]:
                        traffic_heatmap[r, c] += 3

                # Track which action points/goals were actually visited during reconnaissance.
                for goal_cell in recon_goal_visit_counts:
                    if goal_cell in occupied_cells:
                        recon_goal_visit_counts[goal_cell] += 1

                # Also count the route the robot currently intends to use.
                # This makes the recon map capture corridors, not just sparse anchor cells.
                if robot.path:
                    for cell in robot.path[robot.path_index:]:
                        r, c = cell

                        if 0 <= r < traffic_heatmap.shape[0] and 0 <= c < traffic_heatmap.shape[1]:
                            traffic_heatmap[r, c] += 1


        if in_recon_phase:
            min_steps_done = step >= MIN_RECON_STEPS
            max_steps_reached = step >= MAX_RECON_STEPS
            coverage_done = recon_coverage_satisfied(recon_goal_visit_counts)

            if min_steps_done and (coverage_done or max_steps_reached):
                attack_phase_started = True
                attack_phase_start_step = step
                log["attack_phase_start_step"] = step                

                candidates = recon_heatmap_attack_candidates(
                    world,
                    goals,
                    robots,
                    traffic_heatmap,
                )

                print("\n--- RECONNAISSANCE COMPLETE ---")
                print(f"Recon ended at step: {step}")
                print(f"Goal coverage satisfied: {coverage_done}")
                print("Recon goal visit counts:")

                for goal_cell, count in sorted(
                    recon_goal_visit_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    print(f"  goal={goal_cell}, visits={count}")

                print("Top learned medium-traffic fake-object candidates:")

                for idx, candidate in enumerate(candidates[:10]):
                    print(
                        f"  {idx + 1}. center={candidate['center_cell']}, "
                        f"avg_traffic={candidate['traffic_score']:.2f}, "
                        f"bottleneck={candidate.get('bottleneck_score', 0.0):.2f}, "
                        f"route_overlap={candidate.get('path_overlap', 0)}, "
                        f"reported_cells={candidate['report_cell_count']}"
                    )

                print("--- ATTACK PHASE STARTING ---\n")
        # 6. Log state.
        log["truth_grid"].append(world.grid.copy())

        log["malicious_fake_objects"].append(
            sorted(active_malicious_fake_objects.keys())
        )
        # Show only the newest attack footprint. Older events remain in the
        # log for audit/playback data, but stacking their TTLs makes the debug
        # map look as if the whole warehouse is under attack.
        latest_overlay_step = max(
            (value[0] for value in active_attack_overlays.values()),
            default=None,
        )
        overlay_groups = {}
        for (attack_type, cell), value in active_attack_overlays.items():
            if value[0] != latest_overlay_step:
                continue
            overlay_groups.setdefault(attack_type, []).append(cell)
        log["attack_overlays"].append([
            {"attack_type": attack_type, "cells": sorted(cells)}
            for attack_type, cells in sorted(overlay_groups.items())
        ])

        log["traffic_heatmap"].append(traffic_heatmap.copy())
        log["phase"].append(
            current_phase
        )

        for robot in robots:
            rid = robot.robot_id
            rlog = log["robots"][rid]

            rlog["position"].append(robot.position)
            rlog["position_xy"].append(tuple(robot.position_xy.tolist()))
            rlog["path"].append(copy.deepcopy(robot.path))
            rlog["belief"].append(robot.belief_map.display_grid())
            local = robot.belief_map.initial_prior.copy()
            direct_mask = robot.belief_map.source == "self_sensor"
            local[direct_mask] = robot.belief_map.belief[direct_mask]
            own_display = ROBOT_BELIEF_DISPLAY.get(rid, DISPLAY_PEER_BELIEF)
            for dr, dc in np.argwhere(direct_mask):
                if CellState(int(robot.belief_map.belief[dr, dc])) in (
                    CellState.OCCUPIED_DYNAMIC,
                    CellState.TEMPORARILY_BLOCKED,
                ):
                    local[dr, dc] = own_display
            rlog["local_belief"].append(local)
            effective = local.copy()
            peer_cells = []
            peer_provenance = []
            for cell, state in robot.defense_runner.effective_cells(step).items():
                r, c = cell
                # Combined-view yellow is reserved for effective blocked peer
                # belief. Free reports are still fused for navigation, but do
                # not paint large areas of otherwise unoccupied floor yellow.
                if (state.has_active_evidence and state.evidence > 0.0
                        and 0 <= r < effective.shape[0]
                        and 0 <= c < effective.shape[1]
                        and not is_blocking_state(robot.belief_map.initial_prior[r, c])):
                    # Own direct sensing remains authoritative in the display.
                    if robot.belief_map.source[r, c] != "self_sensor":
                        source_id = state.dominant_source
                        effective[r, c] = ROBOT_BELIEF_DISPLAY.get(
                            source_id, DISPLAY_PEER_BELIEF
                        )
                        peer_cells.append((r, c, state.claim, state.routing_cost, state.evidence))
                        peer_provenance.append({
                            "cell": (r, c),
                            "senders": sorted({
                                claim.sender_id
                                for claim in robot.defense_runner.claims_for(cell)
                                if step - claim.timestamp <= robot.defense_runner.config.max_claim_age
                            }),
                            "claim": state.claim,
                            "evidence": state.evidence,
                            "dominant_source": source_id,
                            "supporting_sources": list(state.supporting_sources),
                            "dominant_trust": (
                                robot.trust_for(source_id)
                                if source_id is not None else None
                            ),
                        })
            rlog["combined_belief"].append(effective)
            rlog["effective_peer_cells"].append(peer_cells)
            rlog.setdefault("peer_provenance", []).append(peer_provenance)
            rlog["direct_blocked_cells"].append([
                (r, c)
                for r in range(robot.belief_map.rows)
                for c in range(robot.belief_map.cols)
                if robot.belief_map.source[r, c] == "self_sensor"
                and CellState(int(robot.belief_map.belief[r, c])) in (
                    CellState.OCCUPIED_DYNAMIC,
                    CellState.TEMPORARILY_BLOCKED,
                )
            ])
            rlog["trust"].append(robot.trust_model.snapshot())
            rlog["accepted_reports"].append(robot.accepted_reports)
            rlog["rejected_reports"].append(robot.rejected_reports)
            rlog["replan_count"].append(robot.replan_count)
            rlog["completed"].append(robot.completed)

            rlog["carrying_item"].append(robot.carrying_item)
            rlog["completed_tasks"].append(robot.completed_tasks)
            rlog["current_goal"].append(robot.goal)        
            rlog["lidar_rays"].append(copy.deepcopy(lidar_rays_by_robot.get(rid, [])))
            rlog["malicious_claim_cells_on_route"].append(
                count_active_malicious_claim_cells_on_route(robot, step)
            )
            rlog["traffic_waits"].append(robot.total_traffic_waits)
            rlog["traffic_deadlock_active"].append(robot.traffic_deadlock_active)
            rlog["active_deadlock_id"].append(robot.active_deadlock_id)
            rlog["traffic_mode"].append(robot.traffic_mode)
            rlog["traffic_replans"].append(robot.traffic_replan_count)

        if all(robot.completed for robot in robots):
            break

    return world, robots, log


def count_active_malicious_claim_cells_on_route(robot, timestamp):
    """Count remaining-route cells currently influenced by malicious claims."""
    if not robot.path:
        return 0

    remaining_footprint_cells = set()
    for anchor in robot.path[robot.path_index:]:
        remaining_footprint_cells.update(robot_footprint_cells(anchor))

    count = 0
    max_age = int(robot.defense_runner.config.max_claim_age)
    for cell in remaining_footprint_cells:
        for claim in robot.defense_runner.claims_for(cell):
            if claim.is_malicious and timestamp - claim.timestamp <= max_age:
                count += 1
                break
    return count


# ============================================================
# Metrics
# ============================================================

def compute_path_distance(position_log):
    if len(position_log) <= 1:
        return 0

    distance = 0

    for i in range(1, len(position_log)):
        r0, c0 = position_log[i - 1]
        r1, c1 = position_log[i]
        distance += abs(r1 - r0) + abs(c1 - c0)

    return distance

def compute_experiment_metrics(robots, log):
    malicious_robot_id = log["malicious_robot_id"]
    tasks_per_robot = int(log.get("tasks_per_robot", TASKS_PER_ROBOT))

    benign_robots = [robot for robot in robots if not robot.is_malicious]
    total_completed = sum(robot.completed_tasks for robot in robots)
    benign_completed = sum(robot.completed_tasks for robot in benign_robots)
    total_possible = max(1, len(robots) * tasks_per_robot)
    benign_possible = max(1, len(benign_robots) * tasks_per_robot)

    events_by_robot = {
        robot.robot_id: log["robots"][robot.robot_id]["events"]
        for robot in robots
    }

    blocked_moves = {rid: events.count("blocked_move") for rid, events in events_by_robot.items()}
    no_path_counts = {rid: events.count("no_path") for rid, events in events_by_robot.items()}
    movement_steps = {
        rid: sum(event in ("moved_cell", "moved_continuous") for event in events)
        for rid, events in events_by_robot.items()
    }
    cell_moves = {rid: events.count("moved_cell") for rid, events in events_by_robot.items()}
    completed_tasks = {robot.robot_id: robot.completed_tasks for robot in robots}
    replans = {robot.robot_id: robot.replan_count for robot in robots}
    distances = {
        robot.robot_id: compute_path_distance(log["robots"][robot.robot_id]["position"])
        for robot in robots
    }
    replans_per_delivery = {
        robot.robot_id: (robot.replan_count / robot.completed_tasks if robot.completed_tasks else None)
        for robot in robots
    }
    movement_steps_per_delivery = {
        robot.robot_id: (movement_steps[robot.robot_id] / robot.completed_tasks if robot.completed_tasks else None)
        for robot in robots
    }

    false_trust_events = [event for event in log.get("trust_events", []) if event["truth_matches"] is False]
    true_trust_events = [event for event in log.get("trust_events", []) if event["truth_matches"] is True]
    malicious_trust_events = [event for event in log.get("trust_events", []) if event["sender_id"] == malicious_robot_id]
    malicious_false_events = [event for event in malicious_trust_events if event["truth_matches"] is False]

    time_to_distrust = None
    for event in malicious_trust_events:
        if event["new_trust"] < TRUST_ACCEPT_THRESHOLD:
            time_to_distrust = event["step"]
            break

    benign_ids = [robot.robot_id for robot in benign_robots]

    attack_start = log.get("attack_phase_start_step")
    if attack_start is None:
        attack_start = len(log.get("truth_grid", []))

    distrust_step_per_robot = {}
    for robot in benign_robots:
        rid = robot.robot_id
        distrust_step = None
        for event in malicious_trust_events:
            if event.get("observer_id") == rid and event["new_trust"] < TRUST_ACCEPT_THRESHOLD:
                distrust_step = event["step"]
                break
        distrust_step_per_robot[rid] = distrust_step

    deliveries_before_attack = {}
    deliveries_after_attack = {}
    deliveries_before_distrust = {}
    deliveries_after_distrust = {}
    first_post_attack_delivery_latency = {}
    first_post_distrust_delivery_latency = {}

    for robot in benign_robots:
        rid = robot.robot_id
        completion_steps = [step for step in robot.delivery_completion_steps if step >= 0]
        deliveries_before_attack[rid] = sum(step < attack_start for step in completion_steps)
        deliveries_after_attack[rid] = sum(step >= attack_start for step in completion_steps)

        post_attack = [step for step in completion_steps if step >= attack_start]
        first_post_attack_delivery_latency[rid] = (
            post_attack[0] - attack_start if post_attack else None
        )

        distrust_step = distrust_step_per_robot[rid]
        if distrust_step is None:
            deliveries_before_distrust[rid] = len(completion_steps)
            deliveries_after_distrust[rid] = 0
            first_post_distrust_delivery_latency[rid] = None
        else:
            deliveries_before_distrust[rid] = sum(step < distrust_step for step in completion_steps)
            deliveries_after_distrust[rid] = sum(step >= distrust_step for step in completion_steps)
            post_distrust = [step for step in completion_steps if step >= distrust_step]
            first_post_distrust_delivery_latency[rid] = (
                post_distrust[0] - distrust_step if post_distrust else None
            )

    replan_reason_counts = {}
    replan_phase_counts = {}
    identical_replans = {}
    next_five_changed_replans = {}
    useful_replan_ratio = {}
    expanded_nodes_total = {}

    for robot in robots:
        rid = robot.robot_id
        reason_counts = {}
        phase_counts = {}
        for event in robot.replan_events:
            reason = event["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            phase = event["phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        replan_reason_counts[rid] = reason_counts
        replan_phase_counts[rid] = phase_counts
        identical = sum(bool(event["identical_path"]) for event in robot.replan_events)
        changed = sum(bool(event["next_five_changed"]) for event in robot.replan_events)
        identical_replans[rid] = identical
        next_five_changed_replans[rid] = changed
        useful_replan_ratio[rid] = (changed / len(robot.replan_events)) if robot.replan_events else None
        expanded_nodes_total[rid] = sum(
            int(event["expanded_nodes"])
            for event in robot.replan_events
            if event.get("expanded_nodes") is not None
        )

    route_exposure_steps = {}
    route_exposure_cell_steps = {}
    peak_malicious_claim_cells_on_route = {}
    for robot in robots:
        rid = robot.robot_id
        series = log["robots"][rid].get("malicious_claim_cells_on_route", [])
        route_exposure_steps[rid] = sum(value > 0 for value in series)
        route_exposure_cell_steps[rid] = sum(series)
        peak_malicious_claim_cells_on_route[rid] = max(series, default=0)

    defense_snapshots = {
        robot.robot_id: robot.defense_runner.snapshot(len(log.get("truth_grid", [])))
        for robot in robots
    }

    return {
        "defense_method": log.get("defense_method"),
        "defense_config": log.get("defense_config", {}),
        "trust_model": TRUST_MODEL_NAME,
        "experiment_mode": log.get("experiment_mode", EXPERIMENT_MODE),
        "attack_mode": ATTACK_MODE,
        "configured_deliveries_per_robot": tasks_per_robot,
        "simulation_steps": len(log.get("truth_grid", [])),
        "all_robot_delivery_success_rate": total_completed / total_possible,
        "benign_delivery_success_rate": benign_completed / benign_possible,
        "benign_mission_complete": all(robot.completed for robot in benign_robots),
        "completed_tasks_per_robot": completed_tasks,
        "grid_distance_per_robot": distances,
        "movement_steps_per_robot": movement_steps,
        "cell_moves_per_robot": cell_moves,
        "replans_per_robot": replans,
        "replans_per_delivery": replans_per_delivery,
        "movement_steps_per_delivery": movement_steps_per_delivery,
        "blocked_moves_per_robot": blocked_moves,
        "no_path_count_per_robot": no_path_counts,
        "benign_total_completed_deliveries": benign_completed,
        "benign_total_replans": sum(replans[rid] for rid in benign_ids),
        "benign_total_grid_distance": sum(distances[rid] for rid in benign_ids),
        "benign_total_movement_steps": sum(movement_steps[rid] for rid in benign_ids),
        "verified_false_reports": len(false_trust_events),
        "verified_true_reports": len(true_trust_events),
        "malicious_verified_false_reports": len(malicious_false_events),
        "time_to_distrust_malicious_robot": time_to_distrust,
        "time_to_distrust_per_benign_robot": distrust_step_per_robot,
        "deliveries_before_attack_per_benign_robot": deliveries_before_attack,
        "deliveries_after_attack_per_benign_robot": deliveries_after_attack,
        "deliveries_before_distrust_per_benign_robot": deliveries_before_distrust,
        "deliveries_after_distrust_per_benign_robot": deliveries_after_distrust,
        "first_post_attack_delivery_latency_per_benign_robot": first_post_attack_delivery_latency,
        "first_post_distrust_delivery_latency_per_benign_robot": first_post_distrust_delivery_latency,
        "replan_reason_counts_per_robot": replan_reason_counts,
        "replan_phase_counts_per_robot": replan_phase_counts,
        "identical_path_replans_per_robot": identical_replans,
        "next_five_changed_replans_per_robot": next_five_changed_replans,
        "useful_replan_ratio_per_robot": useful_replan_ratio,
        "source_linked_replan_suppressed_per_robot": {
            robot.robot_id: dict(robot.source_linked_replan_suppressed)
            for robot in robots
        },
        "source_linked_material_replans_per_robot": {
            robot.robot_id: sum(
                1 for event in robot.replan_events
                if "source_linked_trust_reweight" in event["reason"]
            )
            for robot in robots
        },
        "source_linked_total_route_risk_released_per_robot": {
            robot.robot_id: sum(
                float(event.get("source_linked_route_risk_drop") or 0.0)
                for event in robot.replan_events
            )
            for robot in robots
        },
        "planner_expanded_nodes_total_per_robot": expanded_nodes_total,
        "malicious_route_exposure_steps_per_robot": route_exposure_steps,
        "malicious_route_exposure_cell_steps_per_robot": route_exposure_cell_steps,
        "peak_malicious_claim_cells_on_route_per_robot": peak_malicious_claim_cells_on_route,
        "benign_deliveries_after_attack": sum(deliveries_after_attack.values()),
        "benign_deliveries_after_distrust": sum(deliveries_after_distrust.values()),
        "benign_identical_path_replans": sum(identical_replans[rid] for rid in benign_ids),
        "benign_next_five_changed_replans": sum(next_five_changed_replans[rid] for rid in benign_ids),
        "benign_planner_expanded_nodes_total": sum(expanded_nodes_total[rid] for rid in benign_ids),
        "defense_runner_snapshots": defense_snapshots,
    }

def print_summary(world, robots, log):
    print("Simulation finished")
    print(f"Grid size: {world.rows} x {world.cols}")
    print(f"Goal cell: {log['goal']}")
    print(f"Planning goals: {log.get('goals', [log['goal']])}")
    print(f"Displayed action cells: {log.get('display_goals', [])}")
    print(f"Malicious robot: R{log['malicious_robot_id']}")
    print(f"Total steps: {len(log['truth_grid'])}")
    print(f"Defense method: {log.get('defense_method')}")
    print(f"Defense parameters: {log.get('defense_config', {})}")
    print(f"Deliveries per robot: {log.get('tasks_per_robot', TASKS_PER_ROBOT)}")
    print(f"Reports sent: {len(log['reports'])}")

    malicious_reports = [r for r in log["reports"] if r["is_malicious"]]
    print(f"Malicious reports sent: {len(malicious_reports)}")

    for robot in robots:
        rid = robot.robot_id
        rlog = log["robots"][rid]

        positions = rlog["position"]

        role = "MALICIOUS" if robot.is_malicious else "VICTIM"

        distance = compute_path_distance(positions)
        completed = bool(rlog["completed"][-1]) if rlog["completed"] else False
        final_position = positions[-1] if positions else robot.position

        print()
        print(f"Robot {rid} ({role})")
        print(f"  Completed: {completed}")

        print(f"  Completed tasks: {robot.completed_tasks}/{len(robot.task_queue)}")
        print(f"  Carrying item: {robot.carrying_item}")

        print(f"  Final position: {final_position}")
        print(f"  Grid distance traveled: {distance}")
        print(f"  Replans: {robot.replan_count}")
        identical_count = sum(event["identical_path"] for event in robot.replan_events)
        next_five_count = sum(event["next_five_changed"] for event in robot.replan_events)
        print(f"  Identical-path replans: {identical_count}")
        print(f"  Replans changing next 5 cells: {next_five_count}")
        print(f"  Delivery completion steps: {robot.delivery_completion_steps}")
        print(f"  Accepted reports: {robot.accepted_reports}")
        print(f"  Rejected reports: {robot.rejected_reports}")
        print(f"  Verified true reports: {robot.verified_true_reports}")
        print(f"  Verified false reports: {robot.verified_false_reports}")

        trust_snapshot = robot.trust_model.snapshot()

        if trust_snapshot:
            print("  Trust scores:")

            for peer_id, value in sorted(trust_snapshot.items()):
                if isinstance(value, dict):
                    score = value.get("score", None)
                    if score is not None:
                        print(f"    R{peer_id}: {score:.2f}")
                    else:
                        print(f"    R{peer_id}: {value}")
                else:
                    print(f"    R{peer_id}: {value:.2f}")
        else:
            print("  Trust scores: none")

    metrics = compute_experiment_metrics(robots, log)

    print("\nExperiment metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


# ============================================================
# Visualization
# ============================================================

DISPLAY_ROBOT = 9
DISPLAY_GOAL = 10
DISPLAY_MALICIOUS_FAKE_OBJECT = 11
DISPLAY_PEER_BELIEF = 12
DISPLAY_FALSE_CLEARANCE = 13
DISPLAY_ROBOT0_BELIEF = 14
DISPLAY_ROBOT1_BELIEF = DISPLAY_PEER_BELIEF
DISPLAY_ROBOT2_BELIEF = 15

ROBOT_COLORS = {0: "#8e24aa", 1: "#fb8c00", 2: "#1976d2"}
ROBOT_BELIEF_DISPLAY = {0: DISPLAY_ROBOT0_BELIEF, 1: DISPLAY_ROBOT1_BELIEF, 2: DISPLAY_ROBOT2_BELIEF}

def expand_fake_object_cells(fake_cells):
    expanded = set()

    for cell in fake_cells:
        for footprint_cell in robot_footprint_cells(cell):
            expanded.add(tuple(footprint_cell))

    return sorted(expanded)

def make_display_array(
    grid,
    robot_positions=None,
    goal=None,
    goals=None,
    malicious_fake_objects=None,
    attack_overlays=None,
):
    """
    Display array for the shared truth/debug map.

    Important:
    malicious_fake_objects are display-only attacker artifacts.
    They are not part of world.grid and do not affect collision truth.

    Values:
        0 free
        1 occupied_static
        2 occupied_dynamic
        3 unknown
        4 temporarily_blocked
        5 congested
        6 pickup
        7 dropoff
        8 charging
        9 robot
        10 goal
        11 malicious fake object overlay
    """
    arr = np.array(grid, dtype=int).copy()

    if malicious_fake_objects:
        for r, c in expand_fake_object_cells(malicious_fake_objects):
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                arr[r, c] = DISPLAY_MALICIOUS_FAKE_OBJECT

    for overlay in attack_overlays or ():
        display_state = (
            DISPLAY_FALSE_CLEARANCE
            if overlay.get("attack_type") == "false_clearance"
            else DISPLAY_MALICIOUS_FAKE_OBJECT
        )
        for r, c in overlay.get("cells", ()):
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                arr[r, c] = display_state

    if goals:
        for gr, gc in goals:
            arr[gr, gc] = DISPLAY_GOAL
    elif goal is not None:
        gr, gc = goal
        arr[gr, gc] = DISPLAY_GOAL

    if robot_positions:
        for _, pos in robot_positions.items():
            r, c = pos
            arr[r, c] = DISPLAY_ROBOT

    return arr

def make_heatmap_overlay(traffic_heatmap, grid):
    """
    Returns a masked heatmap so traffic is visible only over non-blocked map cells.
    Static obstacles are masked out.

    Higher values mean more benign robot traffic during reconnaissance.
    """
    heat = np.array(traffic_heatmap, dtype=float).copy()

    blocked_mask = np.zeros_like(heat, dtype=bool)

    rows, cols = heat.shape

    for r in range(rows):
        for c in range(cols):
            if is_blocking_state(grid[r, c]):
                blocked_mask[r, c] = True

    heat[blocked_mask] = np.nan

    return heat

def make_belief_display_array(
    belief_grid,
    robot_position=None,
    goal=None,
    goals=None,
    malicious_fake_objects=None,
    attack_overlays=None,
):
    """
    Display array for a robot belief map.

    Normal robots should not receive malicious_fake_objects here. If they
    accept a fake blocked report, the combined map uses the normal source
    color; the animation adds a red outline to identify its malicious origin.

    Malicious robots may receive malicious_fake_objects here so their own
    view shows the fake object in red.
    """
    arr = np.array(belief_grid, dtype=int).copy()

    if malicious_fake_objects:
        for r, c in malicious_fake_objects:
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                arr[r, c] = DISPLAY_MALICIOUS_FAKE_OBJECT

    for overlay in attack_overlays or ():
        display_state = (
            DISPLAY_FALSE_CLEARANCE
            if overlay.get("attack_type") == "false_clearance"
            else DISPLAY_MALICIOUS_FAKE_OBJECT
        )
        for r, c in overlay.get("cells", ()):
            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                arr[r, c] = display_state

    if goals:
        for gr, gc in goals:
            arr[gr, gc] = DISPLAY_GOAL
    elif goal is not None:
        gr, gc = goal
        arr[gr, gc] = DISPLAY_GOAL

    if robot_position is not None:
        r, c = robot_position
        arr[r, c] = DISPLAY_ROBOT

    return arr


def draw_attack_outlines(ax, cells, color="#d32f2f"):
    """Outline maliciously sourced cells that reached a victim's map."""
    patches = []
    for r, c in cells:
        patch = plt.Rectangle(
            (c - 0.5, r - 0.5),
            1,
            1,
            fill=False,
            edgecolor=color,
            linewidth=2.4,
            zorder=12,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def draw_path(ax, path, color="black", linewidth=1.8, alpha=0.8):
    if not path:
        return None

    rows = [cell[0] for cell in path]
    cols = [cell[1] for cell in path]

    # imshow uses x=col, y=row
    return ax.plot(
        cols,
        rows,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )[0]

def draw_lidar_rays(ax, lidar_rays, color="cyan", linewidth=0.45, alpha=0.18):
    """
    Draw lidar rays on a map panel.

    Points are stored in world coordinates, where x=columns and y=rows.
    imshow displays grid cells by col/row indices, so we convert world
    coordinates into display coordinates by subtracting 0.5 cell.
    """
    lines = []

    if not lidar_rays:
        return lines

    for ray in lidar_rays:
        if len(ray) < 2:
            continue

        xs = [(point[0] / CELL_SIZE) - 0.5 for point in ray]
        ys = [(point[1] / CELL_SIZE) - 0.5 for point in ray]

        line, = ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        lines.append(line)

    return lines

def draw_robot_footprint(ax, anchor_cell, **kwargs):
    """
    Draws the robot footprint as a rectangle.

    anchor_cell is the top-left grid cell of the robot footprint.
    """
    r, c = anchor_cell

    visual_width = ROBOT_FOOTPRINT_COLS * ROBOT_VISUAL_SCALE
    visual_height = ROBOT_FOOTPRINT_ROWS * ROBOT_VISUAL_SCALE
    center_x = c - 0.5 + ROBOT_FOOTPRINT_COLS / 2.0
    center_y = r - 0.5 + ROBOT_FOOTPRINT_ROWS / 2.0

    rect = plt.Rectangle(
        (center_x - visual_width / 2.0, center_y - visual_height / 2.0),
        visual_width,
        visual_height,
        fill=False,
        linewidth=2.0,
        zorder=10,
        **kwargs,
    )

    ax.add_patch(rect)
    return rect

def animate(world, robots, log, map_view=None):
    colors = [
        "#ffffff",  # 0 free
        "#222222",  # 1 occupied_static
        "#555555",  # 2 occupied_dynamic
        "#bdbdbd",  # 3 unknown
        "#66bb6a",  # 4 temporarily_blocked, normal believed blockage
        "#f9a825",  # 5 congested
        "#795548",  # 6 pickup
        "#26a69a",  # 7 dropoff
        "#607d8b",  # 8 charging
        "#00e5ff",  # 9 robot
        "#ffeb3b",  # 10 goal
        "#e53935",  # 11 malicious fake object overlay
        "#fb8c00",  # 12 robot 1 / orange provenance
        "#ef9a9a",  # 13 false-clearance attack overlay
        "#8e24aa",  # 14 robot 0 / purple provenance
        "#1976d2",  # 15 robot 2 / blue provenance
    ]

    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    num_panels = 1 + len(robots)

    fig, axes = plt.subplots(
        1,
        num_panels,
        figsize=(7 * num_panels, 8.6),
        squeeze=False,
    )

    axes = axes[0]

    truth_ax = axes[0]
    belief_axes = {
        robot.robot_id: axes[idx + 1]
        for idx, robot in enumerate(robots)
    }

    truth_ax.set_title("Ground Truth Map", fontsize=13, pad=6)
    truth_ax.set_xlabel("col", fontsize=11)
    truth_ax.set_ylabel("row", fontsize=11)
    truth_ax.tick_params(labelsize=10)

    initial_positions = {}

    truth_img = truth_ax.imshow(
        make_display_array(
            log["truth_grid"][0],
            initial_positions,
            log["goal"],
            goals=log.get("display_goals"),
            attack_overlays=log.get("attack_overlays", [[]])[0],
        ),
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    truth_path_lines = []
    truth_robot_patches = {}
    truth_lidar_lines = []

    for robot in robots:
        rid = robot.robot_id
        anchor_cell = log["robots"][rid]["position"][0]

        edge_color = ROBOT_COLORS.get(rid, "#555555")

        patch = draw_robot_footprint(
            truth_ax,
            anchor_cell,
            edgecolor=edge_color,
            alpha=0.95,
        )

        truth_robot_patches[rid] = patch

    belief_imgs = {}
    belief_path_lines = {}
    belief_robot_patches = {}
    belief_attack_outline_patches = {}

    for robot in robots:
        rid = robot.robot_id
        ax = belief_axes[rid]

        role = "MALICIOUS" if robot.is_malicious else "VICTIM"
        selected_view = map_view or log.get("map_view", "combined")
        view_label = "Combined Belief Map" if selected_view == "combined" else "Local Observation Map"
        ax.set_title(
            f"Robot {rid}\n{view_label}\n({role})",
            fontsize=13,
            pad=6,
            color=ROBOT_COLORS.get(rid, "#555555"),
        )
        ax.set_xlabel("col", fontsize=11)
        ax.set_ylabel("row", fontsize=11)
        ax.tick_params(labelsize=10)

        first_belief = log["robots"][rid].get(
            "combined_belief" if selected_view == "combined" else "local_belief",
            log["robots"][rid]["belief"],
        )[0]
        initial_attack_overlays = (
            log.get("attack_overlays", [[]])[0]
            if robot.is_malicious
            else None
        )

        belief_imgs[rid] = ax.imshow(
            make_belief_display_array(
                first_belief,
                robot_position=None,
                goal=log["goal"],
                goals=log.get("display_goals"),
                attack_overlays=initial_attack_overlays,
            ),
            cmap=cmap,
            norm=norm,
            origin="upper",
        )

        anchor_cell = log["robots"][rid]["position"][0]

        edge_color = ROBOT_COLORS.get(rid, "#555555")

        patch = draw_robot_footprint(
            ax,
            anchor_cell,
            edgecolor=edge_color,
            alpha=0.95,
        )

        belief_robot_patches[rid] = patch
        belief_attack_outline_patches[rid] = []

        belief_path_lines[rid] = []

    # Reserve a dedicated lower band for status, sharing guidance, controls,
    # and the legend so wide multi-panel figures do not overlap or clip them.
    status_text = fig.text(0.02, 0.285, "", fontsize=10, va="top", family="DejaVu Sans Mono")
    sharing_text = fig.text(
        0.02,
        0.14,
        "Peer sharing: occupied cells use the source robot's color. "
        "When multiple trusted robots support a cell, the highest-trust "
        "source color is displayed.",
        fontsize=10,
        va="top",
    )

    max_frames = len(log["truth_grid"])
    display_index = 0
    first_tick = True
    controls_ax = fig.add_axes((0.02, 0.035, 0.30, 0.12))
    controls_ax.set_title("Playback controls", fontsize=10, loc="left", pad=3)
    controls_ax.set_xticks([])
    controls_ax.set_yticks([])
    speed_ax = fig.add_axes((0.035, 0.055, 0.14, 0.075))
    speed_ax.set_title("Speed", fontsize=10, loc="left", pad=2)
    speed = RadioButtons(speed_ax, ("0.5x", "1x", "2x", "5x"), active=1)
    for label in speed.labels:
        label.set_fontsize(10)
    pause_ax = fig.add_axes((0.20, 0.070, 0.085, 0.045))
    pause_button = Button(pause_ax, "Pause", color="#eeeeee", hovercolor="#d0d0d0")
    pause_button.label.set_fontsize(10)
    paused = False

    def toggle_pause(_event):
        nonlocal paused
        paused = not paused
        pause_button.label.set_text("Resume" if paused else "Pause")

    pause_button.on_clicked(toggle_pause)

    def selected_multiplier():
        try:
            # Playback labels are intentionally twice the frame-advance rate:
            # the former 0.5x behavior is now the normal 1x setting.
            return float(speed.value_selected.rstrip("x")) / 2.0
        except (TypeError, ValueError):
            return 1

    fig.legend(
        handles=[
            Patch(facecolor="#222222", label="Static obstacle"),
            Patch(facecolor=ROBOT_COLORS[0], label="Robot 0 source (purple)"),
            Patch(facecolor=ROBOT_COLORS[1], label="Robot 1 source (orange)"),
            Patch(facecolor=ROBOT_COLORS[2], label="Robot 2 source (blue)"),
            Patch(facecolor="#ffeb3b", label="Goal/checkpoint"),
            Patch(facecolor="#e53935", label="Attack overlay"),
            Patch(facecolor="#ef9a9a", label="False clearance"),
            Patch(facecolor="none", edgecolor="#d32f2f", label="Malicious claim in victim map"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.145),
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    def update(frame):
        nonlocal truth_path_lines, truth_lidar_lines, display_index, first_tick
        if first_tick:
            first_tick = False
        elif not paused:
            display_index = min(max_frames - 1, display_index + selected_multiplier())
        frame = int(display_index)
        artists = []

        robot_positions = {}

        truth_img.set_data(
            make_display_array(
                log["truth_grid"][frame],
                robot_positions,
                log["goal"],
                goals=log.get("display_goals"),
                attack_overlays=log.get("attack_overlays", [[]])[frame],
            )
        )
        artists.append(truth_img)

        for robot in robots:
            rid = robot.robot_id
            r, c = log["robots"][rid]["position"][frame]

            truth_robot_patches[rid].set_xy((c - 0.5, r - 0.5))

            artists.append(truth_robot_patches[rid])

        for line in truth_path_lines:
            line.remove()
        truth_path_lines = []

        for line in truth_lidar_lines:
            line.remove()
        truth_lidar_lines = []

        if SHOW_LIDAR_RAYS:
            for robot in robots:
                rid = robot.robot_id
                lidar_rays = log["robots"][rid]["lidar_rays"][frame]

                ray_color = "cyan" if not robot.is_malicious else "magenta"

                new_lidar_lines = draw_lidar_rays(
                    truth_ax,
                    lidar_rays,
                    color=ray_color,
                    linewidth=0.35,
                    alpha=0.16,
                )

                truth_lidar_lines.extend(new_lidar_lines)
                artists.extend(new_lidar_lines)

        for robot in robots:
            rid = robot.robot_id
            path = log["robots"][rid]["path"][frame]

            if path:
                color = ROBOT_COLORS.get(rid, "#555555")
                line = draw_path(
                    truth_ax,
                    path,
                    color=color,
                    linewidth=1.2,
                    alpha=0.45,
                )
                truth_path_lines.append(line)
                artists.append(line)

        for robot in robots:
            rid = robot.robot_id
            belief = log["robots"][rid].get(
                "combined_belief" if selected_view == "combined" else "local_belief",
                log["robots"][rid]["belief"],
            )[frame]
            position = log["robots"][rid]["position"][frame]

            attack_overlay = (
                log.get("attack_overlays", [[]])[frame]
                if robot.is_malicious
                else None
            )

            belief_imgs[rid].set_data(
                make_belief_display_array(
                    belief,
                    robot_position=None,
                    goal=log["goal"],
                    goals=log.get("display_goals"),
                    attack_overlays=attack_overlay,
                )
            )
            artists.append(belief_imgs[rid])

            r, c = log["robots"][rid]["position"][frame]

            belief_robot_patches[rid].set_xy((c - 0.5, r - 0.5))

            artists.append(belief_robot_patches[rid])

            for outline in belief_attack_outline_patches[rid]:
                outline.remove()
            belief_attack_outline_patches[rid] = []

            if selected_view == "combined" and not robot.is_malicious:
                provenance_frames = log["robots"][rid].get("peer_provenance", [])
                provenance_at_frame = (
                    provenance_frames[frame]
                    if frame < len(provenance_frames)
                    else ()
                )
                malicious_cells = [
                    tuple(provenance["cell"])
                    for provenance in provenance_at_frame
                    if 0 in provenance.get("senders", ())
                ]
                belief_attack_outline_patches[rid] = draw_attack_outlines(
                    belief_axes[rid], malicious_cells
                )
                artists.extend(belief_attack_outline_patches[rid])

            for line in belief_path_lines[rid]:
                line.remove()
            belief_path_lines[rid] = []

            path = log["robots"][rid]["path"][frame]

            if path:
                color = ROBOT_COLORS.get(rid, "#555555")
                line = draw_path(
                    belief_axes[rid],
                    path,
                    color=color,
                    linewidth=1.2,
                    alpha=0.45,
                )
                belief_path_lines[rid].append(line)
                artists.append(line)

        malicious_robot_id = log["malicious_robot_id"]

        report_count = sum(1 for r in log["reports"] if r["step"] <= frame)
        malicious_report_count = sum(
            1 for r in log["reports"]
            if r["step"] <= frame and r["is_malicious"]
        )

        grace_active = frame < SPAWN_COLLISION_GRACE_STEPS

        phase = log.get("phase", ["UNKNOWN"] * max_frames)[frame]

        attack_start = log.get("attack_phase_start_step")
        attack_start_text = attack_start if attack_start is not None else "not yet"

        actual_step = frame
        latest = None
        for event in log.get("attack_events", ()):
            if int(event.get("step", -1)) <= actual_step:
                if latest is None or int(event["step"]) >= int(latest["step"]):
                    latest = event
        friendly = {"fake_obstacle": "Fake Obstacle", "false_clearance": "False Clearance", "stale_reassertion": "Stale Reassertion"}
        latest_text = "None" if latest is None else f"{friendly.get(latest.get('attack_type'), latest.get('attack_type'))} - Step {latest['step']}"
        overlay_count = sum(
            len(item.get("cells", ()))
            for item in log.get("attack_overlays", [[]])[frame]
        )
        status_lines = [
            " | ".join([
                f"Step: {actual_step}",
                f"Phase: {phase}",
                f"Attack starts: {attack_start_text}",
                f"Spawn grace: {grace_active}",
            ]),
            " | ".join([
                f"Reports: {report_count}",
                f"Malicious reports: {malicious_report_count}",
                f"Attack overlays: {overlay_count}",
                f"Malicious robot: R{malicious_robot_id}",
                f"Latest attack: {latest_text}",
            ]),
        ]

        for robot in robots:
            rid = robot.robot_id
            accepted = log["robots"][rid]["accepted_reports"][frame]
            rejected = log["robots"][rid]["rejected_reports"][frame]
            replans = log["robots"][rid]["replan_count"][frame]
            completed = log["robots"][rid]["completed"][frame]

            carrying = log["robots"][rid]["carrying_item"][frame]
            completed_tasks = log["robots"][rid]["completed_tasks"][frame]    

            status_lines.append(
                f"R{rid}: tasks={completed_tasks}, carrying={carrying}, "
                f"accepted={accepted}, rejected={rejected}, "
                f"replans={replans}, done={completed}"
            )

        status_text.set_text("\n".join(status_lines))
        artists.extend((status_text, sharing_text))

        return artists

    anim = FuncAnimation(
        fig,
        update,
        # The slowest setting advances by 0.25 frame per callback; allocate
        # enough callbacks that every playback speed reaches the final frame.
        frames=(max_frames * 4) + 1,
        interval=ANIMATION_INTERVAL_MS,
        blit=False,
        repeat=False,
    )

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.32, top=0.88, wspace=0.12)
    plt.show()

    return anim

def show_recon_heatmap(world, log, output_path=None):
    """
    Shows the learned reconnaissance heatmap at the end of phase 1.

    This is a static diagnostic view: where benign robots actually moved before
    the attacker began injecting fake obstacles.
    """
    if "traffic_heatmap" not in log or not log["traffic_heatmap"]:
        print("No traffic heatmap found in log.")
        return

    attack_start = log.get("attack_phase_start_step")

    if attack_start is None:
        recon_frame = len(log["traffic_heatmap"]) - 1
    else:
        recon_frame = max(0, min(attack_start, len(log["traffic_heatmap"]) - 1))

    heat = make_heatmap_overlay(
        log["traffic_heatmap"][recon_frame],
        log["truth_grid"][recon_frame],
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(
        make_display_array(
            log["truth_grid"][recon_frame],
            robot_positions=None,
            goal=log["goal"],
            goals=log.get("display_goals"),
        ),
        origin="upper",
        alpha=0.35,
    )

    heat_img = ax.imshow(
        heat,
        origin="upper",
        alpha=0.80,
        cmap="hot",
    )

    ax.set_title(
        f"Reconnaissance Heatmap at Step {recon_frame} "
        f"(higher = more benign traffic)"
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    fig.colorbar(
        heat_img,
        ax=ax,
        label="benign traffic count",
    )

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-robot shared-map attack and defense simulator"
    )

    parser.add_argument("--map-npy", type=str, default=None)
    parser.add_argument("--map-movingai", type=str, default=None)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--defense-method", choices=DEFENSE_METHODS, default=DEFENSE_METHOD)
    parser.add_argument("--trust-threshold", type=float, default=TRUST_ACCEPT_THRESHOLD)

    parser.add_argument("--decay-rate", type=float, default=0.006)
    parser.add_argument("--cost-scale", type=float, default=14.0)
    parser.add_argument("--cost-exponent", type=float, default=1.5)
    parser.add_argument("--blocked-probability-threshold", type=float, default=0.70)
    parser.add_argument("--max-claim-age", type=int, default=900)
    parser.add_argument("--congested-impact", type=float, default=0.50)
    parser.add_argument("--duplicate-window-steps", type=int, default=0)

    parser.add_argument("--deliveries-per-robot", type=int, default=TASKS_PER_ROBOT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--experiment-mode", choices=("clean", "attack"), default=EXPERIMENT_MODE)

    args = parser.parse_args()
    if args.map_npy and args.map_movingai:
        parser.error("use only one of --map-npy or --map-movingai")
    if args.deliveries_per_robot < 1:
        parser.error("--deliveries-per-robot must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if not 0.0 <= args.trust_threshold <= 1.0:
        parser.error("--trust-threshold must be in [0, 1]")
    return args

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    if args.map_npy is not None or args.map_movingai is not None:
        if args.map_movingai is not None:
            prior_grid = load_grid_from_movingai_map(args.map_movingai)
        else:
            prior_grid = load_grid_from_npy(args.map_npy)

        if ENABLE_AUTO_TEMP_OBJECTS_FOR_LOADED_MAPS:
            grid = make_dynamic_grid_with_auto_temporary_objects(prior_grid)
        else:
            grid = prior_grid.copy()
    else:
        prior_grid = make_demo_static_grid()
        grid = make_demo_dynamic_grid(prior_grid)

    defense_config = {
        "trust_threshold": args.trust_threshold,
        "decay_rate": args.decay_rate,
        "cost_scale": args.cost_scale,
        "cost_exponent": args.cost_exponent,
        "blocked_probability_threshold": args.blocked_probability_threshold,
        "max_claim_age": args.max_claim_age,
        "congested_impact": args.congested_impact,
        "duplicate_window_steps": args.duplicate_window_steps,
    }

    world, robots, log = run_simulation(
        grid=grid,
        prior_grid=prior_grid,
        defense_method=args.defense_method,
        defense_config=defense_config,
        tasks_per_robot=args.deliveries_per_robot,
        max_steps=args.max_steps,
        random_seed=args.random_seed,
        experiment_mode=args.experiment_mode,
    )

    # DEBUG: inspect why the malicious robot did or did not move
    from collections import Counter

    rid = log["malicious_robot_id"]

    print("\n--- MALICIOUS ROBOT DEBUG ---")
    print("malicious robot:", rid)
    print(Counter(log["robots"][rid]["events"]))
    print("start:", log["robots"][rid]["position"][0])
    print("first 20 positions:", log["robots"][rid]["position"][:20])
    print("first 20 goals:", log["robots"][rid]["current_goal"][:20])
    print("first 20 carrying:", log["robots"][rid]["carrying_item"][:20])
    print("first 20 replans:", log["robots"][rid]["replan_count"][:20])
    print("--- END DEBUG ---\n")

    print_summary(world, robots, log)

    if SHOW_ANIMATION and not args.no_animation:
        show_recon_heatmap(world, log)
        animate(world, robots, log)
