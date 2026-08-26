"""Multi-robot traffic coordination ported from main's sim2 intent model.

Modular robots occupy one cell each, but the approval/yield/deadlock rules match
the vertex, swap, reservation, and deadlock behavior exercised on main.

Narrow corridors are single-file: a robot that sees another body in the hold
zone waits at the mouth, and a non-owner already inside reverses out until
the corridor is clear. Head-on swaps yield immediately instead of freezing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .planning import astar
from .robot import ModularRobot

TRAFFIC_DEADLOCK_WAIT_THRESHOLD = 10
TRAFFIC_JOINT_REPEAT_THRESHOLD = 5
TRAFFIC_YIELD_SEARCH_RADIUS = 20


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    return [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]


def _can_enter(world, cell: tuple[int, int], step: int, occupied: set[tuple[int, int]]) -> bool:
    from .models import ClaimType

    row, col = cell
    if not (0 <= row < world.static_grid.shape[0] and 0 <= col < world.static_grid.shape[1]):
        return False
    if cell in occupied:
        return False
    return world.state(cell, step) == ClaimType.FREE


def other_robot_footprints(robot: ModularRobot, robots: list[ModularRobot]) -> set[tuple[int, int]]:
    """Cells occupied now, plus the cell each other robot intends to step into."""
    cells: set[tuple[int, int]] = set()
    for other in robots:
        if other is robot or other.completed:
            continue
        cells.update(other.reserved_cells())
    return cells


def _traffic_yield_target(robot: ModularRobot, robots: list[ModularRobot], world, step: int):
    occupied = other_robot_footprints(robot, robots)
    candidates: list[tuple[tuple[int, int], int]] = []
    for distance, previous in enumerate(reversed(robot.position_history), start=1):
        if tuple(previous) != tuple(robot.position):
            candidates.append((tuple(previous), distance))
    frontier = [(tuple(robot.position), 0)]
    seen = {tuple(robot.position)}
    while frontier:
        cell, distance = frontier.pop(0)
        if distance >= TRAFFIC_YIELD_SEARCH_RADIUS:
            continue
        for neighbor in _neighbors(cell):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append((neighbor, distance + 1))
            candidates.append((neighbor, distance + 1))
    valid = []
    for candidate, distance in candidates:
        if candidate == tuple(robot.position) or candidate in occupied:
            continue
        if not _can_enter(world, candidate, step, occupied):
            continue
        degree = sum(_can_enter(world, neighbor, step, occupied) for neighbor in _neighbors(candidate))
        clearance = sum(
            _can_enter(world, (candidate[0] + drow, candidate[1] + dcol), step, occupied)
            for drow in (-1, 0, 1)
            for dcol in (-1, 0, 1)
            if (drow, dcol) != (0, 0)
        )
        valid.append((candidate, distance, degree, clearance))
    if not valid:
        return None
    passing = [item for item in valid if item[2] >= 3]
    pool = passing or valid
    return max(pool, key=lambda item: (item[2], item[3], -item[1]))[0]


def _reverse_out_cell(robot: ModularRobot, world, step: int, occupied: set[tuple[int, int]]):
    """Back out of a one-cell corridor along recent history, then to a wider cell."""
    for previous in reversed(robot.position_history):
        cell = tuple(previous)
        if cell == tuple(robot.position) or cell in occupied:
            continue
        if not _can_enter(world, cell, step, occupied):
            continue
        degree = sum(_can_enter(world, neighbor, step, occupied) for neighbor in _neighbors(cell))
        if degree >= 3:
            return cell
    for previous in reversed(robot.position_history):
        cell = tuple(previous)
        if cell != tuple(robot.position) and _can_enter(world, cell, step, occupied):
            return cell
    return None


def _start_robot_yield(
    robot: ModularRobot,
    blocker_id: int | None,
    blocked_cell: tuple[int, int] | None,
    robots: list[ModularRobot],
    world,
    step: int,
):
    occupied = other_robot_footprints(robot, robots)
    target = _traffic_yield_target(robot, robots, world, step) or _reverse_out_cell(robot, world, step, occupied)
    if target is None:
        return None
    robot.traffic_mode = "YIELDING"
    robot.traffic_blocked_by = blocker_id
    robot.active_yield_target = tuple(target)
    robot.yield_blocked_cell = tuple(blocked_cell) if blocked_cell else None
    robot.yield_conflict_cells = frozenset({tuple(blocked_cell)}) if blocked_cell is not None else frozenset()
    robot.saved_yield_goal = tuple(robot.goal)
    robot.saved_yield_path = list(robot.path or ())
    robot.path = astar(
        robot.position,
        target,
        lambda cell: 1.0 if _can_enter(world, cell, step, occupied) else float("inf"),
    )
    if not robot.path:
        robot.traffic_mode = "NORMAL"
        robot.active_yield_target = None
        robot.saved_yield_path = None
        robot.saved_yield_goal = None
        return None
    robot.traffic_yield_count += 1
    return {
        "step": step,
        "event_type": "traffic_yield_started",
        "robot_id": robot.robot_id,
        "other_robot_ids": (blocker_id,) if blocker_id is not None else (),
        "requested_cell": blocked_cell,
        "yield_target": tuple(target),
        "deadlock_id": robot.active_deadlock_id,
    }


def _restore_robot_goal_after_yield(robot: ModularRobot, step: int):
    if robot.saved_yield_path is None and robot.saved_yield_goal is None:
        return None
    deadlock_id = robot.active_deadlock_id
    robot.traffic_deadlock_active = False
    robot.active_deadlock_id = None
    robot.traffic_mode = "NORMAL"
    robot.traffic_blocked_by = None
    robot.active_yield_target = None
    robot.yield_blocked_cell = None
    robot.yield_conflict_cells = frozenset()
    robot.path = None
    robot.saved_yield_path = None
    robot.saved_yield_goal = None
    robot.consecutive_traffic_waits = 0
    return {
        "step": step,
        "event_type": "traffic_deadlock_recovered" if deadlock_id is not None else "traffic_yield_completed",
        "robot_id": robot.robot_id,
        "other_robot_ids": (),
        "deadlock_id": deadlock_id,
    }


def _resume_cell(robot: ModularRobot) -> tuple[int, int] | None:
    if robot.saved_yield_path:
        return tuple(robot.saved_yield_path[0])
    if robot.saved_yield_goal is not None:
        return tuple(robot.saved_yield_goal)
    return None


@dataclass
class TrafficState:
    next_deadlock_id: int = 1
    last_joint_positions: tuple | None = None
    same_joint_state_streak: int = 0
    corridor_by_cell: dict = field(default_factory=dict)
    corridor_segments: dict = field(default_factory=dict)


def build_narrow_corridor_topology(grid):
    """Return corridor membership for one-cell-wide segments between junctions."""
    rows, cols = grid.shape
    free = {
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if not grid[row, col]
    }
    neighbors = lambda cell: [candidate for candidate in _neighbors(cell) if candidate in free]
    degree = {cell: len(neighbors(cell)) for cell in free}
    endpoints = {cell for cell in free if degree[cell] != 2}
    segments = {}
    corridor_by_cell = {}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
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
            hold = list(chain[1:-1])
            for end in (chain[0], chain[-1]):
                if degree.get(end, 0) <= 2:
                    hold.append(end)
            corridor_id = f"C{corridor_index}"
            corridor_index += 1
            segments[corridor_id] = {
                "cells": tuple(chain),
                "hold_cells": frozenset(hold),
                "endpoint_a": chain[0],
                "endpoint_b": chain[-1],
                "owner_robot_id": None,
            }
            for cell in chain:
                corridor_by_cell[cell] = corridor_id
    return corridor_by_cell, segments


def _propose_intent(robot: ModularRobot) -> dict:
    current = {tuple(robot.position)}
    target_cell = robot.proposed_next_cell()
    target = {target_cell} if target_cell is not None else set()
    return {
        "robot": robot,
        "current": current,
        "target": target,
        "target_anchor": target_cell,
        "approved": False,
    }


def _corridor_hold(state: TrafficState, cell: tuple[int, int] | None) -> frozenset[tuple[int, int]]:
    if cell is None:
        return frozenset()
    corridor_id = state.corridor_by_cell.get(tuple(cell))
    if not corridor_id:
        return frozenset()
    segment = state.corridor_segments.get(corridor_id) or {}
    return frozenset(segment.get("hold_cells") or ())


def _corridor_occupants(hold: frozenset[tuple[int, int]], robots: list[ModularRobot], exclude: ModularRobot | None = None):
    return [
        robot
        for robot in robots
        if robot is not exclude and not robot.completed and tuple(robot.position) in hold
    ]


def _try_approve_yield_step(robot: ModularRobot, intents: dict, approved: dict, world, step: int, robots: list[ModularRobot]) -> bool:
    nxt = robot.proposed_next_cell()
    if nxt is None:
        return False
    occupied = other_robot_footprints(robot, robots)
    if not _can_enter(world, nxt, step, occupied):
        return False
    for other in intents.values():
        if other["robot"] is robot:
            continue
        if other["approved"] and nxt in other["target"]:
            return False
    approved[robot.robot_id] = True
    item = intents[robot.robot_id]
    item["approved"] = True
    item["target"] = {nxt}
    item["target_anchor"] = nxt
    robot.consecutive_traffic_waits = 0
    return True


def coordinate_robot_intents(
    robots: list[ModularRobot],
    world,
    step: int,
    traffic_state: TrafficState | None = None,
) -> tuple[dict[int, bool], list[dict]]:
    """Approve frozen movement intents before any robot commits motion."""
    state = traffic_state or TrafficState()
    if not state.corridor_by_cell:
        state.corridor_by_cell, state.corridor_segments = build_narrow_corridor_topology(world.static_grid)
    events: list[dict] = []
    robots_by_id = {robot.robot_id: robot for robot in robots}

    for idle in robots:
        if not idle.completed or idle.traffic_mode != "NORMAL":
            continue
        for active in robots:
            if active is idle or active.completed:
                continue
            requested = active.proposed_next_cell()
            if requested is None or tuple(requested) != tuple(idle.position):
                continue
            parked = _start_robot_yield(idle, active.robot_id, requested, robots, world, step)
            if parked:
                idle.saved_yield_goal = None
                idle.saved_yield_path = None
                parked = dict(parked)
                parked["reason"] = "completed_robot_parking"
                events.append(parked)
            break

    for robot in robots:
        if robot.completed or robot.traffic_mode not in ("NORMAL", "YIELDING", "YIELDING_PARKED"):
            continue
        if (
            robot.traffic_mode == "YIELDING"
            and robot.active_yield_target is not None
            and tuple(robot.position) == tuple(robot.active_yield_target)
        ):
            robot.traffic_mode = "YIELDING_PARKED"
            robot.path = None

    for robot in robots:
        if robot.traffic_mode != "YIELDING_PARKED" or robot.saved_yield_goal is None:
            continue
        other_positions = {tuple(other.position) for other in robots if other is not robot}
        resume = _resume_cell(robot)
        partner_needs_parked_cell = any(
            other.proposed_next_cell() == tuple(robot.position)
            for other in robots
            if other is not robot
        )
        if partner_needs_parked_cell:
            continue
        if resume is not None and resume in other_positions:
            continue
        recovered = _restore_robot_goal_after_yield(robot, step)
        if recovered:
            events.append(recovered)

    intents = {robot.robot_id: _propose_intent(robot) for robot in robots}
    joint = tuple(sorted((robot.robot_id, tuple(robot.position)) for robot in robots))
    if state.last_joint_positions == joint:
        state.same_joint_state_streak += 1
    else:
        state.same_joint_state_streak = 1
    state.last_joint_positions = joint
    repeated_joint = state.same_joint_state_streak >= TRAFFIC_JOINT_REPEAT_THRESHOLD

    ordered = sorted(
        intents.values(),
        key=lambda item: (-item["robot"].consecutive_traffic_waits, item["robot"].robot_id),
    )
    swap_pairs: set[frozenset[int]] = set()
    for left in intents.values():
        for right in intents.values():
            if left is right:
                continue
            if left["target"] & right["current"] and right["target"] & left["current"]:
                swap_pairs.add(frozenset((left["robot"].robot_id, right["robot"].robot_id)))

    approved: dict[int, bool] = {}
    denied: list[dict] = []
    for item in ordered:
        robot = item["robot"]
        target = item["target"]
        if not target:
            approved[robot.robot_id] = True
            item["approved"] = True
            continue
        conflict_kind = None
        blockers: list[int] = []
        for other in intents.values():
            if other is item:
                continue
            other_robot = other["robot"]
            pair = frozenset((robot.robot_id, other_robot.robot_id))
            if pair in swap_pairs:
                conflict_kind = "traffic_swap_conflict"
                blockers.append(other_robot.robot_id)
                break
            if target & other["target"] and other["approved"]:
                conflict_kind = "traffic_vertex_conflict"
                blockers.append(other_robot.robot_id)
                break
            if target & other["current"] and (not other["approved"] or not other["target"]):
                conflict_kind = "traffic_reservation_conflict"
                blockers.append(other_robot.robot_id)
                break
        if conflict_kind is None and item["target_anchor"] is not None:
            hold = _corridor_hold(state, tuple(item["target_anchor"]))
            if hold:
                occupants = _corridor_occupants(hold, robots, exclude=robot)
                entering_hold = tuple(item["target_anchor"]) in hold and tuple(robot.position) not in hold
                if entering_hold and occupants:
                    conflict_kind = "traffic_corridor_conflict"
                    blockers = [other.robot_id for other in occupants]
        if conflict_kind:
            approved[robot.robot_id] = False
            item["approved"] = False
            robot.consecutive_traffic_waits += 1
            robot.traffic_blocked_by = blockers[0] if blockers else None
            event = {
                "step": step,
                "event_type": conflict_kind,
                "robot_id": robot.robot_id,
                "other_robot_ids": tuple(blockers),
                "requested_cell": item["target_anchor"],
                "wait_age": robot.consecutive_traffic_waits,
            }
            events.append(event)
            denied.append(event)
            if (
                robot.consecutive_traffic_waits >= TRAFFIC_DEADLOCK_WAIT_THRESHOLD or repeated_joint
            ) and not robot.traffic_deadlock_active:
                robot.traffic_deadlock_active = True
                number = int(state.next_deadlock_id)
                state.next_deadlock_id = number + 1
                robot.active_deadlock_id = f"deadlock-{number:06d}"
                events.append(
                    {
                        "step": step,
                        "event_type": "traffic_deadlock_detected",
                        "robot_id": robot.robot_id,
                        "other_robot_ids": tuple(blockers),
                        "requested_cell": item["target_anchor"],
                        "wait_age": robot.consecutive_traffic_waits,
                        "deadlock_id": robot.active_deadlock_id,
                    }
                )
        else:
            approved[robot.robot_id] = True
            item["approved"] = True

    def _yield_if_possible(robot: ModularRobot, blocker_id: int | None, requested) -> bool:
        if robot.traffic_mode != "NORMAL":
            return False
        yielded = _start_robot_yield(robot, blocker_id, requested, robots, world, step)
        if not yielded:
            return False
        events.append(yielded)
        _try_approve_yield_step(robot, intents, approved, world, step, robots)
        return True

    for pair in swap_pairs:
        pair_robots = [robots_by_id[robot_id] for robot_id in pair if robot_id in robots_by_id]
        candidates = [robot for robot in pair_robots if robot.traffic_mode == "NORMAL"]
        if not candidates:
            continue
        requested = next(
            (
                event.get("requested_cell")
                for event in reversed(denied)
                if event.get("robot_id") in pair
            ),
            None,
        )
        ordered_yielders = sorted(
            candidates,
            key=lambda robot: (-robot.consecutive_traffic_waits, -robot.robot_id),
        )
        blocker = next((robot.robot_id for robot in pair_robots if robot is not ordered_yielders[0]), None)
        for yielder in ordered_yielders:
            if _yield_if_possible(yielder, blocker, requested):
                break

    for robot in robots:
        if robot.completed or robot.traffic_mode != "NORMAL":
            continue
        hold = _corridor_hold(state, tuple(robot.position))
        if not hold or tuple(robot.position) not in hold:
            continue
        occupants = _corridor_occupants(hold, robots)
        if len(occupants) < 2:
            continue
        owner = min(occupants, key=lambda item: item.robot_id)
        if robot is owner:
            continue
        requested = robot.proposed_next_cell()
        _yield_if_possible(robot, owner.robot_id, requested)

    deadlocked = [
        robot
        for robot in robots
        if robot.traffic_deadlock_active and robot.traffic_mode == "NORMAL"
    ]
    if deadlocked and (
        repeated_joint
        or any(robot.consecutive_traffic_waits >= TRAFFIC_DEADLOCK_WAIT_THRESHOLD for robot in deadlocked)
    ):
        yielding = min(deadlocked, key=lambda robot: (robot.consecutive_traffic_waits, robot.robot_id))
        requested = next(
            (
                event.get("requested_cell")
                for event in reversed(events)
                if event.get("robot_id") == yielding.robot_id
                and str(event.get("event_type", "")).startswith("traffic_")
            ),
            None,
        )
        _yield_if_possible(yielding, yielding.traffic_blocked_by, requested)

    return approved, events


def summarize_traffic_events(events: list[dict]) -> dict[str, int]:
    counts = {
        "vertex_conflicts_detected": 0,
        "head_on_swap_conflicts_detected": 0,
        "reservation_conflicts_detected": 0,
        "traffic_yield_events": 0,
        "traffic_yields_completed": 0,
        "deadlocks_detected": 0,
        "deadlocks_recovered": 0,
        "robot_overlap_violations": 0,
        "corridor_entry_denied": 0,
    }
    mapping = {
        "traffic_vertex_conflict": "vertex_conflicts_detected",
        "traffic_swap_conflict": "head_on_swap_conflicts_detected",
        "traffic_reservation_conflict": "reservation_conflicts_detected",
        "traffic_yield_started": "traffic_yield_events",
        "traffic_yield_completed": "traffic_yields_completed",
        "traffic_overlap_violation": "robot_overlap_violations",
        "traffic_corridor_conflict": "corridor_entry_denied",
    }
    detected_ids: set[str] = set()
    recovered_ids: set[str] = set()
    for event in events:
        event_type = str(event.get("event_type"))
        key = mapping.get(event_type)
        if key:
            counts[key] += 1
        deadlock_id = event.get("deadlock_id")
        if not deadlock_id:
            continue
        deadlock_id = str(deadlock_id)
        if event_type == "traffic_deadlock_detected":
            detected_ids.add(deadlock_id)
        elif event_type == "traffic_deadlock_recovered":
            recovered_ids.add(deadlock_id)
    # Deadlock statistics are episode counts, not raw event counts.  Only a
    # recovery paired with a previously detected deadlock is a recovery.
    counts["deadlocks_detected"] = len(detected_ids)
    counts["deadlocks_recovered"] = len(detected_ids & recovered_ids)
    return counts
