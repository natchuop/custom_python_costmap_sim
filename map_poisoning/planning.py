"""Small deterministic A* planner used by the headless experimental core."""
from __future__ import annotations

import heapq
import math

Cell = tuple[int, int]


def astar(start: Cell, goal: Cell, traversable_cost) -> list[Cell] | None:
    """Return a four-neighbour path, excluding ``start``, or None when blocked.

    The frontier stores the g-cost that produced each heap entry.  Weighted
    grids can discover a cheaper path to a cell after an older entry has
    already been queued; stale entries are skipped instead of re-expanding the
    same cell repeatedly.  This preserves deterministic A* behavior while
    preventing severe late-run slowdowns on changing costmaps.
    """
    frontier: list[tuple[float, int, float, Cell]] = [(0.0, 0, 0.0, start)]
    costs = {start: 0.0}
    parents: dict[Cell, Cell] = {}
    sequence = 0

    while frontier:
        _, _, popped_cost, cell = heapq.heappop(frontier)
        best_cost = costs.get(cell, math.inf)
        if popped_cost != best_cost:
            continue
        if cell == goal:
            path = []
            while cell != start:
                path.append(cell)
                cell = parents[cell]
            return list(reversed(path))

        for delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            step_cost = traversable_cost(nxt)
            if math.isinf(step_cost):
                continue
            candidate = popped_cost + step_cost
            if candidate < costs.get(nxt, math.inf):
                costs[nxt] = candidate
                parents[nxt] = cell
                sequence += 1
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(frontier, (candidate + heuristic, sequence, candidate, nxt))
    return None
