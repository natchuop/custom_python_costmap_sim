"""Small deterministic A* planner used by the headless experimental core."""
from __future__ import annotations

import heapq
import math

Cell = tuple[int, int]

def astar(start: Cell, goal: Cell, traversable_cost) -> list[Cell] | None:
    """Return a four-neighbour path, excluding ``start``, or None when blocked."""
    frontier: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    costs = {start: 0.0}; parents: dict[Cell, Cell] = {}; sequence = 0
    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell == goal:
            path=[]
            while cell != start: path.append(cell); cell=parents[cell]
            return list(reversed(path))
        for delta in ((-1,0),(0,-1),(0,1),(1,0)):
            nxt=(cell[0]+delta[0],cell[1]+delta[1]); cost=traversable_cost(nxt)
            if math.isinf(cost): continue
            candidate=costs[cell]+cost
            if candidate < costs.get(nxt, math.inf):
                costs[nxt]=candidate; parents[nxt]=cell; sequence+=1
                heuristic=abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1])
                heapq.heappush(frontier,(candidate+heuristic,sequence,nxt))
    return None
