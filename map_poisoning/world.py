"""Truth-map loading and deterministic temporary obstacle episodes."""
from __future__ import annotations
import numpy as np
from .models import ClaimType, TemporaryObstacleEpisode
from .rng import named_rng

def demo_grid(rows: int = 32, cols: int = 42) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=np.uint8)
    grid[[0, -1], :] = 1; grid[:, [0, -1]] = 1
    grid[8:25, 14] = 1; grid[8:25, 28] = 1; grid[16, 14:29] = 1
    grid[11, 14] = 0; grid[21, 14] = 0; grid[12, 28] = 0; grid[23, 28] = 0
    return grid

class World:
    def __init__(self, static_grid: np.ndarray, episodes: tuple[TemporaryObstacleEpisode, ...]):
        self.static_grid = static_grid
        self.episodes = episodes

    def truth_grid(self, step: int) -> np.ndarray:
        """Return the binary physical occupancy grid at ``step``.

        The modular simulator deliberately has no dependency on the former
        continuous-motion engine: zero is free and one is physically blocked.
        """
        grid = np.array(self.static_grid, dtype=np.uint8)
        for episode in self.episodes:
            if episode.appearance_step <= step < episode.clearance_step:
                for cell in episode.cells:
                    grid[cell] = 1
        return grid

    def state(self, cell: tuple[int, int], step: int) -> ClaimType:
        r, c = cell
        if not (0 <= r < self.static_grid.shape[0] and 0 <= c < self.static_grid.shape[1]) or self.static_grid[cell]: return ClaimType.BLOCKED
        return ClaimType.BLOCKED if any(cell in e.cells and e.appearance_step <= step < e.clearance_step for e in self.episodes) else ClaimType.FREE
    def active(self, step: int): return [e for e in self.episodes if e.appearance_step <= step < e.clearance_step]
    def cleared(self, step: int): return [e for e in self.episodes if e.clearance_step <= step]

def make_episodes(grid: np.ndarray, seed: int, total_steps: int, period: int = 150) -> tuple[TemporaryObstacleEpisode, ...]:
    from .obstacles import author_temporary_obstacle_episodes
    return author_temporary_obstacle_episodes(grid, named_rng(seed, "temporary_obstacles"), total_steps, period)
