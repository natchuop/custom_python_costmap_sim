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
        self._activation_steps: dict[str, int] = {}
        self._effective_active_by_step: dict[int, frozenset[str]] = {}
        self.deferred_activation_steps = 0
        self.deferred_episode_ids: set[str] = set()

    def begin_step(self, step: int, occupied_cells=()) -> np.ndarray:
        """Resolve obstacle onset without spawning a footprint on a robot.

        If a footprint is occupied when it is due to appear, activation yields
        until the whole footprint is empty. The authored clearance time remains
        unchanged, and the same rule is applied to every replay.
        """
        occupied = {tuple(cell) for cell in occupied_cells}
        active_ids: set[str] = set()
        for episode in self.episodes:
            if not episode.appearance_step <= step < episode.clearance_step:
                continue
            if episode.episode_id not in self._activation_steps:
                if occupied.intersection(episode.cells):
                    self.deferred_activation_steps += 1
                    self.deferred_episode_ids.add(episode.episode_id)
                    continue
                self._activation_steps[episode.episode_id] = step
            active_ids.add(episode.episode_id)
        self._effective_active_by_step[int(step)] = frozenset(active_ids)
        return self.truth_grid(step)

    def _active_episodes(self, step: int):
        effective = self._effective_active_by_step.get(int(step))
        if effective is None:
            return [
                episode for episode in self.episodes
                if episode.appearance_step <= step < episode.clearance_step
            ]
        return [episode for episode in self.episodes if episode.episode_id in effective]

    def activation_step(self, episode_id: str) -> int | None:
        return self._activation_steps.get(str(episode_id))

    def truth_grid(self, step: int) -> np.ndarray:
        """Return the binary physical occupancy grid at ``step``.

        The modular simulator deliberately has no dependency on the former
        continuous-motion engine: zero is free and one is physically blocked.
        """
        grid = np.array(self.static_grid, dtype=np.uint8)
        for episode in self._active_episodes(step):
            for cell in episode.cells:
                grid[cell] = 1
        return grid

    def state(self, cell: tuple[int, int], step: int) -> ClaimType:
        r, c = cell
        if not (0 <= r < self.static_grid.shape[0] and 0 <= c < self.static_grid.shape[1]) or self.static_grid[cell]: return ClaimType.BLOCKED
        return ClaimType.BLOCKED if any(cell in e.cells for e in self._active_episodes(step)) else ClaimType.FREE
    def active(self, step: int): return self._active_episodes(step)
    def cleared(self, step: int): return [e for e in self.episodes if e.clearance_step <= step]

def make_episodes(grid: np.ndarray, seed: int, total_steps: int, period: int = 150) -> tuple[TemporaryObstacleEpisode, ...]:
    from .obstacles import author_temporary_obstacle_episodes
    return author_temporary_obstacle_episodes(grid, named_rng(seed, "temporary_obstacles"), total_steps, period)
