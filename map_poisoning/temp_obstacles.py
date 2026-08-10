"""Deterministic temporary-obstacle schedule shared with legacy ``sim2``."""
from __future__ import annotations

import numpy as np

from .models import TemporaryObstacleEpisode


def export_temp_episodes(
    static_grid: np.ndarray,
    seed: int,
    total_steps: int,
    change_period: int | None = None,
) -> tuple[TemporaryObstacleEpisode, ...]:
    """Mirror ``TemporaryBlockageManager`` reshuffles for manifest replay parity."""
    import sim2

    manager = sim2.TemporaryBlockageManager(
        np.asarray(static_grid, dtype=int),
        change_period=change_period or sim2.TEMP_BLOCKAGE_CHANGE_PERIOD_STEPS,
        seed=seed,
    )
    period = manager.change_period
    boundaries = [0] + [step for step in range(period, total_steps, period)] + [total_steps]
    episodes: list[TemporaryObstacleEpisode] = []

    for index, start in enumerate(boundaries[:-1]):
        end = boundaries[index + 1]
        if start > 0:
            manager.refresh_active_blockages()
        for pool_index in sorted(manager.active_indices):
            footprint_cells, _ = manager.current_footprints.get(pool_index, manager.pool[pool_index])
            unique = tuple(dict.fromkeys(tuple(cell) for cell in footprint_cells))
            if unique:
                episodes.append(
                    TemporaryObstacleEpisode(
                        f"legacy-temp-{pool_index:02}-segment-{index:03}",
                        unique,
                        start,
                        end,
                    )
                )
    return tuple(episodes)
