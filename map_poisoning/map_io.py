"""Map loaders, including the upstream project's default warehouse map."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def load_npy(path: str | Path) -> np.ndarray:
    grid=np.asarray(np.load(path))
    if grid.ndim != 2: raise ValueError("map NPY must be a two-dimensional grid")
    return (grid != 0).astype(np.uint8)

def load_movingai(path: str | Path) -> np.ndarray:
    lines=Path(path).read_text(encoding="utf-8").splitlines()
    try: start=next(i for i,line in enumerate(lines) if line.strip().lower()=="map") + 1
    except StopIteration as exc: raise ValueError("MovingAI map is missing its 'map' header") from exc
    rows=[line.rstrip() for line in lines[start:] if line.rstrip()]
    if not rows or len({len(row) for row in rows}) != 1: raise ValueError("MovingAI map has inconsistent rows")
    return np.array([[0 if char in ".gGsS" else 1 for char in row] for row in rows],dtype=np.uint8)


def default_warehouse_map() -> np.ndarray:
    """Convert ``warehouse-world/maps/005/map_rotated.pgm`` in memory.

    The upstream repository ignores generated ``converted_maps``.  Using its
    checked-in source map here makes the same rotated warehouse the default
    without requiring a manual conversion step.
    """
    from convert_maps import add_static_boundary, downsample_ros_map_pixels, pixels_to_static_grid, read_pgm

    root = Path(__file__).resolve().parents[1]
    source = root / "warehouse-world" / "maps" / "005" / "map_rotated.pgm"
    if not source.exists():
        raise FileNotFoundError(f"default warehouse map is missing: {source}")
    pixels = read_pgm(source)
    grid = add_static_boundary(
        pixels_to_static_grid(
            downsample_ros_map_pixels(pixels, 8),
            unknown_as_blocked=True,
            occupied_pixel_threshold=80,
            free_pixel_threshold=230,
        )
    )
    # The converted source map traps the attacker spawn at (6, 8) behind a
    # three-cell horizontal wall (rows 10--12).  This one-cell corridor is the
    # smallest static-map correction that connects its bay to the warehouse
    # floor while preserving the rest of the upstream layout.
    grid[10:13, 8] = 0
    return grid
