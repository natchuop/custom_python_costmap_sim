"""Map loaders for NumPy occupancy grids and MovingAI maps."""
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
