import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================
# Simulator cell states
# Keep these aligned with your simulator's CellState enum.
# ============================================================

class CellState:
    FREE = 0
    OCCUPIED_STATIC = 1
    OCCUPIED_DYNAMIC = 2
    UNKNOWN = 3
    TEMPORARILY_BLOCKED = 4
    CONGESTED = 5
    PICKUP = 6
    DROPOFF = 7
    CHARGING = 8


CELL_STATE_NAMES = {
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


# ============================================================
# File discovery
# ============================================================

def find_map_inputs(input_dir: Path):
    """
    Finds .pgm files recursively.

    If a matching .yaml exists in the same folder, it is used.
    Matching examples:
        map.pgm + map.yaml
        world.pgm + world.yaml

    If no matching YAML exists, the converter still works using defaults.
    """
    pgm_files = sorted(input_dir.rglob("*.pgm"))

    results = []

    for pgm_path in pgm_files:
        same_stem_yaml = pgm_path.with_suffix(".yaml")
        map_yaml = pgm_path.parent / "map.yaml"

        yaml_path = None

        if same_stem_yaml.exists():
            yaml_path = same_stem_yaml
        elif map_yaml.exists():
            yaml_path = map_yaml

        results.append((pgm_path, yaml_path))

    return results


# ============================================================
# ROS map loading
# ============================================================

def load_yaml_metadata(yaml_path: Path | None):
    if yaml_path is None:
        return {
            "image": None,
            "resolution": None,
            "origin": None,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
            "negate": 0,
        }

    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    if meta is None:
        meta = {}

    return {
        "image": meta.get("image"),
        "resolution": meta.get("resolution"),
        "origin": meta.get("origin"),
        "occupied_thresh": meta.get("occupied_thresh", 0.65),
        "free_thresh": meta.get("free_thresh", 0.196),
        "negate": meta.get("negate", 0),
    }


def read_pgm(pgm_path: Path):
    """
    Reads a PGM image as grayscale.

    Common ROS map values:
        0   = occupied
        205 = unknown
        254 = free

    But thresholds are safer because not every map is polite.
    """
    img = Image.open(pgm_path).convert("L")
    return np.array(img, dtype=np.uint8)


# ============================================================
# Downsampling
# ============================================================

def downsample_ros_map_pixels(pixels: np.ndarray, factor: int):
    """
    Conservative downsampling for occupancy maps.

    Rule:
        - If any source cell in a block is occupied, output occupied.
        - Else if any source cell is unknown, output unknown.
        - Else output free.

    This preserves walls better than averaging.
    Averaging occupancy maps is how thin walls quietly vanish,
    because apparently geometry also has trust issues.
    """
    if factor <= 1:
        return pixels.copy()

    rows, cols = pixels.shape

    new_rows = rows // factor
    new_cols = cols // factor

    if new_rows <= 0 or new_cols <= 0:
        raise ValueError(
            f"Downsample factor {factor} is too large for map shape {pixels.shape}"
        )

    cropped = pixels[:new_rows * factor, :new_cols * factor]
    blocks = cropped.reshape(new_rows, factor, new_cols, factor)

    # These masks assume common ROS PGM values.
    # We also use thresholding later, but this preserves exact ROS values.
    occupied_mask = np.any(blocks < 80, axis=(1, 3))
    unknown_mask = np.any((blocks >= 80) & (blocks <= 230), axis=(1, 3))

    down = np.full((new_rows, new_cols), 254, dtype=np.uint8)
    down[unknown_mask] = 205
    down[occupied_mask] = 0

    return down


# ============================================================
# Pixel-to-grid conversion
# ============================================================

def pixels_to_static_grid(
    pixels: np.ndarray,
    unknown_as_blocked: bool = True,
    occupied_pixel_threshold: int = 80,
    free_pixel_threshold: int = 230,
):
    """
    Converts grayscale PGM pixels into simulator cell states.

    Default convention:
        dark pixels     -> occupied_static
        light pixels    -> free
        middle/gray     -> unknown or occupied_static

    unknown_as_blocked=True is safer for imported ROS maps because unknown
    areas often represent outside-map void, not explorable warehouse mystery.
    """
    static_grid = np.full(pixels.shape, CellState.UNKNOWN, dtype=np.int16)

    occupied_mask = pixels < occupied_pixel_threshold
    free_mask = pixels > free_pixel_threshold
    unknown_mask = ~(occupied_mask | free_mask)

    static_grid[occupied_mask] = CellState.OCCUPIED_STATIC
    static_grid[free_mask] = CellState.FREE

    if unknown_as_blocked:
        static_grid[unknown_mask] = CellState.OCCUPIED_STATIC
    else:
        static_grid[unknown_mask] = CellState.UNKNOWN

    return static_grid


# ============================================================
# Optional cleanup
# ============================================================

def add_static_boundary(static_grid: np.ndarray):
    """
    Forces map boundaries to be occupied.

    This prevents agents from planning off the edge of the universe,
    which is bad robotics and worse philosophy.
    """
    grid = static_grid.copy()

    grid[0, :] = CellState.OCCUPIED_STATIC
    grid[-1, :] = CellState.OCCUPIED_STATIC
    grid[:, 0] = CellState.OCCUPIED_STATIC
    grid[:, -1] = CellState.OCCUPIED_STATIC

    return grid


def find_free_cell(static_grid: np.ndarray, prefer: str):
    rows, cols = static_grid.shape

    if prefer == "top_left":
        row_range = range(rows)
        col_range = range(cols)
    elif prefer == "top_right":
        row_range = range(rows)
        col_range = range(cols - 1, -1, -1)
    elif prefer == "bottom_left":
        row_range = range(rows - 1, -1, -1)
        col_range = range(cols)
    elif prefer == "bottom_right":
        row_range = range(rows - 1, -1, -1)
        col_range = range(cols - 1, -1, -1)
    else:
        row_range = range(rows)
        col_range = range(cols)

    for r in row_range:
        for c in col_range:
            if static_grid[r, c] == CellState.FREE:
                return [int(r), int(c)]

    return None


def summarize_grid(static_grid: np.ndarray):
    values, counts = np.unique(static_grid, return_counts=True)

    summary = {}

    total = static_grid.size

    for value, count in zip(values, counts):
        value = int(value)
        count = int(count)

        summary[CELL_STATE_NAMES.get(value, str(value))] = {
            "count": count,
            "percent": 100.0 * count / total,
        }

    return summary


# ============================================================
# Preview image
# ============================================================

def save_preview(static_grid: np.ndarray, output_path: Path):
    if plt is None:
        return

    # Preview colors:
    # free = white
    # occupied = black
    # unknown = gray
    preview = np.zeros((*static_grid.shape, 3), dtype=np.uint8)

    preview[:, :] = [180, 180, 180]  # unknown gray

    preview[static_grid == CellState.FREE] = [255, 255, 255]
    preview[static_grid == CellState.OCCUPIED_STATIC] = [0, 0, 0]
    preview[static_grid == CellState.PICKUP] = [70, 180, 70]
    preview[static_grid == CellState.DROPOFF] = [70, 120, 255]
    preview[static_grid == CellState.CHARGING] = [160, 70, 180]

    plt.figure(figsize=(8, 8))
    plt.imshow(preview, origin="upper")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# Conversion
# ============================================================

def convert_one_map(
    pgm_path: Path,
    yaml_path: Path | None,
    output_dir: Path,
    input_root: Path,
    downsample_factor: int,
    unknown_as_blocked: bool,
    add_boundary: bool,
    occupied_pixel_threshold: int,
    free_pixel_threshold: int,
):
    meta = load_yaml_metadata(yaml_path)
    pixels = read_pgm(pgm_path)

    original_shape = pixels.shape

    pixels_down = downsample_ros_map_pixels(pixels, downsample_factor)

    static_grid = pixels_to_static_grid(
        pixels_down,
        unknown_as_blocked=unknown_as_blocked,
        occupied_pixel_threshold=occupied_pixel_threshold,
        free_pixel_threshold=free_pixel_threshold,
    )

    if add_boundary:
        static_grid = add_static_boundary(static_grid)

    relative_parent = pgm_path.parent.relative_to(input_root)
    safe_name = "_".join(relative_parent.parts)

    if not safe_name:
        safe_name = pgm_path.stem
    else:
        safe_name = f"{safe_name}_{pgm_path.stem}"

    map_output_dir = output_dir / safe_name
    map_output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = map_output_dir / "static_grid.npy"
    metadata_path = map_output_dir / "metadata.json"
    preview_path = map_output_dir / "preview.png"

    np.save(npy_path, static_grid)

    start_top_left = find_free_cell(static_grid, "top_left")
    start_bottom_left = find_free_cell(static_grid, "bottom_left")
    goal_bottom_right = find_free_cell(static_grid, "bottom_right")
    goal_top_right = find_free_cell(static_grid, "top_right")

    converted_resolution = None
    if meta.get("resolution") is not None:
        converted_resolution = float(meta["resolution"]) * downsample_factor

    metadata = {
        "source_pgm": str(pgm_path),
        "source_yaml": str(yaml_path) if yaml_path is not None else None,
        "output_npy": str(npy_path),
        "preview_png": str(preview_path),
        "original_shape_rows_cols": list(original_shape),
        "converted_shape_rows_cols": list(static_grid.shape),
        "downsample_factor": downsample_factor,
        "original_resolution_m_per_cell": meta.get("resolution"),
        "converted_resolution_m_per_cell": converted_resolution,
        "origin": meta.get("origin"),
        "occupied_thresh": meta.get("occupied_thresh"),
        "free_thresh": meta.get("free_thresh"),
        "negate": meta.get("negate"),
        "unknown_as_blocked": unknown_as_blocked,
        "added_static_boundary": add_boundary,
        "occupied_pixel_threshold": occupied_pixel_threshold,
        "free_pixel_threshold": free_pixel_threshold,
        "cell_state_ids": CELL_STATE_NAMES,
        "summary": summarize_grid(static_grid),
        "suggested_cells": {
            "start_top_left": start_top_left,
            "start_bottom_left": start_bottom_left,
            "goal_bottom_right": goal_bottom_right,
            "goal_top_right": goal_top_right,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_preview(static_grid, preview_path)

    return metadata


def convert_maps(args):
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    map_inputs = find_map_inputs(input_dir)

    if not map_inputs:
        raise RuntimeError(f"No .pgm files found under: {input_dir}")

    all_metadata = []

    print(f"Found {len(map_inputs)} .pgm map(s)")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    for idx, (pgm_path, yaml_path) in enumerate(map_inputs, start=1):
        print(f"[{idx}/{len(map_inputs)}] Converting:")
        print(f"  PGM:  {pgm_path}")
        print(f"  YAML: {yaml_path if yaml_path else 'none'}")

        metadata = convert_one_map(
            pgm_path=pgm_path,
            yaml_path=yaml_path,
            output_dir=output_dir,
            input_root=input_dir,
            downsample_factor=args.downsample,
            unknown_as_blocked=args.unknown_as_blocked,
            add_boundary=args.add_boundary,
            occupied_pixel_threshold=args.occupied_pixel_threshold,
            free_pixel_threshold=args.free_pixel_threshold,
        )

        all_metadata.append(metadata)

        print(f"  Saved: {metadata['output_npy']}")
        print(f"  Shape: {metadata['converted_shape_rows_cols']}")
        print(f"  Preview: {metadata['preview_png']}")
        print()

    index_path = output_dir / "converted_maps_index.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print("Done.")
    print(f"Index saved to: {index_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ROS-style .pgm/.yaml occupancy maps into simulator .npy static grids."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing .pgm files. Searched recursively.",
    )

    parser.add_argument(
        "--output",
        default="converted_maps",
        help="Output folder for converted maps.",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="Downsample factor. Example: 8 turns 1536x1504 into about 192x188.",
    )

    parser.add_argument(
        "--unknown-as-free",
        action="store_true",
        help="Treat unknown gray cells as free/unknown instead of blocked. Not recommended at first.",
    )

    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="Do not force map edges to occupied.",
    )

    parser.add_argument(
        "--occupied-pixel-threshold",
        type=int,
        default=80,
        help="Pixels below this are occupied.",
    )

    parser.add_argument(
        "--free-pixel-threshold",
        type=int,
        default=230,
        help="Pixels above this are free.",
    )

    args = parser.parse_args()

    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    args.unknown_as_blocked = not args.unknown_as_free
    args.add_boundary = not args.no_boundary

    return args


if __name__ == "__main__":
    convert_maps(parse_args())