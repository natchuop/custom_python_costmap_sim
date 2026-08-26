"""Live reconnaissance heatmap and per-robot belief-map windows."""
from __future__ import annotations

import textwrap

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.widgets import Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np

from .models import AttackType, ClaimType
from .sensing import lidar_observations


DISPLAY_FREE = 0
DISPLAY_STATIC = 1
DISPLAY_DYNAMIC = 2
DISPLAY_UNKNOWN = 3
DISPLAY_BLOCKED = 4
DISPLAY_CONGESTED = 5
DISPLAY_PICKUP = 6
DISPLAY_DROPOFF = 7
DISPLAY_ROBOT = 9
DISPLAY_GOAL = 10
DISPLAY_FAKE = 11
DISPLAY_R1 = 12
DISPLAY_FALSE_CLEARANCE = 13
DISPLAY_R0 = 14
DISPLAY_R2 = 15

ROBOT_COLORS = {0: "#8e24aa", 1: "#fb8c00", 2: "#1976d2"}
ROBOT_DISPLAY = {0: DISPLAY_R0, 1: DISPLAY_R1, 2: DISPLAY_R2}
LIDAR_RANGE_CELLS = 5

_COLORS = [
    "#ffffff",
    "#222222",
    "#43a047",
    "#bdbdbd",
    "#66bb6a",
    "#f9a825",
    "#795548",
    "#26a69a",
    "#607d8b",
    "#00e5ff",
    "#ffeb3b",
    "#e53935",
    "#fb8c00",
    "#ef9a9a",
    "#8e24aa",
    "#1976d2",
]
_CMAP = ListedColormap(_COLORS)
_NORM = BoundaryNorm(np.arange(-0.5, len(_COLORS) + 0.5, 1), _CMAP.N)
_ATTACK_LABELS = {
    AttackType.FAKE_OBSTACLE: "Fake Obstacle",
    AttackType.FALSE_CLEARANCE: "False Clearance",
    AttackType.STALE_REASSERTION: "Stale Reassertion",
}


def _can_show() -> bool:
    backend = str(plt.get_backend() or "").lower()
    return backend not in {"agg", "pdf", "svg", "template", "cairo"}


def _goals(robots):
    goals = []
    for robot in robots:
        if robot.completed:
            continue
        task = robot.tasks[robot.task_index]
        goals.append(task.dropoff if robot.carrying else task.pickup)
    return goals


def _events_up_to(log, step):
    return [event for event in log.get("attack_events", ()) if int(event.step) <= step]


def _latest_attack(log, step):
    latest = None
    for event in _events_up_to(log, step):
        if latest is None or int(event.step) >= int(latest.step):
            latest = event
    return latest


def _overlay_groups(log, step):
    latest = _latest_attack(log, step)
    if latest is None:
        return []
    return [{"attack_type": latest.attack_type, "cells": list(latest.cells)}]


def _paint_overlays(arr, overlays, *, attacker_view: bool):
    if not attacker_view:
        return arr
    for overlay in overlays or ():
        display = (
            DISPLAY_FALSE_CLEARANCE
            if overlay.get("attack_type") == AttackType.FALSE_CLEARANCE
            else DISPLAY_FAKE
        )
        for cell in overlay.get("cells", ()):
            if 0 <= cell[0] < arr.shape[0] and 0 <= cell[1] < arr.shape[1] and arr[cell] != DISPLAY_STATIC:
                arr[cell] = display
    return arr


def _dominant_blocked_sender(robot, items, step):
    """Choose a displayed BLOCKED source only when fusion says it matters."""
    if robot.fusion.method == "majority_vote":
        if robot.fusion.vote(items[0].report.target_cell, step) <= 0:
            return None
    elif robot.fusion.routing_cost(items[0].report.target_cell, step) <= 1.0 + 1e-9:
        return None
    blocked = []
    for item in items:
        report = item.report
        if int(report.claim) != int(ClaimType.BLOCKED):
            continue
        if step - int(report.observation_step) >= robot.fusion.max_claim_age:
            continue
        weight = robot.fusion.operational_weight(report, step)
        if weight <= 1e-12:
            continue
        # Display attribution follows actual operational influence rather than
        # current trust. This is important for the trust-agnostic baselines and
        # for Trust Fused, whose old reports intentionally retain trust-at-report.
        blocked.append((weight, int(report.observation_step), -int(report.sender_id), report.sender_id))
    if not blocked:
        return None
    blocked.sort(reverse=True)
    return blocked[0][-1]

def _paint_trusted_attack_reports(arr, robot, log, step, threshold: float) -> None:
    """Paint active trusted attacker BLOCKED fusion claims on victim belief maps."""
    attacker = log.get("malicious_robot_id")
    if attacker is None or robot.robot_id == attacker:
        return
    if robot.trust.score(attacker) < threshold:
        return
    color = ROBOT_DISPLAY.get(attacker, DISPLAY_BLOCKED)
    for cell, items in robot.fusion.claims.items():
        if arr[cell] == DISPLAY_STATIC or not items:
            continue
        if not any(
            item.report.sender_id == attacker and item.report.claim == ClaimType.BLOCKED
            for item in items
        ):
            continue
        if _fresh_direct_free(robot, cell, step):
            arr[cell] = DISPLAY_FREE
            continue
        arr[cell] = color


def truth_display_grid(world, robots, log, step):
    grid = world.truth_grid(step)
    arr = np.zeros(grid.shape, dtype=np.int16)
    arr[world.static_grid.astype(bool)] = DISPLAY_STATIC
    arr[(grid == 1) & (world.static_grid == 0)] = DISPLAY_DYNAMIC
    _paint_overlays(arr, _overlay_groups(log, step), attacker_view=True)
    for cell in _goals(robots):
        if arr[cell] == DISPLAY_FREE:
            arr[cell] = DISPLAY_GOAL
    return arr


def _direct_display_state(robot, cell, step, display_age):
    return robot.belief.display_state(cell, step, max_age=display_age)


def _fresh_direct_free(robot, cell, step) -> bool:
    claim, freshness = robot.belief.observation_status(cell, step)
    return freshness == "current" and claim == ClaimType.FREE


def _explored_clear(robot, cell, step, display_age) -> bool:
    return _direct_display_state(robot, cell, step, display_age) == ClaimType.FREE


def local_display_grid(robot, world, step):
    rows, cols = world.static_grid.shape
    arr = np.full((rows, cols), DISPLAY_UNKNOWN, dtype=np.int16)
    arr[world.static_grid.astype(bool)] = DISPLAY_STATIC
    own = ROBOT_DISPLAY.get(robot.robot_id, DISPLAY_BLOCKED)
    display_age = robot.fusion.max_claim_age
    for cell in robot.belief.direct:
        if arr[cell] == DISPLAY_STATIC:
            continue
        claim = _direct_display_state(robot, cell, step, display_age)
        if claim is None:
            continue
        if claim == ClaimType.FREE:
            arr[cell] = DISPLAY_FREE
        else:
            arr[cell] = own
    if not robot.completed:
        task = robot.tasks[robot.task_index]
        goal = task.dropoff if robot.carrying else task.pickup
        marker = DISPLAY_DROPOFF if robot.carrying else DISPLAY_PICKUP
        if arr[goal] != DISPLAY_STATIC:
            arr[goal] = marker
    return arr


def combined_display_grid(robot, world, log, step, robots):
    arr = local_display_grid(robot, world, step)
    robot.fusion.set_time(step)
    own = ROBOT_DISPLAY.get(robot.robot_id, DISPLAY_BLOCKED)
    display_age = robot.fusion.max_claim_age
    claims_by_cell = robot.fusion.claims
    for cell, items in claims_by_cell.items():
        if arr[cell] == DISPLAY_STATIC or not items:
            continue
        # Only a fresh direct clear sighting clears trusted peer obstacles.
        if _fresh_direct_free(robot, cell, step):
            arr[cell] = DISPLAY_FREE
            continue
        direct = _direct_display_state(robot, cell, step, display_age)
        if direct == ClaimType.BLOCKED:
            arr[cell] = own
            continue
        sender = _dominant_blocked_sender(robot, items, step)
        if sender is not None:
            arr[cell] = ROBOT_DISPLAY.get(sender, DISPLAY_BLOCKED)
        elif _explored_clear(robot, cell, step, display_age):
            arr[cell] = DISPLAY_FREE
    attacker = log.get("malicious_robot_id")
    _paint_overlays(arr, _overlay_groups(log, step), attacker_view=robot.robot_id == attacker)


    if not robot.completed:
        task = robot.tasks[robot.task_index]
        goal = task.dropoff if robot.carrying else task.pickup
        marker = DISPLAY_DROPOFF if robot.carrying else DISPLAY_PICKUP
        if arr[goal] != DISPLAY_STATIC:
            arr[goal] = marker
    return arr


def belief_display_grid(robot, world, log, step, map_view="combined", robots=None):
    if map_view == "local":
        arr = local_display_grid(robot, world, step)
        attacker = log.get("malicious_robot_id")
        _paint_overlays(arr, _overlay_groups(log, step), attacker_view=robot.robot_id == attacker)
        return arr
    return combined_display_grid(robot, world, log, step, robots or (robot,))


def init_live_log(log, world, robots, config, manifest) -> None:
    rows, cols = world.static_grid.shape
    log["live"] = {
        "truth": [],
        "beliefs": {robot.robot_id: [] for robot in robots},
        "local_beliefs": {robot.robot_id: [] for robot in robots},
        "combined_beliefs": {robot.robot_id: [] for robot in robots},
        "positions": {robot.robot_id: [] for robot in robots},
        "paths": {robot.robot_id: [] for robot in robots},
        "trust": {robot.robot_id: [] for robot in robots},
        "pairwise_trust": [],
        "pairwise_source_memory": [],
        "trusted": {robot.robot_id: [] for robot in robots},
        "deliveries": {robot.robot_id: [] for robot in robots},
        "carrying": {robot.robot_id: [] for robot in robots},
        "accepted": {robot.robot_id: [] for robot in robots},
        "rejected": {robot.robot_id: [] for robot in robots},
        "replans": {robot.robot_id: [] for robot in robots},
        "completed": {robot.robot_id: [] for robot in robots},
        "report_counts": [],
        "malicious_report_counts": [],
        "heatmap": np.zeros((rows, cols), dtype=np.int32),
        "heatmap_static_grid": np.asarray(world.static_grid, dtype=np.uint8).copy(),
        "recon_heatmap": (
            None
            if manifest.reconnaissance_heatmap is None
            else np.asarray(manifest.reconnaissance_heatmap, dtype=np.int32)
        ),
        "heatmap_reference_steps": int(manifest.phase_boundaries.get("total", config.phases.total_steps)),
        "recon_end": config.phases.recon_steps,
        "attack_start": config.phases.recon_steps,
        "threshold": config.trust.threshold,
        "method": log.get("defense_method"),
        "map_view": config.visualization.map_view,
        "seed": config.seed,
        "lidar_range_cells": int(config.lidar_range_cells),
    }
    log["attack_events"] = manifest.attack_events
    log["benign_robot_ids"] = manifest.benign_robot_ids


def record_live_frame(log, world, robots, step, phase) -> None:
    live = log.get("live")
    if live is None:
        return
    live["truth"].append(truth_display_grid(world, robots, log, step))
    attacker = log["malicious_robot_id"]
    map_view = live.get("map_view", "combined")
    snapshot = {}
    memory_snapshot = {}
    for robot in robots:
        rid = robot.robot_id
        local = local_display_grid(robot, world, step)
        combined = combined_display_grid(robot, world, log, step, robots)
        live["local_beliefs"][rid].append(local)
        live["combined_beliefs"][rid].append(combined)
        selected = local if map_view == "local" else combined
        live["beliefs"][rid].append(selected)
        live["positions"][rid].append(robot.position)
        live["paths"][rid].append(list(robot.path or ()))
        attacker_trust = robot.trust.score(attacker)
        attacker_memory = robot.trust.memory_score(attacker)
        live["trust"][rid].append(attacker_trust)
        effective_attacker_trust = min(attacker_trust, attacker_memory) if live.get("method") == "source_memory" else attacker_trust
        live["trusted"][rid].append(effective_attacker_trust >= live["threshold"])
        live["deliveries"][rid].append(robot.deliveries_completed)
        live["carrying"][rid].append(robot.carrying)
        live["accepted"][rid].append(robot.accepted_reports)
        live["rejected"][rid].append(robot.rejected_reports)
        live["replans"][rid].append(robot.total_replans)
        live["completed"][rid].append(robot.completed)
        snapshot[rid] = {other.robot_id: robot.trust.score(other.robot_id) for other in robots if other.robot_id != rid}
        memory_snapshot[rid] = {other.robot_id: robot.trust.memory_score(other.robot_id) for other in robots if other.robot_id != rid}
        if rid != attacker:
            live["heatmap"][robot.position] += 1
    live["pairwise_trust"].append(snapshot)
    live["pairwise_source_memory"].append(memory_snapshot)
    live["report_counts"].append(int(log.get("report_count_total", 0)))
    live["malicious_report_counts"].append(int(log.get("malicious_report_count_total", 0)))
    if step + 1 == live["recon_end"] and live["recon_heatmap"] is None:
        live["recon_heatmap"] = live["heatmap"].copy()


def _draw_path(ax, path, color):
    line, = ax.plot([], [], color=color, linewidth=1.2, alpha=0.45)
    if path:
        line.set_data([cell[1] for cell in path], [cell[0] for cell in path])
    return line


def _lidar_items(readings):
    if isinstance(readings, dict):
        return list(readings.items())
    return [(cell, None) for cell in readings]


def _draw_lidar(ax, origin, readings, color, alpha=0.22, radius=LIDAR_RANGE_CELLS):
    # Allocate enough artists for every lattice cell in the configured sensor
    # circle; this avoids losing rays when later frames expose more cells.
    radius = max(1, int(radius))
    max_count = sum(
        1 for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    )
    lines = []
    for _ in range(max_count):
        line, = ax.plot([], [], color=color, linewidth=0.4, alpha=alpha)
        lines.append(line)
    _update_lidar(lines, origin, readings, base_alpha=alpha)
    return lines


def _update_path(line, path):
    if path:
        line.set_data([cell[1] for cell in path], [cell[0] for cell in path])
        line.set_visible(True)
    else:
        line.set_data([], [])
        line.set_visible(False)


def _update_lidar(lines, origin, readings, base_alpha=0.22):
    r0, c0 = origin
    seen = _lidar_items(readings)
    for line, (cell, reading) in zip(lines, seen):
        line.set_data([c0, cell[1]], [r0, cell[0]])
        confidence = float(getattr(reading, "sensor_confidence", 1.0)) if reading is not None else 1.0
        line.set_alpha(max(0.06, base_alpha * confidence))
        line.set_visible(True)
    for line in lines[len(seen):]:
        line.set_visible(False)

def draw_attack_outlines(ax, cells, color="#d32f2f"):
    cell_set = {tuple(cell) for cell in cells}
    remaining = set(cell_set)
    outlines = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        segments = []
        for row, col in component:
            edges = (
                ((col - 0.5, row - 0.5), (col + 0.5, row - 0.5), (row - 1, col)),
                ((col - 0.5, row + 0.5), (col + 0.5, row + 0.5), (row + 1, col)),
                ((col - 0.5, row - 0.5), (col - 0.5, row + 0.5), (row, col - 1)),
                ((col + 0.5, row - 0.5), (col + 0.5, row + 0.5), (row, col + 1)),
            )
            for start_pt, end_pt, neighbor in edges:
                if neighbor not in component:
                    segments.append((start_pt, end_pt))
        outline = LineCollection(segments, colors=color, linewidths=1.8, linestyles=":", zorder=12)
        ax.add_collection(outline)
        outlines.append(outline)
    return outlines


def _set_window_title(fig, title: str) -> None:
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def show_traffic_heatmap(log, *, show=True):
    live = log.get("live") or {}
    heat = live.get("recon_heatmap")
    if heat is None:
        heat = live.get("heatmap")
    frames = live.get("truth") or []
    static_grid = live.get("heatmap_static_grid")
    if static_grid is not None:
        static_grid = np.asarray(static_grid, dtype=np.uint8)
        truth = np.zeros(static_grid.shape, dtype=np.int16)
        truth[static_grid.astype(bool)] = DISPLAY_STATIC
    else:
        # Backward-compatible fallback for logs created before the static
        # reference layer was recorded.
        recon_end = int(live.get("recon_end") or 0)
        truth = frames[min(max(recon_end - 1, 0), len(frames) - 1)] if frames else None
    if heat is None or truth is None:
        print("No traffic heatmap was recorded. Turn on live maps and run again.")
        return None
    seed = live.get("seed")
    seed_label = f"seed {seed}" if seed is not None else "seed unknown"
    overlay = np.array(heat, dtype=float)
    overlay[truth == DISPLAY_STATIC] = np.nan
    valid = overlay[np.isfinite(overlay)]
    vmax = float(np.nanpercentile(valid, 99.0)) if valid.size else None
    if vmax is not None and vmax <= 0:
        vmax = None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(truth, cmap=_CMAP, norm=_NORM, origin="upper", alpha=0.35)
    image = ax.imshow(overlay, origin="upper", alpha=0.82, cmap="hot", vmin=0, vmax=vmax)
    reference_steps = int(live.get("heatmap_reference_steps") or recon_end)
    title = f"Attack-free reference heatmap | {seed_label} | {reference_steps} clean steps"
    ax.set(title=title, xlabel="col", ylabel="row")
    _set_window_title(fig, f"Attack-free reference heatmap | {seed_label}")
    fig.colorbar(image, ax=ax, label="benign traffic count")
    fig.tight_layout()
    if show and _can_show():
        fig.show()
    return fig


def show_belief_maps(log, world, robots, *, show=True, interval_ms=80):
    live = log.get("live")
    if not live or not live["truth"]:
        print("No live map frames were recorded. Turn on live maps and run again.")
        return None
    robots = sorted(robots, key=lambda item: item.robot_id)
    n_frames = len(live["truth"])
    selected_view = live.get("map_view", "combined")
    view_label = "Combined Belief Map" if selected_view == "combined" else "Local Observation Map"
    attacker = log["malicious_robot_id"]
    threshold = float(live.get("threshold", 0.55))
    method = str(live.get("method") or "unknown")
    lidar_range = int(live.get("lidar_range_cells", LIDAR_RANGE_CELLS))
    belief_source = live["combined_beliefs"] if selected_view == "combined" else live["local_beliefs"]
    if not belief_source.get(robots[0].robot_id):
        belief_source = live["beliefs"]

    fig = plt.figure(figsize=(16, 11), constrained_layout=False)
    seed = live.get("seed")
    seed_label = f"seed {seed}" if seed is not None else "seed unknown"
    _set_window_title(fig, f"Belief maps | {seed_label} | {method}")
    fig.suptitle(f"Belief maps | {seed_label} | {method}", fontsize=14, y=0.98)
    layout = fig.add_gridspec(
        4, 3,
        width_ratios=(1.0, 1.0, 0.82),
        height_ratios=(1.0, 1.0, 0.28, 0.18),
        left=0.045, right=0.98, top=0.91, bottom=0.055,
        hspace=0.36, wspace=0.28,
    )
    map_axes = [
        fig.add_subplot(layout[0, 0]),
        fig.add_subplot(layout[0, 1]),
        fig.add_subplot(layout[1, 0]),
        fig.add_subplot(layout[1, 1]),
    ]
    truth_ax = map_axes[0]
    belief_axes = {robot.robot_id: map_axes[min(index + 1, len(map_axes) - 1)] for index, robot in enumerate(robots)}
    status_ax = fig.add_subplot(layout[0, 2])
    trust_ax = fig.add_subplot(layout[1, 2])
    legend_ax = fig.add_subplot(layout[2, 0:2])
    sharing_ax = fig.add_subplot(layout[3, 0:2])
    latest_attack_ax = fig.add_subplot(layout[2:4, 2])
    for panel in (status_ax, trust_ax, legend_ax, sharing_ax, latest_attack_ax):
        panel.set_axis_off()
        panel.patch.set_facecolor("#f5f5f5")

    truth_ax.set_title("Ground Truth Map", fontsize=13, pad=6)
    truth_ax.set_xlabel("col")
    truth_ax.set_ylabel("row")
    truth_img = truth_ax.imshow(live["truth"][0], cmap=_CMAP, norm=_NORM, origin="upper")
    truth_path_lines = {}
    truth_robot_patches = {}
    truth_lidar_lines = {}
    belief_imgs = {}
    belief_path_lines = {}
    belief_lidar_lines = {}
    belief_lidar_range = {}
    belief_robot_patches = {}
    belief_attack_outlines = {robot.robot_id: [] for robot in robots}

    for robot in robots:
        rid = robot.robot_id
        color = ROBOT_COLORS.get(rid, "#555555")
        r, c = live["positions"][rid][0]
        patch = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, linewidth=2.0, edgecolor=color, zorder=10)
        truth_ax.add_patch(patch)
        truth_robot_patches[rid] = patch
        truth_path_lines[rid] = _draw_path(truth_ax, live["paths"][rid][0], color)
        truth_lidar_lines[rid] = []

    for robot in robots:
        rid = robot.robot_id
        ax = belief_axes[rid]
        role = "MALICIOUS" if rid == attacker else "VICTIM"
        ax.set_title(
            f"Robot {rid} | {view_label}\n({role}) - LiDAR {lidar_range:g} cells",
            fontsize=11, pad=5, color=ROBOT_COLORS.get(rid, "#555555"),
        )
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        first = belief_source[rid][0]
        belief_imgs[rid] = ax.imshow(first, cmap=_CMAP, norm=_NORM, origin="upper")
        r, c = live["positions"][rid][0]
        color = ROBOT_COLORS.get(rid, "#555555")
        range_circle = Circle((c, r), lidar_range, fill=False, edgecolor=color, linestyle="--", linewidth=0.8, alpha=0.45, zorder=2)
        ax.add_patch(range_circle)
        belief_lidar_range[rid] = range_circle
        patch = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, linewidth=2.0, edgecolor=color, zorder=10)
        ax.add_patch(patch)
        belief_robot_patches[rid] = patch
        belief_path_lines[rid] = _draw_path(ax, live["paths"][rid][0], color)
        belief_lidar_lines[rid] = []
        ax.set_xlim(-0.5, first.shape[1] - 0.5)
        ax.set_ylim(first.shape[0] - 0.5, -0.5)
        ax.set_autoscale_on(False)

    for ax in map_axes[:2]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    status_ax.set_title("Simulation status | Playback", fontsize=10, loc="left", pad=3)
    trust_ax.set_title(f"Robot trust level | {method}", fontsize=10, loc="left", pad=3)
    legend_ax.set_title("Map legend", fontsize=10, loc="left", pad=2)
    sharing_ax.set_title("Peer observations", fontsize=10, loc="left", pad=1)
    latest_attack_ax.set_title("Latest attack", fontsize=10, loc="left", pad=3)
    status_text = status_ax.text(0.03, 0.94, "", fontsize=7.2, va="top", family="DejaVu Sans Mono", transform=status_ax.transAxes)
    trust_threshold_text = trust_ax.text(0.05, 0.91, f"Threshold: {threshold:.2f}", fontsize=9.5, va="top", transform=trust_ax.transAxes)
    trust_pairs = [
        (observer.robot_id, sender.robot_id)
        for observer in robots
        for sender in robots
        if sender.robot_id != observer.robot_id
    ]
    source_memory_table = method == "source_memory"
    if source_memory_table:
        trust_table = trust_ax.table(
            cellText=[[f"R{observer}", f"R{sender}", f"{threshold:.2f}", f"{threshold:.2f}", "ACTIVE"] for observer, sender in trust_pairs],
            colLabels=("Observer", "Sender", "Trust", "Memory", "State"),
            colWidths=(0.17, 0.17, 0.17, 0.17, 0.22),
            cellLoc="left",
            colLoc="left",
            bbox=(0.02, 0.12, 0.96, 0.68),
        )
    else:
        trust_table = trust_ax.table(
            cellText=[[f"R{observer}", f"R{sender}", f"{threshold:.2f}", "TRUSTED"] for observer, sender in trust_pairs],
            colLabels=("Observer", "Sender", "Score", "State"),
            colWidths=(0.20, 0.20, 0.18, 0.27),
            cellLoc="left",
            colLoc="left",
            bbox=(0.04, 0.12, 0.92, 0.68),
        )
    trust_table.auto_set_font_size(False)
    trust_table.set_fontsize(8.5)
    latest_attack_text = latest_attack_ax.text(0.05, 0.90, "", fontsize=10, va="top", family="DejaVu Sans Mono", transform=latest_attack_ax.transAxes)
    sharing_ax.text(
        0.01, 0.50,
        "Valid known FREE paints white; gray is unknown or expired. "
        "Operationally active attacker BLOCKED claims use Robot 0 purple; Source Memory hides them while IGNORED.",
        fontsize=8.5, va="center", transform=sharing_ax.transAxes,
    )
    legend_ax.legend(
        handles=[
            Patch(facecolor="#ffffff", edgecolor="#cccccc", label="Explored clear"),
            Patch(facecolor="#bdbdbd", label="Unknown / expired"),
            Patch(facecolor="#222222", label="Static obstacle"),
            Patch(facecolor="#43a047", label="Temporary physical obstacle"),
            Patch(facecolor=ROBOT_COLORS[0], label="Robot 0 source (purple)"),
            Patch(facecolor=ROBOT_COLORS[1], label="Robot 1 source (orange)"),
            Patch(facecolor=ROBOT_COLORS[2], label="Robot 2 source (blue)"),
            Patch(facecolor="#ffeb3b", label="Goal/checkpoint"),
            Patch(facecolor="#e53935", label="Attack claim overlay (not physical)"),
            Patch(facecolor="#ef9a9a", label="False clearance"),
            Patch(facecolor="none", edgecolor="#d32f2f", linestyle=":", linewidth=1.8, label="Recent R0 attack perimeter"),
        ],
        loc="upper left", ncol=4, fontsize=7.5, frameon=False,
    )

    controls_position = status_ax.get_position()
    speed_ax = fig.add_axes((
        controls_position.x0 + controls_position.width * 0.06,
        controls_position.y0 + controls_position.height * 0.04,
        controls_position.width * 0.40,
        controls_position.height * 0.30,
    ))
    speed_ax.set_title("Speed", fontsize=8, loc="left", pad=1)
    speed = RadioButtons(speed_ax, ("0.5x", "1x", "2x", "5x", "20x"), active=1)
    pause_ax = fig.add_axes((
        controls_position.x0 + controls_position.width * 0.56,
        controls_position.y0 + controls_position.height * 0.10,
        controls_position.width * 0.36,
        controls_position.height * 0.22,
    ))
    pause_button = Button(pause_ax, "Pause", color="#eeeeee", hovercolor="#d0d0d0")
    paused = False
    display_index = 0.0
    first_tick = True

    def toggle_pause(_event):
        nonlocal paused
        paused = not paused
        pause_button.label.set_text("Resume" if paused else "Pause")

    pause_button.on_clicked(toggle_pause)

    def selected_multiplier():
        try:
            return float(speed.value_selected.rstrip("x")) / 2.0
        except (TypeError, ValueError):
            return 1.0

    def wrap_panel_lines(lines, width):
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(str(line), width=width, break_long_words=False, break_on_hyphens=False) or [""])
        return wrapped

    def lidar_cells(frame, robot_id):
        static = world.static_grid
        truth_grid = np.array(static, dtype=np.uint8)
        truth_grid[live["truth"][frame] == DISPLAY_DYNAMIC] = 1
        positions = {robot.robot_id: live["positions"][robot.robot_id][frame] for robot in robots}
        others = [pos for rid, pos in positions.items() if rid != robot_id]
        return lidar_observations(truth_grid, positions[robot_id], others, radius=lidar_range)

    def update(_frame):
        nonlocal display_index, first_tick
        if first_tick:
            first_tick = False
        elif not paused:
            display_index = min(n_frames - 1, display_index + selected_multiplier())
        frame = int(display_index)
        artists = [truth_img, trust_threshold_text, trust_table, status_text, latest_attack_text]
        truth_img.set_data(live["truth"][frame])
        phase = log["phase"][frame] if frame < len(log["phase"]) else ""
        for robot in robots:
            rid = robot.robot_id
            r, c = live["positions"][rid][frame]
            color = "magenta" if rid == attacker else "#00bcd4"
            seen = lidar_cells(frame, rid)
            if not truth_lidar_lines[rid]:
                truth_lidar_lines[rid] = _draw_lidar(truth_ax, (r, c), seen, color, alpha=0.16, radius=lidar_range)
            else:
                _update_lidar(truth_lidar_lines[rid], (r, c), seen)
            artists.extend(truth_lidar_lines[rid])
            truth_robot_patches[rid].set_xy((c - 0.5, r - 0.5))
            artists.append(truth_robot_patches[rid])
            _update_path(truth_path_lines[rid], live["paths"][rid][frame])
            artists.append(truth_path_lines[rid])

            belief_imgs[rid].set_data(belief_source[rid][frame])
            artists.append(belief_imgs[rid])
            belief_lidar_range[rid].center = (c, r)
            artists.append(belief_lidar_range[rid])
            if not belief_lidar_lines[rid]:
                belief_lidar_lines[rid] = _draw_lidar(belief_axes[rid], (r, c), seen, ROBOT_COLORS.get(rid, "#00bcd4"), radius=lidar_range)
            else:
                _update_lidar(belief_lidar_lines[rid], (r, c), seen)
            artists.extend(belief_lidar_lines[rid])
            belief_robot_patches[rid].set_xy((c - 0.5, r - 0.5))
            artists.append(belief_robot_patches[rid])
            _update_path(belief_path_lines[rid], live["paths"][rid][frame])
            artists.append(belief_path_lines[rid])
            for outline in belief_attack_outlines[rid]:
                outline.remove()
            belief_attack_outlines[rid] = []
            if selected_view == "combined" and rid != attacker:
                latest = _latest_attack(log, frame)
                if latest is not None and latest.attack_type != AttackType.FALSE_CLEARANCE:
                    belief_attack_outlines[rid] = draw_attack_outlines(belief_axes[rid], latest.cells)
                    artists.extend(belief_attack_outlines[rid])

        report_count = live.get("report_counts", [0])[frame] if live.get("report_counts") else 0
        malicious_report_count = live.get("malicious_report_counts", [0])[frame] if live.get("malicious_report_counts") else 0
        latest = _latest_attack(log, frame)
        overlay_count = sum(len(item.get("cells", ())) for item in _overlay_groups(log, frame))
        status_lines = [
            f"Step: {frame}",
            f"Phase: {phase}",
            f"Attack starts: {live.get('attack_start', 'not yet')}",
            f"Reports: {report_count}",
            f"Malicious reports: {malicious_report_count}",
            f"Attack overlays: {overlay_count}",
            f"Malicious robot: R{attacker}",
        ]
        if latest is None:
            latest_attack_lines = [
                "No attack has occurred yet.",
                f"Attack phase starts: {live.get('attack_start', 'not yet')}",
            ]
        else:
            latest_attack_lines = [
                f"Type: {_ATTACK_LABELS.get(latest.attack_type, latest.attack_type)}",
                f"Step: {latest.step}",
                f"Attacker: R{latest.sender_id}",
                f"Event: {latest.event_id}",
                f"Reported cells: {len(latest.cells)}",
                f"Reports sent: {len(latest.report_ids)}",
            ]
        snapshot = live["pairwise_trust"][frame] if frame < len(live["pairwise_trust"]) else {}
        memory_snapshot = live.get("pairwise_source_memory", [])
        memory_snapshot = memory_snapshot[frame] if frame < len(memory_snapshot) else {}
        for row, (observer_id, sender_id) in enumerate(trust_pairs, start=1):
            value = float((snapshot.get(observer_id) or {}).get(sender_id, threshold))
            memory_value = float((memory_snapshot.get(observer_id) or {}).get(sender_id, value))
            effective = min(value, memory_value) if source_memory_table else value
            state = ("ACTIVE" if effective >= threshold else "IGNORED") if source_memory_table else ("TRUSTED" if value >= threshold else "DISTRUSTED")
            trust_table[(row, 0)].get_text().set_text(f"R{observer_id}")
            trust_table[(row, 1)].get_text().set_text(f"R{sender_id}")
            trust_table[(row, 2)].get_text().set_text(f"{value:.2f}")
            state_col = 3
            if source_memory_table:
                trust_table[(row, 3)].get_text().set_text(f"{memory_value:.2f}")
                state_col = 4
            trust_table[(row, state_col)].get_text().set_text(state)
            trust_table[(row, state_col)].get_text().set_color("#2e7d32" if effective >= threshold else "#c62828")
        for robot in robots:
            rid = robot.robot_id
            status_lines.append(
                f"R{rid}: tasks={live['deliveries'][rid][frame]} "
                f"carry={'Y' if live['carrying'][rid][frame] else 'N'} "
                f"acc={live['accepted'][rid][frame]} rej={live['rejected'][rid][frame]} "
                f"replans={live['replans'][rid][frame]} "
                f"done={'Y' if live['completed'][rid][frame] else 'N'}"
            )
        status_text.set_text("\n".join(wrap_panel_lines(status_lines, 58)))
        latest_attack_text.set_text("\n".join(wrap_panel_lines(latest_attack_lines, 36)))
        return artists

    anim = FuncAnimation(fig, update, frames=(n_frames * 4) + 1, interval=interval_ms, blit=False, repeat=False)
    fig._modular_animation = anim
    fig._speed_control = speed
    fig._pause_control = pause_button
    if show and _can_show():
        fig.show()
    return fig, anim


def show_live_windows(log, world, robots, *, block=True):
    """Show the shared attack-free heatmap first, then live belief maps."""
    heatmap = show_traffic_heatmap(log, show=True)
    if block and _can_show():
        print("Close the attack-free reference heatmap to start live map playback.", flush=True)
        plt.show()
    maps = show_belief_maps(log, world, robots, show=True)
    if block and _can_show():
        plt.show()
    return heatmap, maps
