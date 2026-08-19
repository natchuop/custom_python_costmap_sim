"""Live traffic heatmap and per-robot belief-map windows."""
from __future__ import annotations

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
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
DISPLAY_GOAL = 10
DISPLAY_FAKE = 11

_COLORS = [
    "#ffffff",
    "#222222",
    "#555555",
    "#bdbdbd",
    "#66bb6a",
    "#f9a825",
    "#2e7d32",
    "#1976d2",
    "#8e24aa",
    "#00e5ff",
    "#ffeb3b",
    "#e53935",
]
_CMAP = ListedColormap(_COLORS)
_NORM = BoundaryNorm(np.arange(-0.5, len(_COLORS) + 0.5, 1), _CMAP.N)


def _goals(robots):
    goals = []
    for robot in robots:
        if robot.completed:
            continue
        task = robot.tasks[robot.task_index]
        goals.append(task.dropoff if robot.carrying else task.pickup)
    return goals


def _fake_cells(log, step):
    cells = []
    for event in log.get("attack_events", ()):
        if event.step > step:
            continue
        if event.attack_type in {AttackType.FAKE_OBSTACLE, AttackType.STALE_REASSERTION}:
            cells.extend(event.cells)
    return cells


def truth_display_grid(world, robots, log, step):
    grid = world.truth_grid(step)
    arr = np.zeros(grid.shape, dtype=np.int16)
    arr[world.static_grid.astype(bool)] = DISPLAY_STATIC
    arr[(grid == 1) & (world.static_grid == 0)] = DISPLAY_DYNAMIC
    for cell in _fake_cells(log, step):
        if 0 <= cell[0] < arr.shape[0] and 0 <= cell[1] < arr.shape[1] and arr[cell] != DISPLAY_STATIC:
            arr[cell] = DISPLAY_FAKE
    for cell in _goals(robots):
        if arr[cell] == DISPLAY_FREE:
            arr[cell] = DISPLAY_GOAL
    return arr


def belief_display_grid(robot, world, log, step):
    rows, cols = world.static_grid.shape
    arr = np.full((rows, cols), DISPLAY_UNKNOWN, dtype=np.int16)
    arr[world.static_grid.astype(bool)] = DISPLAY_STATIC
    robot.fusion.set_time(step)
    for cell, items in robot.fusion.claims.items():
        if arr[cell] == DISPLAY_STATIC or not items:
            continue
        if robot.fusion.blocked(cell, step) or robot.fusion.probability(cell, step) >= 0.5:
            arr[cell] = DISPLAY_BLOCKED
    for cell, observation in robot.belief.direct.items():
        claim, freshness = robot.belief.observation_status(cell, step)
        if freshness != "fresh" or claim is None:
            continue
        arr[cell] = DISPLAY_FREE if claim == ClaimType.FREE else DISPLAY_BLOCKED
    if robot.robot_id == log["malicious_robot_id"]:
        for cell in _fake_cells(log, step):
            if arr[cell] != DISPLAY_STATIC:
                arr[cell] = DISPLAY_FAKE
    if not robot.completed:
        task = robot.tasks[robot.task_index]
        goal = task.dropoff if robot.carrying else task.pickup
        marker = DISPLAY_DROPOFF if robot.carrying else DISPLAY_PICKUP
        if arr[goal] != DISPLAY_STATIC:
            arr[goal] = marker
    return arr


def init_live_log(log, world, robots, config, manifest) -> None:
    rows, cols = world.static_grid.shape
    log["live"] = {
        "truth": [],
        "beliefs": {robot.robot_id: [] for robot in robots},
        "positions": {robot.robot_id: [] for robot in robots},
        "paths": {robot.robot_id: [] for robot in robots},
        "trust": {robot.robot_id: [] for robot in robots},
        "trusted": {robot.robot_id: [] for robot in robots},
        "deliveries": {robot.robot_id: [] for robot in robots},
        "carrying": {robot.robot_id: [] for robot in robots},
        "heatmap": np.zeros((rows, cols), dtype=np.int32),
        "recon_heatmap": None,
        "recon_end": config.phases.recon_steps,
        "attack_start": config.phases.recon_steps,
        "threshold": config.trust.threshold,
        "method": log.get("defense_method"),
    }
    log["attack_events"] = manifest.attack_events
    log["benign_robot_ids"] = manifest.benign_robot_ids


def record_live_frame(log, world, robots, step, phase) -> None:
    live = log.get("live")
    if live is None:
        return
    live["truth"].append(truth_display_grid(world, robots, log, step))
    attacker = log["malicious_robot_id"]
    for robot in robots:
        rid = robot.robot_id
        live["beliefs"][rid].append(belief_display_grid(robot, world, log, step))
        live["positions"][rid].append(robot.position)
        live["paths"][rid].append(list(robot.path or ()))
        live["trust"][rid].append(robot.trust.score(attacker))
        live["trusted"][rid].append(robot.trust.score(attacker) >= live["threshold"])
        live["deliveries"][rid].append(robot.deliveries_completed)
        live["carrying"][rid].append(robot.carrying)
        if rid != attacker:
            live["heatmap"][robot.position] += 1
    if step + 1 == live["recon_end"]:
        live["recon_heatmap"] = live["heatmap"].copy()


def _draw_path(ax, path, color):
    if not path:
        return None
    rows = [cell[0] for cell in path]
    cols = [cell[1] for cell in path]
    return ax.plot(cols, rows, color=color, linewidth=1.3, alpha=0.55)[0]


def _draw_lidar(ax, origin, cells, color):
    lines = []
    r0, c0 = origin
    for row, col in cells:
        line, = ax.plot([c0, col], [r0, row], color=color, linewidth=0.4, alpha=0.18)
        lines.append(line)
    return lines


def show_traffic_heatmap(log, *, show=True):
    live = log.get("live") or {}
    heat = live.get("recon_heatmap")
    if heat is None:
        heat = live.get("heatmap")
    frames = live.get("truth") or []
    recon_end = int(live.get("recon_end") or 0)
    truth = frames[min(max(recon_end - 1, 0), len(frames) - 1)] if frames else None
    if heat is None or truth is None:
        print("No traffic heatmap was recorded. Turn on live maps and run again.")
        return None
    overlay = np.array(heat, dtype=float)
    overlay[truth == DISPLAY_STATIC] = np.nan
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(truth, cmap=_CMAP, norm=_NORM, origin="upper", alpha=0.35)
    image = ax.imshow(overlay, origin="upper", alpha=0.82, cmap="hot")
    ax.set(title=f"Traffic heatmap at end of reconnaissance (step {recon_end})", xlabel="col", ylabel="row")
    fig.colorbar(image, ax=ax, label="benign visits")
    fig.tight_layout()
    if show:
        fig.show()
    return fig


def show_belief_maps(log, world, robots, *, show=True, interval_ms=50):
    live = log.get("live")
    if not live or not live["truth"]:
        print("No live map frames were recorded. Turn on live maps and run again.")
        return None
    robots = sorted(robots, key=lambda item: item.robot_id)
    n_frames = len(live["truth"])
    stride = max(1, n_frames // 400)
    frames = list(range(0, n_frames, stride))
    num_panels = 1 + len(robots)
    fig = plt.figure(figsize=(4.6 * num_panels, 8.2))
    spec = fig.add_gridspec(2, num_panels, height_ratios=(0.22, 1.0), hspace=0.14, wspace=0.08)
    trust_ax = fig.add_subplot(spec[0, :])
    trust_ax.set_axis_off()
    axes = [fig.add_subplot(spec[1, index]) for index in range(num_panels)]
    truth_ax, belief_axes = axes[0], axes[1:]
    truth_ax.set_title("Ground truth")
    truth_img = truth_ax.imshow(live["truth"][0], cmap=_CMAP, norm=_NORM, origin="upper")
    patches = {}
    belief_imgs = {}
    path_lines = {robot.robot_id: [] for robot in robots}
    lidar_lines = []
    attacker = log["malicious_robot_id"]
    for robot, ax in zip(robots, belief_axes):
        role = "MALICIOUS" if robot.robot_id == attacker else "BENIGN"
        ax.set_title(f"R{robot.robot_id} belief ({role})")
        belief_imgs[robot.robot_id] = ax.imshow(
            live["beliefs"][robot.robot_id][0], cmap=_CMAP, norm=_NORM, origin="upper"
        )
    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")
    for robot in robots:
        r, c = live["positions"][robot.robot_id][0]
        color = "purple" if robot.robot_id == attacker else "blue"
        ax = belief_axes[robots.index(robot)]
        for target in (truth_ax, ax):
            patch = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, linewidth=2.0, edgecolor=color, zorder=10)
            target.add_patch(patch)
            patches.setdefault(robot.robot_id, []).append(patch)
    trust_panel = trust_ax.text(
        0.01, 0.92, "", fontsize=10, family="monospace", va="top", ha="left",
        transform=trust_ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#777777", "pad": 5},
    )
    status = fig.text(0.02, 0.02, "", fontsize=10)

    def update(frame):
        nonlocal lidar_lines
        artists = [truth_img, trust_panel, status]
        truth_img.set_data(live["truth"][frame])
        phase = log["phase"][frame] if frame < len(log["phase"]) else ""
        for robot in robots:
            rid = robot.robot_id
            r, c = live["positions"][rid][frame]
            belief_imgs[rid].set_data(live["beliefs"][rid][frame])
            artists.append(belief_imgs[rid])
            for patch in patches[rid]:
                patch.set_xy((c - 0.5, r - 0.5))
                artists.append(patch)
            for line in path_lines[rid]:
                line.remove()
            path_lines[rid] = []
            color = "black" if rid == attacker else "blue"
            line = _draw_path(truth_ax, live["paths"][rid][frame], color)
            if line is not None:
                path_lines[rid].append(line)
                artists.append(line)
            line = _draw_path(belief_axes[robots.index(robot)], live["paths"][rid][frame], color)
            if line is not None:
                path_lines[rid].append(line)
                artists.append(line)
        for line in lidar_lines:
            line.remove()
        lidar_lines = []
        static = world.static_grid
        truth_grid = np.array(static, dtype=np.uint8)
        truth_grid[live["truth"][frame] == DISPLAY_DYNAMIC] = 1
        positions = {robot.robot_id: live["positions"][robot.robot_id][frame] for robot in robots}
        for robot in robots:
            others = [pos for rid, pos in positions.items() if rid != robot.robot_id]
            seen = lidar_observations(truth_grid, positions[robot.robot_id], others)
            color = "magenta" if robot.robot_id == attacker else "#008b8b"
            lidar_lines.extend(_draw_lidar(truth_ax, positions[robot.robot_id], seen, color))
        artists.extend(lidar_lines)
        lines = [f"ATTACKER R{attacker} TRUST  |  {live['method']}  |  {phase}  |  step {frame}"]
        for robot in robots:
            if robot.robot_id == attacker:
                continue
            rid = robot.robot_id
            state = "TRUSTED" if live["trusted"][rid][frame] else "DISTRUSTED"
            lines.append(
                f"R{rid} -> R{attacker}: {live['trust'][rid][frame]:.3f} {state} | "
                f"deliveries={live['deliveries'][rid][frame]} | carrying={live['carrying'][rid][frame]}"
            )
        trust_panel.set_text("\n".join(lines))
        status.set_text(
            "  ".join(
                f"R{robot.robot_id}@({live['positions'][robot.robot_id][frame][0]},{live['positions'][robot.robot_id][frame][1]})"
                for robot in robots
            )
        )
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.10)
    fig._modular_animation = anim
    if show:
        fig.show()
    return fig, anim


def show_live_windows(log, world, robots, *, block=True):
    """Open the traffic heatmap and the four live map windows."""
    heatmap = show_traffic_heatmap(log, show=True)
    maps = show_belief_maps(log, world, robots, show=True)
    if block:
        plt.show()
    return heatmap, maps
