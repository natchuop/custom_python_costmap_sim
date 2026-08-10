"""CSV-driven plots and factual reports for completed simulation runs.

The reporter intentionally reads the stable CSV outputs rather than simulator
objects.  This keeps it useful for old results and for regenerating plots after
the simulation has finished.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHOD_ORDER = ("full_trust", "majority_vote", "trust_fused", "source_linked", "soft_probability")
REPLAN_REASON_BIN_STEPS = 100
RUN_PLOTS = (
    "01_attacker_trust_over_time.png",
    "02_fake_claim_influence.png",
    "02b_attacker_route_cost_influence.png",
    "03_delivery_progress.png",
    "04_replans_over_time.png",
    "04b_replan_reasons_over_time.png",
    "04c_replan_productivity.png",
    "05_navigation_health.png",
    "06_robot_trajectories.png",
    "06b_robot_trajectories_by_phase.png",
    "07_event_timeline.png",
    "08_traffic_health.png",
)
COMPARISON_PLOTS = (
    "01_deliveries_by_method.png",
    "02_replans_by_method.png",
    "03_no_path_and_blockage.png",
    "04_attack_resilience.png",
    "05_trust_detection.png",
    "06_fake_influence_over_time.png",
    "07_route_influence_over_time.png",
    "08_traffic_overhead.png",
)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int(value, default=None):
    try:
        return default if value in (None, "") else int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default=None):
    try:
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value):
    return parse_float(value, None)


def parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def parse_tuple(value, default=None):
    if value in (None, ""):
        return default
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (tuple, list)):
            return tuple(int(item) for item in parsed)
    except (ValueError, SyntaxError, TypeError):
        pass
    return default


def _robot_color(robot_id):
    return f"C{int(robot_id) % 10}"


def _robot_role_label(data, robot_id):
    malicious = parse_int(data.summary.get("malicious_robot_id"), 0)
    return f"R{robot_id} attacker" if int(robot_id) == malicious else f"R{robot_id} benign"


def _safe_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        return ax.legend(**kwargs)
    return None


def _figure_legend(fig, axes, **kwargs):
    handles = []
    labels = []
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        return fig.legend(handles, labels, **kwargs)
    return None


@dataclass
class RunReportData:
    directory: Path
    summary: dict[str, str]
    timeseries: list[dict[str, str]]
    events: list[dict[str, str]]
    manifest: dict
    warnings: list[str]


def load_run_data(run_directory: str | Path) -> RunReportData:
    directory = Path(run_directory)
    summaries = read_csv_rows(directory / "run_summary.csv")
    manifest = {}
    manifest_path = directory / "scenario_manifest.json"
    if not manifest_path.exists() and (directory.parent / "scenario_manifest.json").exists():
        # Comparison runs keep one shared manifest at the comparison root.
        manifest_path = directory.parent / "scenario_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    warnings = []
    if not summaries:
        warnings.append("run_summary.csv is missing or empty")
    return RunReportData(
        directory=directory,
        summary=summaries[0] if summaries else {},
        timeseries=read_csv_rows(directory / "robot_timeseries.csv"),
        events=read_csv_rows(directory / "events.csv"),
        manifest=manifest,
        warnings=warnings,
    )


def discover_method_runs(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(
        (child for child in root.iterdir() if child.is_dir() and (child / "run_summary.csv").exists()),
        key=lambda path: (METHOD_ORDER.index(path.name) if path.name in METHOD_ORDER else len(METHOD_ORDER), path.name),
    ) if root.exists() else []


def _series(rows, robot_id, field, parser=parse_float):
    selected = [row for row in rows if parse_int(row.get("robot_id")) == robot_id]
    selected.sort(key=lambda row: parse_int(row.get("step"), 0))
    values = [(parse_int(row.get("step"), 0), parser(row.get(field))) for row in selected]
    return [(step, value) for step, value in values if value is not None]


def _robot_ids(rows):
    return sorted({parse_int(row.get("robot_id")) for row in rows if parse_int(row.get("robot_id")) is not None})


def _benign_ids(data: RunReportData):
    malicious = parse_int(data.summary.get("malicious_robot_id"), 0)
    return [rid for rid in _robot_ids(data.timeseries) if rid != malicious]


def _phase_changes(rows):
    if not rows:
        return []
    robot_rows = [row for row in rows if parse_int(row.get("robot_id")) == _robot_ids(rows)[0]]
    robot_rows.sort(key=lambda row: parse_int(row.get("step"), 0))
    changes = []
    previous = None
    for row in robot_rows:
        phase = row.get("phase", "")
        if phase and phase != previous:
            changes.append((parse_int(row.get("step"), 0), phase))
            previous = phase
    return changes


def _phase_intervals(rows):
    changes = _phase_changes(rows)
    if not changes:
        return []
    steps = [parse_int(row.get("step"), 0) for row in rows]
    end_step = max(steps) if steps else changes[-1][0]
    intervals = []
    for index, (start, phase) in enumerate(changes):
        end = changes[index + 1][0] if index + 1 < len(changes) else end_step
        intervals.append((start, end, phase))
    return intervals


def _decorate_phases(ax, rows, *, labels=True):
    """Shade phase intervals inside the axes; never place labels above titles."""
    colors = {"RECONNAISSANCE": "#6baed6", "ATTACK": "#fd8d3c", "RECOVERY": "#74c476"}
    for start, end, phase in _phase_intervals(rows):
        ax.axvspan(start, end, color=colors.get(phase, "#bdbdbd"), alpha=0.07, zorder=0)
        ax.axvline(start, color="0.55", linestyle="--", linewidth=0.7, alpha=0.6)
        if labels:
            short = phase.replace("RECONNAISSANCE", "RECON")
            ax.text((start + end) / 2, 0.97, short, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, alpha=0.8)
    return _phase_intervals(rows)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _title(data, title):
    method = data.summary.get("method", "unknown")
    seed = data.summary.get("seed", "unknown")
    return f"{title} — {method}, seed {seed}"


def _plot_trust(data, path):
    field = "attacker_trust"
    if not data.timeseries or not any(field in row for row in data.timeseries):
        data.warnings.append("trust plot skipped: attacker_trust is absent")
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    for rid in _benign_ids(data):
        values = _series(data.timeseries, rid, field)
        if values:
            ax.plot([x for x, _ in values], [y for _, y in values], color=_robot_color(rid), label=_robot_role_label(data, rid))
    threshold = next((parse_float(row.get("trust_threshold")) for row in data.timeseries if row.get("trust_threshold") not in (None, "")), None)
    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle=":", label=f"threshold={threshold:.3f}")
    _decorate_phases(ax, data.timeseries)
    marker_labels = set()
    for event in data.events:
        kind = event.get("kind")
        if kind not in {"attacker_distrusted", "attacker_retrusted"}:
            continue
        step = parse_int(event.get("step")); robot_id = parse_int(event.get("robot_id"))
        if step is None or robot_id is None:
            continue
        marker = "v" if kind == "attacker_distrusted" else "^"
        label = None
        current_trust = parse_float(event.get("current_trust"))
        if current_trust is not None:
            ax.scatter(step, current_trust, marker=marker, color=_robot_color(robot_id), s=42, zorder=4)
    ax.set(title=_title(data, "Attacker trust over time"), xlabel="Simulation step", ylabel="Trust [0, 1]", ylim=(0, 1.05))
    handles = [Line2D([0], [0], color=_robot_color(rid), label=_robot_role_label(data, rid)) for rid in _benign_ids(data)]
    if threshold is not None:
        handles.append(Line2D([0], [0], color="black", linestyle=":", label=f"threshold={threshold:.3f}"))
    handles.extend([Line2D([0], [0], marker="v", color="black", linestyle="None", label="distrust"), Line2D([0], [0], marker="^", color="black", linestyle="None", label="retrust")])
    if handles:
        ax.legend(handles=handles)
    ax.grid(alpha=0.25)
    _save(fig, path)
    return True


def _plot_influence(data, path):
    required = {"active_fake_claim_count", "influential_fake_claim_count"}
    if not data.timeseries or not required.issubset(data.timeseries[0]):
        data.warnings.append("fake influence plot skipped: influence columns are absent")
        return False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    benign = _benign_ids(data)
    if benign:
        reference = dict(_series(data.timeseries, benign[0], "active_fake_claim_count", parse_int))
        disagreements = 0
        for rid in benign[1:]:
            if dict(_series(data.timeseries, rid, "active_fake_claim_count", parse_int)) != reference:
                disagreements += 1
        if disagreements:
            data.warnings.append("stored fake claim counts disagree across benign robots; using first benign robot")
        if reference:
            top.step(sorted(reference), [reference[x] for x in sorted(reference)], where="post", label="Stored / unexpired fake claims")
    for rid in benign:
        values = _series(data.timeseries, rid, "influential_fake_claim_count", parse_int)
        if values:
            bottom.step([x for x, _ in values], [y for _, y in values], where="post", color=_robot_color(rid), label=f"R{rid} influential")
    _decorate_phases(top, data.timeseries); _decorate_phases(bottom, data.timeseries)
    top.set(title="Stored / unexpired fake claims", ylabel="Stored claims")
    bottom.set(title="Currently influential fake cells", xlabel="Simulation step", ylabel="Influential cells")
    for axis in (top, bottom):
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
        axis.grid(alpha=0.25)
    from matplotlib.ticker import MaxNLocator
    top.yaxis.set_major_locator(MaxNLocator(integer=True)); bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(_title(data, "Fake claim influence over time"))
    _save(fig, path)
    return True


def _plot_route_cost(data, path):
    field = "attacker_attributable_cost_on_route"
    if not data.timeseries or field not in data.timeseries[0]:
        data.warnings.append("route influence plot skipped: corrected route-cost field is absent")
        return False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for rid in _benign_ids(data):
        values = _series(data.timeseries, rid, field)
        if values and any(value for _, value in values): top.plot([x for x, _ in values], [y for _, y in values], color=_robot_color(rid), label=f"R{rid} benign")
        affected = _series(data.timeseries, rid, "preferred_route_affected_by_attacker", lambda value: int(parse_bool(value)))
        if affected and any(value for _, value in affected): bottom.step([x for x, _ in affected], [y for _, y in affected], where="post", color=_robot_color(rid), label=f"R{rid} benign")
    _decorate_phases(top, data.timeseries); _decorate_phases(bottom, data.timeseries)
    top.set(title="Attacker-attributable cost on stored route", ylabel="Cost delta")
    bottom.set(title="Preferred route affected by attacker", xlabel="Simulation step", ylabel="Affected [0/1]", ylim=(-0.05, 1.05))
    if not any(value for rid in _benign_ids(data) for _, value in _series(data.timeseries, rid, field)):
        top.set_ylim(0, 1)
        top.text(0.5, 0.5, "No attacker-attributable cost on stored routes in this run", transform=top.transAxes, ha="center", va="center")
    if not any(value for rid in _benign_ids(data) for _, value in _series(data.timeseries, rid, "preferred_route_affected_by_attacker", lambda value: int(parse_bool(value)))):
        bottom.text(0.5, 0.5, "Preferred route was never changed by current attacker evidence", transform=bottom.transAxes, ha="center", va="center")
    for axis in (top, bottom):
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle(_title(data, "Attacker route-cost influence"))
    _save(fig, path)
    return True


def _plot_progress(data, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for rid in _robot_ids(data.timeseries):
        values = _series(data.timeseries, rid, "deliveries_completed", parse_int)
        if values:
            role = "attacker" if rid == parse_int(data.summary.get("malicious_robot_id"), 0) else "benign"
            ax.step([x for x, _ in values], [y for _, y in values], where="post", color=_robot_color(rid), label=f"R{rid} {role}")
    _decorate_phases(ax, data.timeseries)
    ax.set(title=_title(data, "Delivery progress"), xlabel="Simulation step", ylabel="Cumulative deliveries")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def _plot_replans(data, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for rid in _benign_ids(data):
        values = _series(data.timeseries, rid, "benign_total_replans", parse_int)
        if values:
            ax.step([x for x, _ in values], [y for _, y in values], where="post", color=_robot_color(rid), label=f"R{rid} benign")
    _decorate_phases(ax, data.timeseries)
    ax.set(title=_title(data, "Cumulative replans"), xlabel="Simulation step", ylabel="Replans")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


REPLAN_REASON_CATEGORIES = {
    "initial/task transition": ("initial_plan", "delivery", "goal", "pre_intent_path_update"),
    "path invalid / empty": ("path_invalid_or_empty", "path_invalid"),
    "real/world blockage": ("blocked_world",),
    "malicious report on route": ("malicious_report_on_route",),
    "honest report on route": ("honest_report_on_route",),
    "source-linked trust reweight": ("source_linked_trust_reweight",),
    "traffic replan": ("traffic_replan",),
    "traffic yield/recovery": ("traffic_yield", "traffic_deadlock"),
    "fallback retry": ("fallback",),
}


def _replan_category(reason):
    reason = str(reason or "")
    for category, tokens in REPLAN_REASON_CATEGORIES.items():
        if any(token in reason for token in tokens):
            return category
    return "other"


def _plot_replan_reasons(data, path):
    events = [event for event in data.events if event.get("kind") == "replan"]
    if not events:
        data.warnings.append("replan reason plot skipped: replan events are absent")
        return False
    fig, axes = plt.subplots(max(1, len(_benign_ids(data))), 1, figsize=(11, 3 * max(1, len(_benign_ids(data)))), squeeze=False, sharex=True)
    categories = list(REPLAN_REASON_CATEGORIES) + ["other"]
    for index, rid in enumerate(_benign_ids(data)):
        ax = axes[index, 0]
        counts = {category: {} for category in categories}
        for event in events:
            if parse_int(event.get("robot_id")) != rid:
                continue
            step = parse_int(event.get("step"), 0)
            bin_start = (step // REPLAN_REASON_BIN_STEPS) * REPLAN_REASON_BIN_STEPS
            category = _replan_category(event.get("reason"))
            counts[category][bin_start] = counts[category].get(bin_start, 0) + 1
        bottom = np.zeros(len({start for values in counts.values() for start in values}))
        starts = sorted({start for values in counts.values() for start in values})
        width = REPLAN_REASON_BIN_STEPS * 0.82
        for category in categories:
            heights = np.array([counts[category].get(start, 0) for start in starts])
            if heights.any():
                ax.bar(np.array(starts) + width / 2, heights, width=width, bottom=bottom, label=category, align="center")
                bottom += heights
        _decorate_phases(ax, data.timeseries)
        ax.set_ylabel(f"R{rid} replans"); ax.grid(alpha=0.25)
        if index == 0:
            pass
    axes[-1, 0].set_xlabel("Simulation step")
    fig.suptitle(_title(data, "Replan reasons over time"))
    axes[-1, 0].set_xlabel(f"Simulation step (stacked bins of {REPLAN_REASON_BIN_STEPS})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, fontsize=7)
    fig.subplots_adjust(top=0.78, hspace=0.22)
    _save(fig, path)
    return True


def _plot_replan_productivity(data, path):
    events = [event for event in data.events if event.get("kind") == "replan"]
    if not events:
        data.warnings.append("replan productivity plot skipped: replan events are absent")
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    productive_values = []
    unchanged_values = []
    for rid in _benign_ids(data):
        selected = [event for event in events if parse_int(event.get("robot_id")) == rid]
        total = len(selected)
        changed = sum(parse_bool(event.get("next_five_changed")) for event in selected)
        identical = sum(parse_bool(event.get("identical_path")) for event in selected)
        labels.append(f"R{rid} benign")
        productive_values.append(changed)
        unchanged_values.append(total - changed)
        ax.text(len(labels) - 1, total, f"total={total}\nproductive={changed / total:.1%}\nexact-identical={identical}" if total else "total=0", ha="center", va="bottom", fontsize=8)
    x = np.arange(len(labels))
    ax.bar(x, productive_values, label="productive / near-term changed")
    ax.bar(x, unchanged_values, bottom=productive_values, label="near-term unchanged")
    max_total = max([productive + unchanged for productive, unchanged in zip(productive_values, unchanged_values)] or [1])
    ax.set_ylim(0, max_total * 1.22)
    ax.set(title=_title(data, "Replan productivity partition"), ylabel="Replan events", xticks=x, xticklabels=labels); ax.grid(axis="y", alpha=0.25); ax.legend()
    _save(fig, path)
    return True


def _plot_navigation(data, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for rid in _benign_ids(data):
        no_path = _series(data.timeseries, rid, "benign_no_path_steps", parse_int)
        blocked = _series(data.timeseries, rid, "benign_blocked_world_steps", parse_int)
        waits = _series(data.timeseries, rid, "benign_traffic_wait_steps", parse_int)
        distance = _series(data.timeseries, rid, "benign_total_distance", parse_float)
        for series, label in ((no_path, "no-path"), (blocked, "blocked-world"), (waits, "traffic waits")):
            if series and any(value for _, value in series): left.plot([x for x, _ in series], [y for _, y in series], label=f"R{rid} {label}")
        if distance: right.plot([x for x, _ in distance], [y for _, y in distance], label=f"R{rid}")
    _decorate_phases(left, data.timeseries); _decorate_phases(right, data.timeseries)
    no_path_total = max((parse_int(row.get("benign_no_path_steps"), 0) for row in data.timeseries), default=0)
    left_title = "Navigation interruptions" if no_path_total else "Navigation interruptions\nNo-path steps: 0"
    left.set(title=left_title, xlabel="Simulation step", ylabel="Cumulative steps")
    right.set(title="Cumulative distance traveled", xlabel="Simulation step", ylabel="Distance")
    handles, labels = left.get_legend_handles_labels()
    if handles:
        left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    _safe_legend(right)
    left.grid(alpha=0.25); right.grid(alpha=0.25)
    fig.subplots_adjust(bottom=0.22)
    fig.suptitle(_title(data, "Navigation health"))
    _save(fig, path)


def _plot_trajectories(data, path):
    grid = np.asarray(data.manifest.get("static_grid", []))
    if grid.size == 0:
        data.warnings.append("trajectory plot skipped: static_grid is absent")
        return False
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(grid, cmap="gray_r", origin="upper")
    malicious = parse_int(data.summary.get("malicious_robot_id"), 0)
    action_rows = [event for event in data.events if event.get("kind") == "robot_action" and event.get("position") not in (None, "")]
    exact = bool(action_rows)
    for rid in _robot_ids(data.timeseries):
        positions = []
        source = action_rows if exact else data.timeseries
        for row in sorted((r for r in source if parse_int(r.get("robot_id")) == rid), key=lambda r: parse_int(r.get("step"), 0)):
            cell = parse_tuple(row.get("position"))
            if cell is not None and (not positions or cell != positions[-1]): positions.append(cell)
        if positions:
            if exact:
                segments = []
                current = [positions[0]]
                for cell in positions[1:]:
                    previous = current[-1]
                    if abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]) <= 1:
                        current.append(cell)
                    else:
                        if len(current) > 1: segments.append(current)
                        data.warnings.append(f"robot R{rid} trajectory jump exceeds one cell; line segment broken")
                        current = [cell]
                if len(current) > 1: segments.append(current)
                for segment in segments:
                    ax.plot([c for r, c in segment], [r for r, c in segment], color=_robot_color(rid), label=_robot_role_label(data, rid))
            else:
                ax.scatter([c for r, c in positions], [r for r, c in positions], color=_robot_color(rid), label=_robot_role_label(data, rid))
            ax.scatter(positions[0][1], positions[0][0], color=_robot_color(rid), marker="o", s=30)
            ax.scatter(positions[-1][1], positions[-1][0], color=_robot_color(rid), marker="x", s=40)
    title = "Executed robot trajectories" if exact else "Sampled robot positions"
    ax.set(title=_title(data, title), xlabel="col", ylabel="row")
    if not exact: data.warnings.append("trajectory fallback: robot_action.position is absent; sampled positions are scatter-only")
    _safe_legend(ax); ax.grid(alpha=0.2)
    _save(fig, path)
    return True


def _plot_trajectories_by_phase(data, path):
    grid = np.asarray(data.manifest.get("static_grid", []))
    if grid.size == 0:
        data.warnings.append("phase trajectory plot skipped: static_grid is absent")
        return False
    phase_names = ("RECONNAISSANCE", "ATTACK", "RECOVERY")
    action_rows = [event for event in data.events if event.get("kind") == "robot_action" and event.get("position") not in (None, "")]
    if not action_rows:
        data.warnings.append("phase trajectory plot skipped: robot_action.position is absent")
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    malicious = parse_int(data.summary.get("malicious_robot_id"), 0)
    for axis, phase in zip(axes, phase_names):
        axis.imshow(grid, cmap="gray_r", origin="upper")
        for rid in _robot_ids(data.timeseries):
            positions = []
            for row in sorted((event for event in action_rows if parse_int(event.get("robot_id")) == rid and event.get("phase") == phase), key=lambda event: parse_int(event.get("step"), 0)):
                cell = parse_tuple(row.get("position"))
                if cell is not None and (not positions or positions[-1] != cell): positions.append(cell)
            if len(positions) < 1:
                continue
            label = f"R{rid} {'attacker' if rid == malicious else 'benign'}"
            for previous, current in zip(positions, positions[1:]):
                if abs(current[0] - previous[0]) + abs(current[1] - previous[1]) <= 1:
                    axis.plot([previous[1], current[1]], [previous[0], current[0]], color=_robot_color(rid), alpha=0.65, linewidth=1.0, label=label)
            axis.scatter(positions[0][1], positions[0][0], color=_robot_color(rid), marker="o", s=24)
            axis.scatter(positions[-1][1], positions[-1][0], color=_robot_color(rid), marker="x", s=30)
        axis.set_title(phase.replace("RECONNAISSANCE", "RECON")); axis.set_xlabel("col"); axis.set_ylabel("row")
    handles = [Line2D([0], [0], color=_robot_color(rid), label=_robot_role_label(data, rid)) for rid in _robot_ids(data.timeseries)]
    handles.extend([Line2D([0], [0], color="0.25", marker="o", linestyle="None", label="phase start"), Line2D([0], [0], color="0.25", marker="x", linestyle="None", label="phase end")])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=min(5, len(handles)), fontsize=8)
    fig.subplots_adjust(bottom=0.16)
    fig.suptitle(_title(data, "Executed robot trajectories by phase"))
    _save(fig, path)
    return True


def _plot_events(data, path):
    trust_rows = {}
    deadlocks = {}
    for event in data.events:
        kind = event.get("kind", "")
        if kind in {"attacker_distrusted", "attacker_retrusted"}:
            # Older event CSVs did not identify the affected robot.  Keep
            # those reports renderable and use the attacker as the fallback.
            robot_id = parse_int(event.get("robot_id"), parse_int(data.summary.get("malicious_robot_id"), 0))
            label = f"R{robot_id} trust"
            trust_rows.setdefault(label, []).append((parse_int(event.get("step"), 0), kind, robot_id))
        elif kind in {"traffic_deadlock_detected", "traffic_deadlock_recovered"}:
            deadlock_id = event.get("deadlock_id") or f"legacy-{event.get('robot_id')}-{event.get('step')}"
            deadlocks.setdefault(deadlock_id, {})[kind] = parse_int(event.get("step"), 0)
            deadlocks[deadlock_id]["robot_id"] = parse_int(event.get("robot_id"))
    deadlock_robot_ids = sorted({episode.get("robot_id") for episode in deadlocks.values() if episode.get("robot_id") is not None})
    deadlock_labels = [f"R{rid} traffic deadlocks" for rid in deadlock_robot_ids] or ["Traffic deadlock episodes"]
    labels = sorted(trust_rows) + deadlock_labels
    fig, ax = plt.subplots(figsize=(11, 4))
    for index, label in enumerate(sorted(trust_rows)):
        for step, kind, robot_id in trust_rows[label]:
            marker = "v" if kind == "attacker_distrusted" else "^"
            ax.scatter(step, index, marker=marker, color=_robot_color(robot_id), s=36, label=f"{label} {'distrust' if marker == 'v' else 'retrust'}")
    for offset, (deadlock_id, episode) in enumerate(sorted(deadlocks.items())):
        start = episode.get("traffic_deadlock_detected")
        end = episode.get("traffic_deadlock_recovered")
        if start is None:
            continue
        robot_id = episode.get("robot_id")
        deadlock_y = len(sorted(trust_rows)) + (deadlock_robot_ids.index(robot_id) if robot_id in deadlock_robot_ids else 0)
        if end is None:
            ax.plot([start, start], [deadlock_y - 0.25, deadlock_y + 0.25], color="#d62728", linewidth=2)
        else:
            ax.plot([start, end], [deadlock_y, deadlock_y], color="#d62728", linewidth=2)
            ax.scatter([start, end], [deadlock_y, deadlock_y], color="#d62728", s=18)
    _decorate_phases(ax, data.timeseries)
    ax.set(title=_title(data, "Major event timeline"), xlabel="Simulation step", yticks=range(len(labels)), yticklabels=labels)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(legend_labels, handles)); ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, path)


def _plot_traffic(data, path):
    duration = []
    for rid in _benign_ids(data):
        values = _series(data.timeseries, rid, "benign_traffic_wait_steps", parse_int)
        duration.append((f"R{rid} waits", values[-1][1] if values else 0))
    events = (("vertex conflicts", "vertex_conflicts_detected"), ("swap conflicts", "head_on_swap_conflicts_detected"), ("reservation conflicts", "reservation_conflicts_detected"), ("traffic replans", "traffic_replans"), ("yield episodes", "traffic_yield_events"), ("deadlocks detected", "deadlocks_detected"), ("deadlocks recovered", "deadlocks_recovered"), ("overlap violations", "robot_overlap_violations"))
    detected = parse_float(data.summary.get("deadlocks_detected"))
    recovered = parse_float(data.summary.get("deadlocks_recovered"))
    if detected is not None and recovered is not None and recovered > detected:
        data.warnings.append("legacy deadlock recovery events exceed detections; recovery count is raw and not episode-safe")
    fig, (left, right) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]})
    left.barh([label for label, value in duration], [value for label, value in duration])
    total_waits = sum(value for label, value in duration)
    labels = [label for label, field in events if data.summary.get(field) not in (None, "")]
    values = [parse_float(data.summary.get(field), 0) for label, field in events if data.summary.get(field) not in (None, "")]
    bars = right.barh(labels, values)
    for bar, value in zip(bars, values):
        right.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:g}", va="center")
    detected = parse_float(data.summary.get("deadlocks_detected"), 0)
    recovered = parse_float(data.summary.get("deadlocks_recovered"), 0)
    if detected > 0:
        right.text(0.98, 0.02, f"deadlock recovery rate = {recovered / detected:.1%}", transform=right.transAxes, ha="right", va="bottom")
    left.set(title=f"Traffic wait burden — benign total: {total_waits} steps", xlabel="Wait steps")
    right.set(title="Discrete traffic events", xlabel="Event count")
    left.grid(axis="x", alpha=0.25); right.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.30, hspace=0.42)
    fig.suptitle(_title(data, "Traffic health"))
    _save(fig, path)


def _write_run_summary(data, plot_names):
    s = data.summary
    influence_diagnostics = []
    reason_diagnostics = []
    planning_diagnostics = []
    traffic_diagnostics = []
    for rid in _benign_ids(data):
        trust = [value for _, value in _series(data.timeseries, rid, "attacker_trust")]
        influential = [value for _, value in _series(data.timeseries, rid, "influential_fake_claim_count", parse_int)]
        route = [value for _, value in _series(data.timeseries, rid, "attacker_attributable_cost_on_route")]
        distrust_steps = [parse_int(event.get("step")) for event in data.events if event.get("kind") == "attacker_distrusted" and parse_int(event.get("robot_id")) == rid]
        influence_diagnostics.append(f"  R{rid}: first distrust={distrust_steps[0] if distrust_steps else 'NA'}, final trust={trust[-1] if trust else 'NA'}, peak influential={max(influential) if influential else 'NA'}, final influential={influential[-1] if influential else 'NA'}, peak route cost={max(route) if route else 'NA'}")
        reasons = [event for event in data.events if event.get("kind") == "replan" and parse_int(event.get("robot_id")) == rid]
        reason_counts = {}
        for event in reasons:
            category = _replan_category(event.get("reason"))
            reason_counts[category] = reason_counts.get(category, 0) + 1
        reason_diagnostics.append(f"  R{rid} replan reasons: {reason_counts}")
        productive = sum(parse_bool(event.get("next_five_changed")) for event in reasons)
        identical = sum(parse_bool(event.get("identical_path")) for event in reasons)
        ratio = productive / len(reasons) if reasons else 0.0
        planning_diagnostics.append(f"  R{rid}: total replans={len(reasons)}, productive={productive} ({ratio:.1%}), exact-identical={identical}")
        waits = _series(data.timeseries, rid, "benign_traffic_wait_steps", parse_int)
        traffic_diagnostics.append(f"  R{rid}: traffic waits={waits[-1][1] if waits else 'NA'}")
    detected = parse_float(s.get("deadlocks_detected"))
    recovered = parse_float(s.get("deadlocks_recovered"))
    recovery_rate = "NA"
    if detected and recovered is not None and 0 <= recovered <= detected:
        recovery_rate = f"{recovered / detected:.1%}"
    lines = [
        f"Method: {s.get('method', 'unknown')}",
        f"Seed: {s.get('seed', 'unknown')}",
        f"Steps: {s.get('steps_completed', 'unknown')}",
        "", "Mission:",
        f"  benign deliveries: {s.get('benign_total_deliveries_completed', 'NA')}",
        f"  benign success rate: {s.get('benign_success_rate', 'NA')}",
        f"  benign deliveries after attack: {s.get('benign_deliveries_after_attack', 'NA')}",
        f"  benign deliveries after distrust: {s.get('benign_deliveries_after_distrust', 'NA')}",
        f"  benign distance: {s.get('benign_total_distance', 'NA')}",
        f"  benign no-path steps: {s.get('benign_no_path_steps', 'NA')}",
        "", "Planning:",
        f"  total replans: {s.get('benign_total_replans', 'NA')}",
        f"  productive replans: {s.get('benign_productive_replans', 'NA')}",
        *planning_diagnostics,
        *reason_diagnostics,
        "", "Trust:",
        f"  time to distrust attacker: {s.get('time_to_distrust_malicious_robot', 'NA')}",
        f"  final attacker trust mean: {s.get('final_attacker_trust_mean', 'NA')}",
        *influence_diagnostics,
        "", "Traffic:",
        f"  traffic waits: {s.get('benign_traffic_wait_steps', 'NA')}",
        *traffic_diagnostics,
        f"  deadlocks detected/recovered: {s.get('deadlocks_detected', 'NA')}/{s.get('deadlocks_recovered', 'NA')}",
        f"  overlap violations: {s.get('robot_overlap_violations', 'NA')}",
        f"  deadlock recovery rate: {recovery_rate}",
        "", "Generated plots:",
        *[f"  {name}" for name in plot_names],
    ]
    (data.directory / "report_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_run_report(run_directory: str | Path, *, formats=("png",)) -> dict:
    data = load_run_data(run_directory)
    plots = data.directory / "plots"
    generated = []
    if "png" in formats:
        funcs = (
            _plot_trust, _plot_influence, _plot_route_cost, _plot_progress,
            _plot_replans, _plot_replan_reasons, _plot_replan_productivity,
            _plot_navigation, _plot_trajectories, _plot_trajectories_by_phase,
            _plot_events, _plot_traffic,
        )
        for name, function in zip(RUN_PLOTS, funcs):
            try:
                function(data, plots / name)
                if (plots / name).exists():
                    generated.append(name)
            except Exception as exc:
                data.warnings.append(f"{name} failed: {exc}")
    manifest = {"directory": str(data.directory), "generated": generated, "warnings": data.warnings}
    (data.directory / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_run_summary(data, generated)
    return manifest


def _method_data(root):
    return [(path.name, load_run_data(path)) for path in discover_method_runs(root)]


def _validate_comparison(runs):
    hashes = {data.summary.get("scenario_manifest_hash") for _, data in runs}
    if not hashes or hashes == {None}:
        hashes = {data.summary.get("manifest_hash") or data.manifest.get("map_hash") for _, data in runs}
        for _, data in runs:
            data.warnings.append("legacy comparison identity: scenario_manifest_hash is absent; using map hash")
    hashes.discard(None)
    if len(hashes) > 1:
        raise ValueError(f"manifest mismatch across comparison runs: {sorted(hashes)}")
    seeds = {data.summary.get("seed") for _, data in runs if data.summary.get("seed")}
    if len(seeds) > 1:
        raise ValueError(f"seed mismatch across comparison runs: {sorted(seeds)}")
    return next(iter(hashes), "NA"), next(iter(seeds), "NA")


def _bar_plot(rows, metrics, title, path):
    methods = [name for name, _ in rows]
    x = np.arange(len(methods)); width = 0.8 / max(1, len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (field, label) in enumerate(metrics):
        values = [parse_float(data.summary.get(field), 0) for _, data in rows]
        ax.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width, label=label)
    ax.set(title=title, ylabel="Count", xticks=x, xticklabels=methods); ax.legend(); ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _comparison_summary(rows):
    output = []
    for method, data in rows:
        values = dict(data.summary); total = parse_float(values.get("benign_total_replans")); productive = parse_float(values.get("benign_productive_replans"))
        deliveries = parse_float(values.get("benign_total_deliveries_completed")); steps = parse_float(values.get("steps_completed"))
        waits = parse_float(values.get("benign_traffic_wait_steps")); detected = parse_float(values.get("deadlocks_detected")); recovered = parse_float(values.get("deadlocks_recovered"))
        values["method"] = method
        values["productive_replan_ratio"] = "" if not total else productive / total
        values["deliveries_per_1000_steps"] = "" if not steps else deliveries / steps * 1000
        values["distance_per_delivery"] = "" if not deliveries else parse_float(values.get("benign_total_distance"), 0) / deliveries
        values["replans_per_delivery"] = "" if not deliveries else total / deliveries
        values["traffic_waits_per_delivery"] = "" if not deliveries else waits / deliveries
        if not detected or recovered is None or recovered > detected:
            values["deadlock_recovery_rate"] = ""
            if recovered is not None and detected is not None and recovered > detected:
                data.warnings.append("legacy deadlock recovery events are not episode-safe")
        else:
            values["deadlock_recovery_rate"] = recovered / detected
        output.append(values)
    return output


def _write_rows(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _plot_comparison_influence(rows, path, field, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for method, data in rows:
        grouped = {}
        benign = set(_benign_ids(data))
        for row in data.timeseries:
            if parse_int(row.get("robot_id")) not in benign:
                continue
            value = parse_float(row.get(field))
            step = parse_int(row.get("step"))
            if value is not None and step is not None:
                grouped.setdefault(step, []).append(value)
        if grouped:
            points = sorted((step, sum(values) / len(values)) for step, values in grouped.items())
            ax.plot([x for x, _ in points], [y for _, y in points], label=method); plotted = True
    if not plotted:
        plt.close(fig); return False
    ax.set(title=title, xlabel="Simulation step", ylabel=ylabel); ax.legend(); ax.grid(alpha=0.25)
    _save(fig, path); return True


def _comparison_split_plot(rows, left_metrics, right_metrics, title, path):
    methods = [name for name, _ in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metrics, subtitle in ((left, left_metrics, "Navigation failures"), (right, right_metrics, "Traffic burden")):
        x = np.arange(len(methods)); width = 0.8 / max(1, len(metrics))
        for index, (field, label) in enumerate(metrics):
            values = [parse_float(data.summary.get(field), 0) for _, data in rows]
            axis.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width, label=label)
        axis.set(title=subtitle, xticks=x, xticklabels=methods); axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title); _save(fig, path)


def _comparison_trust_plot(rows, path):
    methods = [name for name, _ in rows]; x = np.arange(len(methods))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    left.bar(x, [parse_float(data.summary.get("time_to_distrust_malicious_robot"), 0) for _, data in rows]); left.set(title="Time to distrust", ylabel="Steps", xticks=x, xticklabels=methods)
    right.bar(x, [parse_float(data.summary.get("final_attacker_trust_mean"), 0) for _, data in rows]); right.set(title="Final attacker trust", ylabel="Trust [0, 1]", ylim=(0, 1), xticks=x, xticklabels=methods)
    fig.suptitle("Trust detection by defense method"); _save(fig, path)


def _comparison_attack_plot(rows, path):
    methods = [name for name, _ in rows]; x = np.arange(len(methods))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    left.bar(x - width / 2, [parse_float(data.summary.get("benign_deliveries_after_attack"), 0) for _, data in rows], width, label="after attack")
    left.bar(x + width / 2, [parse_float(data.summary.get("benign_deliveries_after_distrust"), 0) for _, data in rows], width, label="after distrust")
    right.bar(x, [parse_float(data.summary.get("malicious_verified_false_reports"), 0) for _, data in rows], label="false reports")
    left.set(title="Deliveries after attack", xticks=x, xticklabels=methods); right.set(title="Malicious false reports", xticks=x, xticklabels=methods)
    left.legend(); right.legend(); fig.suptitle("Attack resilience by defense method"); _save(fig, path)


def _comparison_traffic_plot(rows, path):
    methods = [name for name, _ in rows]; x = np.arange(len(methods))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    left.bar(x, [parse_float(data.summary.get("benign_traffic_wait_steps"), 0) for _, data in rows]); left.set(title="Traffic wait burden", ylabel="Wait steps", xticks=x, xticklabels=methods)
    metrics = (("traffic_replans", "replans"), ("traffic_yield_events", "yields"), ("deadlocks_detected", "deadlocks"), ("deadlocks_recovered", "recoveries"), ("robot_overlap_violations", "overlaps"))
    width = 0.8 / len(metrics)
    for index, (field, label) in enumerate(metrics):
        right.bar(x + (index - 2) * width, [parse_float(data.summary.get(field), 0) for _, data in rows], width, label=label)
    right.set(title="Discrete traffic events", ylabel="Events", xticks=x, xticklabels=methods); right.legend(fontsize=8)
    fig.suptitle("Traffic overhead by defense method"); _save(fig, path)


def _write_comparison_report(root, rows, summary_rows, manifest_hash, seed, warnings):
    def best(field, lower=True):
        values = [(name, parse_float(data.summary.get(field))) for name, data in rows]
        values = [(name, value) for name, value in values if value is not None]
        if not values: return "NA"
        return min(values, key=lambda item: item[1])[0] if lower else max(values, key=lambda item: item[1])[0]
    lines = [
        "This report describes one controlled comparison run; it makes no statistical claims.",
        "", "Scenario:", f"  manifest hash: {manifest_hash}", f"  seed: {seed}", f"  methods: {', '.join(name for name, _ in rows)}",
        "", "Mission outcomes:", f"  lowest no-path steps in this run: {best('benign_no_path_steps')}", f"  highest delivery count in this run: {best('benign_total_deliveries_completed', lower=False)}",
        "", "Planning:", f"  lowest total replans in this run: {best('benign_total_replans')}",
        "", "Attack influence:", f"  earliest/lowest trust timing is reported in comparison_summary.csv; no causal ranking is asserted.",
        "", "Traffic:", f"  lowest traffic waits in this run: {best('benign_traffic_wait_steps')}",
    ]
    if warnings:
        lines.extend(["", "Warnings:", *[f"  {warning}" for warning in warnings]])
    (root / "comparison_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_comparison_report(comparison_directory: str | Path, *, formats=("png",)) -> dict:
    root = Path(comparison_directory)
    rows = _method_data(root)
    if not rows:
        raise ValueError(f"no method run directories found under {root}")
    manifest_hash, seed = _validate_comparison(rows)
    warnings = []
    summary_rows = _comparison_summary(rows)
    for _, data in rows:
        warnings.extend(data.warnings)
    plot_root = root / "comparison_plots"
    generated = []
    if "png" in formats:
        jobs = (
            ("01_deliveries_by_method.png", lambda p: _bar_plot(rows, (("benign_total_deliveries_completed", "total deliveries"), ("benign_deliveries_after_attack", "after attack"), ("benign_deliveries_after_distrust", "after distrust")), "Deliveries by defense method", p)),
            ("02_replans_by_method.png", lambda p: _bar_plot(rows, (("benign_total_replans", "total replans"), ("benign_productive_replans", "productive replans")), "Replans by defense method", p)),
            ("03_no_path_and_blockage.png", lambda p: _comparison_split_plot(rows, (("benign_no_path_steps", "no-path"), ("benign_blocked_world", "blocked world")), (("benign_traffic_wait_steps", "traffic waits"),), "No-path, blockage, and traffic burden", p)),
            ("04_attack_resilience.png", lambda p: _comparison_attack_plot(rows, p)),
            ("05_trust_detection.png", lambda p: _comparison_trust_plot(rows, p)),
            ("06_fake_influence_over_time.png", lambda p: _plot_comparison_influence(rows, p, "influential_fake_claim_count", "Fake influence over time by method", "Mean influential fake cells")),
            ("07_route_influence_over_time.png", lambda p: _plot_comparison_influence(rows, p, "attacker_attributable_cost_on_route", "Attacker route influence over time", "Mean attacker cost on stored route")),
            ("08_traffic_overhead.png", lambda p: _comparison_traffic_plot(rows, p)),
        )
        for name, job in jobs:
            try:
                result = job(plot_root / name)
                if result is not False: generated.append(name)
            except Exception as exc:
                warnings.append(f"{name} failed: {exc}")
    _write_rows(root / "comparison_summary.csv", summary_rows)
    _write_comparison_report(root, rows, summary_rows, manifest_hash, seed, warnings)
    manifest = {"directory": str(root), "generated": generated, "warnings": warnings, "methods": [name for name, _ in rows]}
    (root / "comparison_plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


MULTISEED_DIRECTIONS = {
    "benign_total_deliveries_completed": ("higher_better", "deliveries"),
    "benign_deliveries_after_attack": ("higher_better", "deliveries"),
    "benign_deliveries_after_distrust": ("higher_better", "deliveries"),
    "benign_success_rate": ("higher_better", "ratio"),
    "deliveries_per_1000_steps": ("higher_better", "deliveries/1000 steps"),
    "benign_no_path_steps": ("lower_better", "steps"),
    "distance_per_delivery": ("lower_better", "distance/delivery"),
    "replans_per_delivery": ("lower_better", "replans/delivery"),
    "traffic_waits_per_delivery": ("lower_better", "waits/delivery"),
    "deadlocks_detected": ("lower_better", "deadlocks"),
    "robot_overlap_violations": ("target_zero", "violations"),
    "intent_commit_mismatches": ("target_zero", "mismatches"),
    "productive_replan_ratio": ("diagnostic", "ratio"),
    "time_to_distrust_malicious_robot": ("diagnostic", "steps"),
    "final_attacker_trust_mean": ("diagnostic", "trust"),
    "recovery_trust_gain": ("diagnostic", "trust"),
    "attack_mean_influential_fake_cells": ("lower_better", "cells"),
    "attack_fraction_samples_influenced": ("lower_better", "ratio"),
    "attack_mean_attacker_route_cost": ("lower_better", "cost"),
    "attack_fraction_route_affected": ("lower_better", "ratio"),
}

def _multiseed_runs(root):
    rows = []
    for seed_dir in sorted(Path(root).glob("seed_*")):
        if not seed_dir.is_dir(): continue
        for method_dir in discover_method_runs(seed_dir):
            data = load_run_data(method_dir)
            if data.summary: rows.append((parse_int(seed_dir.name.split("_")[-1]), method_dir.name, data))
    return rows

def _seed_metric(data, field):
    value = parse_float(data.summary.get(field))
    if value is not None: return value
    total = parse_float(data.summary.get("benign_total_replans")); productive = parse_float(data.summary.get("benign_productive_replans")); deliveries = parse_float(data.summary.get("benign_total_deliveries_completed")); steps = parse_float(data.summary.get("steps_completed")); waits = parse_float(data.summary.get("benign_traffic_wait_steps"))
    derived = {"productive_replan_ratio": productive / total if total else None, "deliveries_per_1000_steps": deliveries / steps * 1000 if steps else None, "distance_per_delivery": parse_float(data.summary.get("benign_total_distance")) / deliveries if deliveries else None, "replans_per_delivery": total / deliveries if deliveries else None, "traffic_waits_per_delivery": waits / deliveries if deliveries else None}
    if field in derived: return derived[field]
    if field == "recovery_trust_gain":
        trust = _series(data.timeseries, parse_int(data.summary.get("malicious_robot_id"), 0), "attacker_trust")
        if trust: return trust[-1][1] - min(value for _, value in trust)
    return None

def _write_multiseed_csvs(root, runs):
    aggregate = Path(root) / "aggregate"; aggregate.mkdir(parents=True, exist_ok=True)
    raw = []
    for seed, method, data in runs:
        row = dict(data.summary)
        row.update({"seed": seed, "method": method, "scenario_manifest_hash": data.summary.get("scenario_manifest_hash", data.summary.get("manifest_hash", ""))})
        for field in MULTISEED_DIRECTIONS: row[field] = _seed_metric(data, field)
        for field in ("benign_traffic_wait_steps", "traffic_replans", "deadlocks_detected", "robot_overlap_violations"):
            row[field] = parse_float(data.summary.get(field))
        raw.append(row)
    _write_rows(aggregate / "multiseed_runs.csv", raw)
    summary_rows = []
    for method in sorted({row["method"] for row in raw}):
        method_rows = [row for row in raw if row["method"] == method]
        for metric, (direction, unit) in MULTISEED_DIRECTIONS.items():
            values = [row[metric] for row in method_rows if row.get(metric) is not None]
            stats = __import__("map_poisoning.statistics", fromlist=["summarize"]).summarize(values)
            summary_rows.append({"method": method, "metric": metric, "direction": direction, "unit": unit, "n": stats["n"], "n_missing": len(method_rows)-stats["n"], **{key: stats[key] for key in ("mean","sample_std","sem","ci95_low","ci95_high","median","min","max")}})
    _write_rows(aggregate / "multiseed_summary.csv", summary_rows)
    paired = []
    source = {row["seed"]: row for row in raw if row["method"] == "source_linked"}
    for baseline in sorted({row["method"] for row in raw if row["method"] != "source_linked"}):
        base = {row["seed"]: row for row in raw if row["method"] == baseline}
        for metric, (direction, _) in MULTISEED_DIRECTIONS.items():
            seeds = sorted(set(source) & set(base)); differences = [source[seed][metric] - base[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None]
            from .statistics import paired_summary
            stats = paired_summary([source[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None], [base[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None])
            improvement = None if direction == "diagnostic" else stats["mean_difference"] * (1 if direction == "higher_better" else -1) if stats["mean_difference"] is not None else None
            paired.append({"baseline_method": baseline, "metric": metric, "n_pairs": stats["n_pairs"], "mean_difference": stats["mean_difference"], "sample_std_difference": stats["sample_std_difference"], "sem_difference": stats["sem_difference"], "ci95_difference_low": stats["ci95_difference_low"], "ci95_difference_high": stats["ci95_difference_high"], "improvement_difference": improvement})
    _write_rows(aggregate / "paired_method_differences.csv", paired)
    return raw, summary_rows, paired

def _multiseed_point_plot(raw, metrics, title, path):
    methods = sorted({row["method"] for row in raw}); fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5), squeeze=False)
    from .statistics import summarize
    for axis, metric in zip(axes[0], metrics):
        for x, method in enumerate(methods):
            values = [row[metric] for row in raw if row["method"] == method and row.get(metric) is not None]; stats = summarize(values)
            if values:
                jitter = np.linspace(-.12, .12, len(values)); axis.scatter(np.full(len(values), x)+jitter, values, s=18, alpha=.65)
                axis.errorbar(x, stats["mean"], yerr=None if stats["ci95_low"] is None else [[stats["mean"]-stats["ci95_low"]],[stats["ci95_high"]-stats["mean"]]], fmt="o", color="black", capsize=4, zorder=3)
        axis.set_title(metric.replace("_", " ")); axis.set_xticks(range(len(methods)), methods, rotation=30, ha="right"); axis.grid(axis="y", alpha=.25)
    fig.suptitle(title); fig.tight_layout(); _save(fig, path)

def _multiseed_timeseries(raw_runs, field, path, title, team_sum=False):
    grouped = {}
    for seed, method, data in raw_runs:
        by_step = {}
        for row in data.timeseries:
            if team_sum:
                by_step.setdefault(parse_int(row.get("step"), 0), []).append(parse_float(row.get(field), 0))
            else:
                if parse_int(row.get("robot_id")) in _benign_ids(data): by_step.setdefault(parse_int(row.get("step"), 0), []).append(parse_float(row.get(field), 0))
        for step, values in by_step.items(): grouped.setdefault(method, {}).setdefault(step, []).append(sum(values) if team_sum else sum(values)/len(values))
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, steps in grouped.items():
        points=[]
        for step, values in sorted(steps.items()):
            from .statistics import summarize
            stats=summarize(values)
            if stats["mean"] is not None: points.append((step, stats))
        if points:
            x=[point[0] for point in points]; y=[point[1]["mean"] for point in points]; ax.plot(x,y,label=method)
            if all(point[1]["ci95_low"] is not None for point in points): ax.fill_between(x,[point[1]["ci95_low"] for point in points],[point[1]["ci95_high"] for point in points],alpha=.15)
    ax.set(title=title,xlabel="Simulation step"); ax.legend(); ax.grid(alpha=.25); _save(fig,path)

def generate_multiseed_report(root: str | Path, *, confidence=0.95, formats=("png",)) -> dict:
    root=Path(root); runs=_multiseed_runs(root)
    if not runs: raise ValueError(f"no completed multi-seed runs found under {root}")
    raw, summary, paired = _write_multiseed_csvs(root, runs); aggregate=root/"aggregate"; plots=aggregate/"plots"; generated=[]
    by_seed = {}
    for seed, method, data in runs: by_seed.setdefault(seed, []).append((method, data))
    complete_seeds = [seed for seed, cells in by_seed.items() if len(cells) == len({method for _, method, _ in runs})]
    hashes = {data.summary.get("map_hash", data.summary.get("manifest_hash")) for _, _, data in runs}
    fairness_ok = all(len({data.summary.get("scenario_manifest_hash", data.summary.get("manifest_hash")) for _, data in cells}) == 1 for cells in by_seed.values())
    commits=[]; dirty=[]; presets=[]; warnings=[]
    for _, _, data in runs:
        metadata_path=data.directory/"run_metadata.json"
        if metadata_path.exists():
            try:
                metadata=json.loads(metadata_path.read_text(encoding="utf-8")); commits.append(metadata.get("git_commit")); dirty.append(metadata.get("git_dirty"))
            except json.JSONDecodeError: warnings.append(f"invalid run metadata: {metadata_path}")
        config_path=data.directory/"effective_config.json"
        if config_path.exists():
            try: presets.append(json.loads(config_path.read_text(encoding="utf-8")).get("scenario_preset"))
            except json.JSONDecodeError: pass
    if len({value for value in commits if value}) > 1: warnings.append("runs from multiple Git commits are mixed")
    if any(value is True for value in dirty): warnings.append("one or more runs were created from a dirty Git worktree")
    if len({value for value in presets if value}) > 1: warnings.append("scenario presets differ across runs")
    warnings.extend([] if fairness_ok else ["scenario_manifest_hash mismatch within a seed"])
    validation = {"valid": fairness_ok and all(len(cells) == len({method for _, method, _ in runs}) for cells in by_seed.values()), "requested_seeds": sorted(by_seed), "complete_seeds": sorted(complete_seeds), "incomplete_seeds": sorted(set(by_seed)-set(complete_seeds),), "per_method_run_counts": {method: sum(row["method"] == method for row in raw) for method in sorted({row["method"] for row in raw})}, "paired_seed_count_by_comparison": {method: sum(seed in {row["seed"] for row in raw if row["method"] == "source_linked"} for seed in {row["seed"] for row in raw if row["method"] == method}) for method in sorted({row["method"] for row in raw if row["method"] != "source_linked"})}, "map_hash": next(iter(hashes), None), "scenario_preset": next(iter({value for value in presets if value}), None), "git_commits_observed": sorted(set(commits)), "git_dirty_flags_observed": sorted(set(dirty), key=str), "warnings": warnings}
    (aggregate / "batch_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    jobs=[("01_deliveries_by_method.png",lambda p:_multiseed_point_plot(raw,["benign_total_deliveries_completed","benign_deliveries_after_attack","benign_deliveries_after_distrust"],"Deliveries by method (seed points and 95% CI)",p)),("02_mission_efficiency_by_method.png",lambda p:_multiseed_point_plot(raw,["distance_per_delivery","replans_per_delivery","traffic_waits_per_delivery"],"Mission efficiency by method",p)),("03_attack_influence_by_method.png",lambda p:_multiseed_point_plot(raw,["attack_mean_influential_fake_cells","attack_fraction_samples_influenced","attack_mean_attacker_route_cost","attack_fraction_route_affected"],"Attack influence by method",p)),("04_trust_diagnostics_by_method.png",lambda p:_multiseed_point_plot(raw,["time_to_distrust_malicious_robot","final_attacker_trust_mean","recovery_trust_gain"],"Trust diagnostics",p)),("05_traffic_burden_by_method.png",lambda p:_multiseed_point_plot(raw,["benign_traffic_wait_steps","deadlocks_detected","traffic_replans","robot_overlap_violations"],"Traffic burden by method",p)),("06_fake_influence_over_time.png",lambda p:_multiseed_timeseries(runs,"influential_fake_claim_count",p,"Fake influence over time",False)),("07_delivery_progress_over_time.png",lambda p:_multiseed_timeseries(runs,"deliveries_completed",p,"Delivery progress over time",True)),("08_replans_over_time.png",lambda p:_multiseed_timeseries(runs,"benign_total_replans",p,"Replans over time",True))]
    if "png" in formats:
        for name,job in jobs:
            try: job(plots/name); generated.append(name)
            except Exception: continue
    if "png" in formats:
        fig, ax=plt.subplots(figsize=(10,6)); selected=[row for row in paired if row["metric"] in {"benign_deliveries_after_attack","deliveries_per_1000_steps","replans_per_delivery","traffic_waits_per_delivery","attack_mean_influential_fake_cells","attack_fraction_route_affected"} and row["improvement_difference"] is not None]; labels=[f"{row['baseline_method']} / {row['metric']}" for row in selected]; means=[row["improvement_difference"] for row in selected]; lows=[row["ci95_difference_low"] for row in selected]; highs=[row["ci95_difference_high"] for row in selected]; y=np.arange(len(labels));
        if labels: ax.errorbar(means,y,xerr=[[m-l if l is not None else 0 for m,l in zip(means,lows)],[h-m if h is not None else 0 for m,h in zip(means,highs)]],fmt="o")
        ax.axvline(0,color="0.4"); ax.set(yticks=y,yticklabels=labels,title="Source-linked paired improvement",xlabel="Positive means source-linked better"); ax.grid(axis="x",alpha=.25); _save(fig,plots/"09_source_linked_paired_differences.png"); generated.append("09_source_linked_paired_differences.png")
        from .statistics import summarize
        methods=sorted({row["method"] for row in raw}); fig, ax=plt.subplots(figsize=(8,6))
        for method in methods:
            xvals=[row["replans_per_delivery"] for row in raw if row["method"]==method and row.get("replans_per_delivery") is not None]; yvals=[row["benign_deliveries_after_attack"] for row in raw if row["method"]==method and row.get("benign_deliveries_after_attack") is not None]
            if xvals and yvals:
                xs=summarize(xvals); ys=summarize(yvals); ax.errorbar(xs["mean"],ys["mean"],xerr=None if xs["ci95_low"] is None else [[xs["mean"]-xs["ci95_low"]],[xs["ci95_high"]-xs["mean"]]],yerr=None if ys["ci95_low"] is None else [[ys["mean"]-ys["ci95_low"]],[ys["ci95_high"]-ys["mean"]]],fmt="o",label=method,capsize=4)
        ax.set(xlabel="Replans per delivery",ylabel="Deliveries after attack",title="Delivery vs replan tradeoff"); _safe_legend(ax); ax.grid(alpha=.25); _save(fig,plots/"10_delivery_vs_replan_tradeoff.png"); generated.append("10_delivery_vs_replan_tradeoff.png")
    _write_multiseed_report(aggregate, raw, paired)
    manifest={"directory":str(root),"generated":generated,"warnings":[],"runs":len(runs)}; (aggregate/"aggregate_plot_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest

def _write_multiseed_report(aggregate, raw, paired):
    methods=sorted({row["method"] for row in raw}); lines=["Multi-seed report", "", "Methods: "+", ".join(methods), f"Completed seed/method runs: {len(raw)}", "", "Mission outcomes:"]
    from .statistics import summarize
    for method in methods:
        values=[row["benign_total_deliveries_completed"] for row in raw if row["method"]==method and row.get("benign_total_deliveries_completed") is not None]; s=summarize(values); lines.append(f"  {method}: mean deliveries={s['mean']} SD={s['sample_std']} 95% CI=[{s['ci95_low']}, {s['ci95_high']}]")
    lines += ["", "Paired source_linked comparisons:"]
    for row in paired:
        if row["metric"]=="benign_deliveries_after_attack" and row["improvement_difference"] is not None: lines.append(f"  source_linked vs {row['baseline_method']}: mean paired improvement={row['improvement_difference']:.4g}, 95% CI [{row['ci95_difference_low']}, {row['ci95_difference_high']}], n={row['n_pairs']}")
    (aggregate/"multiseed_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv=None):
    parser = __import__("argparse").ArgumentParser(description="Generate plots and reports from simulation CSV output")
    parser.add_argument("path")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--multiseed", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.path)
    if args.multiseed:
        result = generate_multiseed_report(root)
    elif args.compare or (not args.run and not (root / "run_summary.csv").exists() and discover_method_runs(root)):
        result = generate_comparison_report(root)
    else:
        result = generate_run_report(root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
