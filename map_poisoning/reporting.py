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


METHOD_ORDER = ("majority_vote", "full_trust", "trust_fused", "source_memory", "soft_probability")


def _ordered_methods(values):
    """Return unique method names with the four primary methods first."""
    unique = list(dict.fromkeys(value for value in values if value))
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(unique, key=lambda method: (rank.get(method, len(METHOD_ORDER)), method))
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
    bottom.set(
        title="Currently navigation-relevant fake claims",
        xlabel="Simulation step",
        ylabel="Influential claims",
    )
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
    "initial/task transition": ("initial_plan", "delivery", "goal", "pre_intent_path_update", "task_transition"),
    "path invalid / empty": ("path_invalid_or_empty", "path_invalid"),
    "real/world blockage": ("blocked_world", "blocked_move"),
    "malicious report on route": ("malicious_report_on_route", "peer_report_on_route"),
    "honest report on route": ("honest_report_on_route",),
    "source-memory trust reweight": ("source_memory_trust_reweight",),
    "direct verification": ("direct_verification",),
    "traffic replan": ("traffic_replan", "traffic_wait_reroute"),
    "traffic yield/recovery": ("traffic_yield", "traffic_deadlock"),
    "fallback retry": ("fallback",),
}


def _replan_category(reason):
    reason = str(reason or "")
    for category, tokens in REPLAN_REASON_CATEGORIES.items():
        if any(token in reason for token in tokens):
            return category
    return "other"


def _event_present(event, key):
    return event.get(key) not in (None, "")


def _replan_was_productive(event):
    if _event_present(event, "next_five_changed"):
        return parse_bool(event.get("next_five_changed"))
    return parse_bool(event.get("changed"))


def _replan_was_identical(event):
    if _event_present(event, "identical_path"):
        return parse_bool(event.get("identical_path"))
    return not _replan_was_productive(event)


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
        changed = sum(_replan_was_productive(event) for event in selected)
        identical = sum(_replan_was_identical(event) for event in selected)
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
        productive = sum(_replan_was_productive(event) for event in reasons)
        identical = sum(_replan_was_identical(event) for event in reasons)
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
        f"  full-cycle duration mean/median/p95: {s.get('benign_delivery_cycle_duration_mean_steps', 'NA')}/{s.get('benign_delivery_cycle_duration_median_steps', 'NA')}/{s.get('benign_delivery_cycle_duration_p95_steps', 'NA')}",
        f"  loaded-leg duration mean/median/p95: {s.get('benign_loaded_delivery_duration_mean_steps', 'NA')}/{s.get('benign_loaded_delivery_duration_median_steps', 'NA')}/{s.get('benign_loaded_delivery_duration_p95_steps', 'NA')}",
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
        f"  time to first victim distrust: {s.get('time_to_distrust_malicious_robot', 'NA')}",
        f"  time to all victims distrust: {s.get('time_to_all_benign_distrust', 'NA')}",
        f"  victims that distrusted attacker: {s.get('distrusted_benign_robot_count', 'NA')}",
        f"  minimum attacker trust mean: {s.get('attacker_min_trust_mean', 'NA')}",
        f"  final attacker trust mean: {s.get('final_attacker_trust_mean', 'NA')}",
        f"  malicious reports operationally ignored: {s.get('malicious_reports_operationally_ignored', 'NA')}",
        f"  attacks causing counterfactual path changes: {s.get('attack_induced_path_changes', 'NA')}",
        f"  route penalty mean/max/total: {s.get('attack_route_penalty_mean', 'NA')}/{s.get('attack_route_penalty_max', 'NA')}/{s.get('attack_route_penalty_total', 'NA')}",
        f"  extra path length mean/max/total: {s.get('attack_extra_path_length_mean', 'NA')}/{s.get('attack_extra_path_length_max', 'NA')}/{s.get('attack_extra_path_length_total', 'NA')}",
        f"  steps route affected by attacker: {s.get('steps_route_affected_by_attacker', 'NA')}",
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
    methods = [name for name, _ in rows]; x = np.arange(len(methods)); width = 0.36
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    first = [parse_float(data.summary.get("time_to_distrust_malicious_robot")) for _, data in rows]
    all_victims = [parse_float(data.summary.get("time_to_all_benign_distrust")) for _, data in rows]
    first_plot = [np.nan if value is None else value for value in first]
    all_plot = [np.nan if value is None else value for value in all_victims]
    left.bar(x - width / 2, first_plot, width, label="first victim")
    left.bar(x + width / 2, all_plot, width, label="all victims")
    left.set(title="Time to distrust", ylabel="Steps", xticks=x, xticklabels=methods); left.legend()
    minimum = [parse_float(data.summary.get("attacker_min_trust_mean")) for _, data in rows]
    final = [parse_float(data.summary.get("final_attacker_trust_mean")) for _, data in rows]
    right.bar(x - width / 2, [np.nan if value is None else value for value in minimum], width, label="minimum")
    right.bar(x + width / 2, [np.nan if value is None else value for value in final], width, label="final")
    right.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    right.set(title="Attacker trust", ylabel="Trust [0, 1]", ylim=(0, 1), xticks=x, xticklabels=methods); right.legend()
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
    "benign_delivery_cycle_duration_mean_steps": ("lower_better", "steps"),
    "benign_delivery_cycle_duration_median_steps": ("lower_better", "steps"),
    "benign_delivery_cycle_duration_p95_steps": ("lower_better", "steps"),
    "benign_loaded_delivery_duration_mean_steps": ("lower_better", "steps"),
    "benign_loaded_delivery_duration_median_steps": ("lower_better", "steps"),
    "benign_loaded_delivery_duration_p95_steps": ("lower_better", "steps"),
    "benign_total_distance": ("lower_better", "cells"),
    "benign_planning_checks": ("lower_better", "checks"),
    "benign_path_changes": ("lower_better", "changes"),
    "temporary_obstacle_replan_checks": ("diagnostic", "checks"),
    "temporary_obstacle_path_changes": ("diagnostic", "changes"),
    "robot_on_route_replan_checks": ("diagnostic", "checks"),
    "robot_on_route_path_changes": ("diagnostic", "changes"),
    "malicious_report_replan_checks": ("lower_better", "checks"),
    "malicious_report_path_changes": ("lower_better", "changes"),
    "fake_obstacle_replan_checks": ("lower_better", "checks"),
    "fake_obstacle_path_changes": ("lower_better", "changes"),
    "false_clearance_replan_checks": ("lower_better", "checks"),
    "false_clearance_path_changes": ("lower_better", "changes"),
    "stale_reassertion_replan_checks": ("lower_better", "checks"),
    "stale_reassertion_path_changes": ("lower_better", "changes"),
    "benign_blocked_moves": ("target_zero", "moves"),
    "replans_per_delivery": ("lower_better", "replans/delivery"),
    "traffic_waits_per_delivery": ("lower_better", "waits/delivery"),
    "deadlocks_detected": ("lower_better", "deadlocks"),
    "robot_overlap_violations": ("target_zero", "violations"),
    "intent_commit_mismatches": ("target_zero", "mismatches"),
    "productive_replan_ratio": ("diagnostic", "ratio"),
    "time_to_distrust_malicious_robot": ("diagnostic", "steps"),
    "time_to_all_benign_distrust": ("diagnostic", "steps"),
    "distrusted_benign_robot_count": ("higher_better", "robots"),
    "attacker_min_trust_mean": ("diagnostic", "trust"),
    "final_attacker_trust_mean": ("diagnostic", "trust"),
    "malicious_reports_operationally_ignored": ("higher_better", "reports"),
    "attack_induced_path_changes": ("lower_better", "attacks"),
    "attack_route_penalty_mean": ("lower_better", "cost"),
    "attack_route_penalty_max": ("lower_better", "cost"),
    "attack_route_penalty_total": ("lower_better", "cost"),
    "attack_extra_path_length_mean": ("lower_better", "cells"),
    "attack_extra_path_length_max": ("lower_better", "cells"),
    "attack_extra_path_length_total": ("lower_better", "cells"),
    "steps_route_affected_by_attacker": ("lower_better", "steps"),
    "recovery_start_attacker_trust_mean": ("diagnostic", "trust"),
    "recovery_trust_gain": ("diagnostic", "trust"),
    "deliveries_during_recon": ("higher_better", "deliveries"),
    "deliveries_during_attack": ("higher_better", "deliveries"),
    "deliveries_during_recovery": ("higher_better", "deliveries"),
    "traffic_wait_steps_during_attack": ("lower_better", "steps"),
    "replans_during_attack": ("lower_better", "replans"),
    "mean_influential_fake_cells_during_attack": ("lower_better", "cells"),
    "attack_mean_influential_fake_cells": ("lower_better", "cells"),
    "attack_fraction_samples_influenced": ("lower_better", "ratio"),
    "attack_mean_attacker_route_cost": ("lower_better", "cost"),
    "attack_fraction_route_affected": ("lower_better", "ratio"),
}

def _multiseed_runs(root):
    """Return only cells authorized by the latest batch_status.csv.

    The directory scan remains a compatibility fallback for older batches, but
    a status file is authoritative and failed/stale cells are excluded.
    """
    root = Path(root)
    status = read_csv_rows(root / "batch_status.csv")
    rows = []
    if status:
        for entry in status:
            if entry.get("status") not in {"completed", "skipped_resume"}: continue
            path = Path(entry.get("output_directory", ""))
            if not path.is_absolute():
                candidate = root / path
                path = candidate if candidate.exists() else Path.cwd() / path
            data = load_run_data(path)
            if data.summary:
                rows.append((parse_int(entry.get("seed")), entry.get("method", path.name), data))
        return rows
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir(): continue
        for method_dir in discover_method_runs(seed_dir):
            data = load_run_data(method_dir)
            if data.summary: rows.append((parse_int(seed_dir.name.split("_")[-1]), method_dir.name, data))
    return rows

def _attack_phase_benign_samples(data):
    malicious = parse_int(data.summary.get("malicious_robot_id"), 0)
    benign = set(_benign_ids(data))
    return [row for row in data.timeseries
            if parse_int(row.get("robot_id")) in benign and row.get("phase") == "ATTACK"]

def _valid_attack_metrics(data):
    rows = _attack_phase_benign_samples(data)
    fake = [parse_float(row.get("influential_fake_claim_count")) for row in rows]
    cost = [parse_float(row.get("attacker_attributable_cost_on_route")) for row in rows]
    fake = [value for value in fake if value is not None]
    cost = [value for value in cost if value is not None]
    affected = [parse_bool(row.get("preferred_route_affected_by_attacker"), None)
                for row in rows]
    affected = [value for value in affected if value is not None]
    return {
        "attack_mean_influential_fake_cells": sum(fake) / len(fake) if fake else None,
        "attack_fraction_samples_influenced": sum(value > 0 for value in fake) / len(fake) if fake else None,
        "attack_mean_attacker_route_cost": sum(cost) / len(cost) if cost else None,
        "attack_fraction_route_affected": sum(affected) / len(affected) if affected else None,
    }

def _recovery_trust_metrics(data):
    benign = _benign_ids(data)
    recovery_steps = [parse_int(row.get("step")) for row in data.timeseries
                      if row.get("phase") == "RECOVERY" and parse_int(row.get("step")) is not None]
    if not recovery_steps: return {"recovery_start_attacker_trust_mean": None, "recovery_trust_gain": None}
    first = min(recovery_steps); starts = []; gains = []
    for rid in benign:
        values = _series(data.timeseries, rid, "attacker_trust")
        start = next((value for step, value in values if step >= first and
                      next((row.get("phase") for row in data.timeseries
                            if parse_int(row.get("robot_id")) == rid and parse_int(row.get("step")) == step), "") == "RECOVERY"), None)
        if start is None: continue
        starts.append(start); gains.append(values[-1][1] - start if values else None)
    gains = [value for value in gains if value is not None]
    return {"recovery_start_attacker_trust_mean": sum(starts) / len(starts) if starts else None,
            "recovery_trust_gain": sum(gains) / len(gains) if gains else None}

def _phase_outcome_metrics(data):
    benign = set(_benign_ids(data)); result = {}
    for phase in ("RECONNAISSANCE", "ATTACK", "RECOVERY"):
        rows = [row for row in data.timeseries if row.get("phase") == phase and parse_int(row.get("robot_id")) in benign]
        if not rows:
            result[f"deliveries_during_{phase.lower().replace('reconnaissance', 'recon')}"] = None
            continue
        steps = sorted({parse_int(row.get("step")) for row in rows if parse_int(row.get("step")) is not None})
        first = min(steps); last = max(steps)
        totals = {}
        for step in (first, last):
            totals[step] = sum(parse_float(row.get("deliveries_completed"), 0) or 0 for row in rows if parse_int(row.get("step")) == step)
        key = {"RECONNAISSANCE": "deliveries_during_recon", "ATTACK": "deliveries_during_attack", "RECOVERY": "deliveries_during_recovery"}[phase]
        result[key] = totals[last] - totals[first]
        if phase == "ATTACK":
            for field, key in (("benign_traffic_wait_steps", "traffic_wait_steps_during_attack"), ("benign_total_replans", "replans_during_attack")):
                values = [(parse_int(row.get("step")), parse_float(row.get(field))) for row in rows]
                values = [(step, value) for step, value in values if step is not None and value is not None]
                result[key] = (max(value for step, value in values if step == last) - min(value for step, value in values if step == first)) if values else None
            fake = [parse_float(row.get("influential_fake_claim_count")) for row in rows]
            fake = [value for value in fake if value is not None]
            result["mean_influential_fake_cells_during_attack"] = sum(fake) / len(fake) if fake else None
    return result

def _seed_metric(data, field):
    value = parse_float(data.summary.get(field))
    if value is not None: return value
    total = parse_float(data.summary.get("benign_total_replans")); productive = parse_float(data.summary.get("benign_productive_replans")); deliveries = parse_float(data.summary.get("benign_total_deliveries_completed")); steps = parse_float(data.summary.get("steps_completed")); waits = parse_float(data.summary.get("benign_traffic_wait_steps"))
    derived = {"productive_replan_ratio": productive / total if total else None, "deliveries_per_1000_steps": deliveries / steps * 1000 if steps else None, "distance_per_delivery": parse_float(data.summary.get("benign_total_distance")) / deliveries if deliveries else None, "replans_per_delivery": total / deliveries if deliveries else None, "traffic_waits_per_delivery": waits / deliveries if deliveries else None}
    if field in derived: return derived[field]
    attack = _valid_attack_metrics(data)
    if field in attack: return attack[field]
    recovery = _recovery_trust_metrics(data)
    if field in recovery: return recovery[field]
    phase = _phase_outcome_metrics(data)
    if field in phase: return phase[field]
    return None


def focal_comparison_method(methods) -> str | None:
    """Choose the method that paired differences are measured against.

    ``source_memory`` remains the default proposed method when it is present.
    Otherwise the last requested method is the comparison focus, so a two-method
    batch such as ``majority_vote,full_trust`` still produces paired statistics.
    """
    ordered = [method for method in methods if method]
    if not ordered:
        return None
    if "source_memory" in ordered:
        return "source_memory"
    unique = list(dict.fromkeys(ordered))
    return unique[-1] if len(unique) >= 2 else None


def _write_multiseed_csvs(root, runs, requested_methods=None):
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
    for method in _ordered_methods(row["method"] for row in raw):
        method_rows = [row for row in raw if row["method"] == method]
        for metric, (direction, unit) in MULTISEED_DIRECTIONS.items():
            values = [row[metric] for row in method_rows if row.get(metric) is not None]
            stats = __import__("map_poisoning.statistics", fromlist=["summarize"]).summarize(values)
            summary_rows.append({"method": method, "metric": metric, "direction": direction, "unit": unit, "n": stats["n"], "n_missing": len(method_rows)-stats["n"], **{key: stats[key] for key in ("mean","sample_std","sem","ci95_low","ci95_high","median","min","max")}})
    _write_rows(aggregate / "multiseed_summary.csv", summary_rows)
    table_fields = (
        "benign_total_deliveries_completed",
        "benign_delivery_cycle_duration_mean_steps",
        "benign_delivery_cycle_duration_median_steps",
        "benign_delivery_cycle_duration_p95_steps",
        "benign_loaded_delivery_duration_mean_steps",
        "benign_loaded_delivery_duration_median_steps",
        "benign_loaded_delivery_duration_p95_steps",
        "benign_total_distance",
        "distance_per_delivery",
        "benign_planning_checks",
        "benign_path_changes",
        "temporary_obstacle_replan_checks",
        "temporary_obstacle_path_changes",
        "robot_on_route_replan_checks",
        "robot_on_route_path_changes",
        "malicious_report_replan_checks",
        "malicious_report_path_changes",
        "fake_obstacle_replan_checks",
        "fake_obstacle_path_changes",
        "false_clearance_replan_checks",
        "false_clearance_path_changes",
        "stale_reassertion_replan_checks",
        "stale_reassertion_path_changes",
        "attack_induced_path_changes",
        "attack_route_penalty_mean",
        "attack_route_penalty_max",
        "attack_route_penalty_total",
        "attack_extra_path_length_mean",
        "attack_extra_path_length_max",
        "attack_extra_path_length_total",
        "steps_route_affected_by_attacker",
        "time_to_distrust_malicious_robot",
        "malicious_reports_operationally_ignored",
        "benign_blocked_moves",
        "benign_no_path_steps",
    )
    from .statistics import summarize
    comparison_table = []
    for method in _ordered_methods(row["method"] for row in raw):
        method_rows = [row for row in raw if row["method"] == method]
        output = {"method": method, "seed_count": len(method_rows)}
        for field in table_fields:
            values = [parse_float(row.get(field)) for row in method_rows]
            values = [value for value in values if value is not None]
            stats = summarize(values)
            output[field] = stats["max"] if field in {"attack_route_penalty_max", "attack_extra_path_length_max"} else stats["mean"]
        comparison_table.append(output)
    _write_rows(aggregate / "method_comparison_table.csv", comparison_table)
    paired = []
    present = list(dict.fromkeys(requested_methods or [])) or _ordered_methods(row["method"] for row in raw)
    present = [method for method in present if any(row["method"] == method for row in raw)] or _ordered_methods(row["method"] for row in raw)
    focal = focal_comparison_method(present)
    source = {row["seed"]: row for row in raw if row["method"] == focal} if focal else {}
    for baseline in [method for method in present if method != focal]:
        base = {row["seed"]: row for row in raw if row["method"] == baseline}
        for metric, (direction, _) in MULTISEED_DIRECTIONS.items():
            seeds = sorted(set(source) & set(base)); differences = [source[seed][metric] - base[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None]
            from .statistics import paired_summary
            stats = paired_summary([source[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None], [base[seed][metric] for seed in seeds if source[seed].get(metric) is not None and base[seed].get(metric) is not None])
            improvement = None if direction == "diagnostic" else stats["mean_difference"] * (1 if direction == "higher_better" else -1) if stats["mean_difference"] is not None else None
            raw_low, raw_high = stats["ci95_difference_low"], stats["ci95_difference_high"]
            oriented = None if direction == "diagnostic" else (1 if direction == "higher_better" else -1)
            paired.append({"focal_method": focal, "baseline_method": baseline, "metric": metric, "direction": direction, "n_pairs": stats["n_pairs"], "mean_difference": stats["mean_difference"], "sample_std_difference": stats["sample_std_difference"], "sem_difference": stats["sem_difference"], "ci95_difference_low": raw_low, "ci95_difference_high": raw_high, "improvement_difference": None if oriented is None or stats["mean_difference"] is None else oriented * stats["mean_difference"], "improvement_ci95_low": None if oriented is None or raw_low is None else (raw_low if oriented == 1 else -raw_high), "improvement_ci95_high": None if oriented is None or raw_high is None else (raw_high if oriented == 1 else -raw_low)})
    _write_rows(aggregate / "paired_method_differences.csv", paired)
    return raw, summary_rows, paired, focal

def _multiseed_point_plot(raw, metrics, title, path):
    methods = _ordered_methods(row["method"] for row in raw); fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5), squeeze=False)
    from .statistics import summarize
    for axis, metric in zip(axes[0], metrics):
        for x, method in enumerate(methods):
            values = [row[metric] for row in raw if row["method"] == method and row.get(metric) is not None]; stats = summarize(values)
            if values:
                jitter = np.linspace(-.12, .12, len(values)); axis.scatter(np.full(len(values), x)+jitter, values, s=18, alpha=.65)
                axis.errorbar(x, stats["mean"], yerr=None if stats["ci95_low"] is None else [[stats["mean"]-stats["ci95_low"]],[stats["ci95_high"]-stats["mean"]]], fmt="o", color="black", capsize=4, zorder=3)
        axis.set_title(metric.replace("_", " ")); axis.set_xticks(range(len(methods)), methods, rotation=30, ha="right"); axis.grid(axis="y", alpha=.25)
    fig.suptitle(title)
    try:
        validation = json.loads((Path(path).parent.parent / "batch_validation.json").read_text(encoding="utf-8"))
        if not validation.get("valid", True): fig.text(.99, .01, f"Incomplete batch: {validation.get('complete_seed_count', 0)}/{len(validation.get('requested_seeds', []))} complete seeds; failed cells excluded", ha="right", fontsize=8, color="firebrick")
    except (OSError, json.JSONDecodeError): pass
    fig.tight_layout(); _save(fig, path)

def _multiseed_timeseries(raw_runs, field, path, title, aggregation="benign_mean"):
    grouped = {}
    for seed, method, data in raw_runs:
        by_step = {}
        for row in data.timeseries:
            rid = parse_int(row.get("robot_id")); value = parse_float(row.get(field))
            if value is None: continue
            if aggregation in {"benign_sum", "benign_mean"} and rid not in _benign_ids(data): continue
            if aggregation == "single" and rid != _benign_ids(data)[0]: continue
            by_step.setdefault(parse_int(row.get("step"), 0), []).append(value)
        for step, values in by_step.items():
            aggregate = sum(values) if aggregation in {"benign_sum", "all_sum"} else sum(values) / len(values)
            grouped.setdefault(method, {}).setdefault(step, []).append(aggregate)
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
    ax.set(title=title,xlabel="Simulation step"); ax.legend(); ax.grid(alpha=.25)
    try:
        validation = json.loads((Path(path).parent.parent / "batch_validation.json").read_text(encoding="utf-8"))
        if not validation.get("valid", True): fig.text(.99, .01, "Incomplete batch: failed cells excluded", ha="right", fontsize=8, color="firebrick")
    except (OSError, json.JSONDecodeError): pass
    _save(fig,path)

def _paired_seed_plot(raw, path):
    methods = [method for method in METHOD_ORDER if method in {row["method"] for row in raw}]
    metric = "deliveries_during_attack"; fig, ax = plt.subplots(figsize=(9, 5))
    for seed in sorted({row["seed"] for row in raw}):
        values = [next((row.get(metric) for row in raw if row["seed"] == seed and row["method"] == method), None) for method in methods]
        if all(value is not None for value in values): ax.plot(range(len(methods)), values, color="0.65", alpha=.35, linewidth=.8)
    from .statistics import summarize
    for index, method in enumerate(methods):
        values = [row[metric] for row in raw if row["method"] == method and row.get(metric) is not None]; stats = summarize(values)
        if values: ax.errorbar(index, stats["mean"], yerr=None if stats["ci95_low"] is None else [[stats["mean"]-stats["ci95_low"]], [stats["ci95_high"]-stats["mean"]]], fmt="o", capsize=4, color=f"C{index}")
    ax.set(title="Paired same-seed outcomes", ylabel="Deliveries during ATTACK", xticks=range(len(methods)), xticklabels=methods); ax.grid(axis="y", alpha=.25); _save(fig, path)

def _experiment_design_plot(path, methods=None):
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off")
    method_label = "   ".join(methods) if methods else "majority_vote   full_trust   trust_fused   source_memory"
    for index, seed in enumerate(("Seed 1", "Seed 2", "...")):
        y = .8 - index * .28; ax.text(.02, y, seed, fontsize=11, weight="bold", va="center")
        ax.text(.18, y, "one scenario manifest", bbox=dict(boxstyle="round", facecolor="#e8f1fb"), va="center")
        ax.annotate("", (.42, y), (.37, y), arrowprops=dict(arrowstyle="->"))
        ax.text(.44, y, method_label, bbox=dict(boxstyle="round", facecolor="#eef7e8"), va="center", fontsize=9)
    ax.text(.5, .08, "Aggregate: per-method mean / sample SD / 95% CI + paired same-seed differences", ha="center", weight="bold")
    ax.set_title("Experiment design — SAME MANIFEST WITHIN EACH SEED"); _save(fig, path)

def _batch_completion_plot(root, path):
    status = read_csv_rows(Path(root) / "batch_status.csv")
    config = {}
    try: config = json.loads((Path(root) / "batch_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): pass
    seeds = [int(value) for value in config.get("seeds", sorted({parse_int(row.get("seed")) for row in status}))]
    methods = list(config.get("methods", _ordered_methods(row.get("method") for row in status)))
    state = {(parse_int(row.get("seed")), row.get("method")): row.get("status") for row in status}
    values = [[{"completed": 1, "skipped_resume": 1, "failed": 0, "pending": -1}.get(state.get((seed, method), "pending"), -1) for method in methods] for seed in seeds]
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), max(5, len(seeds) * .25)))
    ax.imshow(values, cmap=plt.get_cmap("RdYlGn", 3), vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=range(len(methods)), xticklabels=methods, yticks=range(len(seeds)), yticklabels=seeds, xlabel="Method", ylabel="Seed", title="Batch completion audit")
    for row_index, seed in enumerate(seeds):
        for col_index, method in enumerate(methods): ax.text(col_index, row_index, state.get((seed, method), "missing"), ha="center", va="center", fontsize=7)
    fig.tight_layout(); _save(fig, path)

def generate_multiseed_report(root: str | Path, *, formats=("png",)) -> dict:
    root=Path(root); runs=_multiseed_runs(root)
    if not runs: raise ValueError(f"no completed multi-seed runs found under {root}")
    config = {}
    try: config = json.loads((root / "batch_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): pass
    requested_methods = list(config.get("methods") or [])
    raw, summary, paired, focal = _write_multiseed_csvs(root, runs, requested_methods); aggregate=root/"aggregate"; plots=aggregate/"plots"; generated=[]; plot_failures=[]
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
    requested_seeds = [int(value) for value in config.get("seeds", sorted(by_seed))]
    requested_methods = list(config.get("methods", _ordered_methods(row["method"] for row in raw)))
    status = read_csv_rows(root / "batch_status.csv")
    status_map = {(parse_int(row.get("seed")), row.get("method")): row for row in status}
    successful_cells = {(seed, method) for (seed, method), row in status_map.items() if row.get("status") in {"completed", "skipped_resume"}}
    failed_cells = [[seed, method] for (seed, method), row in status_map.items() if row.get("status") == "failed"]
    missing_cells = [[seed, method] for seed in requested_seeds for method in requested_methods if (seed, method) not in successful_cells and [seed, method] not in failed_cells]
    incomplete = sorted({seed for seed, _ in failed_cells + missing_cells})
    included_configs = {row.get("experiment_config_hash") for row in status if row.get("status") in {"completed", "skipped_resume"} and row.get("experiment_config_hash")}
    config_mismatch = len(included_configs) > 1
    map_mismatch = len({value for value in hashes if value}) > 1
    preset_values = {value for value in presets if value}; preset_mismatch = len(preset_values) > 1
    validation = {"valid": bool(status) and not failed_cells and not missing_cells and fairness_ok and not config_mismatch and not map_mismatch and not preset_mismatch,
                  "requested_seeds": requested_seeds, "requested_methods": requested_methods,
                  "requested_cell_count": len(requested_seeds) * len(requested_methods), "completed_cell_count": sum(1 for row in status if row.get("status") in {"completed", "skipped_resume"}), "failed_cell_count": len(failed_cells), "missing_cell_count": len(missing_cells),
                  "complete_seeds": sorted(set(requested_seeds) - set(incomplete)), "incomplete_seeds": incomplete, "failed_cells": failed_cells, "missing_cells": missing_cells,
                  "per_method_run_counts": {method: sum(row["method"] == method for row in raw) for method in requested_methods}, "paired_seed_count_by_comparison": {method: sum(1 for seed in requested_seeds if focal and (seed, focal) in successful_cells and (seed, method) in successful_cells) for method in requested_methods if method != focal},
                  "focal_method": focal,
                  "map_hashes_observed": sorted(value for value in hashes if value), "scenario_presets_observed": sorted(preset_values), "experiment_config_hashes_observed": sorted(included_configs), "git_commits_observed": sorted(set(commits)), "git_dirty_flags_observed": sorted(set(dirty), key=str), "warnings": warnings}
    if config_mismatch: validation["warnings"].append("experiment_config_hash differs across included cells")
    if map_mismatch: validation["warnings"].append("map hashes differ across the batch")
    if preset_mismatch: validation["warnings"].append("scenario presets differ across the batch")
    validation.update({"batch_success_rate": validation["completed_cell_count"] / validation["requested_cell_count"] if validation["requested_cell_count"] else None,
                       "complete_seed_count": len(validation["complete_seeds"]), "incomplete_seed_count": len(validation["incomplete_seeds"]),
                       "failed_seed_count": len({seed for seed, _ in failed_cells}),
                       "failure_reason_counts": {reason: sum(1 for row in status if row.get("status") == "failed" and row.get("error") == reason) for reason in sorted({row.get("error") for row in status if row.get("status") == "failed" and row.get("error")})}})
    (aggregate / "batch_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    jobs=[("01_deliveries_by_method.png",lambda p:_multiseed_point_plot(raw,["benign_total_deliveries_completed","benign_deliveries_after_attack","benign_deliveries_after_distrust"],"Deliveries by method (seed points and 95% CI)",p)),("02_mission_efficiency_by_method.png",lambda p:_multiseed_point_plot(raw,["distance_per_delivery","replans_per_delivery","traffic_waits_per_delivery"],"Mission efficiency by method",p)),("03_attack_influence_by_method.png",lambda p:_multiseed_point_plot(raw,["attack_mean_influential_fake_cells","attack_fraction_samples_influenced","attack_mean_attacker_route_cost","attack_fraction_route_affected"],"Attack influence by method",p)),("04_trust_diagnostics_by_method.png",lambda p:_multiseed_point_plot(raw,["time_to_distrust_malicious_robot","time_to_all_benign_distrust","attacker_min_trust_mean","final_attacker_trust_mean"],"Trust diagnostics",p)),("05_traffic_burden_by_method.png",lambda p:_multiseed_point_plot(raw,["benign_traffic_wait_steps","deadlocks_detected","traffic_replans","robot_overlap_violations"],"Traffic burden by method",p)),("06_fake_influence_over_time.png",lambda p:_multiseed_timeseries(runs,"influential_fake_claim_count",p,"Fake influence over time","benign_mean")),("07_delivery_progress_over_time.png",lambda p:_multiseed_timeseries(runs,"deliveries_completed",p,"Benign delivery progress over time","benign_sum")),("08_replans_over_time.png",lambda p:_multiseed_timeseries(runs,"benign_total_replans",p,"Benign replans over time","benign_sum")),("14_route_cause_attribution.png",lambda p:_multiseed_point_plot(raw,["malicious_report_path_changes","fake_obstacle_path_changes","stale_reassertion_path_changes","temporary_obstacle_path_changes"],"Path changes by attributable cause",p))]
    if "png" in formats:
        jobs.insert(0, ("00_batch_completion.png", lambda p: _batch_completion_plot(root, p)))
        for name,job in jobs:
            try: job(plots/name); generated.append(name)
            except Exception as exc:
                plot_failures.append({"filename": name, "error": str(exc)}); print(f"[report] FAILED {name}: {exc}")
    if "png" in formats:
        fig, ax=plt.subplots(figsize=(10,6)); selected=[row for row in paired if row["metric"] in {"benign_deliveries_after_attack","deliveries_per_1000_steps","replans_per_delivery","traffic_waits_per_delivery","attack_mean_influential_fake_cells","attack_fraction_route_affected"} and row["improvement_difference"] is not None]; labels=[f"{row['baseline_method']} / {row['metric']}" for row in selected]; means=[row["improvement_difference"] for row in selected]; lows=[row["improvement_ci95_low"] for row in selected]; highs=[row["improvement_ci95_high"] for row in selected]; y=np.arange(len(labels));
        if labels: ax.errorbar(means,y,xerr=[[m-l if l is not None else 0 for m,l in zip(means,lows)],[h-m if h is not None else 0 for m,h in zip(means,highs)]],fmt="o")
        focal_label = focal or "focal method"
        ax.axvline(0,color="0.4"); ax.set(yticks=y,yticklabels=labels or ["(no paired methods)"],title=f"{focal_label} paired improvement",xlabel=f"Positive means {focal_label} better"); ax.grid(axis="x",alpha=.25); _save(fig,plots/"09_paired_method_differences.png"); generated.append("09_paired_method_differences.png")
        if focal == "source_memory":
            import shutil
            shutil.copyfile(plots/"09_paired_method_differences.png", plots/"09_source_memory_paired_differences.png")
            generated.append("09_source_memory_paired_differences.png")
        from .statistics import summarize
        methods=_ordered_methods(row["method"] for row in raw); fig, ax=plt.subplots(figsize=(8,6))
        for method in methods:
            xvals=[row["replans_per_delivery"] for row in raw if row["method"]==method and row.get("replans_per_delivery") is not None]; yvals=[row["benign_deliveries_after_attack"] for row in raw if row["method"]==method and row.get("benign_deliveries_after_attack") is not None]
            if xvals and yvals:
                xs=summarize(xvals); ys=summarize(yvals); ax.errorbar(xs["mean"],ys["mean"],xerr=None if xs["ci95_low"] is None else [[xs["mean"]-xs["ci95_low"]],[xs["ci95_high"]-xs["mean"]]],yerr=None if ys["ci95_low"] is None else [[ys["mean"]-ys["ci95_low"]],[ys["ci95_high"]-ys["mean"]]],fmt="o",label=method,capsize=4)
        ax.set(xlabel="Replans per delivery",ylabel="Deliveries after attack",title="Delivery vs replan tradeoff"); _safe_legend(ax); ax.grid(alpha=.25); _save(fig,plots/"10_delivery_vs_replan_tradeoff.png"); generated.append("10_delivery_vs_replan_tradeoff.png")
        try:
            _paired_seed_plot(raw, plots / "11_paired_seed_outcomes.png"); generated.append("11_paired_seed_outcomes.png")
            _multiseed_point_plot(raw, ["deliveries_during_attack", "deliveries_during_recovery", "traffic_wait_steps_during_attack", "replans_during_attack"], "Phase outcomes by method", plots / "12_phase_outcomes_by_method.png"); generated.append("12_phase_outcomes_by_method.png")
            _experiment_design_plot(plots / "13_experiment_design.png", requested_methods or _ordered_methods(row["method"] for row in raw)); generated.append("13_experiment_design.png")
        except Exception as exc:
            plot_failures.append({"filename": "11-13 aggregate plots", "error": str(exc)}); print(f"[report] FAILED 11-13 aggregate plots: {exc}")
    _write_multiseed_report(aggregate, raw, paired, focal)
    manifest={"directory":str(root),"generated":generated,"failed":plot_failures,"warnings":[],"runs":len(runs)}; (aggregate/"aggregate_plot_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest

def _write_multiseed_report(aggregate, raw, paired, focal=None):
    methods=_ordered_methods(row["method"] for row in raw)
    try: validation = json.loads((aggregate / "batch_validation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): validation = {}
    status = "VALID" if validation.get("valid") else "INVALID / INCOMPLETE"
    lines=[f"BATCH STATUS: {status}", f"Completed cells: {validation.get('completed_cell_count', len(raw))} / {validation.get('requested_cell_count', len(raw))}", f"Complete seeds: {len(validation.get('complete_seeds', []))} / {len(validation.get('requested_seeds', []))}"]
    if validation.get("failed_cells"): lines.append("Failed seeds: " + ", ".join(map(str, sorted({cell[0] for cell in validation["failed_cells"]}))))
    lines += ["Statistical summaries below exclude failed cells.", "", "Multi-seed report", "", "Methods: "+", ".join(methods), f"Completed seed/method runs: {len(raw)}", "", "Mission outcomes:"]
    from .statistics import summarize
    for method in methods:
        values=[row["benign_total_deliveries_completed"] for row in raw if row["method"]==method and row.get("benign_total_deliveries_completed") is not None]; s=summarize(values); lines.append(f"  {method}: mean deliveries={s['mean']} SD={s['sample_std']} 95% CI=[{s['ci95_low']}, {s['ci95_high']}]")
    focal_label = focal or validation.get("focal_method") or "source_memory"
    lines += ["", f"Paired {focal_label} comparisons:"]
    for row in paired:
        if row["metric"]=="benign_deliveries_after_attack" and row["improvement_difference"] is not None: lines.append(f"  {focal_label} vs {row['baseline_method']}: mean paired improvement={row['improvement_difference']:.4g}, 95% CI [{row['improvement_ci95_low']}, {row['improvement_ci95_high']}], n={row['n_pairs']} (paired same-seed difference)")
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
