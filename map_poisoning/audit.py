"""Small deterministic audits for manifests and fixed-stream replays."""
from __future__ import annotations

from .models import AttackType, ClaimType


def audit_manifest(manifest) -> dict:
    errors: list[str] = []
    if manifest.malicious_robot_id != 0 or tuple(manifest.benign_robot_ids) != (1, 2):
        errors.append("team must be attacker 0 and benign robots 1, 2")
    seen: set[str] = set()
    expected_labels = {
        report_id: (event, cell)
        for event in manifest.attack_events
        for cell, report_id in zip(event.cells, event.report_ids)
    }
    labels_by_id = {label.report_id: label for label in manifest.report_audit_labels}
    if expected_labels and set(labels_by_id) != set(expected_labels):
        errors.append("attack report audit labels do not match the authored attack stream")
    for report_id, (event, cell) in expected_labels.items():
        label = labels_by_id.get(report_id)
        if label is None:
            continue
        blocked_at_observation = bool(manifest.static_grid[cell[0]][cell[1]]) or any(
            cell in episode.cells
            and episode.appearance_step <= event.observation_step < episode.clearance_step
            for episode in manifest.obstacle_episodes
        )
        expected_state = ClaimType.BLOCKED if blocked_at_observation else ClaimType.FREE
        if (
            not label.is_malicious
            or label.attack_type != event.attack_type
            or label.obstacle_episode_id != event.obstacle_episode_id
            or label.actual_state_at_observation != expected_state
        ):
            errors.append(f"invalid audit label {report_id}")
    rows, cols = manifest.map_shape
    for event in manifest.attack_events:
        if event.sender_id != 0 or tuple(event.recipients) != (1, 2):
            errors.append(f"invalid attacker stream on {event.event_id}")
        if event.observation_step > event.step:
            errors.append(f"unordered timestamps on {event.event_id}")
        for cell, report_id in zip(event.cells, event.report_ids):
            if report_id in seen:
                errors.append(f"duplicate report id {report_id}")
            seen.add(report_id)
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                errors.append(f"out-of-bounds cell {cell}")
            active = any(cell in e.cells and e.appearance_step <= event.step < e.clearance_step for e in manifest.obstacle_episodes)
            static_blocked = bool(manifest.static_grid[cell[0]][cell[1]])
            if event.attack_type == AttackType.FAKE_OBSTACLE and (static_blocked or active or event.claim != ClaimType.BLOCKED):
                errors.append(f"invalid fake obstacle {event.event_id}")
            if event.attack_type == AttackType.FALSE_CLEARANCE and (not active or event.claim != ClaimType.FREE):
                errors.append(f"invalid false clearance {event.event_id}")
            if event.attack_type == AttackType.STALE_REASSERTION and (active or event.claim != ClaimType.BLOCKED):
                errors.append(f"invalid stale reassertion {event.event_id}")
    for report in manifest.honest_attacker_reports:
        if report.report_id in seen:
            errors.append(f"duplicate report id {report.report_id}")
        seen.add(report.report_id)
        if report.observation_step < 0:
            errors.append(f"invalid observation time {report.report_id}")
        if not 0.0 <= report.sensor_confidence <= 1.0:
            errors.append(f"invalid sensor confidence {report.report_id}")
    return {
        "passed": not errors,
        "errors": errors,
        "scenario_id": manifest.scenario_id,
        "protocol_id": manifest.protocol_id,
        "attacker_report_count": len(seen),
        "warnings": list(manifest.authoring_warnings),
    }


def attacker_stream_signature(manifest) -> tuple:
    reports = [
        (
            report.report_id,
            report.sender_id,
            report.target_cell,
            int(report.claim),
            report.observation_step,
            report.sensor_confidence,
            report.scenario_event_id,
        )
        for report in manifest.honest_attacker_reports
    ]
    reports += [
        (report_id, event.sender_id, cell, int(event.claim), event.observation_step, 1.0, event.event_id)
        for event in manifest.attack_events
        for cell, report_id in zip(event.cells, event.report_ids)
    ]
    return tuple(reports)
