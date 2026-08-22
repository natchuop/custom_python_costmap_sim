import numpy as np

from map_poisoning.audit import audit_manifest
from map_poisoning.config import AttackConfig, PhaseConfig, SimulationConfig
from map_poisoning.models import AttackType
from map_poisoning.scenario import author_manifest


def _grid():
    grid = np.zeros((17, 17), dtype=np.uint8)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    return grid


def _config(enabled):
    return SimulationConfig(
        seed=15,
        phases=PhaseConfig(200, 500, 50),
        attacks=AttackConfig(enabled=tuple(enabled), interval_min=50, interval_max=50),
        deliveries_per_robot=2,
    )


def _types(manifest):
    return [event.attack_type for event in manifest.attack_events]


def _steps(manifest):
    return [event.step for event in manifest.attack_events]


def test_disabled_attack_is_replaced_at_the_same_schedule_slot():
    grid = _grid()
    full = author_manifest(_config(kind.value for kind in AttackType), grid)
    no_stale = author_manifest(
        _config((AttackType.FAKE_OBSTACLE.value, AttackType.FALSE_CLEARANCE.value)), grid
    )
    assert AttackType.STALE_REASSERTION not in _types(no_stale)
    assert set(_types(no_stale)) <= {AttackType.FAKE_OBSTACLE, AttackType.FALSE_CLEARANCE}
    assert _steps(no_stale) == _steps(full)
    assert len(no_stale.attack_events) == len(full.attack_events)
    for original, replacement in zip(full.attack_events, no_stale.attack_events):
        if original.attack_type == AttackType.STALE_REASSERTION:
            assert replacement.attack_type in {AttackType.FAKE_OBSTACLE, AttackType.FALSE_CLEARANCE}
        else:
            assert replacement.attack_type == original.attack_type


def test_single_enabled_attack_fills_slots_that_would_have_been_other_types():
    grid = _grid()
    full = author_manifest(_config(kind.value for kind in AttackType), grid)
    fake_only = author_manifest(_config((AttackType.FAKE_OBSTACLE.value,)), grid)
    assert _types(fake_only)
    assert set(_types(fake_only)) == {AttackType.FAKE_OBSTACLE}
    assert _steps(fake_only) == _steps(full)
    assert any(kind != AttackType.FAKE_OBSTACLE for kind in _types(full))


def test_each_pairwise_subset_never_emits_the_disabled_type():
    grid = _grid()
    all_kinds = tuple(AttackType)
    for disabled in all_kinds:
        enabled = tuple(kind.value for kind in all_kinds if kind != disabled)
        manifest = author_manifest(_config(enabled), grid)
        assert disabled not in _types(manifest)
        assert _types(manifest)
        assert audit_manifest(manifest)["passed"]


def test_no_enabled_attacks_authors_an_empty_stream():
    manifest = author_manifest(_config(()), _grid())
    assert manifest.attack_events == ()
    assert manifest.obstacle_episodes
