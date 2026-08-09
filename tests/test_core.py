from dataclasses import replace
import contextlib
import io
from map_poisoning.config import AttackConfig, FusionConfig, PhaseConfig, SimulationConfig, VisualizationConfig
from map_poisoning.fusion import FusionEngine
from map_poisoning.models import ClaimReport, ClaimType, DeliveryTask, VerificationOutcome
from map_poisoning.scenario import author_manifest
from map_poisoning.models import AttackType
from map_poisoning.rollout import run_manifest_rollout
from map_poisoning.sensing import lidar_observations
from map_poisoning.world import demo_grid
import sim2
from sim2 import CellState, TemporaryBlockageManager
from map_poisoning.belief import RobotBeliefMap
from map_poisoning.robot import ModularRobot
from map_poisoning.trust import BayesianTrustModel
import numpy as np

def test_same_seed_same_manifest():
    assert author_manifest(SimulationConfig()).to_dict() == author_manifest(SimulationConfig()).to_dict()


def test_seed_selects_attack_types_per_event():
    seen = set()
    for seed in range(12):
        config = SimulationConfig(
            seed=seed,
            phases=PhaseConfig(20, 80, 20),
            attacks=AttackConfig(interval_min=10, interval_max=10),
        )
        manifest = author_manifest(config)
        assert manifest.attack_events
        assert {event.attack_type for event in manifest.attack_events} <= set(AttackType)
        seen.update(event.attack_type for event in manifest.attack_events)
    assert seen == set(AttackType)


def test_lidar_detects_temporary_obstacle_in_view():
    grid = np.zeros((10, 10), dtype=int)
    grid[[0, -1], :] = 1
    grid[:, [0, -1]] = 1
    grid[5, 4] = int(CellState.TEMPORARILY_BLOCKED)
    observations = lidar_observations(grid, (5, 2))
    assert observations[(5, 4)] == ClaimType.BLOCKED


def test_temporary_objects_use_seeded_movement_modes():
    manager = TemporaryBlockageManager(demo_grid(24, 30), active_count=1, change_period=10, seed=3)
    before_index = next(iter(manager.active_indices))
    before = manager.current_footprints[before_index][0]
    manager.refresh_active_blockages()
    after_index = next(iter(manager.active_indices))
    after = manager.current_footprints[after_index][0]
    assert after_index == before_index
    assert manager.movement_decisions[after_index] in {"shift", "teleport", "unchanged"}
    assert len(before) == len(after)
    assert manager.current_footprints[after_index][1] == manager.pool[after_index][1]


def test_each_attack_type_reaches_benign_replanning():
    for seed in (0, 2, 4):
        config = SimulationConfig(
            seed=seed,
            phases=PhaseConfig(20, 80, 20),
            attacks=AttackConfig(interval_min=10, interval_max=10),
            visualization=VisualizationConfig(False),
            max_steps=100,
            deliveries_per_robot=1,
        )
        manifest = author_manifest(config)
        _, robots, log = run_manifest_rollout(config, manifest, "source_linked")
        assert any(report["is_malicious"] for report in log["reports"])
        first_attack_step = manifest.attack_events[0].step
        assert any(
            record["step"] >= first_attack_step
            for robot in robots if not robot.is_malicious
            for record in robot.replan_events
        )


def test_fake_obstacles_use_enlarged_footprints():
    config = SimulationConfig(
        seed=0,
        phases=PhaseConfig(20, 80, 20),
        attacks=AttackConfig(interval_min=10, interval_max=10),
    )
    manifest = author_manifest(config)
    fake_events = [event for event in manifest.attack_events if event.attack_type == AttackType.FAKE_OBSTACLE]
    assert fake_events
    assert max(len(event.cells) for event in fake_events) >= 15
    assert all(not manifest.static_grid[r][c] for event in fake_events for r, c in event.cells)

def test_source_linked_is_retroactive_and_trust_fused_is_not():
    trust={0:.7}; score=lambda sender: trust[sender]
    report=ClaimReport("r",0,(1,1),ClaimType.BLOCKED,0,0)
    linked=FusionEngine("source_linked",score); fused=FusionEngine("trust_fused",score)
    linked.add(report); fused.add(report); before_linked=linked.evidence((1,1),0); before_fused=fused.evidence((1,1),0)
    trust[0]=.1
    assert linked.evidence((1,1),0) < before_linked
    assert fused.evidence((1,1),0) == before_fused

def test_fusion_effect_delta_can_be_collected():
    trust={0:.7}; score=lambda sender: trust[sender]
    report=ClaimReport("r",0,(1,1),ClaimType.BLOCKED,0,0)
    linked=FusionEngine("source_linked",score); fused=FusionEngine("trust_fused",score)
    linked.add(report); fused.add(report)
    linked_before, fused_before = linked.evidence((1,1),0), fused.evidence((1,1),0)
    trust[0]=.1
    assert linked.evidence((1,1),0) - linked_before < 0
    assert fused.evidence((1,1),0) - fused_before == 0

def test_recipients_keep_independent_belief_and_trust_state():
    grid=np.zeros((8,8),dtype=np.uint8)
    task=(DeliveryTask("t",(1,2),(6,6)),)
    first_trust, second_trust=BayesianTrustModel(), BayesianTrustModel()
    first=ModularRobot(1,(1,1),task,RobotBeliefMap(grid),first_trust,FusionEngine("source_linked",first_trust.score),.55,"auto_soft")
    second=ModularRobot(2,(1,1),task,RobotBeliefMap(grid),second_trust,FusionEngine("source_linked",second_trust.score),.55,"auto_soft")
    report=ClaimReport("r",0,(3,3),ClaimType.BLOCKED,0,0)
    first.receive(report); first.process_inbox(0)
    assert first.fusion.claims[(3,3)]
    assert (3,3) not in second.fusion.claims
    first.trust.update(0, VerificationOutcome.CONTRADICTED_FRESH)
    assert first.trust.score(0) < second.trust.score(0)


def test_benign_shared_blocked_observations_reach_combined_map():
    for method in ("source_linked", "soft_probability"):
        with contextlib.redirect_stdout(io.StringIO()):
            _, robots, log = sim2.run_simulation(
                tasks_per_robot=100,
                max_steps=300,
                random_seed=15,
                experiment_mode="clean",
                defense_method=method,
                map_view="combined",
            )

        for receiver, sender in ((1, 2), (2, 1)):
            accepted_blocked = {
                claim.target_cell
                for claims in robots[receiver].defense_runner.claims_by_cell.values()
                for claim in claims
                if claim.sender_id == sender and claim.claim == sim2.ClaimType.BLOCKED
            }
            displayed_orange = {
                tuple(cell)
                for frame in log["robots"][receiver]["combined_belief"]
                for cell in np.argwhere(frame == sim2.DISPLAY_PEER_BELIEF)
            }
            assert accepted_blocked & displayed_orange
