"""CSV adapter around the unchanged legacy simulator during behavior migration."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sim2

from .config import SimulationConfig
from .metrics import CsvMetrics
from .models import AttackEvent, AttackType, ClaimType
from .rng import named_rng
from .scenario import SCHEMA_VERSION, ScenarioManifest


def author_legacy_manifest(config: SimulationConfig, grid=None) -> ScenarioManifest:
    """Freeze legacy heatmap-selected fake footprints after one clean rollout."""
    old_phase=(sim2.MIN_RECON_STEPS,sim2.MAX_RECON_STEPS)
    sim2.MIN_RECON_STEPS=config.phases.recon_steps
    sim2.MAX_RECON_STEPS=config.phases.recon_steps
    try:
        world,robots,log=sim2.run_simulation(grid=grid,prior_grid=grid,tasks_per_robot=config.deliveries_per_robot,max_steps=config.phases.recon_steps + 1,random_seed=config.seed,experiment_mode="clean")
    finally:
        sim2.MIN_RECON_STEPS,sim2.MAX_RECON_STEPS=old_phase
    candidates=sim2.recon_heatmap_attack_candidates(world,log["goals"],robots,log["traffic_heatmap"][-1])
    if not candidates:
        old_overlap=sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP
        sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP=False
        try:
            candidates=sim2.recon_heatmap_attack_candidates(world,log["goals"],robots,log["traffic_heatmap"][-1])
        finally:
            sim2.ATTACK_REQUIRE_CURRENT_ROUTE_OVERLAP=old_overlap
    if not candidates: raise RuntimeError("clean legacy rollout produced no manifest attack candidates")
    attacker=log["malicious_robot_id"]; recipients=tuple(r.robot_id for r in robots if not r.is_malicious)
    rng=named_rng(config.seed,"legacy_manifest_scheduler"); events=[]; metadata=[]; warnings=[]; uses={}; selected_centers=[]; step=config.phases.recon_steps+rng.randint(config.attacks.interval_min,config.attacks.interval_max); index=0
    while step < config.phases.recon_steps+config.phases.attack_steps:
        pool=candidates[:config.attacks.candidate_top_k]
        eligible=[candidate for candidate in pool if uses.get(tuple(candidate["center_cell"]),0) < config.attacks.max_uses_per_footprint and all(tuple(candidate["center_cell"]) == old or abs(candidate["center_cell"][0]-old[0])+abs(candidate["center_cell"][1]-old[1]) >= config.attacks.min_center_spacing for old in selected_centers)]
        if not eligible:
            warnings.append("concentrated_attack_manifest: diversity limits exhausted")
            break
        candidate=eligible[rng.randrange(len(eligible))]
        cells=tuple(tuple(cell) for cell in candidate["report_cells"]); ids=tuple(f"report-{index:04}-{cell_index:02}" for cell_index in range(len(cells)))
        events.append(AttackEvent(f"attack-{index:04}",step,AttackType.FAKE_OBSTACLE,cells,ClaimType.BLOCKED,step,attacker,recipients,ids))
        center=tuple(candidate["center_cell"]); metadata.append({"candidate_id":f"legacy-{index:04}","center":center,"footprint_cells":cells,"route_overlap":candidate["path_overlap"],"traffic_score":candidate["traffic_score"],"bottleneck_score":candidate["bottleneck_score"],"estimated_detour_score":candidate["path_proximity_score"],"rank":candidates.index(candidate)+1,"selection_weight":1/len(eligible),"prior_use_count":uses.get(center,0)})
        uses[center]=uses.get(center,0)+1; selected_centers.append(center)
        index+=1; step+=rng.randint(config.attacks.interval_min,config.attacks.interval_max)
    if len(set(selected_centers)) < config.attacks.min_unique_footprints: warnings.append("concentrated_attack_manifest: minimum unique footprint count not met")
    prior=robots[0].belief_map.initial_prior
    import hashlib
    return ScenarioManifest(SCHEMA_VERSION,config.seed,{"legacy_manifest_scheduler":config.seed},hashlib.sha256(prior.tobytes()).hexdigest(),tuple(prior.shape),tuple(tuple(int(v) for v in row) for row in prior),{"reconnaissance_end":config.phases.recon_steps,"attack_end":config.phases.recon_steps+config.phases.attack_steps,"total":config.phases.total_steps},attacker,recipients,(),tuple(events),scenario_id=f"legacy-{config.seed}",protocol_id="original_legacy_cli",candidate_metadata=tuple(metadata),authoring_warnings=tuple(dict.fromkeys(warnings)))


def replay_legacy(config: SimulationConfig, method: str, output_directory: Path, manifest=None):
    """Run the proven legacy loop and export its behavior to the modular CSV schema."""
    max_steps=config.max_steps or config.phases.total_steps
    # Lock legacy's otherwise adaptive phase controller to the configured
    # experiment boundaries.  Restore globals immediately after the run so
    # importing sim2 remains backwards compatible.
    old_phase=(sim2.MIN_RECON_STEPS,sim2.MAX_RECON_STEPS,sim2.ATTACK_BURST_DURATION_STEPS)
    old_trust=(sim2.TRUST_MODEL_NAME,sim2.TRUST_INITIAL_VALUE,sim2.TRUST_ACCEPT_THRESHOLD,sim2.TRUST_REWARD,sim2.TRUST_PENALTY,sim2.TRUST_BAYES_PRIOR_ALPHA,sim2.TRUST_BAYES_PRIOR_BETA)
    sim2.MIN_RECON_STEPS=config.phases.recon_steps
    sim2.MAX_RECON_STEPS=config.phases.recon_steps
    sim2.ATTACK_BURST_DURATION_STEPS=config.phases.attack_steps
    sim2.TRUST_MODEL_NAME=config.trust.model
    sim2.TRUST_INITIAL_VALUE=config.trust.prior_alpha/(config.trust.prior_alpha+config.trust.prior_beta)
    sim2.TRUST_ACCEPT_THRESHOLD=config.trust.threshold
    sim2.TRUST_BAYES_PRIOR_ALPHA=config.trust.prior_alpha
    sim2.TRUST_BAYES_PRIOR_BETA=config.trust.prior_beta
    try:
        static_grid=np.asarray(manifest.static_grid,dtype=np.uint8) if manifest else None
        defense_config={
            "decay_rate": config.fusion.decay_rate,
            "cost_scale": config.fusion.cost_scale,
            "cost_exponent": config.fusion.cost_exponent,
            "blocked_probability_threshold": config.fusion.blocked_probability_threshold,
            "max_claim_age": config.fusion.max_claim_age,
            "congested_impact": config.fusion.congested_impact,
            "duplicate_window_steps": config.fusion.duplicate_window_steps,
        }
        world, robots, log=sim2.run_simulation(grid=static_grid,prior_grid=static_grid,defense_method=method,defense_config=defense_config,tasks_per_robot=config.deliveries_per_robot,max_steps=max_steps,random_seed=config.seed,experiment_mode="attack",attack_events=(manifest.attack_events if manifest else None))
    finally:
        sim2.MIN_RECON_STEPS,sim2.MAX_RECON_STEPS,sim2.ATTACK_BURST_DURATION_STEPS=old_phase
        sim2.TRUST_MODEL_NAME,sim2.TRUST_INITIAL_VALUE,sim2.TRUST_ACCEPT_THRESHOLD,sim2.TRUST_REWARD,sim2.TRUST_PENALTY,sim2.TRUST_BAYES_PRIOR_ALPHA,sim2.TRUST_BAYES_PRIOR_BETA=old_trust
    # The legacy renderer is the only complete visualizer at present.  It
    # provides the recon heatmap followed by the ground-truth/belief-map
    # animation, using exactly the rollout just written to the CSV files.
    if config.visualization.animation:
        sim2.show_recon_heatmap(world,log)
        sim2.animate(world,robots,log)
    calculated=sim2.compute_experiment_metrics(robots,log); collector=CsvMetrics(); malicious=log["malicious_robot_id"]
    for report in log["reports"]:
        collector.event(report["step"],"report_sent",method=method,**{key:value for key,value in report.items() if key != "step"})
    for event in log.get("trust_events",[]):
        collector.event(event["step"],"trust_update",method=method,**{key:value for key,value in event.items() if key != "step"})
    for robot in robots:
        rid=robot.robot_id; rlog=log["robots"][rid]
        for step,event in enumerate(rlog["events"]): collector.event(step,"robot_action",method=method,robot_id=rid,action=event)
        for step in range(0,len(rlog["position"]),config.logging.timeseries_period_steps):
            collector.sample(step=step,phase=log["phase"][step],method=method,robot_id=rid,position=rlog["position"][step],goal=rlog["current_goal"][step],deliveries_completed=rlog["completed_tasks"][step],benign_no_path_steps=sum(x=="no_path" for x in rlog["events"][:step+1]),benign_movement_steps=sum(x in ("moved_cell","moved_continuous") for x in rlog["events"][:step+1]),benign_total_distance=sim2.compute_path_distance(rlog["position"][:step+1]),benign_total_replans=rlog["replan_count"][step],attacker_trust=rlog["trust"][step].get(malicious),malicious_claim_cells_on_route=rlog["malicious_claim_cells_on_route"][step])
    benign=[robot for robot in robots if not robot.is_malicious]
    summary={"method":method,"engine":"legacy_compatibility","seed":config.seed,"steps_completed":len(log["truth_grid"]),"attack_actions":sum(bool(report.get("is_malicious")) for report in log["reports"]),"benign_total_deliveries_completed":calculated["benign_total_completed_deliveries"],"benign_success_rate":calculated["benign_delivery_success_rate"],"benign_deliveries_after_attack":calculated["benign_deliveries_after_attack"],"benign_deliveries_after_distrust":calculated["benign_deliveries_after_distrust"],"benign_no_path_steps":sum(calculated["no_path_count_per_robot"][r.robot_id] for r in benign),"benign_movement_steps":calculated["benign_total_movement_steps"],"benign_total_distance":calculated["benign_total_grid_distance"],"benign_total_replans":calculated["benign_total_replans"],"benign_productive_replans":calculated["benign_next_five_changed_replans"],"benign_blocked_moves":sum(calculated["blocked_moves_per_robot"][r.robot_id] for r in benign),"time_to_distrust_malicious_robot":calculated["time_to_distrust_malicious_robot"],"malicious_verified_false_reports":calculated["malicious_verified_false_reports"]}
    output_directory.mkdir(parents=True,exist_ok=True); collector.write(output_directory,summary)
    effective=config.to_dict() | {"effective_engine":"legacy_compatibility","effective_method":method,"legacy_defense_config":defense_config,"legacy_trust_model":type(next(robot for robot in robots if not robot.is_malicious).trust_model).__name__}
    CsvMetrics.config(output_directory/"effective_config.json",effective)
    return SimpleNamespace(output_directory=output_directory, method=method, summary=summary)
