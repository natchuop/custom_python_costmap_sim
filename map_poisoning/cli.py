"""CLI parser shared with the GUI launcher."""
from __future__ import annotations
import argparse
from .config import ALL_METHODS, MAP_VIEWS, AttackConfig, FusionConfig, LoggingConfig, PhaseConfig, SimulationConfig, TrustConfig, VisualizationConfig

def parser():
    p=argparse.ArgumentParser(description="Modular multi-robot map-poisoning simulator")
    p.add_argument("--headless",action="store_true",help="Run without importing Tkinter")
    p.add_argument("--compare",action="store_true",help="Replay one manifest across --comparison-methods")
    p.add_argument("--manifest-only",action="store_true",help="Author and save a manifest without replaying it")
    p.add_argument("--manifest",dest="manifest_path")
    p.add_argument("--map-npy"); p.add_argument("--map-movingai")
    p.add_argument("--output-directory",default="outputs/simulation_results")
    p.add_argument("--seed",type=int,default=12); p.add_argument("--defense-method",choices=ALL_METHODS,default="trust_threshold")
    p.add_argument("--comparison-methods",default="trust_fused",help="comma-separated methods used with --compare")
    p.add_argument("--trust-model",choices=("bayesian","scalar"),default="scalar"); p.add_argument("--trust-threshold",type=float,default=0.55)
    p.add_argument("--attacks",default="fake_obstacle,false_clearance,stale_reassertion",help="comma separated, or 'none'")
    p.add_argument("--recon-steps",type=int,default=450); p.add_argument("--attack-steps",type=int,default=1200); p.add_argument("--recovery-steps",type=int,default=750); p.add_argument("--max-steps",type=int)
    p.add_argument("--deliveries-per-robot",type=int,default=100)
    p.add_argument("--attack-interval-min",type=int,default=30); p.add_argument("--attack-interval-max",type=int,default=30)
    p.add_argument("--map-view", choices=MAP_VIEWS, default="combined", help="belief visualization: combined peer/local or local observations")
    p.add_argument("--temp-obstacle-interval", type=int, default=150, help="steps between temporary-obstacle movements")
    p.add_argument("--no-animation",action="store_true"); return p

def config_from_args(args):
    enabled=() if args.attacks == "none" else tuple(x for x in args.attacks.split(",") if x)
    comparison_methods = tuple(item.strip() for item in args.comparison_methods.split(",") if item.strip())
    return SimulationConfig(seed=args.seed,phases=PhaseConfig(args.recon_steps,args.attack_steps,args.recovery_steps),attacks=AttackConfig(enabled=enabled,interval_min=args.attack_interval_min,interval_max=args.attack_interval_max),trust=TrustConfig(model=args.trust_model, threshold=args.trust_threshold),fusion=FusionConfig(method=args.defense_method),logging=LoggingConfig(args.output_directory),visualization=VisualizationConfig(not args.no_animation, args.map_view),comparison_methods=comparison_methods,manifest_path=args.manifest_path,map_npy=args.map_npy,map_movingai=args.map_movingai,max_steps=args.max_steps,deliveries_per_robot=args.deliveries_per_robot,temporary_blockage_change_period_steps=args.temp_obstacle_interval)
