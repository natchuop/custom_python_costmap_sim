"""CLI parser shared with the GUI launcher."""
from __future__ import annotations
import argparse
from dataclasses import replace
from .config import ALL_METHODS, AttackConfig, FusionConfig, LoggingConfig, PhaseConfig, SimulationConfig, TrustConfig, VisualizationConfig

def parser():
    p=argparse.ArgumentParser(description="Modular multi-robot map-poisoning simulator")
    p.add_argument("--headless",action="store_true",help="Run without importing Tkinter")
    p.add_argument("--compare",action="store_true",help="Replay one manifest across primary defense methods")
    p.add_argument("--manifest-only",action="store_true",help="Author and save a manifest without replaying it")
    p.add_argument("--manifest",dest="manifest_path")
    p.add_argument("--map-npy"); p.add_argument("--map-movingai")
    p.add_argument("--engine",choices=("legacy","modular"),default="legacy",help="legacy preserves the proven simulator while modular extraction continues")
    p.add_argument("--output-directory",default="outputs")
    p.add_argument("--seed",type=int,default=15); p.add_argument("--defense-method",choices=ALL_METHODS,default="source_linked")
    p.add_argument("--trust-model",choices=("bayesian","scalar"),default="bayesian"); p.add_argument("--admission-policy",choices=("auto_soft","accept_all","hard_reject"),default="auto_soft")
    p.add_argument("--attacks",default="fake_obstacle,false_clearance,stale_reassertion",help="comma separated, or 'none'")
    p.add_argument("--recon-steps",type=int,default=450); p.add_argument("--attack-steps",type=int,default=1200); p.add_argument("--recovery-steps",type=int,default=750); p.add_argument("--max-steps",type=int)
    p.add_argument("--deliveries-per-robot",type=int,default=100)
    p.add_argument("--no-animation",action="store_true"); return p

def config_from_args(args):
    enabled=() if args.attacks == "none" else tuple(x for x in args.attacks.split(",") if x)
    return SimulationConfig(seed=args.seed,phases=PhaseConfig(args.recon_steps,args.attack_steps,args.recovery_steps),attacks=AttackConfig(enabled=enabled),trust=TrustConfig(model=args.trust_model),fusion=FusionConfig(method=args.defense_method,admission_policy=args.admission_policy),logging=LoggingConfig(args.output_directory),visualization=VisualizationConfig(not args.no_animation),manifest_path=args.manifest_path,map_npy=args.map_npy,map_movingai=args.map_movingai,max_steps=args.max_steps,deliveries_per_robot=args.deliveries_per_robot,engine=args.engine)
