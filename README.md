# NSF REU custom simulator

This folder contains the modular multi-robot simulator, its fixed-manifest replay and
defense methods, the ROS occupancy-map converter, and the AWS RoboMaker warehouse world.

## Run

From PowerShell in this folder:

```powershell
python .\install_dependencies.py --install
python .\main.py
```

Or use the launcher:

```powershell
.\run_sim.ps1
```

For a quick headless test:

```powershell
.\run_sim.ps1 -NoAnimation -MaxSteps 10 -DeliveriesPerRobot 1
```

Headless single run, manifest authoring, and fixed-manifest comparison:

```powershell
python .\main.py --headless --no-animation
python .\main.py --headless --manifest-only --output-directory outputs\scenario
python .\main.py --headless --compare --manifest outputs\scenario\scenario_manifest.json --output-directory outputs\comparison
```

Fixed experimental scenarios are available for the converted warehouse maps. The
preset validates the map shape/hash and uses the same starts and delivery points
for every seed and defense method:

```powershell
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_002_map\static_grid.npy --scenario-preset warehouse_002 --seed 15 --deliveries-per-robot 2 --defense-method source_memory --output-directory outputs\map002
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_005_map\static_grid.npy --scenario-preset warehouse_005 --seed 15 --deliveries-per-robot 2 --defense-method source_memory --output-directory outputs\map005
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_005_map_rotated\static_grid.npy --scenario-preset warehouse_005_rotated --seed 15 --deliveries-per-robot 2 --defense-method source_memory --output-directory outputs\map005_rotated
```

The fixed coordinates and converted-map SHA-256 hashes are defined in
`map_poisoning/scenario_presets.py`. The rotated preset is explicit; the default
warehouse authoring path remains unchanged when no custom map is supplied.

Run the dependency check before installation or on a new machine:

```powershell
python .\install_dependencies.py --install --include-dev
```

The public entry point is the native modular package (`main.py` and
`map_poisoning/`). It owns manifest authoring, sensing, peer delivery, fusion,
planning, and metric collection.

The primary comparison methods are `latest_report`, `majority_vote`, `full_trust`,
`trust_fused`, and `source_memory`. `latest_report` is the categorical auto-accept
baseline: the newest active peer report determines FREE or BLOCKED regardless of
trust, with an exact-timestamp conflict treated as unknown. It does not compute an
occupancy probability. Current local LiDAR remains authoritative for every method.
Optional additional methods are `hard_threshold`, `soft_probability`, `time_decay`,
and `trust_threshold`; they are only run when explicitly selected.

Current primary defaults are 300 reconnaissance steps, 1700 attack steps, and
500 recovery steps (2500 total), with attacks scheduled every 35--40 steps.
Manifest authoring first runs one deterministic attack-free 2500-step reference.
Its benign traffic produces the one shared heatmap, while each proposed attack
uses the position, visibility, and route from the matching reference step. Fake
footprints must have a positive finite clean-reference detour and first enter the
intended victim's LiDAR 15--40 future steps later. Strategic pickup/dropoff points
and a fixed mix of long, medium, short, and corridor routes are seed-dependent.
The authored manifest is then replayed unchanged by every defense method. Robots use
a 360-degree Euclidean line-of-sight
LiDAR with a five-cell range; observation confidence falls from 1.0 near the robot
to 0.60 at range five. Direct and peer occupancy memories use a shared 300-step
linear lifetime, with current LiDAR observations authoritative. Bayesian trust is
the default (`alpha=9`, `beta=1`, evidence cap 12, confirmation multiplier 0.25,
contradiction multiplier 6.0, distrust threshold 0.50). The reduced positive
multiplier slows recovery after detected deception; scalar trust likewise uses a
0.005 positive reward.
`source_memory` applies immediate trust loss to historical reports but rehabilitates
them gradually with a default trust-memory recovery rate of 0.05. Reports are still
received for audit under `accept_all`, but Source Memory gives a distrusted source zero
operational map influence; Trust Fused similarly ignores new reports received while
the sender is below the threshold. Majority Vote and Full Trust remain trust-agnostic
baselines by design.

Temporary physical obstacles use collision-safe onset: if any robot already
occupies a scheduled footprint, that obstacle yields until the complete footprint
is empty while retaining its authored clearance time. Outputs record deferred
activation steps and verify that no false-clearance attack was injected before its
referenced physical obstacle actually activated.

Navigation replans immediately for real route-invalidating events and also performs
a common 25-step route-optimization check for all five primary methods so gradual
age/trust changes can reveal a better path. `planning_checks` and actual
`path_changes` are logged separately. Run summaries also separate route changes
caused by temporary physical obstacles, other robots, malicious reports, and each
attack type. Attacker attribution is applied only to offline metrics after the
operational decision; it cannot influence navigation.

To compare the primary methods on one fixed seed-10 manifest:

```powershell
python .\main.py --headless --manifest-only --seed 10 --no-animation --output-directory outputs\seed10_manifest
python .\main.py --headless --compare --manifest outputs\seed10_manifest\scenario_manifest.json --seed 10 --no-animation --output-directory outputs\seed10_compare
```

The old path under `C:\Users\ashut\...` is not needed; all paths here are
relative to this project.

## Results and plots

Unless you pass `--output-directory`, results are written to a named folder:

- single run: `outputs\runs\<method>_seed<N>_<map>\`
- same-seed method comparison: `outputs\comparisons\seed<N>_<map>\`
- multi-seed batch: `outputs\multiseed\<method>_seeds<spec>_<map>\`

The GUI fills this path automatically when you change method, seed, or map.
PNG diagrams are in `plots\` for a single run, `comparison_plots\` plus each
method's `plots\` for a comparison, and `aggregate\plots\` for multi-seed.

Completed runs also write `run_summary.csv`, `events.csv`, `robot_timeseries.csv`,
`report_summary.txt`, and `plot_manifest.json`.

Delivery reporting separates the loaded pickup-to-dropoff leg from the complete
task cycle and reports mean, median, and p95 durations. Route-impact reporting
includes event-level counterfactual penalty, extra path length, induced path
changes, and steps during which attacker evidence affects a route.

Do not use delivery count alone to judge defense separation. It is a coarse outcome
over roughly 50--60 completed deliveries and is also affected by shared physical
obstacles and traffic. Pair it with p95 cycle time, attacker route penalty,
route-affected steps, malicious-report path changes, ignored-report counts, no-path
steps, and traffic counters. P95 is the duration at or below which 95% of completed
cycles finished.

Disable report generation with `--no-plots`:

```powershell
python .\main.py --headless --no-animation --no-plots --max-steps 10
```

Regenerate reports from existing CSV output without rerunning a simulation:

```powershell
python -m map_poisoning.reporting outputs\source_memory
python -m map_poisoning.reporting --compare outputs\comparison
```

Multi-seed experiments author one scenario manifest per seed and replay every
requested method against that seed's manifest:

```powershell
python .\main.py --headless --no-animation --seeds 1-3 --compare `
  --map-npy .\converted_maps\maps_005_map\static_grid.npy `
  --scenario-preset warehouse_005 --output-directory outputs\map005_multiseed
python .\main.py --headless --no-animation --seeds 1-30 --methods source_memory `
  --map-npy .\converted_maps\maps_005_map\static_grid.npy `
  --scenario-preset warehouse_005 --output-directory outputs\map005_source_memory_multiseed
```

Use `--resume` to skip only completed cells with matching seed, method,
scenario-manifest, and experiment-configuration hashes. Add `--per-run-plots`
to generate individual reports; otherwise multi-seed mode writes CSVs and only
the aggregate plots. `--no-plots` still writes all aggregate CSVs.

Regenerate an aggregate report without rerunning simulations:

```powershell
python -m map_poisoning.reporting --multiseed outputs\map005_multiseed
```

Aggregate results are stored below `aggregate\`, including
`multiseed_runs.csv`, `multiseed_summary.csv`, `paired_method_differences.csv`,
`batch_validation.json`, the wide `method_comparison_table.csv`, and aggregate plots.

## Rebuild converted maps

```powershell
.\.venv\Scripts\python.exe .\convert_maps.py `
  --input .\warehouse-world `
  --output .\converted_maps `
  --downsample 8
```

The AWS repository is checked out at its `ros1` branch because the default
branch is an archive notice. The native Python simulator works on Windows;
ROS/Gazebo itself is not required for this simulator.
