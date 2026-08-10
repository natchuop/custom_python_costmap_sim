# Multi-robot warehouse simulator

This project simulates three warehouse robots completing delivery tasks while sharing obstacle observations. It includes LiDAR sensing, trust updates, peer belief fusion, malicious map-poisoning attacks, route replanning, temporary obstacle movement, metrics, and live playback.

## Installation

Use Python 3.10 or newer. The repository includes one dependency checker and installer:

```powershell
python .\install_dependencies.py --install
```

For development and testing dependencies, include pytest:

```powershell
python .\install_dependencies.py --install --include-dev
```

On a machine without a graphical desktop, skip the Tk window check:

```powershell
python .\install_dependencies.py --install --include-dev --skip-gui
```

The installer checks Python, NumPy, Matplotlib, Pillow, PyYAML, Tkinter, basic
file access, project files, and small package smoke tests.

## Code overview

The project has two cooperating layers:

- `sim2.py` contains the validated simulation engine. It owns the grid world,
  robot motion, LiDAR observations, temporary obstacles, attack behavior,
  belief maps, route planning, trust state, collision coordination, logging,
  and optional playback.
- `map_poisoning/` provides the modular experiment layer. It converts command
  line or GUI settings into a configuration, authors or loads a scenario
  manifest, replays the manifest with different defense methods, and writes
  metrics and reports. Its rollout adapter uses the simulator engine for the
  actual robot run.

The main files are:

| File or folder | Purpose |
| --- | --- |
| `main.py` | Entry point for GUI, single headless, comparison, and multi-seed runs. |
| `map_poisoning/cli.py` and `config.py` | Define command-line options and typed experiment settings. |
| `map_poisoning/scenario.py` | Creates, validates, saves, and loads deterministic manifests. |
| `map_poisoning/application.py`, `simulation.py`, and `rollout.py` | Load maps, replay methods, collect results, and coordinate output folders. |
| `map_poisoning/map_io.py`, `world.py`, `temp_obstacles.py`, and `scenario_presets.py` | Load supported map formats and define reusable warehouse and obstacle setup. |
| `map_poisoning/models.py` | Shared data types for claims, attacks, observations, tasks, and events. |
| `map_poisoning/robot.py` | Modular robot state, sensing, report handling, verification, and replanning. |
| `map_poisoning/sensing.py`, `planning.py`, `belief.py`, `fusion.py`, and `trust.py` | Implement LiDAR conversion, A* planning, local beliefs, peer evidence, and trust updates. |
| `defense_method_runner.py` | Applies the selected defense policy to stored peer claims and produces route influence. |
| `map_poisoning/reporting.py`, `metrics.py`, and `batch.py` | Write CSV/JSON results, plots, comparisons, and multi-seed summaries. |
| `map_poisoning/ui.py` | Provides the configuration form used by the non-headless entry point. |
| `tests/` | Unit and integration tests for sensing, attacks, trust, fusion, manifests, reports, and batch runs. |

## How the simulation works internally

Each run keeps separate versions of the world state:

- The static grid is the warehouse layout and prior known to the robots.
- The truth grid adds temporary obstacles and the actual current state used by
  the simulator and sensors.
- Each robot has its own belief map. A robot plans from that belief, not from
  the full truth grid.
- Peer claims are stored separately from direct observations, so a claim can
  later be verified and its sender's trust can change.

The normal data flow is:

1. `main.py` builds a `SimulationConfig` and selects the map and defense
   method.
2. A scenario manifest fixes the map, robot roles, starts, delivery queues,
   temporary-obstacle episodes, attack events, and random seeds.
3. The rollout restores the manifest and calls `sim2.run_simulation` with the
   selected defense configuration.
4. On each step, temporary obstacles and scheduled attacks update the truth
   state. Robots use ray-cast LiDAR to make direct free/blocked observations.
5. Robots queue observations for communication. Broadcast reports are placed
   in peer inboxes and are processed through the admission policy and the
   selected defense method.
6. The fusion layer converts accepted peer evidence into hard blocks or route
   costs, depending on the defense method. A* then plans around the resulting
   belief and robots replan when their route becomes invalid or affected by a
   new claim.
7. Robots move, coordinate around shared traffic, complete pickup/dropoff
   tasks, and log actions, reports, trust, replans, and deliveries.
8. When a robot later senses a reported cell, the report is verified. Confirmed
   reports reward the sender and contradicted reports lower trust; methods such
   as `source_linked` can then update the influence of older reports and trigger
   another route replan.
9. The completed log is converted into summaries, time series, event files,
   and optional plots.

Robot 0 is the malicious robot in the standard scenario, while Robots 1 and 2
are benign. The attack layer supports fake obstacles, false clearance, and
stale reassertion. Attack events include provenance and audit information so
that fake evidence, peer delivery, trust changes, and route effects can be
checked after the run.

## Running the simulator



### Interactive GUI for Running the Simulation

```powershell
python .\main.py
```

The GUI lets you configure:

- seed
- output parent directory, defaulting to `outputs\simulation_results`; results
  are grouped automatically under `seed_<N>`
- one or more defense methods to run
- combined or local observations
- phase lengths and delivery count
- attack intervals and enabled attack types
- temporary-obstacle movement interval
- trust model, trust threshold, and admission policy
- optional fixed manifest
- animation and playback display

Selecting one defense method displays the full simulation for that method. Selecting multiple methods authors one scenario, replays every selected method on that same scenario headlessly, and saves all results without opening animation windows. Each selected method gets its own result subdirectory.

For a multi-method run, the shared reconnaissance traffic heatmap used by the manifest is also written once at the comparison-folder level as `traffic_heatmap.png` and `traffic_heatmap.npy`.

### Full terminal run command without using the GUI

```powershell
python .\main.py --headless --no-animation `
  --seed 12 `
  --defense-method trust_threshold `
  --output-directory outputs\simulation_results\trust_threshold
```

The default full run is 2,400 steps:


| Phase          | Steps |
| -------------- | ----- |
| Reconnaissance | 450   |
| Attack         | 1,200 |
| Recovery       | 750   |


The default seed is `12`. Other defaults include attack attempts every 30 steps, temporary-obstacle movement every 150 steps, 100 deliveries per robot, combined observations, and `trust_threshold` as the single-run defense method.

Use `--max-steps` only for development smoke runs; it truncates the rollout:

```powershell
python .\main.py --headless --no-animation --max-steps 300
```



### PowerShell helper

`run_sim.ps1` runs a converted occupancy grid through the headless entry point:

```powershell
.\run_sim.ps1 -NoAnimation -MaxSteps 2400 -DeliveriesPerRobot 100
```

It expects a `.venv` and a converted map. The normal `main.py` run uses the checked-in default warehouse map and does not require this helper.

## What is a scenario manifest?

A scenario manifest is a versioned JSON description of one complete simulation scenario. It is the scenario's fixed experimental record: it describes the warehouse map and the events that should occur, while the defense method and other method-specific settings are supplied separately at replay time. The manifest is saved as `scenario_manifest.json` in the run output directory.

The manifest can include:

- the static grid, map hash, map dimensions, and schema version;
- the master seed and named derived seeds used to author the scenario;
- phase boundaries and the malicious and benign robot IDs;
- each robot's starting position and delivery-task queue;
- temporary-obstacle episodes, including when they appear and clear;
- scheduled attack events, attacker positions, honest reports, and audit labels;
- reconnaissance metadata, candidate attack footprints, and authoring warnings.

The simulator authors a manifest automatically when a run starts. You can also author one without running the simulation by using `--manifest-only`, or load an existing file with `--manifest`. Loading a manifest reuses its recorded map, starts, tasks, obstacle history, attack stream, and related scenario data rather than generating a new scenario from the seed. The audit report checks that the manifest is internally consistent before replay.

Manifests are especially useful for fair comparisons. To compare defense methods, author one manifest and replay every method against that same file. Then differences in delivery, replanning, trust, and attack metrics are caused by the defense policy instead of by different robot starts, tasks, obstacles, or attack randomness. The manifest does not lock the defense method; one file can therefore be reused across many methods.

## Simulation lifecycle

1. A static warehouse map is loaded.
2. A seeded manifest fixes robot starts, delivery queues, temporary-obstacle
  episodes, attack events, and audit labels.
3. Robots explore during reconnaissance and produce a traffic heatmap.
4. Robots sense locally, queue observations, and broadcast them on communication
  ticks. Reports are marked transmitted only after they are actually sent.
5. Each defense method combines direct observations and peer reports according
  to its policy and current trust.
6. The malicious robot performs feasible fake-obstacle, false-clearance, or
  stale-reassertion attacks during the attack phase.
7. Robots verify reports when later sensing the same cells. Trust changes can
  activate/deactivate stored evidence and trigger replanning.
8. Attack injection stops and the robots continue through recovery while
  completing delivery tasks.

When animation is enabled, the reconnaissance heatmap opens first, followed by the live playback window. The playback shows the ground-truth map and one map per robot. Robot source colors are purple for robot 0, orange for robot 1, and blue for robot 2. On combined maps, an occupied shared cell uses the source robot’s color; if several trusted robots support it, the highest-trust source color is shown. Yellow marks goals/checkpoints, red marks attack overlays, and light red marks false-clearance overlays.

Playback uses the completed log, so changing speed or pausing does not change
the simulation results.

## Defense methods

All of these methods are implemented and selectable:


| Method             | Behavior                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `trust_threshold`  | Default. Reports remain stored and verifiable, but senders below the current trust threshold have zero operational influence. |
| `full_trust`       | Uses report confidence without trust or age weighting.                                                                        |
| `majority_vote`    | Uses one discrete vote per sender; a positive blocked majority becomes a hard block.                                          |
| `trust_fused`      | Uses sender trust at report-receipt time.                                                                                     |
| `source_linked`    | Uses current sender trust and report age, so old reports change influence when trust changes.                                 |
| `hard_threshold`   | Converts occupancy probability above its threshold into a hard obstacle.                                                      |
| `soft_probability` | Converts peer occupancy evidence into a continuous traversal cost.                                                            |
| `time_decay`       | Decays report influence by age while ignoring sender trust.                                                                   |


Example:

```powershell
python .\main.py --headless --no-animation `
  --defense-method trust_threshold `
  --trust-threshold 0.55
```

The default admission policy is `accept_all`, allowing below-threshold reports
to remain available for later verification and trust learning. Use
`--admission-policy hard_reject` when those reports should be rejected instead.

## Comparing methods on one manifest

Author a scenario without running it:

```powershell
python .\main.py --headless --manifest-only --seed 12 `
  --output-directory outputs\simulation_results\seed12_manifest
```

Replay selected methods on that exact scenario:

```powershell
python .\main.py --headless --compare --seed 12 `
  --comparison-methods trust_threshold,source_linked,soft_probability,full_trust `
  --manifest outputs\simulation_results\seed12_manifest\scenario_manifest.json `
  --no-animation `
  --output-directory outputs\simulation_results\seed12_comparison
```

Using one manifest keeps starts, tasks, temporary-object histories, attack
events, and random choices consistent between methods.

## Output files

A run writes these files to its output directory:


| File                     | Contents                                                                     |
| ------------------------ | ---------------------------------------------------------------------------- |
| `scenario_manifest.json` | Seeded scenario, tasks, temporary episodes, and attack events.               |
| `run_summary.csv`        | Final delivery, no-path, movement, replan, trust, and attack metrics.        |
| `robot_timeseries.csv`   | Periodic per-robot positions, goals, phases, deliveries, trust, and replans. |
| `events.csv`             | Reports, trust updates, robot actions, and fusion events.                    |
| `run_config.json`        | Canonical run configuration shared by the manifest and comparison.           |
| `effective_config.json`  | Method-specific configuration inside each method result folder.              |
| `audit_report.json`      | Manifest consistency and attack-audit results.                               |
| `run_metadata.json`      | Platform, Python, Git, scenario, and manifest metadata.                      |
| `traffic_heatmap.png`    | Shared reconnaissance traffic heatmap for multi-method comparisons.          |
| `traffic_heatmap.npy`    | Numeric version of that shared heatmap for further analysis.                 |


Important summary fields include:

- `benign_total_deliveries_completed`;
- `benign_no_path_steps`;
- `benign_total_replans`;
- `benign_malicious_report_replans`;
- `benign_malicious_route_replans`;
- `attack_actions`;
- `final_attacker_trust_mean`;
- `steps_completed`.

In comparison mode, each method has its own subdirectory under the selected
seed directory. The run-level `run_metadata.json` records the exact UTC time
the outputs were collected.

## Optional map conversion

The default run uses the checked-in warehouse map. To convert additional
ROS-style `.pgm`/`.yaml` maps into simulator `.npy` grids:

```powershell
python .\convert_maps.py `
  --input .\warehouse-world `
  --output .\converted_maps `
  --downsample 8
```

Then provide one map source to `main.py`:

```powershell
python .\main.py --headless --no-animation `
  --map-npy .\converted_maps\maps_005_map_rotated\static_grid.npy
```

Use only one of `--map-npy` and `--map-movingai` per run.

## Verification

```powershell
python -m pytest -q
python -m compileall -q .
git diff --check
```

For a full non-visual verification run:

```powershell
python .\main.py --headless --no-animation --seed 12 `
  --defense-method trust_threshold `
  --output-directory outputs\simulation_results\full_trust_threshold
```

The simulator is deterministic for the same map, manifest, seed, and method
configuration. Use a new output directory when you want to preserve earlier
results.
