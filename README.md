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
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_002_map\static_grid.npy --scenario-preset warehouse_002 --seed 15 --deliveries-per-robot 2 --defense-method source_linked --output-directory outputs\map002
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_005_map\static_grid.npy --scenario-preset warehouse_005 --seed 15 --deliveries-per-robot 2 --defense-method source_linked --output-directory outputs\map005
python .\main.py --headless --no-animation --map-npy .\converted_maps\maps_005_map_rotated\static_grid.npy --scenario-preset warehouse_005_rotated --seed 15 --deliveries-per-robot 2 --defense-method source_linked --output-directory outputs\map005_rotated
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

The primary comparison methods are `full_trust`, `majority_vote`, `trust_fused`,
and `source_linked`. Optional additional methods are `hard_threshold`,
`soft_probability`, and `time_decay`; they are only run when explicitly selected.

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

Disable report generation with `--no-plots`:

```powershell
python .\main.py --headless --no-animation --no-plots --max-steps 10
```

Regenerate reports from existing CSV output without rerunning a simulation:

```powershell
python -m map_poisoning.reporting outputs\source_linked
python -m map_poisoning.reporting --compare outputs\comparison
```

Multi-seed experiments author one scenario manifest per seed and replay every
requested method against that seed's manifest:

```powershell
python .\main.py --headless --no-animation --seeds 1-3 --compare `
  --map-npy .\converted_maps\maps_005_map\static_grid.npy `
  --scenario-preset warehouse_005 --output-directory outputs\map005_multiseed
python .\main.py --headless --no-animation --seeds 1-30 --methods source_linked `
  --map-npy .\converted_maps\maps_005_map\static_grid.npy `
  --scenario-preset warehouse_005 --output-directory outputs\map005_source_linked_multiseed
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
`batch_validation.json`, and the ten aggregate plots.

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
