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

The public entry point is the modular package (`main.py` and `map_poisoning/`).  The
current replay deliberately keeps `sim2.py` and `defense_method_runner.py` as internal
behavioral components: they provide the validated continuous-motion, warehouse-layout,
LiDAR, and fusion implementations used by the modular manifest adapter.  They are not
separate selectable engines.

To compare all five supported methods on one fixed seed-10 manifest:

```powershell
python .\main.py --headless --manifest-only --seed 10 --no-animation --output-directory outputs\seed10_manifest
python .\main.py --headless --compare --manifest outputs\seed10_manifest\scenario_manifest.json --seed 10 --no-animation --output-directory outputs\seed10_compare
```

The old path under `C:\Users\ashut\...` is not needed; all paths here are
relative to this project.

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
