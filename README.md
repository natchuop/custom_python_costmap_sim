# NSF REU custom simulator

This folder contains the standalone multi-robot simulator, the defense runner,
the ROS occupancy-map converter, and the AWS RoboMaker small warehouse world.

## Run

From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe .\sim2.py --map-npy ".\converted_maps\maps_005_map_rotated\static_grid.npy"
```

Or use the launcher:

```powershell
.\run_sim.ps1
```

For a quick headless test:

```powershell
.\run_sim.ps1 -NoAnimation -MaxSteps 10 -DeliveriesPerRobot 1
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
