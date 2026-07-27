param(
    [ValidateSet("maps_005_map_rotated", "maps_005_map", "maps_002_map")]
    [string]$Map = "maps_005_map_rotated",
    [switch]$NoAnimation,
    [int]$MaxSteps = 2400,
    [int]$DeliveriesPerRobot = 100
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$mapPath = Join-Path $projectRoot "converted_maps\$Map\static_grid.npy"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run: py -3.14 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
if (-not (Test-Path -LiteralPath $mapPath)) {
    throw "Map not found: $mapPath. Run: .\.venv\Scripts\python.exe .\convert_maps.py --input .\warehouse-world --output .\converted_maps --downsample 8"
}

$args = @("$projectRoot\sim2.py", "--map-npy", $mapPath, "--max-steps", $MaxSteps, "--deliveries-per-robot", $DeliveriesPerRobot)
if ($NoAnimation) { $args += "--no-animation" }
& $python @args
exit $LASTEXITCODE
