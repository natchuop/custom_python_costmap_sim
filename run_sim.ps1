param(
    [ValidateSet("default_warehouse", "maps_005_map_rotated", "maps_005_map", "maps_002_map")]
    [string]$Map = "default_warehouse",
    [switch]$NoAnimation,
    [int]$MaxSteps = 2400,
    [int]$DeliveriesPerRobot = 100
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run: py -3.14 -m venv .venv; .\.venv\Scripts\python.exe .\install_dependencies.py --install"
}
$out = Join-Path $projectRoot "outputs\runs\source_memory_seed15_$Map"
$args = @("$projectRoot\main.py", "--headless", "--max-steps", $MaxSteps, "--output-directory", $out)
if ($Map -ne "default_warehouse") {
    $mapPath = Join-Path $projectRoot "converted_maps\$Map\static_grid.npy"
    if (-not (Test-Path -LiteralPath $mapPath)) {
        throw "Map not found: $mapPath. Run: .\.venv\Scripts\python.exe .\convert_maps.py --input .\warehouse-world --output .\converted_maps --downsample 8"
    }
    $args += @("--map-npy", $mapPath)
}
if ($NoAnimation) { $args += "--no-animation" }
& $python @args
exit $LASTEXITCODE
