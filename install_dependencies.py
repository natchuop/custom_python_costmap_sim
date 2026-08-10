#!/usr/bin/env python3
"""Install and verify dependencies for custom_map_poisoning_costmap.

This file intentionally uses only the Python standard library so it can run
before the project's third-party packages are installed.

Typical use:
    python install_dependencies.py

Install missing pip packages automatically, then re-check:
    python install_dependencies.py --install

For CI or a computer without a graphical desktop:
    python install_dependencies.py --skip-gui

Include the recommended test dependency:
    python install_dependencies.py --include-dev
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

MIN_PYTHON = (3, 10)


@dataclass(frozen=True)
class Dependency:
    display_name: str
    import_name: str
    pip_name: Optional[str]
    required: bool = True


@dataclass
class CheckResult:
    name: str
    status: str  # PASS, WARN, FAIL
    detail: str
    install_name: Optional[str] = None


RUNTIME_DEPENDENCIES = (
    Dependency("NumPy", "numpy", "numpy"),
    Dependency("Matplotlib", "matplotlib", "matplotlib"),
    Dependency("Pillow", "PIL", "Pillow"),
    Dependency("PyYAML", "yaml", "PyYAML"),
)

DEV_DEPENDENCIES = (
    Dependency("pytest", "pytest", "pytest", required=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and verify the Python and GUI dependencies required by the simulator."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install missing pip packages into the current Python environment, then verify.",
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Also check the recommended pytest development dependency.",
    )
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Import tkinter but do not create a test window. Use for CI/headless systems.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root to inspect. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as JSON instead of the normal readable summary.",
    )
    return parser.parse_args()


def package_version(import_name: str, pip_name: Optional[str]) -> str:
    candidates = [name for name in (pip_name, import_name) if name]
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "version unavailable"


def check_python() -> CheckResult:
    current = sys.version_info[:3]
    detail = (
        f"Python {current[0]}.{current[1]}.{current[2]} at {sys.executable} "
        f"({platform.system()} {platform.machine()})"
    )
    if current[:2] < MIN_PYTHON:
        return CheckResult(
            "Python",
            "FAIL",
            detail
            + f". Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required.",
        )
    return CheckResult("Python", "PASS", detail)


def check_dependency(dependency: Dependency) -> CheckResult:
    try:
        importlib.import_module(dependency.import_name)
    except Exception as exc:  # Import errors can include missing binary libraries.
        status = "FAIL" if dependency.required else "WARN"
        return CheckResult(
            dependency.display_name,
            status,
            f"Could not import {dependency.import_name!r}: {exc}",
            install_name=dependency.pip_name,
        )

    version = package_version(dependency.import_name, dependency.pip_name)
    return CheckResult(
        dependency.display_name,
        "PASS",
        f"Imported {dependency.import_name!r} successfully ({version}).",
    )


def tkinter_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return (
            "Use the official Python installer and enable the Tcl/Tk and IDLE feature "
            "(Modify the existing Python installation if necessary)."
        )
    if system == "Darwin":
        return (
            "The simplest fix is usually installing Python from python.org. For Homebrew "
            "Python, install the matching python-tk formula shown by `brew search python-tk`."
        )
    return "Install the operating system package that provides Python tkinter/Tk support."


def check_tkinter(skip_window: bool) -> CheckResult:
    try:
        import tkinter as tk
    except Exception as exc:
        return CheckResult(
            "Tkinter",
            "FAIL",
            f"Could not import tkinter: {exc}. {tkinter_install_hint()}",
        )

    if skip_window:
        return CheckResult(
            "Tkinter",
            "PASS",
            f"Imported tkinter successfully (Tk {tk.TkVersion}); window test skipped.",
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.update()
        root.destroy()
    except Exception as exc:
        return CheckResult(
            "Tkinter GUI",
            "FAIL",
            f"tkinter imported, but a test window could not be created: {exc}. "
            f"{tkinter_install_hint()} Use --skip-gui only on a deliberately headless system.",
        )

    return CheckResult(
        "Tkinter GUI",
        "PASS",
        f"Imported tkinter and created a hidden test window successfully (Tk {tk.TkVersion}).",
    )


def run_smoke_test(name: str, function: Callable[[], str]) -> CheckResult:
    try:
        detail = function()
    except Exception as exc:
        return CheckResult(name, "FAIL", f"Smoke test failed: {exc}")
    return CheckResult(name, "PASS", detail)


def smoke_test_numpy() -> str:
    import numpy as np

    rng_a = np.random.default_rng(15)
    rng_b = np.random.default_rng(15)
    values_a = rng_a.integers(0, 100, size=8)
    values_b = rng_b.integers(0, 100, size=8)
    if not np.array_equal(values_a, values_b):
        raise RuntimeError("Seeded NumPy random generators were not repeatable.")
    grid = np.zeros((4, 6), dtype=int)
    if grid.shape != (4, 6):
        raise RuntimeError("Unexpected NumPy array shape.")
    return "Array creation and seeded RNG repeatability passed."


def smoke_test_matplotlib() -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "matplotlib_smoke.png"
        fig.savefig(output)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Matplotlib did not create a PNG file.")
    plt.close(fig)
    return "Created and saved a test figure successfully."


def smoke_test_pillow() -> str:
    from PIL import Image

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "pillow_smoke.png"
        image = Image.new("RGB", (8, 8), (255, 255, 255))
        image.save(output)
        with Image.open(output) as loaded:
            loaded.verify()
    return "Created, saved, and verified a test image successfully."


def smoke_test_yaml() -> str:
    import yaml

    parsed = yaml.safe_load("seed: 15\nattacks:\n  - fake_obstacle\n")
    if parsed != {"seed": 15, "attacks": ["fake_obstacle"]}:
        raise RuntimeError(f"Unexpected YAML result: {parsed!r}")
    return "Parsed a small YAML document successfully."


def check_project_files(project_root: Path) -> list[CheckResult]:
    root = project_root.expanduser().resolve()
    expected = ("main.py", "map_poisoning", "sim2.py", "defense_method_runner.py")
    results: list[CheckResult] = []

    if not root.exists():
        return [CheckResult("Project root", "FAIL", f"Directory does not exist: {root}")]

    results.append(CheckResult("Project root", "PASS", str(root)))
    for filename in expected:
        path = root / filename
        status = "PASS" if path.exists() else "WARN"
        detail = f"Found {path}" if path.exists() else f"Not found: {path}"
        results.append(CheckResult(filename, status, detail))

    map_files = list(root.glob("converted_maps/**/*.npy")) + list(root.glob("**/*.map"))
    if map_files:
        results.append(
            CheckResult(
                "Map assets",
                "PASS",
                f"Found {len(map_files)} .npy/.map file(s); example: {map_files[0]}",
            )
        )
    else:
        results.append(
            CheckResult(
                "Map assets",
                "WARN",
                "No .npy or MovingAI .map files were found. The built-in demo map can still run. "
                "Converted warehouse maps require the warehouse-world submodule and convert_maps.py.",
            )
        )

    return results


def install_packages(package_names: list[str]) -> bool:
    unique_names = sorted(set(package_names), key=str.lower)
    if not unique_names:
        return True

    command = [sys.executable, "-m", "pip", "install", *unique_names]
    print("\nInstalling missing pip packages with:")
    print("  " + " ".join(command))
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"Could not start pip: {exc}")
        return False
    return completed.returncode == 0


def collect_results(args: argparse.Namespace) -> list[CheckResult]:
    results = [check_python()]

    dependencies = list(RUNTIME_DEPENDENCIES)
    if args.include_dev:
        dependencies.extend(DEV_DEPENDENCIES)

    dependency_results = [check_dependency(item) for item in dependencies]
    results.extend(dependency_results)
    results.append(check_tkinter(args.skip_gui))

    imported = {result.name: result.status == "PASS" for result in dependency_results}
    if imported.get("NumPy"):
        results.append(run_smoke_test("NumPy smoke test", smoke_test_numpy))
    if imported.get("Matplotlib"):
        results.append(run_smoke_test("Matplotlib smoke test", smoke_test_matplotlib))
    if imported.get("Pillow"):
        results.append(run_smoke_test("Pillow smoke test", smoke_test_pillow))
    if imported.get("PyYAML"):
        results.append(run_smoke_test("PyYAML smoke test", smoke_test_yaml))

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "write_test.txt"
            test_file.write_text("ok", encoding="utf-8")
            if test_file.read_text(encoding="utf-8") != "ok":
                raise RuntimeError("read-back content differed")
        results.append(CheckResult("Filesystem", "PASS", "Temporary file write/read passed."))
    except Exception as exc:
        results.append(CheckResult("Filesystem", "FAIL", f"Temporary write test failed: {exc}"))

    results.extend(check_project_files(args.project_root))
    return results


def missing_installable_packages(results: list[CheckResult]) -> list[str]:
    return [
        result.install_name
        for result in results
        if result.status in {"FAIL", "WARN"} and result.install_name
    ]


def print_human_report(results: list[CheckResult]) -> None:
    print("\ncustom_map_poisoning_costmap dependency check")
    print("=" * 52)
    for result in results:
        print(f"[{result.status:4}] {result.name}: {result.detail}")

    fail_count = sum(result.status == "FAIL" for result in results)
    warn_count = sum(result.status == "WARN" for result in results)
    print("-" * 52)
    if fail_count == 0:
        print(f"Required checks passed. Warnings: {warn_count}.")
    else:
        print(f"Required check failures: {fail_count}. Warnings: {warn_count}.")
        print(f"Install missing packages with: {sys.executable} install_dependencies.py --install")


def main() -> int:
    args = parse_args()
    results = collect_results(args)

    if args.install:
        packages = missing_installable_packages(results)
        if packages:
            install_ok = install_packages(packages)
            if not install_ok:
                return 1
            importlib.invalidate_caches()
            results = collect_results(args)

    if args.json:
        payload = {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "project_root": str(args.project_root.expanduser().resolve()),
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_human_report(results)

    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
