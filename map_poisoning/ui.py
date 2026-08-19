"""Tkinter launcher for interactive simulator runs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .application import run
from .cli import config_from_args, result_location_message, suggested_output_directory
from .config import ALL_METHODS
from .map_io import load_npy
from .models import AttackType
from .scenario_presets import PRESETS, preset_for_hash, preset_for_id, validate_fixed_preset


MAP_OPTIONS = {
    "Default warehouse": (None, None),
    "Map 002 (converted)": ("converted_maps/maps_002_map/static_grid.npy", "warehouse_002"),
    "Map 005 (converted)": ("converted_maps/maps_005_map/static_grid.npy", "warehouse_005"),
    "Map 005 rotated (converted)": (
        "converted_maps/maps_005_map_rotated/static_grid.npy", "warehouse_005_rotated"
    ),
}


def _preset_for_map_path(map_path: str | None) -> str | None:
    if not map_path or not Path(map_path).exists():
        return None
    grid = load_npy(map_path)
    preset = preset_for_hash(hashlib.sha256(grid.tobytes()).hexdigest())
    return preset.preset_id if preset else None


def _map_option_for_args(args) -> str:
    if args.map_npy:
        requested = Path(args.map_npy).resolve()
        for label, (path, _) in MAP_OPTIONS.items():
            if path and Path(path).resolve() == requested:
                return label
        return "Custom NPY map"
    return "Default warehouse"


def validate_gui_map_preset(map_path: str | None, preset_id: str | None) -> None:
    """Validate the GUI-selected map/preset before starting a run."""
    if not preset_id:
        known = _preset_for_map_path(map_path)
        if known:
            raise ValueError(f"selected map matches scenario preset {known}; select that preset")
        return
    if not map_path:
        raise ValueError("a fixed scenario preset requires a converted NPY map")
    path = Path(map_path)
    if not path.exists():
        raise ValueError(f"selected map does not exist: {map_path}")
    validate_fixed_preset(load_npy(path), preset_for_id(preset_id))


def launch(args) -> None:
    """Show the interactive run configuration window in one compact view."""
    root = tk.Tk()
    root.title("Modular Map-Poisoning Simulator")
    root.geometry("760x780")
    root.minsize(680, 710)
    form = ttk.Frame(root, padding=14)
    form.pack(fill="both", expand=True)

    initial_preset = args.scenario_preset or _preset_for_map_path(args.map_npy) or ""
    values = {
        "map": tk.StringVar(value=_map_option_for_args(args)),
        "map_path": tk.StringVar(value=args.map_npy or ""),
        "scenario_preset": tk.StringVar(value=initial_preset),
        "seed": tk.StringVar(value=str(args.seed)),
        "method": tk.StringVar(value=args.defense_method),
        "trust_model": tk.StringVar(value=args.trust_model),
        "admission_policy": tk.StringVar(value=args.admission_policy),
        "output": tk.StringVar(value=args.output_directory or ""),
        "manifest": tk.StringVar(value=args.manifest_path or ""),
        "recon": tk.StringVar(value=str(args.recon_steps)),
        "attack": tk.StringVar(value=str(args.attack_steps)),
        "recovery": tk.StringVar(value=str(args.recovery_steps)),
        "deliveries": tk.StringVar(value=str(args.deliveries_per_robot)),
        "max_steps": tk.StringVar(value="" if args.max_steps is None else str(args.max_steps)),
        "interval_min": tk.StringVar(value=str(args.attack_interval_min)),
        "interval_max": tk.StringVar(value=str(args.attack_interval_max)),
        "compare": tk.BooleanVar(value=args.compare),
        "multi_seed": tk.BooleanVar(value=bool(getattr(args, "seeds", None))),
        "seeds": tk.StringVar(value=getattr(args, "seeds", None) or "1-3"),
        "live_view": tk.BooleanVar(value=False),
    }
    selected_attacks = set() if args.attacks == "none" else set(args.attacks.split(","))
    attack_enabled = {
        attack.value: tk.BooleanVar(value=attack.value in selected_attacks)
        for attack in AttackType
    }

    def entry(label, key, row):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=values[key], width=43).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=4
        )

    def dropdown(label, key, choices, row):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            form, textvariable=values[key], values=choices, state="readonly", width=40
        ).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)

    ttk.Label(form, text="Run configuration", font=("TkDefaultFont", 12, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 7)
    )
    dropdown("Map", "map", tuple(MAP_OPTIONS) + ("Custom NPY map",), 1)
    entry("Map NPY path", "map_path", 2)
    dropdown("Scenario preset", "scenario_preset", ("",) + tuple(PRESETS), 3)
    map_status = ttk.Label(form)
    map_status.grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 4))
    dropdown("Defense method", "method", ALL_METHODS, 5)
    entry("Seed", "seed", 6)
    entry("Output directory (auto-named)", "output", 7)
    entry("Fixed manifest (optional)", "manifest", 8)
    ttk.Checkbutton(form, text="Compare all primary defenses", variable=values["compare"]).grid(
        row=9, column=0, columnspan=3, sticky="w", pady=(7, 2)
    )
    ttk.Checkbutton(form, text="Multi-seed experiment", variable=values["multi_seed"]).grid(row=10, column=0, sticky="w", pady=2)
    ttk.Entry(form, textvariable=values["seeds"], width=18).grid(row=10, column=1, sticky="w", pady=2)
    seed_preview = ttk.Label(form); seed_preview.grid(row=10, column=2, sticky="w", pady=2)
    ttk.Checkbutton(
        form,
        text="Show live maps (traffic heatmap + 4 belief windows)",
        variable=values["live_view"],
    ).grid(row=11, column=0, columnspan=3, sticky="w", pady=2)

    ttk.Separator(form).grid(row=12, column=0, columnspan=3, sticky="ew", pady=9)
    ttk.Label(form, text="Experiment settings", font=("TkDefaultFont", 12, "bold")).grid(
        row=13, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    entry("Reconnaissance steps", "recon", 14)
    entry("Poisoning steps", "attack", 15)
    entry("Recovery steps", "recovery", 16)
    entry("Deliveries per robot", "deliveries", 17)
    entry("Maximum steps (optional)", "max_steps", 18)
    entry("Attack interval: minimum steps", "interval_min", 19)
    entry("Attack interval: maximum steps", "interval_max", 20)
    dropdown("Trust model", "trust_model", ("bayesian", "scalar"), 21)
    dropdown(
        "Admission policy", "admission_policy", ("accept_all", "hard_reject", "auto_soft"), 22
    )
    attacks_frame = ttk.LabelFrame(form, text="Enabled attack types", padding=7)
    attacks_frame.grid(row=23, column=0, columnspan=3, sticky="ew", pady=(9, 0))
    for column, attack in enumerate(AttackType):
        ttk.Checkbutton(
            attacks_frame, text=attack.value.replace("_", " ").title(),
            variable=attack_enabled[attack.value],
        ).grid(row=0, column=column, sticky="w", padx=(0, 12))
    form.columnconfigure(1, weight=1)

    def select_map(*_):
        path, preset = MAP_OPTIONS.get(values["map"].get(), (None, None))
        if values["map"].get() != "Custom NPY map":
            values["map_path"].set(path or "")
            values["scenario_preset"].set(preset or "")
        map_status.configure(
            text=f"Selected experimental geometry: {preset or 'default warehouse behavior'}"
        )

    def execute() -> None:
        try:
            args.map_npy = values["map_path"].get().strip() or None
            args.map_movingai = None
            args.scenario_preset = values["scenario_preset"].get() or None
            validate_gui_map_preset(args.map_npy, args.scenario_preset)
            args.seed = int(values["seed"].get())
            args.defense_method = values["method"].get()
            args.output_directory = values["output"].get().strip() or None
            args.manifest_path = values["manifest"].get().strip() or None
            args.compare = values["compare"].get()
            args.seeds = values["seeds"].get().strip() if values["multi_seed"].get() else None
            args.no_animation = not values["live_view"].get()
            args.recon_steps = int(values["recon"].get())
            args.attack_steps = int(values["attack"].get())
            args.recovery_steps = int(values["recovery"].get())
            args.deliveries_per_robot = int(values["deliveries"].get())
            max_steps = values["max_steps"].get().strip()
            args.max_steps = int(max_steps) if max_steps else None
            args.attack_interval_min = int(values["interval_min"].get())
            args.attack_interval_max = int(values["interval_max"].get())
            args.trust_model = values["trust_model"].get()
            args.admission_policy = values["admission_policy"].get()
            enabled = [name for name, variable in attack_enabled.items() if variable.get()]
            args.attacks = ",".join(enabled) if enabled else "none"
            config = config_from_args(args)
            values["output"].set(config.logging.output_directory)
        except Exception as exc:
            messagebox.showerror("Unable to run", str(exc))
            return
        print(f"Writing results to {config.logging.output_directory}", flush=True)
        live = bool(values["live_view"].get()) and not args.seeds
        if values["live_view"].get() and args.seeds:
            print("Live maps are skipped for multi-seed runs.", flush=True)
        if live:
            print("Live heatmap and belief-map windows will open after the run.", flush=True)
        else:
            print("Progress prints in this terminal. PNG diagrams are written when the run finishes.", flush=True)
        run_button.configure(state="disabled")
        status_label.configure(
            text="Running live maps..." if live else "Running... watch the terminal."
        )
        compare = bool(args.compare)
        multi_seed = bool(args.seeds)

        def finish(error=None) -> None:
            run_button.configure(state="normal")
            if error is not None:
                status_label.configure(text="Run failed.")
                messagebox.showerror("Unable to run", str(error))
                return
            message = result_location_message(
                config.logging.output_directory, compare=compare, multi_seed=multi_seed,
            )
            status_label.configure(text=f"Done. Diagrams: {config.logging.output_directory}\\plots")
            print(message, flush=True)
            messagebox.showinfo("Completed", message)

        def work() -> None:
            try:
                if args.seeds:
                    from .batch import parse_seed_spec, run_multiseed
                    methods = config.comparison_methods if compare else (config.fusion.method,)
                    run_multiseed(config, parse_seed_spec(args.seeds), methods=methods, comparison=compare, generate_per_run_plots=False)
                else:
                    run(config, comparison=compare)
            except Exception as exc:
                return exc
            return None

        if live:
            finish(work())
        else:
            def background() -> None:
                error = work()
                root.after(0, lambda: finish(error))
            threading.Thread(target=background, daemon=True).start()

    footer = ttk.Frame(form)
    footer.grid(row=26, column=0, columnspan=3, sticky="ew", pady=(16, 0))
    status_label = ttk.Label(footer, text="PNG diagrams are saved in a plots folder inside the output directory.")
    status_label.pack(side="left")
    run_button = ttk.Button(footer, text="Run", command=execute)
    run_button.pack(side="right")
    last_auto_output = {"value": ""}

    def current_auto_output() -> str:
        try:
            seed = int(values["seed"].get().strip() or "15")
        except ValueError:
            seed = 15
        seeds = values["seeds"].get().strip() if values["multi_seed"].get() else None
        return suggested_output_directory(
            method=values["method"].get() or "source_linked",
            seed=seed,
            seeds=seeds or None,
            compare=bool(values["compare"].get()),
            map_npy=values["map_path"].get().strip() or None,
            scenario_preset=values["scenario_preset"].get().strip() or None,
            enabled_attacks=tuple(name for name, variable in attack_enabled.items() if variable.get()),
        )

    def output_should_auto_update() -> bool:
        current = values["output"].get().strip()
        return current in {"", "outputs", last_auto_output["value"]}

    def refresh_output(*_):
        if not output_should_auto_update():
            return
        path = current_auto_output()
        last_auto_output["value"] = path
        if values["output"].get() != path:
            values["output"].set(path)

    values["map"].trace_add("write", select_map)
    for key in ("map", "map_path", "scenario_preset", "method", "seed", "seeds", "compare", "multi_seed"):
        values[key].trace_add("write", refresh_output)
    for variable in attack_enabled.values():
        variable.trace_add("write", refresh_output)
    refresh_output()
    def update_preview(*_):
        try:
            from .batch import parse_seed_spec
            count = len(parse_seed_spec(values["seeds"].get()))
            methods = 4 if values["compare"].get() else 1
            seed_preview.configure(text=f"{count} seeds x {methods} methods = {count * methods} simulations" if values["multi_seed"].get() else "")
        except Exception:
            seed_preview.configure(text="")
    values["seeds"].trace_add("write", update_preview)
    values["multi_seed"].trace_add("write", update_preview)
    values["compare"].trace_add("write", update_preview)
    select_map()
    update_preview()
    root.mainloop()
