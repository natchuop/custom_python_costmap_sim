"""Tkinter launcher for interactive simulator runs."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import threading

from .application import run
from .cli import config_from_args, result_location_message, suggested_output_directory
from .config import ALL_METHODS
from .map_io import load_npy
from .models import AttackType
from .reporting import REFERENCE_FIGURE_METHODS, generate_reference_report
from .scenario_presets import PRESETS, map_path_for_preset, preset_for_hash, preset_for_id, validate_fixed_preset


MAP_OPTIONS = {
    "Default warehouse": (None, None),
    "Map 002 (converted)": (map_path_for_preset("warehouse_002"), "warehouse_002"),
    "Map 005 (converted)": (map_path_for_preset("warehouse_005"), "warehouse_005"),
    "Map 005 rotated (converted)": (map_path_for_preset("warehouse_005_rotated"), "warehouse_005_rotated"),
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


def is_physical_ai_method_selection(methods) -> bool:
    """Return whether the UI selection is the four-method reference workflow."""
    return set(methods) == set(REFERENCE_FIGURE_METHODS)


def generate_physical_ai_report(output_directory: str | Path) -> dict:
    """Delegate UI reporting to the canonical aggregate reference reporter."""
    return generate_reference_report(output_directory)


def run_physical_ai_workflow(config, seeds, *, full_suite=False):
    """Run the selected Physical AI workflow and render its current batch."""
    from .batch import run_multiseed

    root = config.logging.output_directory
    if full_suite:
        from .reference_experiments import run_reference_suite

        run_reference_suite(
            config,
            tuple(seeds),
            root,
            include_sweeps=True,
            measure_runtime=config.logging.measure_fusion_runtime,
            generate_report=False,
        )
    else:
        report_config = replace(
            config,
            logging=replace(config.logging, generate_plots=False),
        )
        run_multiseed(
            report_config,
            tuple(seeds),
            methods=REFERENCE_FIGURE_METHODS,
            comparison=True,
            generate_per_run_plots=False,
        )
    return generate_physical_ai_report(root)


def launch(args) -> None:
    """Show the interactive run configuration window."""
    # Keep map/preset validation importable on headless Python builds. Tk is
    # required only when the user actually launches the interactive window.
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Modular Map-Poisoning Simulator")
    root.geometry("820x900")
    root.minsize(700, 620)

    shell = ttk.Frame(root, padding=8)
    shell.pack(fill="both", expand=True)
    body = ttk.Frame(shell)
    body.pack(fill="both", expand=True)
    canvas = tk.Canvas(body, highlightthickness=0, borderwidth=0)
    scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    form = ttk.Frame(canvas, padding=14)
    form_window = canvas.create_window((0, 0), window=form, anchor="nw")

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def resize_form(event):
        canvas.itemconfigure(form_window, width=event.width)

    form.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", resize_form)

    def scroll_with_mouse(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")

    canvas.bind_all("<MouseWheel>", scroll_with_mouse)

    initial_preset = args.scenario_preset or _preset_for_map_path(args.map_npy) or ""
    values = {
        "map": tk.StringVar(value=_map_option_for_args(args)),
        "map_path": tk.StringVar(value=args.map_npy or ""),
        "scenario_preset": tk.StringVar(value=initial_preset),
        "seed": tk.StringVar(value=str(args.seed)),
        "trust_model": tk.StringVar(value=args.trust_model),
        "trust_threshold": tk.StringVar(value=str(getattr(args, "trust_threshold", 0.50))),
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
        "visibility_min": tk.StringVar(value=str(getattr(args, "attack_visibility_min", 15))),
        "visibility_max": tk.StringVar(value=str(getattr(args, "attack_visibility_max", 40))),
        "temp_interval": tk.StringVar(value=str(getattr(args, "temp_obstacle_interval", 150))),
        "map_view": tk.StringVar(
            value=("Combined observations" if getattr(args, "map_view", "combined") == "combined" else "Local observations")
        ),
        "multi_seed": tk.BooleanVar(value=bool(getattr(args, "seeds", None))),
        "seeds": tk.StringVar(value=getattr(args, "seeds", None) or "1-3"),
        "live_view": tk.BooleanVar(value=False),
        "full_physical_ai_suite": tk.BooleanVar(value=False),
    }
    initial_methods = {args.defense_method}
    extra = getattr(args, "comparison_methods", None)
    if args.compare:
        if extra:
            initial_methods.update(item.strip() for item in str(extra).split(",") if item.strip())
        else:
            initial_methods.update(REFERENCE_FIGURE_METHODS)
    method_enabled = {
        method: tk.BooleanVar(value=method in initial_methods)
        for method in ALL_METHODS
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
    methods_frame = ttk.LabelFrame(form, text="Defense methods to run", padding=7)
    methods_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(2, 4))
    ttk.Label(
        methods_frame,
        text=(
            "Select one or more methods. One selection shows the full live playback "
            "for that method. Multiple selections replay the same manifest and save "
            "each method under the output folder."
        ),
        wraplength=620,
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    for index, method in enumerate(ALL_METHODS):
        ttk.Checkbutton(
            methods_frame,
            text=method,
            variable=method_enabled[method],
        ).grid(row=1 + index // 2, column=index % 2, sticky="w", padx=(0, 24), pady=1)
    entry("Seed", "seed", 6)
    entry("Output directory (auto-named)", "output", 7)
    entry("Fixed manifest (optional)", "manifest", 8)
    ttk.Checkbutton(form, text="Multi-seed experiment", variable=values["multi_seed"]).grid(row=10, column=0, sticky="w", pady=2)
    ttk.Entry(form, textvariable=values["seeds"], width=18).grid(row=10, column=1, sticky="w", pady=2)
    seed_preview = ttk.Label(form); seed_preview.grid(row=10, column=2, sticky="w", pady=2)
    ttk.Checkbutton(
        form,
        text="Show live maps (recon heatmap first, then 4 belief windows)",
        variable=values["live_view"],
    ).grid(row=11, column=0, columnspan=3, sticky="w", pady=2)
    ttk.Checkbutton(
        form,
        text="Run full Physical AI suite (includes Fig. 4 and Fig. 9 sweeps; may take hours)",
        variable=values["full_physical_ai_suite"],
    ).grid(row=12, column=0, columnspan=3, sticky="w", pady=2)
    dropdown("Belief map view", "map_view", ("Combined observations", "Local observations"), 13)
    ttk.Label(
        form,
        text=(
            "Live maps open the shared attack-free reference heatmap first. Close that window "
            "to start ground-truth and per-robot belief playback with the trust panel."
        ),
        wraplength=650,
        justify="left",
    ).grid(row=14, column=0, columnspan=3, sticky="w", pady=3)

    ttk.Separator(form).grid(row=14, column=0, columnspan=3, sticky="ew", pady=9)
    ttk.Label(form, text="Experiment settings", font=("TkDefaultFont", 12, "bold")).grid(
        row=15, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    entry("Reconnaissance steps", "recon", 16)
    entry("Poisoning steps", "attack", 17)
    entry("Recovery steps", "recovery", 18)
    entry("Deliveries per robot", "deliveries", 19)
    entry("Maximum steps (optional)", "max_steps", 20)
    entry("Attack interval: minimum steps", "interval_min", 21)
    entry("Attack interval: maximum steps", "interval_max", 22)
    entry("Fake target: minimum steps to visibility", "visibility_min", 23)
    entry("Fake target: maximum steps to visibility", "visibility_max", 24)
    entry("Temporary obstacle movement interval", "temp_interval", 25)
    dropdown("Trust model", "trust_model", ("bayesian", "scalar"), 26)
    entry("Trust threshold", "trust_threshold", 27)
    dropdown(
        "Admission policy", "admission_policy", ("accept_all", "hard_reject", "auto_soft"), 28
    )
    attacks_frame = ttk.LabelFrame(form, text="Enabled attack types", padding=7)
    attacks_frame.grid(row=29, column=0, columnspan=3, sticky="ew", pady=(9, 0))
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
            selected_methods = tuple(method for method in ALL_METHODS if method_enabled[method].get())
            if not selected_methods:
                raise ValueError("Select at least one defense method to run.")
            args.defense_method = selected_methods[0]
            args.compare = len(selected_methods) > 1
            args.comparison_methods = ",".join(selected_methods) if args.compare else None
            full_suite = bool(values["full_physical_ai_suite"].get())
            if full_suite and not is_physical_ai_method_selection(selected_methods):
                raise ValueError("Run full Physical AI suite requires exactly the four Physical AI methods.")
            args.output_directory = values["output"].get().strip() or None
            args.manifest_path = values["manifest"].get().strip() or None
            args.seeds = values["seeds"].get().strip() if (values["multi_seed"].get() or full_suite) else None
            args.no_animation = (not values["live_view"].get()) or args.compare
            args.map_view = "combined" if values["map_view"].get() == "Combined observations" else "local"
            args.recon_steps = int(values["recon"].get())
            args.attack_steps = int(values["attack"].get())
            args.recovery_steps = int(values["recovery"].get())
            args.deliveries_per_robot = int(values["deliveries"].get())
            max_steps = values["max_steps"].get().strip()
            args.max_steps = int(max_steps) if max_steps else None
            args.attack_interval_min = int(values["interval_min"].get())
            args.attack_interval_max = int(values["interval_max"].get())
            args.attack_visibility_min = int(values["visibility_min"].get())
            args.attack_visibility_max = int(values["visibility_max"].get())
            args.temp_obstacle_interval = int(values["temp_interval"].get())
            args.trust_model = values["trust_model"].get()
            args.trust_threshold = float(values["trust_threshold"].get())
            args.admission_policy = values["admission_policy"].get()
            enabled = [name for name, variable in attack_enabled.items() if variable.get()]
            args.attacks = ",".join(enabled) if enabled else "none"
            config = config_from_args(args)
            values["output"].set(config.logging.output_directory)
        except Exception as exc:
            messagebox.showerror("Unable to run", str(exc))
            return
        print(f"Writing results to {config.logging.output_directory}", flush=True)
        print(
            f"Configured steps: recon={config.phases.recon_steps}, "
            f"attack={config.phases.attack_steps}, recovery={config.phases.recovery_steps}, "
            f"effective_total={config.total_steps}",
            flush=True,
        )
        live = bool(values["live_view"].get()) and not args.seeds and not args.compare
        if values["live_view"].get() and (args.seeds or args.compare):
            print("Live maps play only for a single-method, single-seed run.", flush=True)
        if live:
            print("Reconnaissance heatmap opens first; close it to play the four belief maps.", flush=True)
        else:
            print("Progress prints in this terminal. PNG diagrams are written when the run finishes.", flush=True)
        run_button.configure(state="disabled")
        status_label.configure(
            text="Running live maps..." if live else "Running... watch the terminal."
        )
        compare = bool(args.compare)
        multi_seed = bool(args.seeds)
        physical_ai_workflow = is_physical_ai_method_selection(selected_methods)

        def finish(error=None) -> None:
            run_button.configure(state="normal")
            if error is not None:
                status_label.configure(text="Run failed.")
                messagebox.showerror("Unable to run", str(error))
                return
            plot_directory = (
                Path(config.logging.output_directory) / "aggregate" / "plots"
                if physical_ai_workflow
                else Path(config.logging.output_directory) / "plots"
            )
            message = (
                f"Created results in {config.logging.output_directory}\n\n"
                f"Physical AI aggregate diagrams:\n{plot_directory}"
                if physical_ai_workflow
                else result_location_message(
                    config.logging.output_directory, compare=compare, multi_seed=multi_seed,
                )
            )
            status_label.configure(text=f"Done. Diagrams: {plot_directory}")
            print(message, flush=True)
            messagebox.showinfo("Completed", message)

        def work() -> None:
            try:
                if physical_ai_workflow:
                    from .batch import parse_seed_spec
                    seeds = parse_seed_spec(args.seeds) if args.seeds else (config.seed,)
                    run_physical_ai_workflow(config, seeds, full_suite=full_suite)
                elif args.seeds:
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

    footer = ttk.Frame(shell, padding=(6, 8, 6, 0))
    footer.pack(side="bottom", fill="x")
    status_label = ttk.Label(
        footer,
        text="Close the heatmap window to start live playback. PNG diagrams are saved in plots/.",
    )
    status_label.pack(side="left")
    run_button = ttk.Button(footer, text="Run", command=execute)
    run_button.pack(side="right")
    last_auto_output = {"value": ""}

    def current_auto_output() -> str:
        try:
            seed = int(values["seed"].get().strip() or "15")
        except ValueError:
            seed = 15
        seeds = values["seeds"].get().strip() if (values["multi_seed"].get() or values["full_physical_ai_suite"].get()) else None
        return suggested_output_directory(
            method=next((method for method in ALL_METHODS if method_enabled[method].get()), "source_memory"),
            seed=seed,
            seeds=seeds or None,
            compare=sum(1 for variable in method_enabled.values() if variable.get()) > 1,
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
    for key in ("map", "map_path", "scenario_preset", "seed", "seeds", "multi_seed"):
        values[key].trace_add("write", refresh_output)
    for variable in list(attack_enabled.values()) + list(method_enabled.values()):
        variable.trace_add("write", refresh_output)
    refresh_output()
    def update_preview(*_):
        try:
            from .batch import parse_seed_spec
            count = len(parse_seed_spec(values["seeds"].get()))
            methods = sum(1 for variable in method_enabled.values() if variable.get()) or 1
            active = values["multi_seed"].get() or values["full_physical_ai_suite"].get()
            seed_preview.configure(text=f"{count} seeds x {methods} methods = {count * methods} simulations" if active else "")
        except Exception:
            seed_preview.configure(text="")
    values["seeds"].trace_add("write", update_preview)
    values["multi_seed"].trace_add("write", update_preview)
    for variable in method_enabled.values():
        variable.trace_add("write", update_preview)
    select_map()
    update_preview()
    root.mainloop()
