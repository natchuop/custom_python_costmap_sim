"""Tkinter launcher for interactive simulator runs."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .application import run
from .cli import config_from_args
from .config import ALL_METHODS
from .models import AttackType


def launch(args) -> None:
    """Show the interactive run configuration window in one compact view."""
    root = tk.Tk()
    root.title("Modular Map-Poisoning Simulator")
    root.geometry("760x900")
    root.minsize(680, 820)
    form = ttk.Frame(root, padding=14)
    form.pack(fill="both", expand=True)

    values = {
        "seed": tk.StringVar(value=str(args.seed)),
        "trust_model": tk.StringVar(value=args.trust_model),
        "trust_threshold": tk.StringVar(value=str(getattr(args, "trust_threshold", 0.55))),
        "output": tk.StringVar(value=args.output_directory),
        "manifest": tk.StringVar(value=args.manifest_path or ""),
        "recon": tk.StringVar(value=str(args.recon_steps)),
        "attack": tk.StringVar(value=str(args.attack_steps)),
        "recovery": tk.StringVar(value=str(args.recovery_steps)),
        "deliveries": tk.StringVar(value=str(args.deliveries_per_robot)),
        "max_steps": tk.StringVar(value="" if args.max_steps is None else str(args.max_steps)),
        "interval_min": tk.StringVar(value=str(args.attack_interval_min)),
        "interval_max": tk.StringVar(value=str(args.attack_interval_max)),
        "temp_interval": tk.StringVar(value=str(getattr(args, "temp_obstacle_interval", 150))),
        "map_view": tk.StringVar(
            value=("Combined observations" if getattr(args, "map_view", "combined") == "combined" else "Local observations")
        ),
    }
    selected_comparison_methods = set(
        getattr(args, "comparison_methods", "trust_fused").split(",")
    )
    method_enabled = {
        method: tk.BooleanVar(value=method in selected_comparison_methods)
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
    entry("Seed", "seed", 1)
    methods_frame = ttk.LabelFrame(form, text="Defense methods to run", padding=7)
    methods_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 4))
    ttk.Label(
        methods_frame,
        text=(
            "Select one or more methods. One selection displays the full "
            "simulation for that method. Multiple selections replay the same "
            "manifest headlessly and save every result to the output folder."
        ),
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    for index, method in enumerate(ALL_METHODS):
        ttk.Checkbutton(
            methods_frame,
            text=method,
            variable=method_enabled[method],
        ).grid(row=1 + index // 2, column=index % 2, sticky="w", padx=(0, 24), pady=1)
    entry("Output parent directory", "output", 7)
    entry("Fixed manifest (optional)", "manifest", 8)
    ttk.Label(
        form,
        text=(
            "One selected method shows its reconnaissance heatmap and full live "
            "playback. Multiple selected methods run headlessly on the same "
            "manifest; results are grouped under a seed folder, each method gets "
            "a separate result folder, and the shared traffic heatmap is saved "
            "at the comparison-folder level."
        ),
        wraplength=650,
        justify="left",
    ).grid(row=9, column=0, columnspan=3, sticky="w", pady=3)
    dropdown("Belief map view", "map_view", ("Combined observations", "Local observations"), 10)

    ttk.Separator(form).grid(row=11, column=0, columnspan=3, sticky="ew", pady=9)
    ttk.Label(form, text="Experiment settings", font=("TkDefaultFont", 12, "bold")).grid(
        row=12, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    entry("Reconnaissance steps", "recon", 13)
    entry("Poisoning steps", "attack", 14)
    entry("Recovery steps", "recovery", 15)
    entry("Deliveries per robot", "deliveries", 16)
    entry("Maximum steps (optional)", "max_steps", 17)
    entry("Attack interval: minimum steps", "interval_min", 18)
    entry("Attack interval: maximum steps", "interval_max", 19)
    entry("Temporary obstacle movement interval", "temp_interval", 20)
    dropdown("Trust model", "trust_model", ("bayesian", "scalar"), 21)
    entry("Trust threshold", "trust_threshold", 22)
    attacks_frame = ttk.LabelFrame(form, text="Enabled attack types", padding=7)
    attacks_frame.grid(row=24, column=0, columnspan=3, sticky="ew", pady=(9, 0))
    for column, attack in enumerate(AttackType):
        ttk.Checkbutton(
            attacks_frame,
            text=attack.value.replace("_", " ").title(),
            variable=attack_enabled[attack.value],
        ).grid(row=0, column=column, sticky="w", padx=(0, 12))

    form.columnconfigure(1, weight=1)

    def execute() -> None:
        try:
            args.seed = int(values["seed"].get())
            selected_methods = tuple(
                method for method in ALL_METHODS if method_enabled[method].get()
            )
            if not selected_methods:
                raise ValueError("Select at least one defense method to run.")
            args.defense_method = selected_methods[0]
            args.comparison_methods = ",".join(selected_methods)
            args.compare = len(selected_methods) > 1
            args.output_directory = values["output"].get()
            args.manifest_path = values["manifest"].get().strip() or None
            # Interactive playback is useful for a single method. Comparison
            # runs stay headless so selecting several methods does not open a
            # full animation window for every replay.
            args.no_animation = len(selected_methods) > 1
            args.map_view = "combined" if values["map_view"].get() == "Combined observations" else "local"
            args.recon_steps = int(values["recon"].get())
            args.attack_steps = int(values["attack"].get())
            args.recovery_steps = int(values["recovery"].get())
            args.deliveries_per_robot = int(values["deliveries"].get())
            max_steps = values["max_steps"].get().strip()
            args.max_steps = int(max_steps) if max_steps else None
            args.attack_interval_min = int(values["interval_min"].get())
            args.attack_interval_max = int(values["interval_max"].get())
            args.temp_obstacle_interval = int(values["temp_interval"].get())
            args.trust_model = values["trust_model"].get()
            args.trust_threshold = float(values["trust_threshold"].get())
            enabled = [name for name, variable in attack_enabled.items() if variable.get()]
            args.attacks = ",".join(enabled) if enabled else "none"

            run(config_from_args(args), comparison=args.compare)
            messagebox.showinfo(
                "Completed",
                f"Created results under {args.output_directory}\\seed_{args.seed}",
            )
        except Exception as exc:
            messagebox.showerror("Unable to run", str(exc))

    footer = ttk.Frame(form)
    footer.grid(row=25, column=0, columnspan=3, sticky="ew", pady=(16, 0))
    ttk.Label(footer, text="Close each Matplotlib window to continue to the next one.").pack(side="left")
    ttk.Button(footer, text="Run", command=execute).pack(side="right")
    root.mainloop()
