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
    root.geometry("720x720")
    root.minsize(640, 650)
    form = ttk.Frame(root, padding=14)
    form.pack(fill="both", expand=True)

    values = {
        "seed": tk.StringVar(value=str(args.seed)),
        "method": tk.StringVar(value=args.defense_method),
        "trust_model": tk.StringVar(value=args.trust_model),
        "admission_policy": tk.StringVar(value=args.admission_policy),
        "output": tk.StringVar(value=args.output_directory),
        "manifest": tk.StringVar(value=args.manifest_path or ""),
        "recon": tk.StringVar(value=str(args.recon_steps)),
        "attack": tk.StringVar(value=str(args.attack_steps)),
        "recovery": tk.StringVar(value=str(args.recovery_steps)),
        "deliveries": tk.StringVar(value=str(args.deliveries_per_robot)),
        "max_steps": tk.StringVar(value="" if args.max_steps is None else str(args.max_steps)),
        "interval_min": tk.StringVar(value=str(args.attack_interval_min)),
        "interval_max": tk.StringVar(value=str(args.attack_interval_max)),
        "compare": tk.BooleanVar(value=args.compare),
        "animation": tk.BooleanVar(value=not args.no_animation),
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
    dropdown("Defense method", "method", ALL_METHODS, 2)
    entry("Output directory", "output", 3)
    entry("Fixed manifest (optional)", "manifest", 4)
    ttk.Checkbutton(form, text="Compare all primary defenses", variable=values["compare"]).grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(7, 2)
    )
    ttk.Checkbutton(
        form,
        text="Show reconnaissance heatmap and simulation animation",
        variable=values["animation"],
    ).grid(row=6, column=0, columnspan=3, sticky="w", pady=2)

    ttk.Separator(form).grid(row=7, column=0, columnspan=3, sticky="ew", pady=9)
    ttk.Label(form, text="Experiment settings", font=("TkDefaultFont", 12, "bold")).grid(
        row=8, column=0, columnspan=3, sticky="w", pady=(0, 4)
    )
    entry("Reconnaissance steps", "recon", 9)
    entry("Poisoning steps", "attack", 10)
    entry("Recovery steps", "recovery", 11)
    entry("Deliveries per robot", "deliveries", 12)
    entry("Maximum steps (optional)", "max_steps", 13)
    entry("Attack interval: minimum steps", "interval_min", 14)
    entry("Attack interval: maximum steps", "interval_max", 15)
    dropdown("Trust model", "trust_model", ("bayesian", "scalar"), 16)
    dropdown(
        "Admission policy", "admission_policy", ("accept_all", "hard_reject", "auto_soft"), 17
    )
    attacks_frame = ttk.LabelFrame(form, text="Enabled attack types", padding=7)
    attacks_frame.grid(row=18, column=0, columnspan=3, sticky="ew", pady=(9, 0))
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
            args.defense_method = values["method"].get()
            args.output_directory = values["output"].get()
            args.manifest_path = values["manifest"].get().strip() or None
            args.compare = values["compare"].get()
            args.no_animation = not values["animation"].get()
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

            run(config_from_args(args), comparison=args.compare)
            messagebox.showinfo("Completed", f"Created results in {args.output_directory}")
        except Exception as exc:
            messagebox.showerror("Unable to run", str(exc))

    footer = ttk.Frame(form)
    footer.grid(row=21, column=0, columnspan=3, sticky="ew", pady=(16, 0))
    ttk.Label(footer, text="Close each Matplotlib window to continue to the next one.").pack(side="left")
    ttk.Button(footer, text="Run", command=execute).pack(side="right")
    root.mainloop()
