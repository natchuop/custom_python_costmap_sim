"""Thin Tkinter launcher.  Import this module only for interactive use."""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, ttk
from .application import run
from .cli import config_from_args

def launch(args) -> None:
    root=tk.Tk(); root.title("Modular Map-Poisoning Simulator"); root.geometry("650x420")
    notebook=ttk.Notebook(root); notebook.pack(fill="both",expand=True,padx=10,pady=10)
    basic=ttk.Frame(notebook,padding=12); advanced=ttk.Frame(notebook,padding=12); notebook.add(basic,text="Run"); notebook.add(advanced,text="Advanced")
    values={"seed":tk.StringVar(value=str(args.seed)),"method":tk.StringVar(value=args.defense_method),"attacks":tk.StringVar(value=args.attacks),"output":tk.StringVar(value=args.output_directory),"recon":tk.StringVar(value=str(args.recon_steps)),"attack":tk.StringVar(value=str(args.attack_steps)),"recovery":tk.StringVar(value=str(args.recovery_steps))}
    def field(parent,label,key,row): ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=5); ttk.Entry(parent,textvariable=values[key],width=42).grid(row=row,column=1,sticky="ew",pady=5)
    field(basic,"Seed", "seed",0); field(basic,"Defense method", "method",1); field(basic,"Attacks", "attacks",2); field(basic,"Output directory", "output",3)
    field(advanced,"Reconnaissance steps", "recon",0); field(advanced,"Poisoning steps", "attack",1); field(advanced,"Recovery steps", "recovery",2)
    basic.columnconfigure(1,weight=1); advanced.columnconfigure(1,weight=1)
    def execute(compare=False):
        try:
            args.seed=int(values["seed"].get()); args.defense_method=values["method"].get(); args.attacks=values["attacks"].get(); args.output_directory=values["output"].get(); args.recon_steps=int(values["recon"].get()); args.attack_steps=int(values["attack"].get()); args.recovery_steps=int(values["recovery"].get())
            result=run(config_from_args(args),comparison=compare); messagebox.showinfo("Completed",f"Created results in {args.output_directory}")
        except Exception as exc: messagebox.showerror("Unable to run",str(exc))
    ttk.Button(basic,text="Run single defense",command=lambda:execute(False)).grid(row=5,column=0,pady=24,sticky="w"); ttk.Button(basic,text="Compare primary defenses",command=lambda:execute(True)).grid(row=5,column=1,pady=24,sticky="e")
    root.mainloop()
