"""Save and load plants as four CSV files, so the tools run on real exports too.

A real plant needs the same columns the generator writes (see SCHEMA). The
hidden ``reason_true`` column is optional and only used for evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .derive import HOURS_MINIMUM, derive_hours
from .generator import Plant

SCHEMA = {
    "machines": ["machine_id", "line_id", "machine_type", "nominal_cycle_s", "night_shift"],
    "stops": ["stop_id", "machine_id", "line_id", "day", "shift", "t_start", "duration_s", "hour_of_day",
              "cycles_since_prev", "spindle_load_before", "tool_age_before", "reason_reported"],
    "cycles": ["cycle_id", "machine_id", "program_id", "day", "shift", "t_end", "cycle_s", "nominal_cycle_s",
               "spindle_load", "spindle_speed", "feed_override", "axis_load_x", "axis_load_y", "axis_load_z",
               "motor_current", "vibration_rms", "spindle_temp", "coolant_pressure", "coolant_temp", "power_kw"],
    "hours": ["machine_id", "line_id", "machine_type", "day", "hour_of_day", "weekday", "shift", "supplier",
              "operator_id", "program_new", "tool_age", "days_since_pm", "ambient_temp", "coolant_conc",
              "unplanned_rate"],
}


def save_plant(plant: Plant, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for t in ("machines", "stops", "cycles", "hours"):
        getattr(plant, t).to_csv(out / f"{t}.csv", index=False)
    meta = {"plant_id": plant.plant_id, "start": plant.start.isoformat(), "days": plant.days, "truth": plant.truth}
    (out / "plant.json").write_text(json.dumps(meta, indent=2))
    return out


def load_plant(src: str | Path) -> Plant:
    src = Path(src)
    meta = json.loads((src / "plant.json").read_text())
    tables = {}
    for t in ("machines", "stops", "cycles"):
        tables[t] = pd.read_csv(src / f"{t}.csv", keep_default_na=False if t == "stops" else True)
        missing = [c for c in SCHEMA[t] if c not in tables[t].columns]
        if missing:
            raise ValueError(f"{t}.csv is missing columns: {missing}")
    hours_path = src / "hours.csv"
    if hours_path.exists():
        hours = pd.read_csv(hours_path)
        missing = [c for c in HOURS_MINIMUM if c not in hours.columns]
        if missing:
            raise ValueError(f"hours.csv is missing columns: {missing}")
    else:
        hours = derive_hours(tables["machines"], tables["stops"], tables["cycles"],
                             start=pd.Timestamp(meta["start"]), days=int(meta["days"]))
    return Plant(plant_id=meta["plant_id"], start=pd.Timestamp(meta["start"]), days=int(meta["days"]),
                 truth=meta.get("truth", {}), hours=hours, **tables)
