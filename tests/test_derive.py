"""Hours derived from stops, cycles and machines, with context columns left absent."""
import numpy as np
import pandas as pd

from plantgen import HOURS_CONTEXT, Plant, derive_hours, generate_plant, load_plant, save_plant
from tools import TOOLS, set_backend


def _reduced_stops(plant):
    drop = [c for c in ("supplier", "operator_id") if c in plant.stops.columns]
    return plant.stops.drop(columns=drop)


def test_derived_hours_match_the_generator_and_omit_context():
    plant = generate_plant("S", seed=3)
    derived = derive_hours(plant.machines, _reduced_stops(plant), plant.cycles, plant.start, plant.days)
    for col in HOURS_CONTEXT:
        assert col not in derived.columns
    keys = ["machine_id", "day", "hour_of_day"]
    merged = plant.hours.merge(derived, on=keys, suffixes=("_o", "_d"))
    assert len(merged) == len(plant.hours)
    # unplanned_rate is stored to 0.01. Re-splitting rounded stop durations lands on that grid.
    both = merged.dropna(subset=["unplanned_rate_o", "unplanned_rate_d"])
    assert np.max(np.abs(both["unplanned_rate_o"] - both["unplanned_rate_d"]).round(2)) <= 0.01
    assert (merged["shift_o"] == merged["shift_d"]).all()
    assert "tool_age" in derived.columns
    assert derived["program_new"].dtype == bool


def test_dropped_context_round_trip_and_tools(tmp_path):
    plant = generate_plant("S", seed=3)
    save_plant(plant, tmp_path)
    hours = pd.read_csv(tmp_path / "hours.csv")
    hours = hours.drop(columns=[c for c in HOURS_CONTEXT if c in hours.columns])
    hours.to_csv(tmp_path / "hours.csv", index=False)
    loaded = load_plant(tmp_path)
    for col in HOURS_CONTEXT:
        assert col not in loaded.hours.columns

    (tmp_path / "hours.csv").unlink()
    derived_plant = load_plant(tmp_path)
    assert "unplanned_rate" in derived_plant.hours.columns
    # supplier and operator_id ride along when the stops file has them. The rest cannot be derived.
    for col in ("days_since_pm", "ambient_temp", "coolant_conc"):
        assert col not in derived_plant.hours.columns

    hours = derive_hours(plant.machines, _reduced_stops(plant), plant.cycles, plant.start, plant.days)
    reduced = Plant(plant_id=plant.plant_id, start=plant.start, days=plant.days, machines=plant.machines,
                    stops=_reduced_stops(plant), cycles=plant.cycles, hours=hours, truth=plant.truth)
    set_backend("stock")
    envs = {name: TOOLS[name](reduced, None) for name in TOOLS}
    for env in envs.values():
        assert env["call_id"] and env["facts"]
    missing = envs["rank_loss_drivers"]["result"]["missing_features"]
    assert missing == list(HOURS_CONTEXT)
    assert "Missing features" in next(f["text"] for f in envs["rank_loss_drivers"]["facts"] if f["id"].endswith("#M"))
