"""Build the machine-hour table from stops, cycles and machines.

A plant that never stored the hour rollup can still run ``rank_loss_drivers``.
Columns that are not in the export stay absent. They are not filled with
placeholders.

Assumptions, used only when the export has no better column:

* If ``stops.reason_true`` is missing, a blank ``reason_reported`` counts as
  unplanned loss. Meals and setups are logged in the synthetic plant; a real
  plant that leaves planned stops blank will look worse than it is.
* ``program_new`` is the first three days after a program_id is first seen on
  that machine. A program already running on day 0 is not new.
* ``tool_age`` is the mean ``cycles.tool_age`` in the hour, carried forward
  when the hour had no finished cycle. If cycles have no tool age, the last
  ``stops.tool_age_before`` is carried forward instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import LOSS_KINDS, PLANNED_KINDS, UNPLANNED_KINDS

# On a full synthetic hour table, and not recoverable from stops, cycles and machines.
HOURS_CONTEXT = ("supplier", "operator_id", "days_since_pm", "ambient_temp", "coolant_conc")
# Enough for load_plant to accept an hours export. Other SCHEMA columns are optional.
HOURS_MINIMUM = ("machine_id", "day", "hour_of_day", "unplanned_rate")
_BLANK = "unplanned_blank"


def derive_hours(machines: pd.DataFrame, stops: pd.DataFrame, cycles: pd.DataFrame,
                 start, days: int | None = None) -> pd.DataFrame:
    start = pd.Timestamp(start)
    if days is None:
        observed = [int(cycles["day"].max()) if len(cycles) else 0, int(stops["day"].max()) if len(stops) else 0]
        days = max(observed) + 1
    n = len(machines)
    n_hours = int(days) * 24
    order = {m: i for i, m in enumerate(machines["machine_id"].to_numpy())}

    stops = stops.loc[stops["machine_id"].isin(order)].copy()
    stop_i = stops["machine_id"].map(order).to_numpy(np.int64)
    kind_col, unplanned, loss, planned = _kinds(stops)
    loss_min = _minutes_by_hour(stops, stop_i, loss, kind_col, n, n_hours)
    unplanned_min = _minutes_by_hour(stops, stop_i, unplanned, kind_col, n, n_hours)
    planned_min = _minutes_by_hour(stops, stop_i, planned, kind_col, n, n_hours)

    cyc = cycles.loc[cycles["machine_id"].isin(order)].copy()
    # Cycles that finish after the last hour are counted in that hour, matching the generator's clip.
    cyc["bin"] = np.clip(cyc["t_end"].to_numpy(np.float64) // 3600, 0, n_hours - 1).astype(np.int64)
    cyc_n = np.zeros(n * n_hours)
    if len(cyc):
        ci = cyc["machine_id"].map(order).to_numpy(np.int64)
        np.add.at(cyc_n, ci * n_hours + cyc["bin"].to_numpy(np.int64), 1)

    sched = (cyc_n > 0) | (loss_min > 0)
    idx = np.flatnonzero(sched)
    mm, hh = idx // n_hours, idx % n_hours
    day = hh // 24
    hod = (hh % 24).astype(np.int64)
    shift = _shift(cyc, order, n, n_hours, idx, hod)
    program_new = _program_new(cyc, order, machines["machine_id"].to_numpy(), n, n_hours, idx, day)
    tool_age = _tool_age(cyc, stops, stop_i, order, n, n_hours, idx, mm)

    out = {
        "machine_id": machines["machine_id"].to_numpy()[mm],
        "line_id": machines["line_id"].to_numpy()[mm],
        "machine_type": machines["machine_type"].to_numpy()[mm],
        "day": day.astype(np.int16),
        "hour_of_day": hod.astype(np.int8),
        "weekday": ((int(start.weekday()) + day) % 7).astype(np.int8),
        "shift": shift,
        "program_new": program_new,
        "unplanned_rate": np.where(
            60 - planned_min[idx] >= 15,
            np.clip(unplanned_min[idx] / np.maximum(60 - planned_min[idx], 1e-9) * 60, 0, 60),
            np.nan,
        ).round(2),
        "stop_minutes": loss_min[idx].round(2),
        "unplanned_minutes": np.clip(unplanned_min[idx], 0, 60).round(2),
        "planned_minutes": np.clip(planned_min[idx], 0, 60).round(2),
    }
    if tool_age is not None:
        out["tool_age"] = tool_age
    for col in ("supplier", "operator_id"):
        if col in stops.columns:
            out[col] = _day_value(stops, stop_i, col, n, n_hours, idx, mm, day)
    return pd.DataFrame(out)


def _kinds(stops: pd.DataFrame):
    if "reason_true" in stops.columns:
        return "reason_true", set(UNPLANNED_KINDS), set(LOSS_KINDS), set(PLANNED_KINDS)
    return "reason_reported", set(UNPLANNED_KINDS) | {_BLANK}, set(LOSS_KINDS) | {_BLANK}, set(PLANNED_KINDS)


def _minutes_by_hour(stops, machine_index, kinds, kind_col, n, n_hours) -> np.ndarray:
    out = np.zeros(n * n_hours)
    if kind_col not in stops.columns or not len(stops):
        return out
    kind = stops[kind_col].fillna("").astype(str).to_numpy()
    if kind_col == "reason_reported":
        kind = np.where(kind == "", _BLANK, kind)
    sel = np.isin(kind, list(kinds))
    if not sel.any():
        return out
    s = stops["t_start"].to_numpy(np.float64)[sel]
    dur = stops["duration_s"].to_numpy(np.float64)[sel]
    e = s + dur
    m = machine_index[sel]
    b0 = (s // 3600).astype(np.int64)
    for j in range(int(((e - s) // 3600).max()) + 2):
        b = b0 + j
        ov = np.clip(np.minimum(e, (b + 1) * 3600) - np.maximum(s, b * 3600), 0, None)
        ok = (ov > 0) & (b >= 0) & (b < n_hours)
        np.add.at(out, m[ok] * n_hours + b[ok], ov[ok] / 60.0)
    return out


def _shift(cyc, order, n, n_hours, idx, hod) -> np.ndarray:
    shift_of = np.full(n * n_hours, None, dtype=object)
    if len(cyc):
        last = cyc.sort_values(["t_end"], kind="stable").groupby(["machine_id", "bin"], sort=False).tail(1)
        slot = last["machine_id"].map(order).to_numpy(np.int64) * n_hours + last["bin"].to_numpy(np.int64)
        shift_of[slot] = last["shift"].to_numpy()
    inferred = np.where((hod >= 6) & (hod < 14), "A", np.where((hod >= 14) & (hod < 22), "B", "C"))
    got = shift_of[idx]
    return np.where(pd.isna(got), inferred, got).astype(str)


def _program_new(cyc, order, machine_ids, n, n_hours, idx, day) -> np.ndarray:
    prog_of = np.full(n * n_hours, None, dtype=object)
    intro: dict[tuple, int] = {}
    if len(cyc):
        last = cyc.sort_values(["t_end"], kind="stable").groupby(["machine_id", "bin"], sort=False).tail(1)
        slot = last["machine_id"].map(order).to_numpy(np.int64) * n_hours + last["bin"].to_numpy(np.int64)
        prog_of[slot] = last["program_id"].to_numpy()
        first = cyc.groupby(["machine_id", "program_id"], sort=False)["day"].min()
        intro = {(m, p): int(d) for (m, p), d in first.items()}
    out = np.zeros(len(idx), dtype=bool)
    if not intro:
        return out
    for i, slot in enumerate(idx):
        program = prog_of[slot]
        if program is None:
            continue
        first_day = intro.get((machine_ids[slot // n_hours], program))
        if first_day is not None and first_day > 0 and int(day[i]) - first_day < 3:
            out[i] = True
    return out


def _tool_age(cyc, stops, stop_i, order, n, n_hours, idx, mm) -> np.ndarray | None:
    values = np.full(n * n_hours, np.nan)
    found = False
    if "tool_age" in cyc.columns and len(cyc):
        found = True
        means = cyc.groupby(["machine_id", "bin"], sort=False)["tool_age"].mean()
        for (machine, binv), value in means.items():
            values[order[machine] * n_hours + int(binv)] = value
    elif "tool_age_before" in stops.columns and len(stops):
        found = True
        b0 = (stops["t_start"].to_numpy(np.float64) // 3600).astype(np.int64)
        ok = (b0 >= 0) & (b0 < n_hours)
        # last stop in the hour wins; stable sort by time then write
        order_t = np.argsort(stops["t_start"].to_numpy(np.float64), kind="stable")
        for j in order_t:
            if not ok[j]:
                continue
            values[stop_i[j] * n_hours + b0[j]] = stops["tool_age_before"].to_numpy(np.float64)[j]
    if not found:
        return None
    ta = np.where(np.isnan(values[idx]), np.nan, values[idx])
    filled = pd.Series(ta).groupby(mm).ffill().groupby(mm).bfill().fillna(0).to_numpy()
    return np.round(filled.astype(np.float64), 1)


def _day_value(stops, stop_i, col, n, n_hours, idx, mm, day) -> np.ndarray:
    """One value per machine-day, taken from the stops of that day. Hours with no stop stay empty."""
    per_day: dict[tuple[int, int], str] = {}
    raw = stops[col].to_numpy()
    sday = stops["day"].to_numpy()
    for i in range(len(stops)):
        per_day[(int(stop_i[i]), int(sday[i]))] = raw[i]
    return np.array([per_day.get((int(m), int(d)), np.nan) for m, d in zip(mm, day)], dtype=object)
