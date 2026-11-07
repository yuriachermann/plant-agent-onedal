"""Synthetic plant generator.

Produces plant-shaped data in the same four tables the edge stores for a
CNC job shop:

* ``machines`` - one row per machine (line, type, nominal cycle, tool life)
* ``stops``    - one row per stop event, with the operator-reported reason
                 (often blank) and the hidden true reason
* ``cycles``   - one row per completed part cycle, with per-cycle summaries
                 of 12 machine signals
* ``hours``    - one row per machine x scheduled hour, with lost minutes and
                 the candidate loss drivers

Faults (cycle-time drift, coolant degradation) and loss drivers are planted
on purpose and returned in ``truth``, so every tool can be scored against a
known answer. Everything is seeded: the same arguments give the same plant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAY = 86_400
SHIFT_WINDOWS = {"A": (6 * 3600, 14 * 3600), "B": (14 * 3600, 22 * 3600), "C": (22 * 3600, 30 * 3600)}
MEAL_AT = {"A": int(11.5 * 3600), "B": int(18.5 * 3600), "C": int(26.0 * 3600)}
SUPPLIERS = np.array(["S1", "S2", "S3", "S4"])
SUPPLIER_P = [0.35, 0.30, 0.20, 0.15]
SIGNALS = [
    "spindle_load", "spindle_speed", "feed_override", "axis_load_x", "axis_load_y",
    "axis_load_z", "motor_current", "vibration_rms", "spindle_temp",
    "coolant_pressure", "coolant_temp", "power_kw",
]
# Kinds that count as losses in OEE availability. Meal breaks are planned time.
LOSS_KINDS = {"micro", "tool_change", "setup", "first_article", "quality_check", "material_wait", "breakdown"}
UNPLANNED_KINDS = {"micro", "quality_check", "material_wait", "breakdown"}
PLANNED_KINDS = {"setup", "first_article", "tool_change", "meal"}
REPORTED_ALWAYS = {"setup", "meal"}  # operators always log these; the rest are often left blank

# Size presets: the three plant sizes used in the talk.
PRESETS = {"S": 12, "M": 40, "L": 120}


@dataclass
class Plant:
    plant_id: str
    start: pd.Timestamp
    days: int
    machines: pd.DataFrame
    stops: pd.DataFrame
    cycles: pd.DataFrame
    hours: pd.DataFrame
    truth: dict = field(default_factory=dict)

    def summary(self) -> dict:
        prod_days = self.cycles["day"].nunique()
        return {
            "plant_id": self.plant_id,
            "machines": int(len(self.machines)),
            "days": int(self.days),
            "stops": int(len(self.stops)),
            "cycles": int(len(self.cycles)),
            "machine_hours": int(len(self.hours)),
            "stops_per_day": round(len(self.stops) / max(prod_days, 1), 1),
            "cycles_per_day": round(len(self.cycles) / max(prod_days, 1), 1),
        }


def _shifts_for(day: int, start_weekday: int, night_line: bool) -> list[str]:
    wd = (start_weekday + day) % 7  # 0 = Monday
    if wd == 6:
        return []
    if wd == 5:
        return ["A"]
    return ["A", "B", "C"] if night_line else ["A", "B"]


def generate_plant(n_machines: int = 40, days: int = 30, seed: int = 7,
                   start: str = "2026-10-05", plant_id: str | None = None) -> Plant:
    """Generate one plant. ``n_machines`` can also be a preset key: 'S', 'M', 'L'."""
    if isinstance(n_machines, str):
        n_machines = PRESETS[n_machines]
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    start_wd = start_ts.weekday()
    n = n_machines
    n_lines = max(1, int(np.ceil(n / 8)))
    night_line_idx = min(2, n_lines - 1)

    # ---------------- machines ----------------
    line_idx = np.arange(n) // 8
    mtype = rng.choice(["mill", "lathe", "5axis"], n, p=[0.45, 0.35, 0.20])
    base_cycle = np.clip(rng.lognormal(np.log(150), 0.45, n), 45, 900)
    tool_life = rng.integers(120, 320, n)
    pm_offset = rng.integers(0, 21, n)  # preventive maintenance every 21 days

    # planted faults
    order = rng.permutation(n)
    n_wear = max(1, round(0.10 * n))
    n_step = max(1, round(0.06 * n))
    n_cool = max(1, round(0.06 * n))
    wear_m = order[:n_wear]
    step_m = order[n_wear:n_wear + n_step]
    cool_m = order[n_wear + n_step:n_wear + n_step + n_cool]
    onset = {int(m): int(days - rng.integers(3, 7)) for m in np.concatenate([wear_m, step_m, cool_m])}

    # programs per machine: (cycle multiplier, load factor, rpm, intro day)
    programs = []
    for m in range(n):
        k = int(rng.integers(3, 7))
        intro = np.zeros(k, dtype=int)
        if rng.random() < 0.5:
            intro[-1] = int(rng.integers(days // 3, days - 3))
        programs.append({
            "mult": rng.uniform(0.7, 1.4, k),
            "load": rng.uniform(0.4, 0.9, k),
            "rpm": rng.uniform(3000, 12000, k),
            "intro": intro,
        })

    machines = pd.DataFrame({
        "machine_id": [f"M{m + 1:03d}" for m in range(n)],
        "line_id": [f"L{li + 1}" for li in line_idx],
        "machine_type": mtype,
        "nominal_cycle_s": base_cycle.round(1),
        "tool_life_cycles": tool_life,
        "programs": [len(p["mult"]) for p in programs],
        "night_shift": line_idx == night_line_idx,
    })

    # ---------------- simulation ----------------
    cyc_parts, stop_parts = [], []
    supplier_by_mday = rng.choice(len(SUPPLIERS), size=(n, days), p=SUPPLIER_P)
    operators_by_line_shift_day = rng.integers(0, 12, size=(n_lines, 3, days))
    shift_code = {"A": 0, "B": 1, "C": 2}

    for m in range(n):
        P = programs[m]
        tool_age = int(rng.integers(0, tool_life[m]))
        prog = 0
        for day in range(days):
            for sh in _shifts_for(day, start_wd, line_idx[m] == night_line_idx):
                avail_progs = np.flatnonzero(P["intro"] <= day)
                changeover = rng.random() < 0.45 or prog not in avail_progs
                if changeover:
                    # newly introduced programs get pulled in as soon as they exist
                    fresh = avail_progs[P["intro"][avail_progs] == day]
                    prog = int(fresh[0]) if len(fresh) else int(rng.choice(avail_progs))
                prog_new = P["intro"][prog] > 0 and day - P["intro"][prog] < 3
                days_since_pm = (day + pm_offset[m]) % 21
                supplier = supplier_by_mday[m, day]
                drift = 1.0
                fo_drop = 0.0
                load_mult = 1.0
                if m in onset and day >= onset[m]:
                    dd = day - onset[m] + 1
                    if m in wear_m:
                        drift = 1 + min(0.013 * dd, 0.12)
                        fo_drop = 100 * (1 - 1 / drift)
                    elif m in step_m:
                        drift = 1.10
                        load_mult = 0.90
                cool_h = 1.0
                if m in cool_m and day >= onset[m]:
                    cool_h = 1 - min(0.05 * (day - onset[m] + 1), 0.4)

                cyc, stp, tool_age = _simulate_shift(
                    rng, m=m, day=day, shift=sh, cnom=base_cycle[m] * P["mult"][prog],
                    tool_life=int(tool_life[m]), tool_age0=tool_age, changeover=changeover,
                    prog_new=bool(prog_new), supplier=int(supplier), days_since_pm=int(days_since_pm),
                    drift=drift,
                )
                if cyc is not None:
                    k = len(cyc["t_end"])
                    cyc.update({
                        "m": np.full(k, m), "day": np.full(k, day), "shift": np.full(k, shift_code[sh]),
                        "prog": np.full(k, prog), "load": np.full(k, P["load"][prog] * load_mult),
                        "rpm": np.full(k, P["rpm"][prog]), "fo_drop": np.full(k, fo_drop),
                        "cool_h": np.full(k, cool_h), "nominal": np.full(k, base_cycle[m] * P["mult"][prog]),
                    })
                    cyc_parts.append(cyc)
                if stp is not None:
                    k = len(stp["t_start"])
                    stp.update({
                        "m": np.full(k, m), "day": np.full(k, day), "shift": np.full(k, shift_code[sh]),
                        "prog": np.full(k, prog), "supplier": np.full(k, supplier),
                        "operator": np.full(k, operators_by_line_shift_day[line_idx[m], shift_code[sh], day]),
                    })
                    stop_parts.append(stp)

    C = {k: np.concatenate([p[k] for p in cyc_parts]) for k in cyc_parts[0]}
    S = {k: np.concatenate([p[k] for p in stop_parts]) for k in stop_parts[0]}

    cycles = _build_cycles(rng, C, machines, programs)
    stops = _build_stops(rng, S, machines, cycles)
    hours = _build_hours(rng, C, S, machines, supplier_by_mday, operators_by_line_shift_day, line_idx,
                         pm_offset, programs, start_wd, days)

    mid = machines["machine_id"].to_numpy()
    truth = {
        "drift_wear": sorted(mid[wear_m].tolist()),
        "drift_step": sorted(mid[step_m].tolist()),
        "coolant_fault": sorted(mid[cool_m].tolist()),
        "onset_day": {mid[m]: d for m, d in onset.items()},
        "loss_drivers": ["supplier", "days_since_pm", "shift", "tool_age"],
        "null_drivers": ["operator_id", "coolant_conc", "weekday"],
    }
    return Plant(plant_id=plant_id or f"plant-{n}m-s{seed}", start=start_ts, days=days,
                 machines=machines, stops=stops, cycles=cycles, hours=hours, truth=truth)


def _simulate_shift(rng, *, m, day, shift, cnom, tool_life, tool_age0, changeover, prog_new,
                    supplier, days_since_pm, drift):
    s0, s1 = SHIFT_WINDOWS[shift]
    span = s1 - s0
    n_est = int(span / cnom) + 4
    c = (cnom * rng.lognormal(0.0, 0.02, n_est) + rng.gamma(2.0, 4.0, n_est)) * drift
    age = (tool_age0 + np.arange(n_est)) % tool_life
    wear = age / tool_life

    pos, dur, kind = [], [], []
    if changeover:
        pos += [0, 0]
        dur += [float(rng.lognormal(np.log(40 * 60 * (1.6 if prog_new else 1.0)), 0.35)),
                float(max(120.0, rng.normal(8 * 60, 90)))]
        kind += ["setup", "first_article"]
    tc = np.flatnonzero((age == 0) & (np.arange(n_est) > 0))
    pos += tc.tolist()
    dur += np.clip(rng.normal(6 * 60, 80, len(tc)), 120, None).tolist()
    kind += ["tool_change"] * len(tc)

    s3 = supplier == 2
    hourly = c / 3600.0
    rates = {
        "micro": 1.6 * (1 + 1.5 * wear ** 2) * (1.5 if s3 else 1.0),
        "quality_check": np.full(n_est, 0.08 * (2.0 if s3 else 1.0) * (1.5 if prog_new else 1.0)),
        "material_wait": np.full(n_est, 0.10 * (3.0 if shift == "C" else 1.0)),
        "breakdown": np.full(n_est, 0.006 * (1 + days_since_pm / 7)),
    }
    for k, r in rates.items():
        hit = np.flatnonzero(rng.random(n_est) < r * hourly)
        if not len(hit):
            continue
        if k == "micro":
            d = np.clip(rng.lognormal(np.log(75), 0.5, len(hit)), 20, 300)
        elif k == "quality_check":
            d = np.clip(rng.normal(8 * 60, 120, len(hit)), 120, None)
        elif k == "material_wait":
            d = np.clip(rng.lognormal(np.log(12 * 60), 0.6, len(hit)), 120, None)
        else:
            d = np.clip(rng.lognormal(np.log(75 * 60), 0.7, len(hit)), 15 * 60, None)
        pos += hit.tolist()
        dur += d.tolist()
        kind += [k] * len(hit)

    def timeline(pos, dur):
        pos = np.asarray(pos, dtype=np.int64)
        dur = np.asarray(dur, dtype=float)
        o = np.argsort(pos, kind="stable")
        pos, dur = pos[o], dur[o]
        cum_c = np.concatenate([[0.0], np.cumsum(c)])
        cum_d = np.cumsum(dur)
        prev_d = cum_d - dur
        stop_start = cum_c[pos] + prev_d
        idx = np.searchsorted(pos, np.arange(n_est), side="right")
        added = np.where(idx > 0, cum_d[np.maximum(idx - 1, 0)] if len(cum_d) else 0.0, 0.0)
        cyc_start = cum_c[:-1] + added
        return o, stop_start, cyc_start

    o, stop_start, cyc_start = timeline(pos, dur)
    meal_rel = MEAL_AT[shift] - s0
    i_meal = int(np.searchsorted(cyc_start, meal_rel))
    if i_meal < n_est:
        pos.append(i_meal)
        dur.append(float(rng.normal(45 * 60, 4 * 60)))
        kind.append("meal")
        o, stop_start, cyc_start = timeline(pos, dur)

    cyc_end = cyc_start + c
    keep = cyc_end <= span
    dur_s = np.asarray(dur)[o]
    kind_s = np.asarray(kind)[o]
    pos_s = np.asarray(pos)[o]
    ks = stop_start < span
    dur_s = np.minimum(dur_s[ks], span - stop_start[ks])
    t_base = day * DAY + s0
    n_done = int(keep.sum())
    tool_age_next = int((tool_age0 + n_done) % tool_life)

    cyc = None
    if n_done:
        cyc = {
            "t_end": t_base + cyc_end[keep], "cycle_s": c[keep], "tool_age": age[keep],
            "wear": wear[keep], "t_in_shift": cyc_start[keep],
        }
    stp = None
    if ks.any():
        # cycles since previous stop: distance in cycle index to the previous stop position
        p = pos_s[ks]
        prev = np.concatenate([[0], p[:-1]])
        stp = {
            "t_start": t_base + stop_start[ks], "duration_s": dur_s, "kind": kind_s[ks],
            # wear of the cycle that ran just before the stop (a tool change resets wear to 0 after it)
            "cycles_since_prev": (p - prev).astype(float), "wear_at_stop": wear[np.clip(p - 1, 0, n_est - 1)],
        }
    return cyc, stp, tool_age_next


def _build_cycles(rng, C, machines, programs) -> pd.DataFrame:
    k = len(C["t_end"])
    L, w = C["load"], C["wear"]
    th = 1 - np.exp(-C["t_in_shift"] / 5400.0)
    amb = np.sin(2 * np.pi * ((C["t_end"] % DAY) / DAY - 0.375))
    h = C["cool_h"]
    nz = lambda s: rng.normal(0, s, k)  # noqa: E731
    sig = {
        "spindle_load": 100 * (0.5 * L + 0.15 * w + 0.03 * th) + nz(2.0),
        "spindle_speed": C["rpm"] * (1 - 0.02 * w) + nz(50),
        "feed_override": 100 - C["fo_drop"] + nz(1.0),
        "axis_load_x": 30 * L * (1 + 0.2 * w) + nz(1.0),
        "axis_load_y": 26 * L * (1 + 0.2 * w) + nz(1.0),
        "axis_load_z": 18 * L * (1 + 0.3 * w) + nz(1.0),
        "motor_current": 8 + 20 * L + 4 * w + nz(0.5),
        "vibration_rms": 1.2 + 1.5 * w + 0.5 * L + nz(0.15),
        "spindle_temp": 30 + 8 * th + 6 * L + 2 * amb + 6 * (1 - h) / 0.4 + nz(0.5),
        "coolant_pressure": 20 * h + nz(0.4),
        "coolant_temp": 22 + 3 * th + 2 * amb + 5 * (1 - h) / 0.4 + nz(0.3),
        "power_kw": 3 + 12 * L + 1.5 * w + nz(0.3),
    }
    mid = machines["machine_id"].to_numpy()
    df = pd.DataFrame({
        "cycle_id": np.arange(k, dtype=np.int64),
        "machine_id": mid[C["m"]],
        "program_id": [f"{mid[m]}-P{p + 1}" for m, p in zip(C["m"], C["prog"])],
        "day": C["day"].astype(np.int16),
        "shift": np.array(["A", "B", "C"])[C["shift"]],
        "t_end": C["t_end"].round(1),
        "cycle_s": C["cycle_s"].round(2),
        "nominal_cycle_s": C["nominal"].round(2),
        "tool_age": C["tool_age"].astype(np.int32),
        **{s: v.round(3) for s, v in sig.items()},
    })
    return df.sort_values(["machine_id", "t_end"], kind="stable").reset_index(drop=True)


def _build_stops(rng, S, machines, cycles) -> pd.DataFrame:
    k = len(S["t_start"])
    mid = machines["machine_id"].to_numpy()
    line = machines["line_id"].to_numpy()
    kind = S["kind"]
    always = np.isin(kind, list(REPORTED_ALWAYS))
    reported = np.where(always | (rng.random(k) < 0.4), kind, "")
    df = pd.DataFrame({
        "stop_id": np.arange(k, dtype=np.int64),
        "machine_id": mid[S["m"]],
        "line_id": line[S["m"]],
        "day": S["day"].astype(np.int16),
        "shift": np.array(["A", "B", "C"])[S["shift"]],
        "t_start": S["t_start"].round(1),
        "duration_s": S["duration_s"].round(1),
        "hour_of_day": ((S["t_start"] % DAY) / 3600.0).round(3),
        "cycles_since_prev": S["cycles_since_prev"],
        "spindle_load_before": (100 * (0.50 + 0.25 * S["wear_at_stop"] ** 2) + rng.normal(0, 3, k)).round(2),
        "tool_age_before": (S["wear_at_stop"] * 100).round(1),  # % of tool life used, from the CNC tool counter
        "supplier": SUPPLIERS[S["supplier"]],
        "operator_id": [f"OP{o + 1:02d}" for o in S["operator"]],
        "reason_reported": reported,
        "reason_true": kind,
    })
    return df.sort_values(["machine_id", "t_start"], kind="stable").reset_index(drop=True)


def _build_hours(rng, C, S, machines, supplier_by_mday, ops, line_idx, pm_offset, programs,
                 start_wd, days) -> pd.DataFrame:
    n = len(machines)
    H = days * 24
    key = lambda m, hr: m * H + hr  # noqa: E731

    # stop minutes, split exactly across hour bins; meal breaks are planned time, not loss
    def minutes_by_hour(kinds):
        sel = np.isin(S["kind"], list(kinds))
        s, e, m = S["t_start"][sel], S["t_start"][sel] + S["duration_s"][sel], S["m"][sel]
        out = np.zeros(n * H)
        b0 = (s // 3600).astype(np.int64)
        for j in range(int(((e - s) // 3600).max()) + 2 if len(s) else 0):
            b = b0 + j
            ov = np.clip(np.minimum(e, (b + 1) * 3600) - np.maximum(s, b * 3600), 0, None)
            ok = (ov > 0) & (b < H)
            np.add.at(out, key(m[ok], b[ok]), ov[ok] / 60.0)
        return out

    stop_min = minutes_by_hour(LOSS_KINDS)
    unplanned_min = minutes_by_hour(UNPLANNED_KINDS)
    planned_min = minutes_by_hour(PLANNED_KINDS)

    hr = (C["t_end"] // 3600).astype(np.int64)
    kk = key(C["m"], np.minimum(hr, H - 1))
    cyc_n = np.bincount(kk, minlength=n * H)
    cyc_loss = np.bincount(kk, weights=(C["cycle_s"] - C["nominal"]) / 60.0, minlength=n * H)
    tool_age_mean = np.bincount(kk, weights=C["tool_age"], minlength=n * H) / np.maximum(cyc_n, 1)
    shift_of = np.full(n * H, -1)
    shift_of[kk] = C["shift"]
    prog_of = np.full(n * H, -1)
    prog_of[kk] = C["prog"]
    # scheduled hours = hours where the machine produced or stood still on a loss
    sched = (cyc_n > 0) | (stop_min > 0)
    idx = np.flatnonzero(sched)
    mm, hh = idx // H, idx % H
    day = hh // 24
    hod = hh % 24
    sh = shift_of[idx]
    # hours with only stops: infer shift from hour of day
    inferred = np.where((hod >= 6) & (hod < 14), 0, np.where((hod >= 14) & (hod < 22), 1, 2))
    sh = np.where(sh < 0, inferred, sh)
    pnew = np.array([
        (p >= 0) and programs[a]["intro"][p] > 0 and d - programs[a]["intro"][p] < 3
        for a, p, d in zip(mm, prog_of[idx], day)
    ])
    lost = stop_min[idx] + np.clip(cyc_loss[idx], 0, None)
    # Tool age when no cycle finished in the hour: the CNC tool counter still holds the last value,
    # so carry it forward per machine instead of writing 0 (0 would leak "machine was down" into the feature).
    ta = np.where(cyc_n[idx] > 0, tool_age_mean[idx], np.nan)
    tool_age_hr = (pd.Series(ta).groupby(mm).ffill().groupby(mm).bfill().fillna(0).to_numpy()).round(1)
    df = pd.DataFrame({
        "machine_id": machines["machine_id"].to_numpy()[mm],
        "line_id": machines["line_id"].to_numpy()[mm],
        "machine_type": machines["machine_type"].to_numpy()[mm],
        "day": day.astype(np.int16),
        "hour_of_day": hod.astype(np.int8),
        "weekday": ((start_wd + day) % 7).astype(np.int8),
        "shift": np.array(["A", "B", "C"])[sh],
        "supplier": SUPPLIERS[supplier_by_mday[mm, day]],
        "operator_id": [f"OP{o + 1:02d}" for o in ops[line_idx[mm], sh, day]],
        "program_new": pnew,
        "tool_age": tool_age_hr,
        "days_since_pm": ((day + pm_offset[mm]) % 21).astype(np.int16),
        "ambient_temp": (24 + 4 * np.sin(2 * np.pi * (hod / 24 - 0.375)) + rng.normal(0, 0.8, len(idx))).round(2),
        "coolant_conc": rng.normal(7.0, 0.6, len(idx)).round(2),
        "cycles": cyc_n[idx],
        "stop_minutes": stop_min[idx].round(2),
        "lost_minutes": np.clip(lost, 0, 60).round(2),
        "unplanned_minutes": np.clip(unplanned_min[idx], 0, 60).round(2),
        "planned_minutes": np.clip(planned_min[idx], 0, 60).round(2),
        # unplanned stop minutes per hour of time the machine was available to run (exposure-normalised);
        # NaN when less than 15 min were available (the hour was mostly a setup or a meal)
        "unplanned_rate": np.where(60 - planned_min[idx] >= 15,
                                   np.clip(unplanned_min[idx] / np.maximum(60 - planned_min[idx], 1e-9) * 60, 0, 60),
                                   np.nan).round(2),
    })
    return df
