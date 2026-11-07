# Running the tools on a plant export

The tools read four tables. `plantgen/io.py` lists every column the synthetic generator writes (`SCHEMA`). A real export does not have to include the context columns. `plantgen/derive.py` builds the machine-hour table from machines, stops and cycles when `hours.csv` is missing, and it leaves out anything it cannot see.

Load a directory with `plantgen.load_plant`. `reason_true` on stops is optional. It is the hidden label the generator uses for scoring. A real export will not have it.

## What has to be there

| Table | Required | If it is missing |
|---|---|---|
| `machines` | `machine_id`, `line_id`, `machine_type`, `nominal_cycle_s`, `night_shift` | The tools cannot scope a line or a machine. |
| `stops` | the `SCHEMA` stop columns, including `t_start` in seconds from `plant.json`'s `start` and `duration_s` | `cluster_stops` has nothing to group. |
| `cycles` | the `SCHEMA` cycle columns, including the 12 signal means | Drift and signal health have nothing to score. `tool_age` is optional and is used only if you have it. |
| `hours` | `machine_id`, `day`, `hour_of_day`, `unplanned_rate` | Derived. See below. |

`plant.json` needs `plant_id`, `start` (ISO timestamp) and `days`. `truth` is only for the synthetic scorer.

## Columns the hour table can and cannot grow

Derived when the source exists:

| Column | Where it comes from |
|---|---|
| `machine_id`, `line_id`, `machine_type` | `machines` |
| `day`, `hour_of_day`, `weekday`, `shift` | `t_start` / `t_end` and `start`. A hour with no cycle takes its shift from the hour of day: A 06–14, B 14–22, C otherwise. |
| `program_new` | True for the first three days after `program_id` is first seen on that machine, and never when that first day is day 0. **Assumption:** this is first-seen, not a hidden introduction date. A program that simply was not cut until day 3 still counts as new here. |
| `tool_age` | Mean of `cycles.tool_age` in the hour, else the last `stops.tool_age_before`, carried forward across empty hours. Cycles that finish after the last hour are counted in that hour. **Assumption** when the cycle column is absent. Timestamps rounded to 0.1 s can move a cycle across an hour boundary. |
| `unplanned_rate` | Unplanned stop minutes per available hour. With `reason_true`, the kinds match the generator. Without it, a blank `reason_reported` counts as unplanned. **Assumption:** meals and setups were logged. Planned stops left blank inflate the rate. |
| `supplier`, `operator_id` | Copied from stops of that machine-day, only when those columns exist. A day with no stop is left empty, not guessed. |

Never created, because nothing in the three tables holds them:

`days_since_pm`, `ambient_temp`, `coolant_conc`.

`rank_loss_drivers` fits on whichever of its features are present and lists the rest in `result.missing_features`. It does not impute them. The model fact (`#M`) names them when the list is not empty.

## Example export

This is a generic TimescaleDB shape, not a vendor schema. Time is `timestamptz`. The plant clock in the CSVs is seconds from `plant.json` `start`. Adjust the zone before you subtract.

```sql
-- Assumption: one row per stop in machine_events, and one row per finished
-- cycle in machine_cycles. Signal columns are already cycle means.
-- Assumption: reason_code is the operator's reason, '' when they left it blank.
-- Assumption: 'meal' and 'setup' are the planned reasons your plant always logs.
--   Anything else blank is treated downstream as unplanned (see above).

-- machines.csv
SELECT machine_id, line_id, machine_type, nominal_cycle_s, night_shift
FROM machines
ORDER BY machine_id;

-- stops.csv
SELECT
    stop_id,
    machine_id,
    line_id,
    floor(extract(epoch FROM (started_at - timestamptz '2026-10-05 00:00+00')) / 86400)::int AS day,
    shift,
    extract(epoch FROM (started_at - timestamptz '2026-10-05 00:00+00')) AS t_start,
    extract(epoch FROM (ended_at - started_at)) AS duration_s,
    extract(hour FROM started_at) + extract(minute FROM started_at) / 60.0 AS hour_of_day,
    cycles_since_prev,
    spindle_load_before,
    tool_age_before,
    coalesce(reason_code, '') AS reason_reported
FROM machine_events
WHERE kind = 'stop'
ORDER BY machine_id, started_at;

-- cycles.csv
SELECT
    cycle_id,
    machine_id,
    program_id,
    floor(extract(epoch FROM (ended_at - timestamptz '2026-10-05 00:00+00')) / 86400)::int AS day,
    shift,
    extract(epoch FROM (ended_at - timestamptz '2026-10-05 00:00+00')) AS t_end,
    cycle_s,
    nominal_cycle_s,
    spindle_load, spindle_speed, feed_override,
    axis_load_x, axis_load_y, axis_load_z,
    motor_current, vibration_rms, spindle_temp,
    coolant_pressure, coolant_temp, power_kw
FROM machine_cycles
ORDER BY machine_id, ended_at;
```

Skip `hours.csv`. `load_plant` builds it. If you already roll hours up in the database, export `unplanned_rate` and any of the driver columns you really have, and leave the others out of the file.
