# Progress

Working log for the oneAPI DevSummit 2026 companion repo. Every number here comes from a command run in this checkout.

## Baseline (before any code change)

Date: 2026-11-07. Host: Windows, Python 3.12.2 in `.venv`. CPU: 12th Gen Intel Core i5-12450H, 12 logical processors, 63.7 GiB RAM. This is not the edge-class target (Intel Core Ultra 5 NUC or a plant PC), so task 0.8 does not run here.

- `pytest -q`: 11 passed in 63.05s.
- `python -m bench.parity`: PASS in 48.5s. Record: `results/parity.json`, checked_at `2026-11-07T16:43:45+00:00`. scikit-learn 1.9.1, scikit-learn-intelex 2026.2.0, tool version 0.1.0. All four tools passed on the plant and line scopes, and the oneDAL repeat matched the first digest.
  - `cluster_stops`: family ARI 1.0, noise agreement 1.0. Plant DBSCAN path `stock-tree`, line path `onedal-brute`.
  - `detect_cycle_drift`: flagged machines match (plant: M003, M006, M022, M028). Max share difference 0.0.
  - `rank_loss_drivers`: top 3 are tool_age, supplier, days_since_pm. Spearman 0.9301 (plant) and 0.8807 (line). Absolute R² difference 0.0017 and 0.007.
  - `signal_health`: flagged machines match. Max share difference 0.0. The dispatch log is empty because every PCA fit stayed on stock (under the 8,000-row route).

## 0.1 Repo hygiene

Plan: add `pyproject.toml` with project metadata, ruff, and the pytest config now in `pytest.ini`. Add `requirements-lock.txt` from `pip freeze`, plus `CHANGELOG.md` and this file. Leave the license unset. Select ruff rules E/F/I/UP/B and ignore the ones that would reflow the existing modules (long fact sentences, lambdas that `best_of` calls immediately, equal-length zips). Fix whatever that selection still reports.

Files: `pyproject.toml`, `requirements-lock.txt`, `CHANGELOG.md`, `PROGRESS.md`, `pytest.ini` (remove), `README.md`, and any file ruff still flags.

Outcome: `ruff check .` clean. `pytest -q` 11 passed in 24.39s. `python -m bench.parity` PASS (metrics unchanged from the baseline record; the rewritten `results/parity.json` was restored so a new timestamp is not committed). Ruff selection is E/F/I/UP/B. Fixes were an f-string with no placeholder, `datetime.UTC`, and import order in `bench/kernels.py`. `tools/signals.py` import order ended up identical to HEAD.

## 0.2 CI

Plan: add `.github/workflows/ci.yml`. Ubuntu matrix Python 3.11 and 3.13 installs the requirements, runs ruff, pytest, then `python -m bench.parity --size S --out "$RUNNER_TEMP"`. Windows job on Python 3.12 runs pytest only. Confirm the same commands pass locally, with parity pointed at a temp directory.

Files: `.github/workflows/ci.yml`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: workflow file written. Locally `ruff check .` was already clean and `pytest -q` was already green in 0.1. `python -m bench.parity --size S --out` a temp directory PASS (exit 0); it did not write `results/`. GitHub itself was not executed from this machine.

## 0.3 Multi-seed evaluation

Plan: score drift, signal health, loss drivers and stop suggestions against `plant.truth` for seeds 1–10 and sizes S and M. Write `results/eval/summary.md` and `.json`. Add a slow test whose floors are the measured means minus 0.10.

Files: `bench/evaluate.py`, `tests/test_evaluate.py`, `results/eval/`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: `results/eval/summary.json`, stock scikit-learn, 20 plants. Drift precision 100% on the 19 plants that flagged anyone; recall mean 0.5333 (0.45 on S, 0.6167 on M). Coolant recall mean 0.9, healthy-machine false positives 0. Planted drivers in the top 3: mean 0.7125 (shift only 0.10). Any null driver reported: 0.25 of runs. Stop coverage mean 0.8724 by count, accuracy 0.965 by count and 0.6145 by minutes. Plant M seed 7: coverage 0.8537 (85.4%), accuracy 0.9799 (98.0%), accuracy by minutes 0.6723 (67.2%). `pytest -m slow` passed in 289.12s.

## 0.4 Real-data path

Plan: derive machine-hours from stops, cycles and machines, leaving context columns absent. Teach `rank_loss_drivers` to fit on any subset and list the missing features. Document the column map and a generic TimescaleDB export. Test a synthetic plant with context dropped, and the case where `hours.csv` is absent.

Files: `plantgen/derive.py`, `plantgen/io.py`, `plantgen/__init__.py`, `tools/drivers.py`, `tools/common.py`, `tools/schemas.json`, `tests/test_derive.py`, `docs/real-data.md`, `results/parity.json` (tool version 0.2.0), `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: derived `unplanned_rate` matches the generator to the stored 0.01 precision (re-splitting rounded durations). Shifts match. `program_new` is first-seen, which is not the generator's hidden intro day. `pytest -q`: 25 passed, 1 deselected, in 37.80s. `python -m bench.parity` PASS, same metrics as the baseline (plant Spearman 0.9301, ΔR² 0.0017, ARI 1.0). Tool version is now 0.2.0.

## 0.5 Benchmark robustness

Plan: add `--seeds` (default remains the single `--seed`, 7), report p50 and p90 beside the existing median, and run kernel sweeps at 1, 2, and all threads in a subprocess that sets OMP/TBB/MKL/OpenBLAS before NumPy imports. Keep the old JSON keys. Re-rendering `results/intel-xeon-2t` bench JSON must still produce the same answer milliseconds.

Files: `bench/run_bench.py`, `bench/report.py`, `bench/kernels.py`, `tests/test_bench.py`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: `tests/test_bench.py` passed, including the Xeon bench JSON (plant M answer milliseconds unchanged) and a synthetic p50–p90 column. Thread caps are checked by `python -m bench.kernels --threads 1 --dump-threads`.

Scratch run after that commit: `python -m bench.run_bench --sizes M --repeats 3 --out results/_scratch` (gitignored), exit 0 in 737s. From `results/_scratch/summary.md`, plant M, seed 7, 3 repeats: one answer 11,500 → 1,721 ms (p50–p90 stock 168–187 / 2,955–3,069 / 7,936–7,946 / 440–442 ms by tool). Suggested routing on this CPU: `PLANT_TOOLS_DBSCAN_BRUTE_MAX=80000`, `PLANT_TOOLS_PCA_MIN_ROWS=8000`. The Xeon file suggests 20000 and 8000 (`results/intel-xeon-2t/summary.md`). Thread sweep speed-ups at the largest input, 1 / 2 / 12 threads: dbscan 0.4 / 0.43 / 0.49, knn 1.76 / 2.78 / 4.41, pca 2.43 / 3.01 / 3.1.

## 0.6 Windows

Plan: read CPU and RAM through PowerShell CIM, skip `/proc` on Windows, and emit ANSI only when stdout is a terminal. Write UTF-8 so σ, × and arrows survive cp1252 consoles.

Files: `bench/run_bench.py` (machine info, in the 0.5 commit), `demo/agent_demo.py`, `bench/parity.py`, `bench/evaluate.py`, `tests/test_platform.py`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: `tests/test_platform.py` passed on this host (12th Gen Intel Core i5-12450H, memory reported). `python -m demo.agent_demo --size S --backend stock --line L1` with stdout redirected: exit 0, no ESC bytes, citation check passed. The default line L3 is absent on a size-S plant (two lines), so that combination still has an empty driver scope.

## 0.7 Deck numbers

Plan: `bench/deck_numbers.py` reads a benchmark directory and writes `deck_numbers.md` with each slide figure and its JSON key. Timings come from that directory's `summary.json`. Parity comes from `results/parity.json` when the tag has none. Stop suggestions for M seed 7 come from `results/eval/summary.json`.

Files: `bench/deck_numbers.py`, `tests/test_deck.py`, `results/intel-xeon-2t/deck_numbers.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: `tests/test_deck.py` passed. The generated table matches the talk figures: one answer S 3,775 → 591 ms, M 4,104 → 1,026 ms, L 4,222 → 1,041 ms; plant M tools 51 → 37, 1,003 → 336, 2,941 → 532, 109 → 122 ms; stop suggestions 85.4%, 98.0%, 67.2%.

## 0.8 Edge run

This machine is a 12th Gen Intel Core i5-12450H (12 logical processors), not an Intel Core Ultra 5 NUC and not a plant edge PC. The full edge benchmark was not run. The size-M scratch numbers above are a laptop reference only. Routing differs from the Xeon: DBSCAN brute max 80,000 here versus 20,000 in `results/intel-xeon-2t/summary.md`. PCA minimum stays 8,000 on both.

## 1.1 signal_health slowdown

Plan: in fresh processes, time stock `signal_health` with and without sklearnex imported, with and without a prior oneDAL forest fit, and with thread env vars set to 1. Record threadpoolctl pools. Change the tool only if a clean fix shows up and parity still passes.

Files: `bench/probe_signal_health.py`, `results/findings/signal-health-slowdown.json`, `docs/findings/signal-health-slowdown.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: not found on this CPU. Stock line-L1 median 328.95 ms; after `import sklearnex` 311.34 ms; after a forest fit 310.63 ms. At 1 thread: 305.2, 205.05, 217.31 ms. threadpoolctl showed OpenBLAS and MSVC OpenMP only, no TBB pool. The on-demand tool in `results/_scratch/summary.md` is faster in the oneDAL process (440 → 284 ms), the opposite of the Xeon (109 → 122 ms). No code change.

## 1.4 DBSCAN heuristic

Plan: run the current `eps` / `min_samples` rule on seeds 1–10, sizes S and M, whole plant and line L1. Report irregular share and suggestion accuracy. Change the rule only if those move around.

Files: `bench/probe_dbscan.py`, `results/findings/dbscan-heuristic.json`, `docs/findings/dbscan-heuristic.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: no change. Irregular share within a scope spans less than 2 points (plant M 24.1–25.0%, line scopes about 10.6–12.2%). Count accuracy stays at or above 92.8%. Plant M is on the eps floor 0.5 with `min_samples` 30. Minute-weighted accuracy still spans roughly 50–80%, which matches the evaluation grid and is not an eps jump. Table: `docs/findings/dbscan-heuristic.md`.

## Session baseline, 2026-11-08

Host: the same 12th Gen Intel Core i5-12450H (8 cores, 12 logical processors, 63.7 GiB RAM), Windows 11, Python 3.12.2 from the main checkout's `.venv`. Power plan "High performance", on AC power. scikit-learn 1.9.1, scikit-learn-intelex 2026.2.0, tool version 0.2.0. Start commit `87f4959`.

- `ruff check .`: all checks passed.
- `pytest -q`: 25 passed, 1 deselected, in 46.33s.
- `python -m bench.parity`: PASS in 45.7s wall. Metrics unchanged from the record in `results/parity.json`: ARI 1.0 and noise agreement 1.0 on both scopes, drift flags M003, M006, M022, M028 with max share difference 0.0, drivers top 3 tool_age, supplier, days_since_pm with Spearman 0.9301 (plant) and 0.8807 (line) and ΔR² 0.0017 and 0.007, signal_health flags M002, M003, M006, M020, M022, M028 with max share difference 0.0. Only `checked_at` changed, so the rewritten `results/parity.json` and `results/parity_report.md` were restored.

## CLAUDE.md

Outcome: `CLAUDE.md` holds the session rules and a one-line-per-folder map. `pytest -q`: 25 passed, 1 deselected.

## 1.2 Null drivers

Plan: add a shadow-feature rule to `rank_loss_drivers` beside the current one. Each feature gets a copy with its rows shuffled, the forest is fitted on features and copies, and a driver is reported only if its permutation importance is above zero and above the strongest copy's. Score both rules with `bench.evaluate` on seeds 1–10, sizes S and M. Keep the rule with fewer null reports, as long as planted drivers do not drop out. Add "planted driver reported" to the evaluation, since a rule that reports more can hide behind top-3 rates.

Files: `tools/drivers.py`, `tools/common.py` (tool version 0.3.0), `bench/evaluate.py`, `tests/test_tools.py`, `results/eval/` (rerun), `results/eval/drivers-shadow/`, `results/parity.json`, `results/parity_report.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: shadow rule rejected. The default is still the 2σ / 20% rule.

- Current rule, `results/eval/summary.json`: nulls reported in 25% of runs (operator 20%, weekday 15%, coolant concentration 0%). Planted drivers in the top 3 71.25%, reported 80.0% (shift 30%). The rerun reproduces every per-run value of the 2026-11-07 grid. One size-M mean, stop coverage by minutes, now rounds to 0.6742 instead of 0.6743 (unrounded 0.67425).
- Shadow rule, `results/eval/drivers-shadow/summary.json`: nulls in 55% of runs (operator 35%, weekday 30%, coolant concentration 5%). Planted drivers in the top 3 70.0%. It reports more planted drivers (90.0%, shift 65%), but it reports 5.7 of the top 6 rows per run against 4.05 for the current rule, so that is not selectivity. Days since PM in the top 3 falls from 85% to 60%, supplier from 90% to 80%. Shift rises from 10% to 40%.
- Drift, signal health and stop results are identical between the two runs.
- `python -m bench.parity` (tool version 0.3.0): PASS, same metrics as the baseline. The gate runs the default rule only. `pytest -q`: 26 passed, 1 deselected, in 36.71s.

## 1.3 Drift recall

Plan: add a per-machine step test beside the kNN score. Divide each cycle time by its program's reference median, take the machine's daily median, and run a one-sided CUSUM over the 7 test days in units of the reference days' standard deviation. Use the textbook slack 0.5σ and alarm 5σ, require the median since the alarm to be at least 2% above the reference, and skip days with fewer than 20 cycles. Flag a machine when either test fires. Merge only if recall rises across seeds and neither precision nor healthy-machine false positives get worse. Add per-type recall and drift healthy-machine false positives to `bench.evaluate`, and a switch that reruns the envelope alone as the baseline.

Files: `tools/drift.py`, `tools/common.py` (tool version 0.4.0), `tools/schemas.json`, `bench/evaluate.py`, `tests/test_tools.py`, `tests/test_evaluate.py`, `results/eval/` (rerun), `results/eval/drift-envelope-only/`, `results/parity.json`, `results/parity_report.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: merged.

- Envelope only, `results/eval/drift-envelope-only/summary.json`: same flagged machines, precision and recall per run as the committed 0.3.0 grid. Recall 53.3% (S 45.0%, M 61.7%), wear 80.0%, step 22.5%. Precision 100% on the 19 plants that flagged anyone. Healthy machines flagged 0%.
- Envelope + step test, `results/eval/summary.json`: recall 98.3% (S 100%, M 96.7%), wear 97.5%, step 100%, lowest plant 66.7% (M seed 3). Precision 100% on all 20 plants. Healthy machines flagged 0%. No coolant-fault machine flagged. Recall rose or stayed equal on every seed and size, and precision did not fall on any. Signal health, drivers and stops are unchanged.
- 78 of the 80 planted faults are flagged. The step test is the only detector for 32 of those 78 (`flagged_by` is `["step"]` in `results/eval/summary.json`).
- `tests/test_evaluate.py`: drift recall floor 0.9833 − 0.10 and a new drift healthy-machine ceiling of 0.05. I checked both assertions against `results/eval/summary.json` (the grid is deterministic). `pytest -m slow` itself was not rerun, because the laptop was on battery.
- `python -m bench.parity` (tool version 0.4.0): PASS. Drift now flags M003, M006, M018, M022, M028 and M040 on plant M seed 7 with both backends (max share difference 0.0). The other tools have the same metrics as before. `pytest -q`: 27 passed. `python -m demo.agent_demo --size M --backend onedal --line L1`: citation check passed with the new step-test sentences.
- During this task the laptop went from AC to battery. That changes timings only; the evaluation is deterministic.

## Stock gap, laptop vs Xeon (prompt task 4)

Plan: rerun the 0.5 command, then probe one factor at a time in fresh processes: `n_jobs` forced to 1 or 2, native pools at 1 thread, the process pinned to P-cores or E-cores, and an EcoQoS opt-out. Also time single-thread calibration loops on one P-core and one E-core, per-call estimator cost, and the stock 1-thread kernel sweep. Record the power plan and AC state. Change the tools only with a fix that treats both backends the same.

Files: `bench/probe_stock_gap.py`, `bench/run_bench.py` and `bench/report.py` (power state), `tests/test_platform.py`, `results/findings/stock-gap-battery.json`, `results/findings/stock-gap-ecoqos-battery.json`, `results/findings/stock-gap-bench-battery/`, `docs/findings/stock-gap.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: explained, no tool change. Everything ran on battery: the laptop was on AC at the session baseline and was unplugged before this task.

- The 0.5 command, plant M, gives 6,011 ms stock and 983 ms oneDAL (6.12×), against 11,500 and 1,721 ms in 0.5 (`results/findings/stock-gap-bench-battery/summary.json`). Both backends were slower on every tool in 0.5, so that run was a machine-state outlier, and its power state was not recorded.
- The remaining gap to the Xeon (4,104 / 1,026 ms, 4.0×) comes from stock: `n_jobs=-1` starts 12 workers on every small kNN query and forest predict. Forcing `n_jobs=1` takes the stock answer from 5,939 to 3,854 ms and oneDAL from 1,288 to 2,288 ms (`results/findings/stock-gap-battery.json`), so it is not a fair fix.
- Native pools at 1 thread: stock 6,057 ms, so OpenBLAS and OpenMP are not the cause. EcoQoS opt-out: 5,603 / 852 ms against 5,525 / 817 ms default in the same probe (`results/findings/stock-gap-ecoqos-battery.json`).
- A single-thread loop runs 1.9× slower on an E-core than on a P-core. Unpinned runs landed on either kind: the oneDAL answer was 1,288 ms in one probe and 817 ms in the next.
- `pytest -q`: 28 passed (the uncommitted MCP test skipped). No change under `tools/`, so no parity run was needed.

## 2.1 MCP server (optional)

Plan: a stdio server on the `mcp` v2 low-level `Server`. Tools and input schemas come from `tools/schemas.json`, with the shared `scope` definition inlined because MCP input schemas must stand alone. Calls run in a worker thread and return `tools.public(envelope)`. Keep `mcp` an optional extra and test with an in-process client.

Files: `agent/__init__.py`, `agent/mcp_server.py`, `tests/test_mcp_server.py`, `pyproject.toml` (`mcp` extra, `agent` package), `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: done. `mcp` 2.3.0 was installed into a scratch directory, not the project `.venv`, so the optional dependency stays optional there. With it on the path, `tests/test_mcp_server.py`: 2 passed. The in-process client lists the four tools, and each call matches a direct `tools.public` call on call id, facts and parity status. Unknown arguments return a tool error. A stdio smoke run, with the server as a subprocess on plant S seed 3, listed the four tools and returned a citable `signal_health` envelope. Without `mcp`, `pytest -q`: 28 passed, 1 skipped. No change under `tools/`.

## Null drivers: day holdout (decision 1)

Plan: add `holdout_by="day"` to `rank_loss_drivers` beside the current random-hour split. It holds out a 25% share of whole days, so no day has hours on both sides of the split. Score it with both reporting rules on seeds 1–10, sizes S and M. Make it the default only if null reports drop and planted drivers hold. Rerun the default grid to check that nothing else moved.

Files: `tools/drivers.py`, `tools/common.py` (tool version 0.5.0), `bench/evaluate.py` (`--drivers-holdout`), `tests/test_tools.py`, `results/eval/` (rerun), `results/eval/drivers-holdout-day/`, `results/eval/drivers-shadow-holdout-day/`, `results/parity.json`, `results/parity_report.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: day holdout rejected. The default stays the random-hour split with the 2σ / 20% rule. The laptop was on AC for this task.

- Default rerun, `results/eval/summary.json`: every run, the aggregate and the per-size blocks match the committed grid. The only change is the new `holdout_by` key.
- Day holdout, 2σ rule, `results/eval/drivers-holdout-day/summary.json`: a null driver is reported in 10% of runs instead of 25% (operator 5%, weekday 5%, coolant concentration 0%). Planted drivers reported fall from 80% to 50% (tool age 75%, supplier 75%, days since PM 25%, shift 25%), and planted drivers in the top 3 from 71.25% to 66.25% (days since PM 45% instead of 85%, shift 45% instead of 10%). It reports 2.45 rows per run against 4.05 (counted from `runs[].drivers.reported`), so most of the drop in nulls comes from reporting less overall.
- Day holdout, shadow rule, `results/eval/drivers-shadow-holdout-day/summary.json`: nulls in 45% of runs (operator 20%, weekday 15%, coolant concentration 25%). Planted drivers reported 80%, top 3 68.75%. That beats the shadow rule on the hour split (55% of runs, `results/eval/drivers-shadow/summary.json`) but not the default.
- The guess in decision 1 holds only in part. Operator and weekday are reported less when no day sits on both sides, but planted drivers that change once a day (days since PM, supplier) lose as much. One reading, not tested: with the hour split, any column that changes once a day can also act as a day marker, for planted and null drivers alike.
- Drift, signal health and stop results are identical in all three grids. `python -m bench.parity` (tool version 0.5.0): PASS, same metrics as before; the gate runs the default split only. `ruff check .` clean. `pytest -q`: 29 passed, 1 skipped (`mcp` not installed).

## Stock gap on AC (decision 4)

Plan: with the laptop plugged in, run the two commands from decision 4 back to back, then compare them with the battery runs and the Xeon. No code change.

Files: `results/findings/stock-gap-ac.json`, `results/findings/stock-gap-bench-ac/`, `docs/findings/stock-gap.md` (section 5), `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: battery accounts for part of the slow 0.5 run, but not for the gap to the Xeon. No tool change.

- Power state: AC online, plan "High performance". The probe recorded the Windows power mode overlay `ded574b5-45a0-4f42-8737-46345c09c238` (Best performance) on both its AC and its battery setting. The battery probe in `results/findings/stock-gap-ecoqos-battery.json` recorded Balanced (all zeros). I did not change any power setting, so this compares battery in Balanced mode with AC in Best performance.
- Benchmark, `results/findings/stock-gap-bench-ac/summary.json`: one answer takes 4,941 ms stock and 878 ms oneDAL (5.62×), against 6,011 and 983 ms on battery (6.12×, `results/findings/stock-gap-bench-battery/summary.json`). Every stock call is 16–22% faster. The Xeon's stock answer is 4,104 ms (`results/intel-xeon-2t/summary.json`), so stock here is still 20% slower, mostly in drift (1,540 vs 1,003 ms). oneDAL is now faster than the Xeon's 1,026 ms.
- Probe, `results/findings/stock-gap-ac.json`: the first case, the unpinned default, ran its single-threaded phases at E-core speed (5,558 / 1,622 ms). The P-core and EcoQoS opt-out cases ran at 4,768 / 916 and 4,760 / 840 ms. `n_jobs=1` gives 2,985 / 1,649 ms: the same trade-off as on battery.
- The one-shot calibration loop on a P-core took 142.85 ms, against 97.0 and 113.83 ms on battery. Two short JSON reads of mine (each under a second) ran in the probe's first 30 seconds, during the calibration and overhead steps, so that calibration may be off. The cases ran after that. The unpinned default oneDAL case, which ran at E-core speed, came after the stock default case.
- `ruff check .` clean. `pytest -q`: 29 passed, 1 skipped (`mcp` not installed). No change under `tools/`, so no parity run was needed.

## EcoQoS and thread placement (follow-up to decision 4)

Plan: make the probe record where the main thread runs, then repeat the default and EcoQoS opt-out cases six times on AC, alternating which runs first. Change only `bench/`; the tools stay as they are.

Files: `bench/probe_stock_gap.py` (`current_cpu`, `placement`, `main_thread` and `main_thread_cpus` per case), `tests/test_platform.py`, `results/findings/stock-gap-ecoqos-ac/run-1.json` to `run-6.json`, `docs/findings/stock-gap.md` (section 6), `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: EcoQoS is the main source of the run-to-run noise on this laptop, not of the gap to the Xeon.

- Without the opt-out, the main thread ran mostly on the E-cores in 10 of 12 processes, whether the default case ran first or second. With the opt-out, it ran on the P-cores in all 12 (83–100% of samples).
- Answer medians over the six runs: default 6,344 ms stock and 1,496 ms oneDAL; opt-out 5,289 and 868 ms. Within each run, the opt-out was faster on oneDAL in 6 of 6 runs and on stock in 5 of 6. The exception is run 3, where the default also ran on the P-cores.
- With the opt-out, stock never beat the Xeon's 4,104 ms (4,338–6,851 ms), so the `n_jobs` explanation of the gap stands.
- Runs 4–6 were slower than runs 1–3 in both cases (opt-out stock 6,099–6,851 against 4,338–4,479 ms). I ran nothing else during the series, and the cause is not recorded.
- Every process was started from the Claude desktop app's shell. A process started from a terminal in the foreground was not tested.
- `ruff check .` clean. `pytest -q`: 30 passed, 1 skipped (`mcp` not installed), with the new placement test in `tests/test_platform.py`. No change under `tools/`, so no parity run was needed.

## EcoQoS opt-out in the benchmark (decision 6, option a)

Plan: move the Windows core helpers from the probe into `bench/hybrid.py`. Make `bench.run_bench` and `bench.kernels` opt each timed process out of EcoQoS unless `PLANT_BENCH_ECOQOS=keep` (`--keep-ecoqos`). Record the opt-out and the main thread's P-core share, and print both in the report. Verify with size-M runs on AC, opt-out against keep. Change only `bench/`.

Files: `bench/hybrid.py` (new), `bench/probe_stock_gap.py` (imports the helpers), `bench/run_bench.py`, `bench/kernels.py`, `bench/report.py`, `tests/test_platform.py`, `tests/test_bench.py`, `results/findings/bench-ecoqos-ac/`, `docs/findings/stock-gap.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: done. Four size-M runs on AC, in the order opt-out, keep, keep, opt-out (`results/findings/bench-ecoqos-ac/<run>/summary.json`):

- Opt-out: main thread on P-cores 100% / 100% and 83% / 100% (stock / oneDAL). Answers 4,672 / 787 ms (5.94×) and 5,020 / 906 ms (5.54×).
- Keep: 83% / 8% and 79% / 100%. Answers 4,497 / 1,732 ms (2.60×) and 5,408 / 846 ms (6.39×). The oneDAL process that ran on the E-cores took about twice as long as the others.
- In this harness, Windows kept the stock process mostly on the P-cores even without the opt-out, unlike in the probe. Two runs per setting show the mechanism works, not a rate.
- The kernel sweeps were skipped in these runs. `bench.kernels` applies the same policy and records `ecoqos_opt_out` in its JSON, but no sweep has run with it yet.
- `ruff check .` clean. `pytest -q`: 32 passed, 1 skipped (`mcp` not installed), with new tests for the opt-out policy and the report header. No change under `tools/`, so no parity run was needed.

## EcoQoS opt-out in the MCP server (decision 6, option b)

Plan: have `agent/mcp_server.py` call the benchmark's opt-out at start-up, before the plant loads, with `--keep-ecoqos` to turn it off. Give `apply_ecoqos_policy` an explicit `keep`, so the benchmark's environment variable does not reach the server. Test without the `mcp` SDK. Then start the server over stdio in both modes and read its throttling state from outside.

Files: `agent/mcp_server.py`, `bench/hybrid.py`, `tests/test_platform.py`, `docs/findings/stock-gap.md`, `README.md`, `CHANGELOG.md`, `PROGRESS.md`.

Outcome: done.

- Stdio smoke run on plant S, seed 3, with `mcp` from the scratch install. `GetProcessInformation(ProcessPowerThrottling)` on the server's interpreter process gave ControlMask 1, StateMask 0 (execution speed controlled, not throttled) by default. With `--keep-ecoqos` it gave ControlMask 0. In both modes the server listed 4 tools, and `signal_health` returned the same call id, marked citable.
- A venv's `python.exe` on Windows is a launcher that starts the base interpreter as a child process. The server code runs in that child, which is where the opt-out applies. The launcher process stays at the Windows default.
- The server's effect on tool-call latency was not measured, here or on the NUC.
- `tests/test_mcp_server.py` with the scratch `mcp` install: 2 passed. `ruff check .` clean. `pytest -q`: 34 passed, 1 skipped (`mcp` not installed in the `.venv`), with new tests for the server's policy and for an explicit `keep`. No change under `tools/`, so no parity run was needed.

## Decisions needed

1. **Resolved 2026-11-09: grouped holdout tested and rejected.** See "Null drivers: day holdout". Holding out whole days cuts runs with a null driver from 25% to 10% but planted drivers reported from 80% to 50%, so the random-hour split stays. `holdout_by="day"` stays available, off by default.
2. **Re-run the Xeon reference before the talk?** `results/intel-xeon-2t/` was measured with tool version 0.2.0. Version 0.4.0 adds the drift step test (more post-processing per call), and the slide plant (M seed 7) now reports 6 drifting machines instead of 4. The tools are now at 0.5.0; the day-holdout option does not change the default path. The deck's timing table and any drift screenshot should come from a fresh `python -m bench.run_bench` on the Xeon.
3. **Resolved 2026-11-08: `main` fast-forwarded.** An uncommitted edit to `results/parity_report.md` in the main checkout (re-encoded `·` separators and a blank line) blocked the fast-forward. On your go-ahead it was discarded and `main` was fast-forwarded to the session branch.
4. **Resolved 2026-11-09: AC rerun done.** See "Stock gap on AC". Plugged in, the answer takes 4,941 ms stock and 878 ms oneDAL (5.62×). Stock is still 20% slower than on the Xeon, so the `n_jobs` explanation stands. One part is still open and needs you. The AC run was in power mode Best performance and the battery runs in Balanced, so battery and power mode are not separated. To separate them, set Settings > System > Power > Power mode to Balanced while plugged in, then rerun `python -m bench.run_bench --sizes M --repeats 3 --skip-kernels --out results/findings/stock-gap-bench-ac-balanced`. I won't change Windows settings myself.
5. **Edge run (plan 0.8) still open.** This machine is the i5-12450H laptop, not the Intel Core Ultra 5 NUC, so prompt task 5 was skipped.
6. **Resolved 2026-11-09: opt out of EcoQoS on Windows.** Without the opt-out, the probe's main thread ran mostly on the E-cores in 10 of 12 processes (`results/findings/stock-gap-ecoqos-ac/`).
   - (a) Done on your go-ahead. `bench.run_bench` and `bench.kernels` opt out on Windows and record it; `--keep-ecoqos` restores the old behaviour. See "EcoQoS opt-out in the benchmark". Laptop timings from before this change ran without the opt-out, so they do not compare directly.
   - (b) Done on your go-ahead. `agent/mcp_server.py` opts out at start-up; `--keep-ecoqos` restores the old behaviour. See "EcoQoS opt-out in the MCP server". The effect is still unmeasured on the NUC.
