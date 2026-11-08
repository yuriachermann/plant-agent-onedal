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

## Decisions needed

1. **Null drivers: grouped holdout?** The shadow rule did not help because operator and weekday beat their shuffled copies in most runs. My guess, not tested: the holdout is a random split of hours, so hours from the same line, shift and day sit on both sides. Operator is drawn per line, shift and day, and weekday also marks days, so both can stand in for day-level effects such as the supplier drawn per machine and day. A holdout split by day would test this. It changes the tool's method and the slide numbers, so it is your call.
2. **Re-run the Xeon reference before the talk?** `results/intel-xeon-2t/` was measured with tool version 0.2.0. Version 0.4.0 adds the drift step test (more post-processing per call), and the slide plant (M seed 7) now reports 6 drifting machines instead of 4. The deck's timing table and any drift screenshot should come from a fresh `python -m bench.run_bench` on the Xeon.
3. **`main` is not fast-forwarded past the null-driver commit.** The main checkout has an uncommitted edit to `results/parity_report.md` (the `·` separators re-encoded and a blank line added before the JSON fence, which looks like an editor saving it). The parity runs in this session rewrite that file, so `git merge --ff-only` refuses. I did not touch your working copy. To bring `main` up to this branch: `git -C <main checkout> checkout -- results/parity_report.md`, then `git -C <main checkout> merge --ff-only claude/plant-agent-onedal-devsummit-354b94`.
4. **AC rerun of the stock-gap probe.** All task-4 timings ran on battery. Plug in and run `python -m bench.probe_stock_gap --repeats 3 --out results/findings/stock-gap-ac.json` and `python -m bench.run_bench --sizes M --repeats 3 --skip-kernels --out results/findings/stock-gap-bench-ac` to separate battery from the rest. I did not change any power setting.
5. **Edge run (plan 0.8) still open.** This machine is the i5-12450H laptop, not the Intel Core Ultra 5 NUC, so prompt task 5 was skipped.
