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
