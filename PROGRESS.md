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

Outcome: `tests/test_bench.py` passed, including the Xeon bench JSON (plant M answer milliseconds unchanged) and a synthetic p50–p90 column. Thread caps are checked by `python -m bench.kernels --threads 1 --dump-threads`. The timed scratch run is recorded below once `results/_scratch` exists.
