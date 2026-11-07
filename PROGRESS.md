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
