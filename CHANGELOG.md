# Changelog

## Unreleased

- Repo hygiene: `pyproject.toml` (metadata, ruff, pytest), `requirements-lock.txt`, this file, and `PROGRESS.md`. License is still unset.
- CI: ruff, pytest, and a size-S parity run on Ubuntu (Python 3.11 and 3.13); pytest on Windows (Python 3.12). Parity writes to the runner temp directory.
- Detection quality over seeds 1–10 and sizes S and M, in `results/eval/summary.json`. `pytest -m slow` checks floors 0.10 below those means.
- Real exports: `plantgen/derive.py` builds machine-hours from stops, cycles and machines. Context columns that are not in the export stay absent. `rank_loss_drivers` (tool version 0.2.0) fits on whichever of its features are present and lists the rest in `result.missing_features`.
- Benchmarks report on-demand p50 and p90, accept `--seeds` without changing the default single seed, and sweep kernels at 1 thread, 2 threads, and all logical CPUs. Older bench JSON still loads.
- Windows: CPU and memory come from CIM via PowerShell, markdown and demo output are UTF-8, and ANSI colour is used only when stdout is a terminal.
- `bench/deck_numbers.py` writes the slide table for a benchmark directory. For `results/intel-xeon-2t` it reproduces the talk figures, including plant M seed 7 stop suggestions from `results/eval/summary.json`.
- Recorded a signal_health thread-pool probe on this 12-thread laptop. The Xeon slowdown did not appear, and no tool change was made. See `docs/findings/signal-health-slowdown.md`.
- DBSCAN `eps` / `min_samples` left unchanged. Across seeds 1–10 the irregular share stays inside a 2-point band per scope. See `docs/findings/dbscan-heuristic.md`.
- `CLAUDE.md` holds the working rules and a repo map for later sessions.
- Null drivers (tool version 0.3.0): `rank_loss_drivers` keeps its reporting rule (above 2σ of permutation noise and at least 20% of the top driver). A shadow-feature rule was tried and rejected: shuffled copies of every feature are fitted with the forest, and a driver is reported only if it beats the strongest copy. It stays available as `null_rule="shadow"`, off by default. `bench.evaluate --drivers-rule` picks the rule and now also reports how often each planted driver is reported. Seeds 1–10, sizes S and M, stock:

  | | 2σ / 20% rule (kept) | Shadow rule |
  |---|---:|---:|
  | Runs with any null driver reported | 25% | 55% |
  | operator / weekday / coolant concentration reported | 20% / 15% / 0% | 35% / 30% / 5% |
  | Planted drivers in the top 3 | 71.25% | 70.0% |
  | tool age / supplier / days since PM / shift in the top 3 | 100% / 90% / 85% / 10% | 100% / 80% / 60% / 40% |
  | Planted drivers reported | 80.0% | 90.0% |

  Sources: `results/eval/summary.json` and `results/eval/drivers-shadow/summary.json`.
