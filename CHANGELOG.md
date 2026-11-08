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
