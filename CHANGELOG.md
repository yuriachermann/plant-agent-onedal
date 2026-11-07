# Changelog

## Unreleased

- Repo hygiene: `pyproject.toml` (metadata, ruff, pytest), `requirements-lock.txt`, this file, and `PROGRESS.md`. License is still unset.
- CI: ruff, pytest, and a size-S parity run on Ubuntu (Python 3.11 and 3.13); pytest on Windows (Python 3.12). Parity writes to the runner temp directory.
- Detection quality over seeds 1–10 and sizes S and M, in `results/eval/summary.json`. `pytest -m slow` checks floors 0.10 below those means.
- Real exports: `plantgen/derive.py` builds machine-hours from stops, cycles and machines. Context columns that are not in the export stay absent. `rank_loss_drivers` (tool version 0.2.0) fits on whichever of its features are present and lists the rest in `result.missing_features`.
- Benchmarks report on-demand p50 and p90, accept `--seeds` without changing the default single seed, and sweep kernels at 1 thread, 2 threads, and all logical CPUs. Older bench JSON still loads.
