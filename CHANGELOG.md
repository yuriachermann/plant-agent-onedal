# Changelog

## Unreleased

- Repo hygiene: `pyproject.toml` (metadata, ruff, pytest), `requirements-lock.txt`, this file, and `PROGRESS.md`. License is still unset.
- CI: ruff, pytest, and a size-S parity run on Ubuntu (Python 3.11 and 3.13); pytest on Windows (Python 3.12). Parity writes to the runner temp directory.
- Detection quality over seeds 1–10 and sizes S and M, in `results/eval/summary.json`. `pytest -m slow` checks floors 0.10 below those means.
