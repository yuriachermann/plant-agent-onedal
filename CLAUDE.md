# CLAUDE.md

Companion repo for the oneAPI DevSummit talk on 1 Dec 2026. Read `README.md`, `PROGRESS.md` and
`CHANGELOG.md` before starting: they log what is done and every number measured so far.

## Rules

- Baseline first: `ruff check .`, `pytest -q`, `python -m bench.parity`. Log the results in `PROGRESS.md`.
- After every change run `pytest -q`. After any change under `tools/`, `python -m bench.parity` must
  PASS, and bump `TOOL_VERSION` in `tools/common.py`.
- Never loosen a tolerance, test assertion or threshold to make something pass. Never change
  `plantgen/` to make a tool score better.
- Every number you write in docs must come from a results file you produced; cite its path. Never
  hand-edit `results/`.
- One task per commit, conventional commit messages. Don't push.
- If a task is blocked or needs a decision from the owner, write it under "Decisions needed" in
  `PROGRESS.md` and move on.

## Repo map

- `plantgen/`: synthetic plant generator with planted faults and drivers (`plant.truth`), CSV I/O, and machine-hours derived from real exports.
- `tools/`: the four agent tools, the backend switch (`backend.py`, the only file that imports estimators), the result envelope and `TOOL_VERSION` (`common.py`), and `schemas.json`.
- `bench/`: parity gate (`parity.py`), benchmark (`run_bench.py`, `report.py`), kernel sweeps (`kernels.py`), detection scoring against planted truth (`evaluate.py`), slide table (`deck_numbers.py`) and one-off probes (`probe_*.py`).
- `demo/`: scripted agent run (`agent_demo.py`) and the citation check (`cite_check.py`).
- `tests/`: fast pytest suite; `pytest -m slow` reruns the seeds 1–10 evaluation grid.
- `docs/`: real-data column mapping (`real-data.md`) and investigation write-ups (`findings/`).
- `results/`: generated outputs only: `parity.json` (read by the tools at run time), `eval/`, `findings/`, and one benchmark directory per CPU tag.
- `.github/workflows/`: CI (ruff, pytest, size-S parity).
