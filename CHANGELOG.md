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
- Drift recall (tool version 0.4.0): `detect_cycle_drift` adds a per-machine step test beside the kNN envelope. Each cycle time is divided by its program's reference median, and a one-sided CUSUM (slack 0.5σ, alarm at 5σ, σ from the 21 reference days) runs on the machine's daily median. It fires when the CUSUM alarms and the median since the alarm is at least 2% above the reference. A machine is flagged when either test fires; `step_test=False` turns it off. Flagged items carry `flagged_by` and `step`, and the fact gives the step change and dates. Seeds 1–10, sizes S and M, stock:

  | | Envelope only | Envelope + step test |
  |---|---:|---:|
  | Recall, mean (S / M) | 53.3% (45.0% / 61.7%) | 98.3% (100% / 96.7%) |
  | Recall, wear / step | 80.0% / 22.5% | 97.5% / 100% |
  | Recall, lowest plant | 0% | 66.7% |
  | Precision, mean and min | 100% (19 plants flagged) | 100% (20 plants flagged) |
  | Healthy machines flagged | 0% | 0% |

  No plant lost recall or precision. Sources: `results/eval/drift-envelope-only/summary.json` and `results/eval/summary.json`. `bench.evaluate` now reports drift recall per fault type and healthy-machine false positives, and `--drift-step-test off` reproduces the envelope-only grid. The slow test's drift recall floor moves up to 0.9833 − 0.10, and it gains a drift healthy-machine ceiling of 0.05.
- Stock gap, laptop vs Xeon: the 11.5 s stock answer from 0.5 does not reproduce. The same command now gives 6,011 ms stock and 983 ms oneDAL on battery (`results/findings/stock-gap-bench-battery/summary.json`). The remaining 6.1× vs 4.0× comes from oneDAL using the laptop's 12 threads, while stock pays thread-pool start-up on many small calls with `n_jobs=-1`. P-core vs E-core placement moves single runs. Native pools and Windows power throttling are not the cause. No tool change: every `n_jobs` value measured helps one backend and hurts the other. `bench/probe_stock_gap.py` repeats the measurement, and `bench.run_bench` now records AC or battery and the power plan on Windows. See `docs/findings/stock-gap.md`.
- Day holdout for drivers (tool version 0.5.0): `rank_loss_drivers` takes `holdout_by="day"`, which holds out a share of whole days instead of random hours. It was tested and stays off by default; `bench.evaluate --drivers-holdout` picks the split. Seeds 1–10, sizes S and M, stock:

  | | Hour split, 2σ / 20% (kept) | Day split, 2σ / 20% | Day split, shadow |
  |---|---:|---:|---:|
  | Runs with any null driver reported | 25% | 10% | 45% |
  | operator / weekday / coolant concentration reported | 20% / 15% / 0% | 5% / 5% / 0% | 20% / 15% / 25% |
  | Planted drivers in the top 3 | 71.25% | 66.25% | 68.75% |
  | Planted drivers reported | 80.0% | 50.0% | 80.0% |
  | tool age / supplier / days since PM / shift reported | 100% / 95% / 95% / 30% | 75% / 75% / 25% / 25% | 100% / 85% / 60% / 75% |

  Sources: `results/eval/summary.json`, `results/eval/drivers-holdout-day/summary.json` and `results/eval/drivers-shadow-holdout-day/summary.json`. The default grid is unchanged run for run.
- Stock gap on AC: the stock-gap probe and the size-M benchmark were rerun with the laptop plugged in (power mode Best performance; the battery runs were in Balanced). One answer takes 4,941 ms stock and 878 ms oneDAL (5.62×), against 6,011 and 983 ms on battery. Stock is still 20% slower than on the Xeon (4,104 ms), and `n_jobs=1` still trades a faster stock for a slower oneDAL (2,985 / 1,649 ms), so there is no tool change. Sources: `results/findings/stock-gap-bench-ac/summary.json` and `results/findings/stock-gap-ac.json`. See section 5 of `docs/findings/stock-gap.md`.
- EcoQoS and thread placement: on Windows, `bench/probe_stock_gap.py` now records which logical processor each case's main thread is on before and after every tool call (`main_thread`, `main_thread_cpus`). Six AC probes of the default case against the EcoQoS opt-out: without the opt-out, the main thread ran mostly on the E-cores in 10 of 12 processes. With it, the main thread stayed on the P-cores in all 12. Median answer: 6,344 vs 5,289 ms stock, 1,496 vs 868 ms oneDAL. This explains the run-to-run noise, not the gap to the Xeon. Sources: `results/findings/stock-gap-ecoqos-ac/run-1.json` to `run-6.json`. See section 6 of `docs/findings/stock-gap.md`. Whether `bench.run_bench` or the MCP server should opt out is decision 6 in `PROGRESS.md`.
- Benchmarks opt out of EcoQoS on Windows: `bench.run_bench` and `bench.kernels` call `SetProcessInformation` so Windows does not throttle the timed processes, and they record `ecoqos_opt_out`. `bench.run_bench` also records the share of on-demand calls whose main thread was on a P-core (`main_thread`), and `bench.report` prints both in the summary header. `--keep-ecoqos` restores the old behaviour. The Windows core helpers moved from the probe to `bench/hybrid.py`. In four size-M runs on AC, the opt-out runs kept their main threads mostly on the P-cores and gave a 5.94× and a 5.54× speed-up. Without the opt-out, one oneDAL process landed on the E-cores and its run's speed-up fell to 2.60× (`results/findings/bench-ecoqos-ac/`). Off Windows nothing changes.
- MCP server: `agent/mcp_server.py` exposes the four tools from `tools/schemas.json` over stdio with the official `mcp` SDK (v2 low-level `Server`), returning `tools.public(envelope)` as text and structured content. Unknown arguments come back as tool errors. `mcp` is an optional extra (`pip install -e ".[mcp]"`), and `tests/test_mcp_server.py` skips without it.
