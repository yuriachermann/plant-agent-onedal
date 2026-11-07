# plant-agent-onedal

**oneDAL-accelerated analytics behind a manufacturing agent's tool interface.**

Companion repo for the talk at the oneAPI DevSummit 2026 (hosted by the UXL Foundation).

The agent's language model never computes a loss itself. It calls four tools, which cluster
machine stops, detect cycle-time drift, rank the drivers of lost production and check signal
health. It then quotes only the facts those tools return, each with a citation id. This repo has:

* **`plantgen/`**: a synthetic plant generator with planted faults and loss drivers, shaped like
  the data a CNC job shop's edge PC stores (stops, part cycles with 12 signals, machine-hours)
* **`tools/`**: the four agent tools in plain scikit-learn, switchable to oneDAL with one import
  (via the Intel Extension for Scikit-learn), with a result envelope that carries provenance and
  parity status, plus JSON tool schemas for the agent
* **`bench/`**: the parity gate (oneDAL vs stock, with tolerances), the benchmark (per tool call,
  per answer, nightly batch, CPU per plant per day) and kernel sweeps that show where oneDAL wins
  and where it does not
* **`demo/`**: a scripted agent run with timings, and the citation check that blocks any number a
  tool did not return

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ruff check .                         # config lives in pyproject.toml
pytest -q                            # ~15 s: parity, determinism, planted faults, citation check
python -m bench.parity               # ~30 s: writes results/parity.json (the agent reads it)
python -m demo.agent_demo --compare  # one question, stock vs oneDAL, side by side
python -m bench.run_bench            # ~10-15 min: full benchmark -> results/<your-cpu>/summary.md
```

oneDAL runs on x86-64 CPUs (Intel and AMD). Run `bench.parity` again after any change to the
scikit-learn or scikit-learn-intelex version: until you do, the envelope marks every oneDAL
result `citable: false`.

## The pattern

1. **The model never computes.** It plans tool calls and writes prose; every number comes from a tool.
2. **Tools return facts with ids.** Each call returns an envelope (`call_id`, scope, parameters,
   input fingerprint, library versions, parity status, timings) and a list of facts as short
   sentences with ids like `drift-09c14fad#D1`. The answer may only quote facts, cited by id.
3. **oneDAL with one import.** `tools/backend.py` is the only file that imports estimators.
   `from sklearnex.cluster import DBSCAN` instead of `from sklearn.cluster import DBSCAN`
   (or `sklearnex.patch_sklearn()` at start-up) is the whole switch.
4. **Route by size.** oneDAL is not faster everywhere. Two measured thresholds send small PCA fits
   and large DBSCAN runs to stock scikit-learn (`ROUTING` in `tools/backend.py`; the benchmark
   report suggests values for your CPU).
5. **Parity gate per library version.** Results must match stock within per-tool tolerances before
   the agent may cite them.
6. **Citation check per answer.** `demo/cite_check.py` blocks any sentence with a number that none
   of its cited facts contains.

## The tools

| Tool | Question it answers | Method | oneDAL estimators |
|---|---|---|---|
| `cluster_stops` | Where did the time go? Which stops have no reason? | k-means families with a suggested reason; DBSCAN for irregular stops | `KMeans`, `DBSCAN` (brute force up to 30k stops) |
| `detect_cycle_drift` | Is anything getting slower? | kNN novelty score per machine x program, last 7 days vs the 21 before | `NearestNeighbors` |
| `rank_loss_drivers` | What goes with unplanned stops? | Random forest on machine-hours + permutation importance on held-out hours | `RandomForestRegressor` (fit and predict) |
| `signal_health` | Is any machine behaving abnormally? | PCA per machine x program, SPE and Hotelling T² | `PCA` (from 8k rows) |

## What changed for oneDAL, and what didn't

| Change | Why |
|---|---|
| Estimator imports moved into `tools/backend.py` | One switch for both backends; stock and oneDAL can run side by side for the parity gate |
| k-means starting centers computed once (`kmeans_plusplus`) and passed to both backends | The libraries' internal k-means++ use different random streams and land on different local optima. With shared centers the stop families match exactly (ARI 1.0) |
| DBSCAN: oneDAL brute force up to 30k points, stock KD-tree above | oneDAL's DBSCAN is brute force; on large, low-dimensional inputs the tree wins. Calling stock directly avoids the ~8% cost of sklearnex's fallback path |
| PCA: stock below 8k rows | Per-call overhead dominates tiny fits |
| Permutation importance on a fixed 3,000-row sample | Keeps cost flat as plants grow; oneDAL's RF predict is fastest at small and medium batches |
| Arrays passed as C-contiguous float64 | Avoids conversion copies on every call |
| **Unchanged** | Tool logic, pandas preparation, outputs, schemas, tests |

`bench/parity.py` also logs every estimator call through sklearnex's dispatcher, so you can see
which calls actually ran on oneDAL and which fell back to stock.

## Parity tolerances

| Tool | Must hold (oneDAL vs stock, same input) |
|---|---|
| `cluster_stops` | family labels ARI ≥ 0.99; DBSCAN noise agreement ≥ 99% |
| `detect_cycle_drift` | same flagged machines; per machine-program share outside envelope within 0.01 |
| `rank_loss_drivers` | same top-3 drivers; Spearman ρ of importances ≥ 0.80; holdout R² within 0.02 |
| `signal_health` | same flagged machines; per machine abnormal share within 0.01 |
| all | the same call twice gives the same result digest |

The random forest is the only tool that is not bit-identical: oneDAL builds its trees with a
different implementation and random stream. That is why it has a rank-based tolerance.

## The synthetic plant

`python -m plantgen --size M --out data/plant-M` writes four CSVs and `plant.json` with the planted truth.

| Size | Machines | Stops/day | Cycles/day |
|---|---:|---:|---:|
| S | 12 | ~520 | ~4,600 |
| M | 40 | ~1,600 | ~11,700 |
| L | 120 | ~4,600 | ~34,600 |

Planted per plant: tool-wear drift (operators lower feed override, so cycle time creeps up),
program-edit steps (+10% cycle time, lower cutting load), coolant-pump degradation (no cycle-time
change), a bad material supplier, a night shift with longer material waits, and maintenance-age
effects on breakdowns. Operator ID and coolant concentration are null drivers.

**Your own data:** export the four tables with the columns in `plantgen/io.py:SCHEMA` and load them
with `plantgen.load_plant(dir)`. `reason_true` is optional.

## Reference results

See `results/intel-xeon-2t/summary.md`: a 2-vCPU cloud Xeon (AVX-512, AMX). That box is not like
a plant edge PC. Run `python -m bench.run_bench` on your own edge hardware: it writes
`results/<your-cpu>/` and prints suggested routing thresholds for that CPU.

## Limits

* The data is synthetic. Shapes, rates and fault signatures come from CNC job shops, but your
  stop mix and signals will differ, so re-check the tolerances and thresholds on real data.
* `rank_loss_drivers` reports association, not cause. On noisy hourly data (holdout R² ≈ 0.05), null
  columns can score a little above zero. The tool reports only drivers above 2σ of permutation
  noise and ≥ 20% of the top driver's importance.
* Timings depend on the CPU, thread count and library versions. The report records all three.

## License

Choose before publishing (Apache-2.0 matches most UXL Foundation projects).
