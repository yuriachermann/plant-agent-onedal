# Stock scikit-learn: laptop vs Xeon

PROGRESS.md 0.5 recorded "one answer" (the four on-demand calls for plant M, seed 7) at 11,500 ms stock and 1,721 ms oneDAL on this laptop, a 12th Gen Intel Core i5-12450H (4 P-cores with HT plus 4 E-cores, 12 logical CPUs, Windows 11). On the 2-vCPU Xeon in `results/intel-xeon-2t/summary.json` it is 4,104 ms and 1,026 ms. So oneDAL looked 6.7× faster here and 4.0× there. This note asks why.

## Short answer

1. **The 11.5 s does not reproduce.** The same command today gives 6,011 ms stock and 983 ms oneDAL (6.12×). In the 0.5 run every on-demand call on both backends was slower, stock by 1.4–2.1× per tool and oneDAL by 1.75× on the answer. The power state was not recorded then. Benchmark runs now record AC or battery and the power plan.
2. **What remains is real: 6.1× here, 4.0× on the Xeon.** oneDAL uses the laptop's 12 threads. Stock spends much of its time in many small estimator calls, and with `n_jobs=-1` each call starts a pool of 12 workers. It barely gains from the extra cores and pays the start-up on every call.
3. **Hybrid cores add noise from run to run.** Single-threaded Python runs about half as fast on an E-core, and the main thread does not always land on a P-core.
4. **Not the cause:** the native thread pools (OpenMP, OpenBLAS) and Windows power throttling (EcoQoS).
5. **No tool change.** Every `n_jobs` setting measured makes one backend faster and the other slower, so none of them is fair to both.
6. **Plugged in, the gap narrows but stays.** On AC the answer takes 4,941 ms stock and 878 ms oneDAL (5.62×). Stock is still 20% slower than on the Xeon, and the probe gives the same picture as on battery (section 5).

Sections 1–4 ran **on battery** (plan "High performance", power mode Balanced, maximum processor state 100% on AC and on battery). Section 5 repeats both runs on AC, where the power mode was Best performance.

## 1. The 0.5 number

`python -m bench.run_bench --sizes M --repeats 3 --skip-kernels --out results/findings/stock-gap-bench-battery` is the 0.5 command without the kernel sweeps (they run after the on-demand calls and do not affect them). On-demand medians, plant M, seed 7:

| Call (one line) | 0.5 stock (PROGRESS.md) | Today stock | Today oneDAL | Xeon stock | Xeon oneDAL |
|---|---:|---:|---:|---:|---:|
| cluster_stops | 168 | 119 | 46 | 51 | 37 |
| detect_cycle_drift | 2,955 | 1,923 | 465 | 1,003 | 336 |
| rank_loss_drivers | 7,936 | 3,750 | 314 | 2,941 | 532 |
| signal_health | 440 | 219 | 159 | 109 | 122 |
| **One answer** | **11,500** | **6,011** | **983** | **4,104** | **1,026** |

Sources: `results/findings/stock-gap-bench-battery/summary.json` (today) and `results/intel-xeon-2t/summary.json` (Xeon). The 0.5 column is from PROGRESS.md 0.5, which read the uncommitted `results/_scratch/summary.md`; its oneDAL answer was 1,721 ms. Today's run uses tool version 0.4.0, whose drift tool does more work (the step test), so the drop is not a code change.

The 0.5 run was slower on both backends and on every tool, which points to the machine's state at the time rather than to the code. Candidates are battery or a lower power mode, heat, other load, or where Windows placed the threads. None of these was recorded, so the cause cannot be settled after the fact.

## 2. Where stock loses on this machine

`python -m bench.probe_stock_gap` (`bench/probe_stock_gap.py`) runs each case in a fresh process per backend: one warm-up, then 3 timed on-demand calls per tool, median reported, split into prep, ML (`ml_*` phases) and post. The tools are not changed. The `njobs*` cases wrap the estimator classes so every instance gets that `n_jobs`, and the `threads1` case sets the OpenMP, BLAS, TBB and DAAL thread variables to 1 before the process starts. Source: `results/findings/stock-gap-battery.json`.

| Case | Stock answer ms | oneDAL answer ms | Speed-up |
|---|---:|---:|---:|
| default (`n_jobs=-1`) | 5,939 | 1,288 | 4.6× |
| `n_jobs=1` | 3,854 | 2,288 | 1.7× |
| `n_jobs=2` | 5,335 | 1,633 | 3.3× |
| native pools at 1 thread | 6,057 | 2,417 | 2.5× |

The two stock tools that make up the answer:

* **detect_cycle_drift** fits a small kNN model per machine and program and queries it twice. In the overhead section of the same file, a fit plus two queries on 800 × 6 reference rows costs 50.68 ms with `n_jobs=-1` and 16.35 ms with `n_jobs=1`. oneDAL costs 8.01 and 9.63 ms. Forcing `n_jobs=1` drops stock drift's ML time from 1,748 to 696 ms. On the Xeon `n_jobs=-1` means 2 workers, so it pays far less per call.
* **rank_loss_drivers** spends most of its time in permutation importance: 61 forest predictions on 3,000 rows. With `n_jobs=-1`, a stock 200-tree predict on 3,000 rows takes 86.65 ms here, against 42.6 ms for the Xeon's `rf_predict` kernel at the same size (`results/intel-xeon-2t/kernels_stock.json`). With `n_jobs=1`, permutation time falls from 3,291 to 1,926 ms. The forest fit goes the other way, 327 to 717 ms, because the trees really do build in parallel.

oneDAL needs those threads: with `n_jobs=1` its forest permutation step rises from 486 to 1,167 ms, because scikit-learn-intelex passes `n_jobs` to oneDAL as its thread count.

The native pools are not the stock problem. threadpoolctl shows OpenBLAS and MSVC OpenMP at 12 threads by default and at 1 in the `threads1` case, where the stock answer moves from 5,939 to 6,057 ms.

The stock kernel sweep at 1 native thread (`results/findings/stock-gap-bench-battery/kernels_stock_t1.json`, `n_jobs` as coded) shows the same pattern against the Xeon's default run. Small calls are slower here: kNN with 2,000 reference rows takes 49.1 ms vs 25.1 ms, a forest predict on 3,000 rows 61.9 vs 42.6 ms. Large fits are faster: a forest fit on 16,000 rows takes 698 vs 1,564 ms.

## 3. P-cores, E-cores and run-to-run noise

A pure-Python loop pinned to one E-core took 186.4 ms against 97.0 ms on a P-core, and a NumPy sort 99.97 against 29.34 ms (calibration in `results/findings/stock-gap-battery.json`). Pinning the whole process (`SetProcessAffinityMask`) gives:

| Case | Stock answer ms | oneDAL answer ms | signal_health outside ML, stock ms |
|---|---:|---:|---:|
| default, first probe | 5,939 | 1,288 | 155.0 |
| P-cores only, first probe | 5,116 | 809 | 86.6 |
| E-cores only, first probe | 6,662 | 1,254 | 157.0 |
| default, second probe | 5,525 | 817 | 81.6 |
| P-cores only, second probe | 5,290 | 877 | 91.8 |
| E-cores only, second probe | 7,255 | 1,331 | 139.7 |

The second probe is `results/findings/stock-gap-ecoqos-battery.json`, a later run the same morning, also on battery. In the first probe the unpinned default ran its single-threaded work at E-core speed. In the second it ran at P-core speed. The same command swung oneDAL from 1,288 to 817 ms. On a hybrid laptop, one benchmark run is not enough.

## 4. Windows power throttling

The `noecoqos` case opts the process out of EcoQoS for itself (`SetProcessInformation` with `ProcessPowerThrottling`). No system setting changes. It made no difference: 5,603 ms stock and 852 ms oneDAL, against 5,525 and 817 ms default in the same probe.

## 5. On AC

On 2026-11-09 the laptop was plugged in, and both runs were repeated back to back with tool version 0.5.0 (for these calls it runs the same code path as 0.4.0):

```bash
python -m bench.probe_stock_gap --repeats 3 --out results/findings/stock-gap-ac.json
python -m bench.run_bench --sizes M --repeats 3 --skip-kernels --out results/findings/stock-gap-bench-ac
```

Both files record AC online and the plan "High performance". The probe also records the power mode overlay. Both its AC and its battery overlay were `ded574b5-45a0-4f42-8737-46345c09c238`, the GUID Windows uses for Best performance. The battery probe that recorded the overlay (`results/findings/stock-gap-ecoqos-battery.json`) had the all-zero GUID, which is Balanced. I did not change the setting. So this section compares battery in Balanced mode with AC in Best performance, not battery alone.

On-demand medians, plant M, seed 7:

| Call (one line) | Battery stock | AC stock | Xeon stock | Battery oneDAL | AC oneDAL | Xeon oneDAL |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops | 119 | 92 | 51 | 46 | 36 | 37 |
| detect_cycle_drift | 1,923 | 1,540 | 1,003 | 465 | 371 | 336 |
| rank_loss_drivers | 3,750 | 3,125 | 2,941 | 314 | 310 | 532 |
| signal_health | 219 | 184 | 109 | 158 | 161 | 122 |
| **One answer** | **6,011** | **4,941** | **4,104** | **983** | **878** | **1,026** |

The answer speed-up is 6.12× on battery, 5.62× on AC and 4.0× on the Xeon. Sources: `results/findings/stock-gap-bench-battery/summary.json`, `results/findings/stock-gap-bench-ac/summary.json` and `results/intel-xeon-2t/summary.json`.

* **Plugged in, stock gains 18% and oneDAL 11%.** Every stock call is 16–22% faster. oneDAL gains only on cluster_stops and drift; drivers and signal_health move by less than 2%. The warm nightly batch drops from 17.1 to 12.3 s stock and from 7.5 to 4.9 s oneDAL. CPU per plant per day drops from 559 to 418 core-s stock and from 166 to 112 core-s oneDAL.
* **Battery is not the gap to the Xeon.** On AC the stock answer is still 20% slower than on the Xeon (4,941 vs 4,104 ms). Most of the difference is in drift (1,540 vs 1,003 ms), the tool with the most small kNN calls. The oneDAL answer is now faster than on the Xeon (878 vs 1,026 ms), so the speed-up stays higher here.

The probe (`results/findings/stock-gap-ac.json`) tells the same story as on battery, once thread placement is taken into account. Its first case, the unpinned default, ran its single-threaded work at E-core speed on both backends: signal_health outside ML took 154 ms stock and 232 ms oneDAL, against 97–127 ms in the other unpinned cases and 179 and 162 ms when pinned to E-cores. Compare the other cases with the P-core and EcoQoS rows, which ran at P-core speed with `n_jobs=-1`:

| Case (AC probe) | Stock answer ms | oneDAL answer ms | Speed-up |
|---|---:|---:|---:|
| default (`n_jobs=-1`), ran at E-core speed | 5,558 | 1,622 | 3.43× |
| P-cores only | 4,768 | 916 | 5.20× |
| EcoQoS opt-out | 4,760 | 840 | 5.67× |
| `n_jobs=1` | 2,985 | 1,649 | 1.81× |
| `n_jobs=2` | 4,200 | 1,174 | 3.58× |
| native pools at 1 thread | 4,564 | 1,640 | 2.78× |
| E-cores only | 7,567 | 1,659 | 4.56× |

* **`n_jobs` is the same trade-off as on battery.** `n_jobs=1` takes the stock answer from about 4,760 ms to 2,985 ms, which beats the Xeon's 4,104 ms. It takes oneDAL from 840–916 ms to 1,649 ms. Per call, a stock kNN fit plus two queries costs 36.41 ms with `n_jobs=-1` and 12.07 ms with `n_jobs=1`; oneDAL costs 12.78 and 9.53 ms. A stock 200-tree predict on 3,000 rows takes 49.0 ms with `n_jobs=-1`, against 86.65 ms on battery and 42.6 ms for the Xeon's kernel. Still no tool change.
* **The native pools are still not the cause.** At 1 thread, stock takes 4,564 ms, close to the P-core-speed cases.
* **Hybrid cores still add noise.** Plugged in, the same unpinned command ran at E-core speed once in this probe, and the benchmark minutes later gave oneDAL 878 ms. The one-shot calibration loops are noisy too. On one P-core the Python loop took 142.85 ms, against 97.0 and 113.83 ms in the two battery probes. The E-core-to-P-core ratio was 1.9×, 1.5× and 1.2× across the three probes. In this probe, two short file reads of mine (each under a second) ran during the first 30 seconds, while the calibration and overhead steps were running. So treat this calibration with care. The per-case phases are the better guide.
* **EcoQoS: open.** On AC the opt-out case (4,760 / 840 ms) matches the P-core case (4,768 / 916 ms). On battery it matched the default, which ran at P-core speed that time. The opt-out ran at P-core speed in both probes that had it. The unpinned default ran at E-core speed in two of three probes. That is too few runs to tell whether EcoQoS sends the process to the E-cores.

## What changed

* No tool change. Every `n_jobs` value measured favours one backend: `n_jobs=1` takes stock from 5,939 to 3,854 ms and oneDAL from 1,288 to 2,288 ms, and `n_jobs=2` lands between the two on both backends. The tools keep `n_jobs=-1`.
* `bench.run_bench` now records `ac_power` and `power_plan` in the machine block on Windows, and `bench.report` prints them in the summary header. The 0.5 run would have shown its power state.
* `bench/probe_stock_gap.py` stays in the repo to repeat this on the NUC or a plant PC.

## For the talk

Quote the speed-up with the machine and its thread count. On a many-thread laptop, oneDAL gains from the extra cores and stock scikit-learn barely does. Run benchmarks on AC power, record the power state, and take the median of more than one run. On this laptop, plugged in, one answer takes 4,941 ms stock and 878 ms oneDAL (5.62×, `results/findings/stock-gap-bench-ac/summary.json`).

## What is still open

* Battery vs power mode. The AC run was also in Best performance mode, while the battery runs were in Balanced, so section 5 does not split the two. One run on AC in Balanced mode would. That is a Windows setting for the owner to change.
* EcoQoS and thread placement. A probe that repeats `--cases default noecoqos` several times would show whether the opt-out keeps an unpinned run on the P-cores.
* Linux on the same laptop (WSL or a live USB), to separate Windows from hardware. Not tried.
