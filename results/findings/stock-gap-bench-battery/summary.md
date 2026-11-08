# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 119 | 46 | 2.59× | 6.33× | 119–122 | 46–48 |
| detect_cycle_drift (kNN) | 1923 | 465 | 4.14× | 6.26× | 1923–1952 | 465–483 |
| rank_loss_drivers (random forest) | 3750 | 314 | 11.96× | 12.5× | 3750–3800 | 314–340 |
| signal_health (PCA) | 219 | 158 | 1.38× | 1.39× | 219–244 | 158–159 |
| **One answer (4 calls)** | **6011** | **983** | **6.12×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 15.1 s | 4.1 s | 3.67× |
| Wall time, warm | 17.1 s | 7.5 s | 2.26× |
| · cluster_stops | 1.77 s | 2.35 s | 0.75× |
| · detect_cycle_drift | 8.63 s | 2.98 s | 2.9× |
| · rank_loss_drivers | 5.75 s | 1.11 s | 5.18× |
| · signal_health | 0.90 s | 1.10 s | 0.82× |
| CPU per plant per day (nightly + 40 questions) | 559 core-s | 166 core-s | 3.37× |
| …at $0.04/core-hour | $0.0062 | $0.0018 | |

