# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2 · on AC, plan High performance, EcoQoS opt-out
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0
**Main thread on P-cores (on-demand calls):** stock 100% · oneDAL 100%

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 92 | 47 | 1.94× | 4.58× | 92–96 | 47–52 |
| detect_cycle_drift (kNN) | 1593 | 326 | 4.89× | 7.88× | 1593–1833 | 326–338 |
| rank_loss_drivers (random forest) | 2841 | 270 | 10.51× | 11.24× | 2841–2883 | 270–273 |
| signal_health (PCA) | 146 | 144 | 1.02× | 1.04× | 146–168 | 144–145 |
| **One answer (4 calls)** | **4672** | **787** | **5.94×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 13.1 s | 4.4 s | 3.0× |
| Wall time, warm | 12.4 s | 4.6 s | 2.71× |
| · cluster_stops | 1.47 s | 1.49 s | 0.99× |
| · detect_cycle_drift | 6.00 s | 1.56 s | 3.85× |
| · rank_loss_drivers | 4.20 s | 0.67 s | 6.24× |
| · signal_health | 0.72 s | 0.84 s | 0.86× |
| CPU per plant per day (nightly + 40 questions) | 410 core-s | 124 core-s | 3.32× |
| …at $0.04/core-hour | $0.0046 | $0.0014 | |

