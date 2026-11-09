# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2 · on AC, plan High performance, EcoQoS left to Windows
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0
**Main thread on P-cores (on-demand calls):** stock 83% · oneDAL 8%

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 94 | 60 | 1.56× | 3.27× | 94–102 | 60–71 |
| detect_cycle_drift (kNN) | 1419 | 574 | 2.47× | 3.91× | 1419–1485 | 574–618 |
| rank_loss_drivers (random forest) | 2816 | 766 | 3.68× | 3.85× | 2816–2924 | 766–839 |
| signal_health (PCA) | 167 | 332 | 0.5× | 0.6× | 167–174 | 332–349 |
| **One answer (4 calls)** | **4497** | **1732** | **2.6×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 15.4 s | 3.6 s | 4.23× |
| Wall time, warm | 18.8 s | 6.9 s | 2.73× |
| · cluster_stops | 2.22 s | 1.64 s | 1.36× |
| · detect_cycle_drift | 9.44 s | 2.64 s | 3.58× |
| · rank_loss_drivers | 6.31 s | 1.67 s | 3.79× |
| · signal_health | 0.87 s | 0.96 s | 0.91× |
| CPU per plant per day (nightly + 40 questions) | 398 core-s | 164 core-s | 2.43× |
| …at $0.04/core-hour | $0.0044 | $0.0018 | |

