# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2 · on AC, plan High performance, EcoQoS left to Windows
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0
**Main thread on P-cores (on-demand calls):** stock 79% · oneDAL 100%

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 106 | 43 | 2.44× | 7.38× | 106–107 | 43–46 |
| detect_cycle_drift (kNN) | 2008 | 372 | 5.4× | 8.0× | 2008–2199 | 372–402 |
| rank_loss_drivers (random forest) | 3128 | 282 | 11.1× | 11.78× | 3128–3148 | 282–285 |
| signal_health (PCA) | 167 | 149 | 1.11× | 1.1× | 167–167 | 149–156 |
| **One answer (4 calls)** | **5408** | **846** | **6.39×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 17.1 s | 4.7 s | 3.67× |
| Wall time, warm | 13.1 s | 3.8 s | 3.43× |
| · cluster_stops | 1.71 s | 0.82 s | 2.09× |
| · detect_cycle_drift | 6.67 s | 1.54 s | 4.33× |
| · rank_loss_drivers | 3.98 s | 0.74 s | 5.37× |
| · signal_health | 0.75 s | 0.72 s | 1.04× |
| CPU per plant per day (nightly + 40 questions) | 409 core-s | 115 core-s | 3.55× |
| …at $0.04/core-hour | $0.0045 | $0.0013 | |

