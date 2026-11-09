# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2 · on AC, plan High performance
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 92 | 36 | 2.56× | 7.58× | 92–94 | 36–39 |
| detect_cycle_drift (kNN) | 1540 | 371 | 4.15× | 6.6× | 1540–1603 | 371–375 |
| rank_loss_drivers (random forest) | 3125 | 310 | 10.07× | 10.72× | 3125–3277 | 310–316 |
| signal_health (PCA) | 184 | 161 | 1.14× | 1.1× | 184–195 | 161–163 |
| **One answer (4 calls)** | **4941** | **878** | **5.62×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 12.4 s | 4.1 s | 3.03× |
| Wall time, warm | 12.3 s | 4.9 s | 2.5× |
| · cluster_stops | 1.11 s | 1.59 s | 0.69× |
| · detect_cycle_drift | 6.37 s | 1.61 s | 3.95× |
| · rank_loss_drivers | 4.09 s | 0.82 s | 5.02× |
| · signal_health | 0.75 s | 0.90 s | 0.83× |
| CPU per plant per day (nightly + 40 questions) | 418 core-s | 112 core-s | 3.73× |
| …at $0.04/core-hour | $0.0046 | $0.0012 | |

