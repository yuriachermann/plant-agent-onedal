# Benchmark: stock scikit-learn vs oneDAL

**Machine:** 12th Gen Intel(R) Core(TM) i5-12450H · 12 logical CPUs · ISA n/a · Windows 11 · Python 3.12.2 · on AC, plan High performance, EcoQoS opt-out
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0
**Main thread on P-cores (on-demand calls):** stock 83% · oneDAL 100%

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 3 warm calls. p50–p90 is the spread of those calls. Nightly = whole plant, all four tools, on the first seed; cold = first run in a fresh process.

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up | Stock p50–p90 | oneDAL p50–p90 |
|---|---:|---:|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 93 | 37 | 2.54× | 5.98× | 93–104 | 37–39 |
| detect_cycle_drift (kNN) | 1484 | 355 | 4.18× | 7.01× | 1484–1590 | 355–365 |
| rank_loss_drivers (random forest) | 3281 | 339 | 9.68× | 10.31× | 3281–3341 | 339–358 |
| signal_health (PCA) | 162 | 175 | 0.93× | 1.03× | 162–172 | 175–209 |
| **One answer (4 calls)** | **5020** | **906** | **5.54×** | | | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 12.4 s | 4.5 s | 2.78× |
| Wall time, warm | 12.2 s | 5.2 s | 2.36× |
| · cluster_stops | 0.92 s | 1.78 s | 0.51× |
| · detect_cycle_drift | 6.75 s | 1.85 s | 3.64× |
| · rank_loss_drivers | 3.86 s | 0.80 s | 4.82× |
| · signal_health | 0.71 s | 0.75 s | 0.95× |
| CPU per plant per day (nightly + 40 questions) | 435 core-s | 147 core-s | 2.96× |
| …at $0.04/core-hour | $0.0048 | $0.0016 | |

