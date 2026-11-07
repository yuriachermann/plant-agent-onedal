# Benchmark: stock scikit-learn vs oneDAL

**Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz · 2 logical CPUs · ISA avx2, avx512f, avx512_vnni, amx_tile · Linux 6.18.44-fc-v300 · Python 3.13.18
**Versions:** scikit-learn 1.9.1 · scikit-learn-intelex 2026.2.0 · oneDAL 2026.2.0

Synthetic plants, 30 days, seed 7. On-demand = one line, median of 5 warm calls. Nightly = whole plant, all four tools; cold = first run in a fresh process.

## Plant S: 12 machines · 522 stops/day · 4635 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up |
|---|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 34 | 27 | 1.27× | 4.45× |
| detect_cycle_drift (kNN) | 603 | 186 | 3.24× | 4.67× |
| rank_loss_drivers (random forest) | 3067 | 304 | 10.08× | 10.39× |
| signal_health (PCA) | 71 | 74 | 0.95× | 0.96× |
| **One answer (4 calls)** | **3775** | **591** | **6.38×** | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 4.8 s | 1.5 s | 3.26× |
| Wall time, warm | 5.2 s | 1.5 s | 3.45× |
| · cluster_stops | 0.20 s | 0.14 s | 1.45× |
| · detect_cycle_drift | 1.59 s | 0.51 s | 3.15× |
| · rank_loss_drivers | 3.22 s | 0.68 s | 4.77× |
| · signal_health | 0.20 s | 0.19 s | 1.03× |
| CPU per plant per day (nightly + 40 questions) | 167 core-s | 41 core-s | 4.04× |
| …at $0.04/core-hour | $0.0019 | $0.0005 | |

## Plant M: 40 machines · 1615 stops/day · 11670 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up |
|---|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 51 | 37 | 1.39× | 3.97× |
| detect_cycle_drift (kNN) | 1003 | 336 | 2.99× | 4.17× |
| rank_loss_drivers (random forest) | 2941 | 532 | 5.53× | 5.65× |
| signal_health (PCA) | 109 | 122 | 0.89× | 0.88× |
| **One answer (4 calls)** | **4104** | **1026** | **4.0×** | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 10.5 s | 4.1 s | 2.56× |
| Wall time, warm | 10.4 s | 4.3 s | 2.4× |
| · cluster_stops | 1.06 s | 0.86 s | 1.23× |
| · detect_cycle_drift | 4.52 s | 1.38 s | 3.28× |
| · rank_loss_drivers | 4.20 s | 1.47 s | 2.86× |
| · signal_health | 0.61 s | 0.62 s | 0.99× |
| CPU per plant per day (nightly + 40 questions) | 202 core-s | 76 core-s | 2.65× |
| …at $0.04/core-hour | $0.0022 | $0.0008 | |

## Plant L: 120 machines · 4562 stops/day · 34629 cycles/day

| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up |
|---|---:|---:|---:|---:|
| cluster_stops (k-means + DBSCAN) (onedal-brute) | 54 | 33 | 1.64× | 4.17× |
| detect_cycle_drift (kNN) | 1006 | 321 | 3.13× | 4.61× |
| rank_loss_drivers (random forest) | 3044 | 543 | 5.61× | 5.71× |
| signal_health (PCA) | 118 | 144 | 0.82× | 0.81× |
| **One answer (4 calls)** | **4222** | **1041** | **4.05×** | |

| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |
|---|---:|---:|---:|
| Wall time, cold | 25.0 s | 12.1 s | 2.06× |
| Wall time, warm | 24.5 s | 11.9 s | 2.05× |
| · cluster_stops | 4.12 s | 4.60 s | 0.9× |
| · detect_cycle_drift | 12.59 s | 3.57 s | 3.53× |
| · rank_loss_drivers | 6.30 s | 2.15 s | 2.93× |
| · signal_health | 1.51 s | 1.62 s | 0.93× |
| CPU per plant per day (nightly + 40 questions) | 224 core-s | 90 core-s | 2.49× |
| …at $0.04/core-hour | $0.0025 | $0.0010 | |

## Kernel sweeps (single calls, large plant data)

**dbscan**

| n | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 1,000 | 13.4 | 1.1 | 11.69× |
| 2,500 | 13.3 | 3.8 | 3.54× |
| 5,000 | 36.7 | 12.6 | 2.93× |
| 10,000 | 84.9 | 48.4 | 1.75× |
| 20,000 | 240.0 | 174.2 | 1.38× |
| 40,000 | 772.4 | 805.4 | 0.96× |
| 80,000 | 2397.6 | 3054.7 | 0.78× |

**kmeans**

| n | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 2,500 | 3.2 | 2.6 | 1.25× |
| 20,000 | 10.6 | 7.7 | 1.38× |
| 118,000 | 147.7 | 123.3 | 1.2× |

**knn**

| n_ref | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 500 | 24.4 | 2.9 | 8.57× |
| 2,000 | 25.1 | 6.8 | 3.69× |
| 8,000 | 95.8 | 25.7 | 3.72× |
| 32,000 | 552.8 | 130.0 | 4.25× |

**rf_fit**

| n | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 4,000 | 421.1 | 92.3 | 4.56× |
| 16,000 | 1564.0 | 347.9 | 4.5× |
| 45,731 | 4621.9 | 1087.3 | 4.25× |

**rf_predict**

| n | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 1 | 32.6 | 0.4 | 83.22× |
| 100 | 32.7 | 1.0 | 31.44× |
| 1,000 | 43.0 | 5.8 | 7.41× |
| 3,000 | 42.6 | 16.0 | 2.67× |
| 10,000 | 63.2 | 53.1 | 1.19× |
| 40,000 | 168.9 | 194.0 | 0.87× |

**pca**

| n | stock ms | oneDAL ms | speed-up |
|---:|---:|---:|---:|
| 500 | 0.5 | 1.1 | 0.42× |
| 2,000 | 0.8 | 1.2 | 0.68× |
| 8,000 | 1.9 | 1.7 | 1.14× |
| 64,000 | 20.7 | 4.5 | 4.62× |

## Suggested routing for this machine
Use oneDAL where it is at least 1.1× faster in the sweeps above, stock elsewhere:

```
PLANT_TOOLS_DBSCAN_BRUTE_MAX=20000
PLANT_TOOLS_PCA_MIN_ROWS=8000
```
Set these as environment variables before running the tools (defaults: 30000 and 8000).

