# Deck numbers

Source directory: `intel-xeon-2t`. Stock → oneDAL. Regenerate with `python -m bench.deck_numbers C:\Users\yuria\Projects\oneAPI\plant-agent-onedal\results\intel-xeon-2t`.

| Figure | Value | Source |
|---|---|---|
| One answer, plant S | 3,775 → 591 ms | `summary.json sizes.S.answer.stock_ms / onedal_ms` |
| One answer, plant M | 4,104 → 1,026 ms | `summary.json sizes.M.answer.stock_ms / onedal_ms` |
| One answer, plant L | 4,222 → 1,041 ms | `summary.json sizes.L.answer.stock_ms / onedal_ms` |
| cluster_stops, plant M | 51 → 37 ms | `summary.json sizes.M.on_demand.cluster_stops.stock_ms / onedal_ms` |
| detect_cycle_drift, plant M | 1,003 → 336 ms | `summary.json sizes.M.on_demand.detect_cycle_drift.stock_ms / onedal_ms` |
| rank_loss_drivers, plant M | 2,941 → 532 ms | `summary.json sizes.M.on_demand.rank_loss_drivers.stock_ms / onedal_ms` |
| signal_health, plant M | 109 → 122 ms | `summary.json sizes.M.on_demand.signal_health.stock_ms / onedal_ms` |
| Nightly cold, plant S | 4.8 → 1.5 s | `summary.json sizes.S.nightly.cold.stock_s / onedal_s` |
| Nightly warm, plant S | 5.2 → 1.5 s | `summary.json sizes.S.nightly.warm.stock_s / onedal_s` |
| CPU per day, plant S | 166 → 41 core-s | `summary.json sizes.S.daily_cpu.stock_core_s / onedal_core_s` |
| Cost per day, plant S | $0.0019 → $0.0005 | `summary.json sizes.S.daily_cpu.stock_usd / onedal_usd` |
| Nightly cold, plant M | 10.4 → 4.1 s | `summary.json sizes.M.nightly.cold.stock_s / onedal_s` |
| Nightly warm, plant M | 10.4 → 4.3 s | `summary.json sizes.M.nightly.warm.stock_s / onedal_s` |
| CPU per day, plant M | 202 → 76 core-s | `summary.json sizes.M.daily_cpu.stock_core_s / onedal_core_s` |
| Cost per day, plant M | $0.0022 → $0.0008 | `summary.json sizes.M.daily_cpu.stock_usd / onedal_usd` |
| Nightly cold, plant L | 25.0 → 12.1 s | `summary.json sizes.L.nightly.cold.stock_s / onedal_s` |
| Nightly warm, plant L | 24.5 → 11.9 s | `summary.json sizes.L.nightly.warm.stock_s / onedal_s` |
| CPU per day, plant L | 224 → 90 core-s | `summary.json sizes.L.daily_cpu.stock_core_s / onedal_core_s` |
| Cost per day, plant L | $0.0025 → $0.0010 | `summary.json sizes.L.daily_cpu.stock_usd / onedal_usd` |
| Kernel dbscan n=1,000 | 11.69× | `summary.json kernels.dbscan[n=1000].speedup` |
| Kernel dbscan n=2,500 | 3.54× | `summary.json kernels.dbscan[n=2500].speedup` |
| Kernel dbscan n=5,000 | 2.93× | `summary.json kernels.dbscan[n=5000].speedup` |
| Kernel dbscan n=10,000 | 1.75× | `summary.json kernels.dbscan[n=10000].speedup` |
| Kernel dbscan n=20,000 | 1.38× | `summary.json kernels.dbscan[n=20000].speedup` |
| Kernel dbscan n=40,000 | 0.96× | `summary.json kernels.dbscan[n=40000].speedup` |
| Kernel dbscan n=80,000 | 0.78× | `summary.json kernels.dbscan[n=80000].speedup` |
| Kernel kmeans n=2,500 | 1.25× | `summary.json kernels.kmeans[n=2500].speedup` |
| Kernel kmeans n=20,000 | 1.38× | `summary.json kernels.kmeans[n=20000].speedup` |
| Kernel kmeans n=118,000 | 1.2× | `summary.json kernels.kmeans[n=118000].speedup` |
| Kernel knn n=500 | 8.57× | `summary.json kernels.knn[n=500].speedup` |
| Kernel knn n=2,000 | 3.69× | `summary.json kernels.knn[n=2000].speedup` |
| Kernel knn n=8,000 | 3.72× | `summary.json kernels.knn[n=8000].speedup` |
| Kernel knn n=32,000 | 4.25× | `summary.json kernels.knn[n=32000].speedup` |
| Kernel rf_fit n=4,000 | 4.56× | `summary.json kernels.rf_fit[n=4000].speedup` |
| Kernel rf_fit n=16,000 | 4.5× | `summary.json kernels.rf_fit[n=16000].speedup` |
| Kernel rf_fit n=45,731 | 4.25× | `summary.json kernels.rf_fit[n=45731].speedup` |
| Kernel rf_predict n=1 | 83.22× | `summary.json kernels.rf_predict[n=1].speedup` |
| Kernel rf_predict n=100 | 31.44× | `summary.json kernels.rf_predict[n=100].speedup` |
| Kernel rf_predict n=1,000 | 7.41× | `summary.json kernels.rf_predict[n=1000].speedup` |
| Kernel rf_predict n=3,000 | 2.67× | `summary.json kernels.rf_predict[n=3000].speedup` |
| Kernel rf_predict n=10,000 | 1.19× | `summary.json kernels.rf_predict[n=10000].speedup` |
| Kernel rf_predict n=40,000 | 0.87× | `summary.json kernels.rf_predict[n=40000].speedup` |
| Kernel pca n=500 | 0.42× | `summary.json kernels.pca[n=500].speedup` |
| Kernel pca n=2,000 | 0.68× | `summary.json kernels.pca[n=2000].speedup` |
| Kernel pca n=8,000 | 1.14× | `summary.json kernels.pca[n=8000].speedup` |
| Kernel pca n=64,000 | 4.62× | `summary.json kernels.pca[n=64000].speedup` |
| Parity record | 2026-11-07T17:29:10+00:00 | `results/parity.json checked_at` |
| Parity cluster_stops (plant) | ARI 1.0, noise agreement 1.0 | `results/parity.json tools.cluster_stops.scopes.plant.family_ari / noise_agreement` |
| Parity cluster_stops (line) | ARI 1.0, noise agreement 1.0 | `results/parity.json tools.cluster_stops.scopes.line.family_ari / noise_agreement` |
| Parity detect_cycle_drift (plant) | flagged ['M003', 'M006', 'M022', 'M028'], max share diff 0.0 | `results/parity.json tools.detect_cycle_drift.scopes.plant.flagged_stock / max_share_abs_diff` |
| Parity detect_cycle_drift (line) | flagged [], max share diff 0.0 | `results/parity.json tools.detect_cycle_drift.scopes.line.flagged_stock / max_share_abs_diff` |
| Parity rank_loss_drivers (plant) | Spearman 0.9301, ΔR² 0.0017, top3 ['tool_age', 'supplier', 'days_since_pm'] | `results/parity.json tools.rank_loss_drivers.scopes.plant.spearman / r2_abs_diff / top3_stock` |
| Parity rank_loss_drivers (line) | Spearman 0.8807, ΔR² 0.007, top3 ['tool_age', 'supplier', 'days_since_pm'] | `results/parity.json tools.rank_loss_drivers.scopes.line.spearman / r2_abs_diff / top3_stock` |
| Parity signal_health (plant) | flagged ['M002', 'M003', 'M006', 'M020', 'M022', 'M028'], max share diff 0.0 | `results/parity.json tools.signal_health.scopes.plant.flagged_stock / max_share_abs_diff` |
| Parity signal_health (line) | flagged [], max share diff 0.0 | `results/parity.json tools.signal_health.scopes.line.flagged_stock / max_share_abs_diff` |
| Stop coverage, M seed 7 | 85.4% | `results/eval/summary.json runs[size=M, seed=7].stops.coverage_count` |
| Stop accuracy, M seed 7 | 98.0% | `results/eval/summary.json runs[size=M, seed=7].stops.accuracy_count` |
| Stop accuracy by minutes, M seed 7 | 67.2% | `results/eval/summary.json runs[size=M, seed=7].stops.accuracy_minutes` |
