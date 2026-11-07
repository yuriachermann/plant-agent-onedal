# Parity gate: oneDAL vs stock scikit-learn

Checked 2026-11-07T17:29:10+00:00 · sklearn 1.9.1 · sklearnex 2026.2.0 · tool version 0.2.0 · plant 40 machines, 30 days

| Tool | Scope | Result | Key metrics | oneDAL dispatch |
|---|---|---|---|---|
| cluster_stops | plant | PASS | `{"family_ari": 1.0, "noise_agreement": 1.0, "dbscan_path": "stock-tree", "deterministic_onedal": true}` | KMeans.fit: 1 acc / 0 fb |
| cluster_stops | line | PASS | `{"family_ari": 1.0, "noise_agreement": 1.0, "dbscan_path": "onedal-brute", "deterministic_onedal": true}` | KMeans.fit: 1 acc / 0 fb; DBSCAN.fit: 1 acc / 0 fb |
| detect_cycle_drift | plant | PASS | `{"flagged_stock": ["M003", "M006", "M022", "M028"], "flagged_onedal": ["M003", "M006", "M022", "M028"], "max_share_abs_diff": 0.0, "compared_units": 121, "deterministic_onedal": true}` | NearestNeighbors.fit: 121 acc / 0 fb; NearestNeighbors.kneighbors: 242 acc / 0 fb |
| detect_cycle_drift | line | PASS | `{"flagged_stock": [], "flagged_onedal": [], "max_share_abs_diff": 0.0, "compared_units": 29, "deterministic_onedal": true}` | NearestNeighbors.fit: 29 acc / 0 fb; NearestNeighbors.kneighbors: 58 acc / 0 fb |
| rank_loss_drivers | plant | PASS | `{"top3_stock": ["tool_age", "supplier", "days_since_pm"], "top3_onedal": ["tool_age", "supplier", "days_since_pm"], "spearman": 0.9301, "r2_abs_diff": 0.0017, "r2_stock": 0.0375, "r2_onedal": 0.0392, "deterministic_onedal": true}` | RandomForestRegressor.fit: 1 acc / 0 fb; RandomForestRegressor.score: 62 acc / 0 fb |
| rank_loss_drivers | line | PASS | `{"top3_stock": ["tool_age", "supplier", "days_since_pm"], "top3_onedal": ["tool_age", "supplier", "days_since_pm"], "spearman": 0.8807, "r2_abs_diff": 0.007, "r2_stock": 0.0599, "r2_onedal": 0.0669, "deterministic_onedal": true}` | RandomForestRegressor.fit: 1 acc / 0 fb; RandomForestRegressor.score: 62 acc / 0 fb |
| signal_health | plant | PASS | `{"flagged_stock": ["M002", "M003", "M006", "M020", "M022", "M028"], "flagged_onedal": ["M002", "M003", "M006", "M020", "M022", "M028"], "max_share_abs_diff": 0.0, "compared_units": 40, "deterministic_onedal": true}` |  |
| signal_health | line | PASS | `{"flagged_stock": [], "flagged_onedal": [], "max_share_abs_diff": 0.0, "compared_units": 8, "deterministic_onedal": true}` |  |

Tolerances:
```json
{
  "cluster_stops": {
    "family_ari_min": 0.99,
    "noise_agreement_min": 0.99
  },
  "detect_cycle_drift": {
    "flagged_equal": true,
    "share_abs_diff_max": 0.01
  },
  "rank_loss_drivers": {
    "top3_equal": true,
    "spearman_min": 0.8,
    "r2_abs_diff_max": 0.02
  },
  "signal_health": {
    "flagged_equal": true,
    "share_abs_diff_max": 0.01
  }
}
```
