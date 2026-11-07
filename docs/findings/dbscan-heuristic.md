# DBSCAN heuristic

`cluster_stops` sets `min_samples = max(10, round(n/1400))` and
`eps = clip(0.5 * (40000/n)^(1/7), 0.5, 1.0)`. The question was whether that
share of irregular stops, and the reason-suggestion accuracy, jump around
when the plant and the scope change.

`bench/probe_dbscan.py` ran seeds 1–10 at sizes S and M, whole plant and line
L1, stock backend. The table is `results/findings/dbscan-heuristic.md`.

| Scope | Irregular share | Coverage | Accuracy | Accuracy by minutes | eps | min_samples |
|---|---|---|---|---|---|---|
| S, whole plant | 14.8% (14.2–15.7) | 87.1% (85.2–89.9) | 96.9% (93.9–98.3) | 62.2% (51.8–78.3) | 0.5835–0.5861 | 10 |
| S, line L1 | 11.4% (10.6–12.2) | 88.8% (85.3–91.3) | 95.6% (94.4–97.8) | 58.2% (50.0–70.7) | 0.6312–0.6354 | 10 |
| M, whole plant | 24.5% (24.1–25.0) | 87.4% (84.9–90.6) | 96.1% (92.8–98.4) | 60.7% (50.7–77.2) | 0.5 | 30 |
| M, line L1 | 11.7% (11.2–12.1) | 87.8% (86.4–89.9) | 96.8% (94.4–98.1) | 64.9% (54.1–79.9) | 0.6314–0.6356 | 10 |

Inside one scope the irregular share moves by less than 2 percentage points
across ten seeds, and count accuracy stays above 92.8%. Plant M sits on the
eps floor of 0.5 because it has about 42,000 stops, just above the 40,000
the formula was calibrated at, and `min_samples` stays 30. Smaller scopes
sit at `min_samples` 10 with eps between 0.58 and 0.64.

Accuracy weighted by minutes swings from about 50% to about 80%. That spread
is the same one `results/eval/summary.json` already reports for the whole
grid, and it tracks a few long stops, not a jump in eps.

The rule stays as it is.
