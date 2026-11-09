# Detection quality against planted truth

Backend `stock`, seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sizes ['S', 'M'], 20 plants, driver rule `sigma`, driver holdout by `day`. Definitions are in `summary.json`.

## Drift (wear + step)

Precision mean 100.0% (min 100.0%) on the 20 plants that flagged at least one machine. Recall mean 98.3% (min 66.7%): wear 97.5%, step 100.0%. Healthy-machine false-positive rate mean 0.0% (max 0.0%). Step test on.

## Signal health

Coolant recall mean 90.0% (min 0.0%). Healthy-machine false-positive rate mean 0.0% (max 0.0%).

## Loss drivers

Share of planted drivers in the top 3: mean 66.2% (min 50.0%). Share of planted drivers reported: mean 50.0%. Runs with any null driver reported: 10.0%.

| Planted driver | Share of runs in the top 3 | Share of runs reported |
|---|---:|---:|
| days_since_pm | 45.0% | 25.0% |
| shift | 45.0% | 25.0% |
| supplier | 85.0% | 75.0% |
| tool_age | 90.0% | 75.0% |

| Null driver | Share of runs reported |
|---|---:|
| coolant_conc | 0.0% |
| operator_id | 5.0% |
| weekday | 5.0% |

## Stop suggestions

| | By count | By minutes |
|---|---:|---:|
| Coverage mean | 87.2% | 66.4% |
| Coverage min | 84.9% | 51.6% |
| Accuracy mean | 96.5% | 61.5% |
| Accuracy min | 92.8% | 50.7% |

Accuracy is the share of suggestions that match `reason_true`, among unjustified stops that received one. Minute weights are stop durations.

## By plant size

| Size | Drift recall | Coolant recall | Planted drivers in top 3 | Stop coverage | Stop accuracy |
|---|---:|---:|---:|---:|---:|
| S | 100.0% | 80.0% | 62.5% | 87.1% | 96.9% |
| M | 96.7% | 100.0% | 70.0% | 87.4% | 96.1% |
