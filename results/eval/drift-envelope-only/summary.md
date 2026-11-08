# Detection quality against planted truth

Backend `stock`, seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sizes ['S', 'M'], 20 plants, driver rule `sigma`. Definitions are in `summary.json`.

## Drift (wear + step)

Precision mean 100.0% (min 100.0%) on the 19 plants that flagged at least one machine. Recall mean 53.3% (min 0.0%): wear 80.0%, step 22.5%. Healthy-machine false-positive rate mean 0.0% (max 0.0%). Step test off.

## Signal health

Coolant recall mean 90.0% (min 0.0%). Healthy-machine false-positive rate mean 0.0% (max 0.0%).

## Loss drivers

Share of planted drivers in the top 3: mean 71.2% (min 50.0%). Share of planted drivers reported: mean 80.0%. Runs with any null driver reported: 25.0%.

| Planted driver | Share of runs in the top 3 | Share of runs reported |
|---|---:|---:|
| days_since_pm | 85.0% | 95.0% |
| shift | 10.0% | 30.0% |
| supplier | 90.0% | 95.0% |
| tool_age | 100.0% | 100.0% |

| Null driver | Share of runs reported |
|---|---:|
| coolant_conc | 0.0% |
| operator_id | 20.0% |
| weekday | 15.0% |

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
| S | 45.0% | 80.0% | 67.5% | 87.1% | 96.9% |
| M | 61.7% | 100.0% | 75.0% | 87.4% | 96.1% |
