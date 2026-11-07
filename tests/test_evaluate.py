"""Full detection grid. Off by default: pytest -m slow.

Floors are the means in results/eval/summary.json (2026-11-07, stock, seeds 1-10, sizes S and M)
minus 0.10. The healthy-machine false-positive ceiling is the measured maximum (0) plus 0.05.
The null-driver ceiling is the measured rate plus 0.10.
"""
import pytest

from bench.evaluate import run

MEASURED_MEAN = {
    "drift_precision": 1.0,
    "drift_recall": 0.5333,
    "coolant_recall": 0.9,
    "planted_in_top3": 0.7125,
    "coverage_count": 0.8724,
    "accuracy_count": 0.965,
    "accuracy_minutes": 0.6145,
}
MEAN_MARGIN = 0.10
NULL_RATE = 0.25
FP_CEILING = 0.05


@pytest.mark.slow
def test_detection_floors(tmp_path):
    summary = run(list(range(1, 11)), ["S", "M"], "stock", tmp_path)
    agg = summary["aggregate"]
    assert agg["drift"]["precision_mean"] >= MEASURED_MEAN["drift_precision"] - MEAN_MARGIN
    assert agg["drift"]["recall_mean"] >= MEASURED_MEAN["drift_recall"] - MEAN_MARGIN
    assert agg["signal_health"]["coolant_recall_mean"] >= MEASURED_MEAN["coolant_recall"] - MEAN_MARGIN
    assert agg["signal_health"]["healthy_false_positive_rate_max"] <= FP_CEILING
    assert agg["drivers"]["planted_in_top3_mean"] >= MEASURED_MEAN["planted_in_top3"] - MEAN_MARGIN
    assert agg["drivers"]["any_null_reported_rate"] <= NULL_RATE + MEAN_MARGIN
    assert agg["stops"]["coverage_count_mean"] >= MEASURED_MEAN["coverage_count"] - MEAN_MARGIN
    assert agg["stops"]["accuracy_count_mean"] >= MEASURED_MEAN["accuracy_count"] - MEAN_MARGIN
    assert agg["stops"]["accuracy_minutes_mean"] >= MEASURED_MEAN["accuracy_minutes"] - MEAN_MARGIN
    m7 = next(r for r in summary["runs"] if r["size"] == "M" and r["seed"] == 7)
    # Slide figures for plant M seed 7, one decimal, from this same grid.
    assert round(m7["stops"]["coverage_count"] * 100, 1) == 85.4
    assert round(m7["stops"]["accuracy_count"] * 100, 1) == 98.0
    assert round(m7["stops"]["accuracy_minutes"] * 100, 1) == 67.2
