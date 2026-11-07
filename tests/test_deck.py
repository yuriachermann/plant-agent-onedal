"""The slide table for the reference Xeon run matches the figures quoted in the talk notes."""
from pathlib import Path

from bench.deck_numbers import build

TAG = Path(__file__).resolve().parents[1] / "results" / "intel-xeon-2t"


def test_xeon_deck_matches_the_slide_figures():
    text = build(TAG)
    for figure in (
        "| One answer, plant S | 3,775 → 591 ms |",
        "| One answer, plant M | 4,104 → 1,026 ms |",
        "| One answer, plant L | 4,222 → 1,041 ms |",
        "| cluster_stops, plant M | 51 → 37 ms |",
        "| detect_cycle_drift, plant M | 1,003 → 336 ms |",
        "| rank_loss_drivers, plant M | 2,941 → 532 ms |",
        "| signal_health, plant M | 109 → 122 ms |",
        "| Stop coverage, M seed 7 | 85.4% |",
        "| Stop accuracy, M seed 7 | 98.0% |",
        "| Stop accuracy by minutes, M seed 7 | 67.2% |",
    ):
        assert figure in text, figure
    assert "summary.json sizes.M.answer.stock_ms" in text
    assert "family_ari" in text
    assert "kernels.dbscan" in text
