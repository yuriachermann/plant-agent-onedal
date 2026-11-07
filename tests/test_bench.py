"""Benchmark helpers: percentiles, old result files, thread caps, Windows CPU info, plain output."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bench.kernels import THREAD_VARS, thread_limit_env
from bench.report import main as report_main
from bench.report import spread
from bench.run_bench import percentile, resolve_seeds

REPO = Path(__file__).resolve().parents[1]
TOOLS = ("cluster_stops", "detect_cycle_drift", "rank_loss_drivers", "signal_health")


def test_resolve_seeds_keeps_the_single_seed_default():
    assert resolve_seeds(7, None) == [7]
    assert resolve_seeds(7, [1, 2, 3]) == [1, 2, 3]


def test_p50_is_the_middle_element_for_an_odd_count():
    xs = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(xs, 0.5) == xs[len(xs) // 2]
    assert percentile(xs, 0.9) == pytest.approx(7.6)


def test_spread_is_missing_on_old_records():
    assert spread({"median_ms": 51.2}) == "n/a"
    assert spread({"median_ms": 51.2, "p50_ms": 51.2, "p90_ms": 80.4}) == "51–80"


def _plant_block(ms: float, p50=None, p90=None) -> dict:
    tool = {"median_ms": ms, "ml_median_ms": ms / 2, "cpu_median_ms": ms, "min_ms": ms, "max_ms": ms}
    if p50 is not None:
        tool["p50_ms"] = p50
        tool["p90_ms"] = p90
    tools = {name: dict(tool) for name in TOOLS}
    phase = {"wall_ms": ms, "cpu_ms": ms, "phases_ms": {}}
    return {
        "versions": {"backend": "stock", "sklearn": "1.9.1", "sklearnex": "2026.2.0", "onedal": "2026.2.0"},
        "machine": {"cpu": "test", "logical_cpus": 2, "isa": [], "mem_gb": 1.0, "os": "test", "python": "3.12.2"},
        "seed": 7, "repeats": 3,
        "sizes": {"S": {
            "plant": {"machines": 12, "stops_per_day": 10.0, "cycles_per_day": 10.0},
            "on_demand": tools,
            "answer_ms": ms * 4,
            "answer_cpu_ms": ms * 4,
            "nightly": {
                "cold": {"wall_ms": 1000.0, "cpu_ms": 1000.0, "tools": {n: dict(phase) for n in TOOLS}},
                "warm": {"wall_ms": 900.0, "cpu_ms": 900.0, "tools": {n: dict(phase) for n in TOOLS}},
            },
        }},
    }


def _write_pair(tmp: Path, stock_ms: float, onedal_ms: float, **extra):
    stock, onedal = _plant_block(stock_ms, **extra), _plant_block(onedal_ms, **extra)
    stock["versions"] = {"backend": "stock", "sklearn": "1.9.1"}
    (tmp / "bench_stock.json").write_text(json.dumps(stock))
    (tmp / "bench_onedal.json").write_text(json.dumps(onedal))


def test_report_prints_p50_p90_and_keeps_median(tmp_path):
    _write_pair(tmp_path, 40, 10, p50=40, p90=55)
    report_main([str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    row = summary["sizes"]["S"]["on_demand"]["cluster_stops"]
    assert row["stock_ms"] == 40
    assert row["onedal_ms"] == 10
    assert row["stock_p90_ms"] == 55
    assert "40–55" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_old_xeon_bench_json_still_parses(tmp_path):
    src = REPO / "results" / "intel-xeon-2t"
    for name in ("bench_stock.json", "bench_onedal.json"):
        shutil.copy(src / name, tmp_path / name)
    report_main([str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    stock = json.loads((src / "bench_stock.json").read_text())
    onedal = json.loads((src / "bench_onedal.json").read_text())
    assert summary["sizes"]["M"]["answer"]["stock_ms"] == stock["sizes"]["M"]["answer_ms"]
    assert summary["sizes"]["M"]["answer"]["onedal_ms"] == onedal["sizes"]["M"]["answer_ms"]
    assert summary["sizes"]["M"]["on_demand"]["rank_loss_drivers"]["stock_ms"] == (
        stock["sizes"]["M"]["on_demand"]["rank_loss_drivers"]["median_ms"])
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "n/a" in md
    assert "thread_sweep" not in summary


def test_thread_sweep_section(tmp_path):
    _write_pair(tmp_path, 40, 10)

    def kernel(ms):
        return {
            "threads": 1, "threads_requested": "1",
            "kernels": {
                "dbscan": [{"n": 1000, "ms": ms}],
                "kmeans": [{"n": 2500, "ms": ms}],
                "knn": [{"n_ref": 500, "ms": ms}],
                "rf_fit": [{"n": 4000, "ms": ms}],
                "rf_predict": [{"n": 1000, "ms": ms}],
                "pca": [{"n": 500, "ms": ms}],
            },
        }

    (tmp_path / "kernels_stock_t1.json").write_text(json.dumps(kernel(10)))
    (tmp_path / "kernels_onedal_t1.json").write_text(json.dumps(kernel(5)))
    report_main([str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["thread_sweep"]["dbscan"]["1"]["largest_speedup"] == 2.0
    assert "Thread sweep" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_thread_limit_env_sets_the_pool_variables():
    env = thread_limit_env("1")
    assert env["PLANT_KERNELS_THREADS"] == "1"
    for key in THREAD_VARS:
        assert env[key] == "1"
    assert thread_limit_env("all")["PLANT_KERNELS_THREADS"] == str(os.cpu_count() or 1)


def test_kernels_apply_thread_cap_in_a_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "bench.kernels", "--threads", "1", "--dump-threads"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    got = json.loads(proc.stdout)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
        assert got[key] == "1"

