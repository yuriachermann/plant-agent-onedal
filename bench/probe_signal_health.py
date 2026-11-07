"""Time stock signal_health in fresh processes under different thread pools.

Importing scikit-learn-intelex loads oneTBB. Stock PCA then shares the machine
with that pool. This probe measures that, it does not change the tool.

    python -m bench.probe_signal_health --out results/findings/signal-health-slowdown.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS", "DAAL_NUM_THREADS",
)
CASES = ("stock", "import_sklearnex", "after_rf", "threads1_stock", "threads1_import_sklearnex", "threads1_after_rf")


def _pools() -> list[dict]:
    from threadpoolctl import threadpool_info
    out = []
    for row in threadpool_info():
        out.append({
            "user_api": row.get("user_api"),
            "internal_api": row.get("internal_api"),
            "num_threads": row.get("num_threads"),
            "prefix": row.get("prefix"),
        })
    return out


def _case(name: str, size: str, repeats: int, seed: int) -> dict:
    limited = name.startswith("threads1_")
    kind = name.removeprefix("threads1_")
    if kind in ("import_sklearnex", "after_rf"):
        import sklearnex  # noqa: F401
    pools_after_import = _pools()
    if kind == "after_rf":
        import numpy as np
        from sklearnex.ensemble import RandomForestRegressor
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.normal(size=(2_000, 8)))
        y = rng.normal(size=2_000)
        RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0, n_jobs=-1).fit(x, y)
    from plantgen import generate_plant
    from tools import TOOLS, set_backend
    set_backend("stock")
    plant = generate_plant(size, seed=seed)
    TOOLS["signal_health"](plant, {"line_id": "L1"})  # warm-up, one line
    walls = []
    for _ in range(repeats):
        t = time.perf_counter()
        TOOLS["signal_health"](plant, {"line_id": "L1"})
        walls.append((time.perf_counter() - t) * 1000)
    walls.sort()
    return {
        "case": name,
        "threads_limited": limited,
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "pools_after_import": pools_after_import,
        "pools_after_run": _pools(),
        "n": len(walls),
        "median_ms": round(walls[len(walls) // 2], 2),
        "min_ms": round(walls[0], 2),
        "max_ms": round(walls[-1], 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--size", default="M")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(REPO / "results" / "findings" / "signal-health-slowdown.json"))
    args = ap.parse_args(argv)
    if args.case:
        json.dump(_case(args.case, args.size, args.repeats, args.seed), sys.stdout)
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        env["PLANT_TOOLS_BACKEND"] = "stock"
        if case.startswith("threads1_"):
            for key in THREAD_VARS:
                env[key] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "bench.probe_signal_health", "--case", case, "--size", args.size,
             "--repeats", str(args.repeats), "--seed", str(args.seed)],
            cwd=REPO, env=env, capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        row = json.loads(proc.stdout)
        rows.append(row)
        print(f"{case:28s} median {row['median_ms']:8.1f} ms", flush=True)
    payload = {"size": args.size, "seed": args.seed, "repeats": args.repeats, "cases": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
