"""Kernel sweeps: where does oneDAL win, and where does it not?

Runs single estimator calls on stop/cycle features from the large plant at
several input sizes. One backend per process (see run_bench.py).

* dbscan:  stock (KD-tree, algorithm='auto') vs oneDAL (brute force)
* kmeans:  fit with a shared k-means++ init
* knn:     fit + query on one machine-program-sized block, at several sizes
* rf_fit:  RandomForestRegressor fit on machine-hours
* rf_predict: batch-size sweep (interactive vs batch inference)
* pca:     fit + transform per machine-program block
"""
from __future__ import annotations

import os
import subprocess
import sys

# Set before NumPy or oneDAL are imported. ``--threads`` re-execs this module
# in a child whose BLAS / TBB pools are already capped.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TBB_NUM_THREADS",
    "DAAL_NUM_THREADS",
)


def thread_limit_env(n: str) -> dict:
    """Environment for a process whose numeric libraries use ``n`` threads.

    ``n`` is a positive integer, or ``"all"`` for the machine's logical CPU count.
    """
    value = str(os.cpu_count() or 1) if n == "all" else str(int(n))
    env = os.environ.copy()
    env["PLANT_KERNELS_THREADS_READY"] = "1"
    env["PLANT_KERNELS_THREADS"] = value
    env["PLANT_KERNELS_THREADS_REQUESTED"] = n
    for key in THREAD_VARS:
        env[key] = value
    return env


def _reexec_for_threads() -> None:
    """Run the sweep in a child with the thread cap set, then exit this process."""
    if os.environ.get("PLANT_KERNELS_THREADS_READY") == "1":
        return
    argv = sys.argv[1:]
    if "--threads" not in argv:
        return
    n = argv[argv.index("--threads") + 1]
    proc = subprocess.run([sys.executable, "-m", "bench.kernels", *argv], env=thread_limit_env(n))
    os._exit(proc.returncode)


if __name__ == "__main__":
    _reexec_for_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from bench.hybrid import apply_ecoqos_policy  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def best_of(f, reps: int) -> float:
    f()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return round(min(t) * 1000, 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stock", "onedal"])
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", help="1, 2, all, or a count. Applied in a subprocess before numeric imports.")
    ap.add_argument("--dump-threads", action="store_true", help="print the thread environment and exit")
    args = ap.parse_args(argv)
    if args.dump_threads:
        print(json.dumps({k: os.environ.get(k) for k in THREAD_VARS}, sort_keys=True))
        return 0
    if not args.backend or not args.out:
        ap.error("--backend and --out are required")

    ecoqos = apply_ecoqos_policy()  # Windows: same EcoQoS policy as bench.run_bench

    from sklearn.cluster import kmeans_plusplus

    from plantgen import SIGNALS, generate_plant
    from tools import estimators
    from tools.stops import stop_features
    est = estimators(args.backend)
    plant = generate_plant("L", seed=args.seed)
    rng = np.random.default_rng(0)
    Xs_all = stop_features(plant.stops)
    requested = os.environ.get("PLANT_KERNELS_THREADS_REQUESTED")
    threads = os.environ.get("PLANT_KERNELS_THREADS")
    res = {"backend": args.backend, "plant": plant.summary(), "kernels": {},
           "threads": int(threads) if threads else None, "threads_requested": requested}
    if ecoqos is not None:
        res["ecoqos_opt_out"] = ecoqos
    K = res["kernels"]

    K["dbscan"] = []
    for n in [1_000, 2_500, 5_000, 10_000, 20_000, 40_000, 80_000]:
        X = np.ascontiguousarray(Xs_all[rng.permutation(len(Xs_all))[:n]])
        alg = "brute" if args.backend == "onedal" else "auto"
        reps = 3 if n <= 20_000 else 1
        K["dbscan"].append({"n": n, "ms": best_of(lambda: est.DBSCAN(eps=0.5, min_samples=30, algorithm=alg, n_jobs=-1).fit(X), reps)})

    K["kmeans"] = []
    for n in [2_500, 20_000, 118_000]:
        X = np.ascontiguousarray(Xs_all[:n])
        c, _ = kmeans_plusplus(X, 10, random_state=0)
        K["kmeans"].append({"n": n, "ms": best_of(lambda: est.KMeans(10, init=c, n_init=1).fit(X), 3)})

    C = plant.cycles
    F = ["cycle_s", "spindle_load", "feed_override", "vibration_rms", "motor_current", "spindle_speed"]
    Xc = C[F].to_numpy(np.float64)
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    K["knn"] = []
    for n_ref in [500, 2_000, 8_000, 32_000]:
        R = np.ascontiguousarray(Xc[:n_ref])
        Q = np.ascontiguousarray(Xc[n_ref:n_ref + n_ref // 3])

        def knn():
            nn = est.NearestNeighbors(n_neighbors=11, n_jobs=-1).fit(R)
            nn.kneighbors(R)
            nn.kneighbors(Q, n_neighbors=10)
        K["knn"].append({"n_ref": n_ref, "ms": best_of(knn, 3)})

    H = plant.hours[plant.hours["unplanned_rate"].notna()]
    import pandas as pd

    from tools.drivers import CATS, NUMS
    Xh = np.column_stack([*[pd.Categorical(H[c]).codes for c in CATS], *[H[c].to_numpy(float) for c in NUMS]]).astype(np.float64)
    yh = H["unplanned_rate"].to_numpy()
    K["rf_fit"] = []
    for n in [4_000, 16_000, len(yh)]:
        X, y = np.ascontiguousarray(Xh[:n]), yh[:n]
        K["rf_fit"].append({"n": n, "ms": best_of(lambda: est.RandomForestRegressor(
            200, max_depth=10, min_samples_leaf=30, max_features=0.5, random_state=0, n_jobs=-1).fit(X, y), 1)})
    rf = est.RandomForestRegressor(200, max_depth=10, min_samples_leaf=30, max_features=0.5, random_state=0,
                                   n_jobs=-1).fit(Xh, yh)
    K["rf_predict"] = []
    for n in [1, 100, 1_000, 3_000, 10_000, 40_000]:
        X = np.ascontiguousarray(Xh[:n])
        K["rf_predict"].append({"n": n, "ms": best_of(lambda: rf.predict(X), 5 if n <= 3_000 else 2)})

    Xp = C[SIGNALS].to_numpy(np.float64)
    Xp = (Xp - Xp.mean(0)) / Xp.std(0)
    K["pca"] = []
    for n in [500, 2_000, 8_000, 64_000]:
        X = np.ascontiguousarray(Xp[:n])

        def pca():
            p = est.PCA(n_components=0.9, svd_solver="full").fit(X)
            p.transform(X)
        K["pca"].append({"n": n, "ms": best_of(pca, 3)})

    Path(args.out).mkdir(parents=True, exist_ok=True)
    suffix = ""
    if requested == "all":
        suffix = "_tall"
    elif requested:
        suffix = f"_t{requested}"
    (Path(args.out) / f"kernels_{args.backend}{suffix}.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.backend}] kernels done" + (f" threads={requested}" if requested else ""), flush=True)


if __name__ == "__main__":
    main()
