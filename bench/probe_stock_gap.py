"""Why is stock scikit-learn so much slower on a 12-thread laptop than on a 2-vCPU Xeon?

Times the four on-demand tool calls (one line, as in ``bench.run_bench``) in a
fresh process per case and backend, split into prep, ML and post phases. The
cases change one thing at a time:

* ``default``:  the tools as they are (``n_jobs=-1`` in drift, drivers, stops)
* ``njobs1``, ``njobs2``: ``n_jobs`` forced to 1 or 2 for every estimator that takes it
* ``threads1``: OpenMP, BLAS and TBB capped at 1 thread, ``n_jobs`` untouched
* ``pcores``, ``ecores``: process pinned to the performance or efficient cores
  (Windows only; core classes from ``GetSystemCpuSetInformation``)
* ``noecoqos``: the process opts out of Windows power throttling (EcoQoS) for
  itself with ``SetProcessInformation``; no system setting changes

It also records the power plan, AC state, thread pools, and a single-thread
calibration (a pure-Python loop and a NumPy sort) pinned to one P-core and
one E-core. On Windows each case also samples the logical processor its main
thread is on just before and just after every tool call, and reports the share
of samples on P-cores (``main_thread``). The tools are not changed.

    python -m bench.probe_stock_gap --out results/findings/stock-gap.json
    python -m bench.probe_stock_gap --cases default noecoqos --out results/findings/x.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS", "DAAL_NUM_THREADS",
)
CASES = ("default", "njobs1", "njobs2", "threads1", "pcores", "ecores", "noecoqos")
TOOL_ORDER = ("cluster_stops", "detect_cycle_drift", "rank_loss_drivers", "signal_health")


# ------------------------------------------------------------------ Windows CPU sets
def cpu_sets() -> list[dict]:
    """Logical processors with their core and efficiency class. Empty off Windows."""
    if platform.system() != "Windows":
        return []
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetSystemCpuSetInformation
    fn.restype = wintypes.BOOL
    fn.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG]
    need = wintypes.ULONG(0)
    fn(None, 0, ctypes.byref(need), None, 0)
    if need.value == 0:
        return []
    buf = (ctypes.c_ubyte * need.value)()
    if not fn(buf, need, ctypes.byref(need), None, 0):
        return []
    raw = bytes(buf)
    out, off = [], 0
    while off < len(raw):
        size = int.from_bytes(raw[off:off + 4], "little")
        if size == 0:
            break
        out.append({"logical": raw[off + 14], "core": raw[off + 15], "efficiency_class": raw[off + 18]})
        off += size
    return out


def core_masks() -> dict[str, int]:
    """Affinity masks for the fastest and slowest efficiency classes (the P-cores and E-cores)."""
    sets = cpu_sets()
    if not sets:
        return {}
    hi = max(s["efficiency_class"] for s in sets)
    lo = min(s["efficiency_class"] for s in sets)
    if hi == lo:
        return {}
    p = sum(1 << s["logical"] for s in sets if s["efficiency_class"] == hi)
    e = sum(1 << s["logical"] for s in sets if s["efficiency_class"] == lo)
    return {"pcores": p, "ecores": e}


_GET_CPU = None


def current_cpu() -> int | None:
    """Logical processor the calling thread is on right now. None off Windows."""
    global _GET_CPU
    if platform.system() != "Windows":
        return None
    if _GET_CPU is None:
        import ctypes
        _GET_CPU = ctypes.WinDLL("kernel32").GetCurrentProcessorNumber
    return int(_GET_CPU())


def placement(cpus: list[int | None], sets: list[dict]) -> dict:
    """Share of main-thread samples on the fastest efficiency class (the P-cores).

    ``pcore_share`` is None when there are no samples or the CPU has one core class.
    """
    cls = {s["logical"]: s["efficiency_class"] for s in sets}
    seen = [cls[c] for c in cpus if c is not None and c in cls]
    if not seen or len(set(cls.values())) < 2:
        return {"samples": len(seen), "pcore_share": None}
    hi = max(cls.values())
    return {"samples": len(seen), "pcore_share": round(sum(c == hi for c in seen) / len(seen), 3)}


def pin(mask: int) -> None:
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")


def opt_out_of_ecoqos() -> None:
    """Ask Windows not to throttle this process's execution speed (EcoQoS). Affects this process only."""
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    process_power_throttling, execution_speed = 4, 0x1
    st = State(1, execution_speed, 0)  # control execution speed, state off = never throttle
    if not k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling, ctypes.byref(st),
                                     ctypes.sizeof(st)):
        raise OSError(ctypes.get_last_error(), "SetProcessInformation(ProcessPowerThrottling) failed")


def power_state() -> dict:
    if platform.system() != "Windows":
        return {}

    def ps(cmd: str) -> str:
        try:
            return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                                  timeout=30, check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=False).stdout.strip()
    ac = ps("(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | Select-Object -First 1).PowerOnline")
    throttle = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_PROCESSOR", "PROCTHROTTLEMAX"],
                              capture_output=True, text=True, check=False).stdout
    lines = [ln.strip() for ln in throttle.splitlines() if "Current AC Power Setting Index" in ln
             or "Current DC Power Setting Index" in ln]
    # Settings > Power mode is an overlay on the plan (Best power efficiency / Balanced / Best performance)
    overlay = {}
    for key in ("ActiveOverlayAcPowerScheme", "ActiveOverlayDcPowerScheme"):
        out = subprocess.run(["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes",
                              "/v", key], capture_output=True, text=True, check=False).stdout
        overlay[key] = next((ln.split()[-1] for ln in out.splitlines() if key in ln), None)
    return {"active_scheme": scheme, "ac_power_online": ac, "max_processor_state": lines, "power_mode_overlay": overlay}


# ------------------------------------------------------------------ child process
def _pools() -> list[dict]:
    from threadpoolctl import threadpool_info
    return [{"user_api": r.get("user_api"), "internal_api": r.get("internal_api"), "num_threads": r.get("num_threads")}
            for r in threadpool_info()]


def _force_njobs(n: int) -> None:
    """Wrap the backend's estimator classes so every instance gets ``n_jobs=n``."""
    from tools import backend as B
    original = B.estimators

    def wrap(cls):
        def make(*a, **kw):
            if "n_jobs" in kw:
                kw["n_jobs"] = n
            return cls(*a, **kw)
        return make

    def patched(name=None):
        ns = original(name)
        for attr in ("DBSCAN", "KMeans", "NearestNeighbors", "RandomForestRegressor", "PCA"):
            setattr(ns, attr, wrap(getattr(ns, attr)))
        if ns.stock is not ns:
            for attr in ("DBSCAN", "KMeans", "NearestNeighbors", "RandomForestRegressor", "PCA"):
                setattr(ns.stock, attr, wrap(getattr(ns.stock, attr)))
        return ns

    B.estimators = patched
    for mod in ("drift", "drivers", "signals", "stops"):
        m = sys.modules.get(f"tools.{mod}")
        if m is not None and hasattr(m, "B"):
            m.B = B


def _calibrate(reps: int = 5) -> dict:
    """Single-thread speed: a pure-Python loop and a NumPy sort, best of ``reps``."""
    import numpy as np

    def loop():
        s = 0
        for i in range(2_000_000):
            s += i * i
        return s

    rng = np.random.default_rng(0)
    arr = rng.random(2_000_000)

    def best(f):
        f()
        t = []
        for _ in range(reps):
            s = time.perf_counter()
            f()
            t.append((time.perf_counter() - s) * 1000)
        return round(min(t), 2)

    return {"python_loop_ms": best(loop), "numpy_sort_ms": best(lambda: np.sort(arr))}


def _child(case: str, backend: str, size: str, seed: int, repeats: int) -> dict:
    masks = core_masks()
    if case in ("pcores", "ecores"):
        if case not in masks:
            return {"case": case, "backend": backend, "skipped": "no hybrid core classes on this CPU"}
        pin(masks[case])
    if case == "noecoqos":
        if platform.system() != "Windows":
            return {"case": case, "backend": backend, "skipped": "Windows only"}
        opt_out_of_ecoqos()
    from bench.run_bench import on_demand_scope
    from plantgen import generate_plant
    from tools import TOOLS, set_backend
    set_backend(backend)
    from tools import estimators
    estimators()
    if case.startswith("njobs"):
        _force_njobs(int(case.removeprefix("njobs")))
    plant = generate_plant(size, seed=seed)
    tools = {}
    where = []  # logical processor of the main thread, sampled outside the timed window
    for name in TOOL_ORDER:
        fn = TOOLS[name]
        scope = on_demand_scope(plant, name)
        fn(plant, scope)  # warm-up
        walls, cpus, phases = [], [], []
        for _ in range(repeats):
            where.append(current_cpu())
            tc, tw = time.process_time(), time.perf_counter()
            env = fn(plant, scope)
            walls.append((time.perf_counter() - tw) * 1000)
            cpus.append((time.process_time() - tc) * 1000)
            where.append(current_cpu())
            phases.append(env["timings_ms"])
        order = sorted(range(len(walls)), key=walls.__getitem__)
        mid = order[len(order) // 2]
        ph = phases[mid]
        tools[name] = {
            "scope": scope,
            "median_ms": round(walls[mid], 2),
            "min_ms": round(min(walls), 2),
            "max_ms": round(max(walls), 2),
            "cpu_ms_at_median": round(cpus[mid], 2),
            "ml_ms_at_median": round(sum(v for k, v in ph.items() if k.startswith("ml_")), 2),
            "phases_ms_at_median": ph,
        }
    return {
        "case": case,
        "backend": backend,
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "affinity_mask": hex(masks[case]) if case in masks else None,
        "pools": _pools(),
        "answer_ms": round(sum(t["median_ms"] for t in tools.values()), 1),
        "main_thread": placement(where, cpu_sets()),
        "main_thread_cpus": where,
        "tools": tools,
    }


def _overhead_child(backend: str, reps: int = 7) -> dict:
    """Per-call cost of the estimator calls the tools repeat, at n_jobs -1, 1 and 2."""
    import numpy as np

    from tools import backend as B
    est = B.estimators(backend)
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.normal(size=(4_000, 12)))
    y = rng.normal(size=4_000)
    Xp = np.ascontiguousarray(X[:3_000])
    R = np.ascontiguousarray(rng.normal(size=(800, 6)))
    Q = np.ascontiguousarray(rng.normal(size=(200, 6)))

    def best(f):
        f()
        t = []
        for _ in range(reps):
            s = time.perf_counter()
            f()
            t.append((time.perf_counter() - s) * 1000)
        t.sort()
        return round(t[len(t) // 2], 2)

    rows = []
    for n_jobs in (-1, 1, 2):
        rf = est.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=30, max_features=0.5,
                                       random_state=0, n_jobs=n_jobs)
        fit_ms = best(lambda: rf.fit(X, y))
        nn = est.NearestNeighbors(n_neighbors=11, n_jobs=n_jobs).fit(R)
        rows.append({
            "n_jobs": n_jobs,
            "rf_fit_4000x12_ms": fit_ms,
            "rf_predict_1_row_ms": best(lambda: rf.predict(Xp[:1])),
            "rf_predict_3000_rows_ms": best(lambda: rf.predict(Xp)),
            "knn_fit_query_800_200_ms": best(lambda: (nn.fit(R), nn.kneighbors(R), nn.kneighbors(Q, n_neighbors=10))),
        })
    return {"backend": backend, "pools": _pools(), "rows": rows}


def _calibration_child(where: str) -> dict:
    sets = cpu_sets()
    if sets:
        cls = max(s["efficiency_class"] for s in sets) if where == "pcore" else min(s["efficiency_class"] for s in sets)
        logical = next(s["logical"] for s in sets if s["efficiency_class"] == cls)
        pin(1 << logical)
    else:
        logical = None
    return {"where": where, "logical_processor": logical, **_calibrate()}


def _spawn(args: list[str], env_extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-m", "bench.probe_stock_gap", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=CASES)
    ap.add_argument("--calibrate", choices=("pcore", "ecore"))
    ap.add_argument("--overhead", action="store_true")
    ap.add_argument("--backend", choices=("stock", "onedal"), default="stock")
    ap.add_argument("--backends", nargs="+", default=["stock", "onedal"])
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--size", default="M")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "results" / "findings" / "stock-gap.json"))
    args = ap.parse_args(argv)

    if args.calibrate:
        print(json.dumps(_calibration_child(args.calibrate)))
        return 0
    if args.overhead:
        import warnings
        warnings.filterwarnings("ignore")
        print(json.dumps(_overhead_child(args.backend), default=str))
        return 0
    if args.child:
        import warnings
        warnings.filterwarnings("ignore")
        print(json.dumps(_child(args.child, args.backend, args.size, args.seed, args.repeats), default=str))
        return 0

    from bench.run_bench import machine_info
    payload = {
        "machine": machine_info(),
        "power": power_state(),
        "cpu_sets": cpu_sets(),
        "size": args.size, "seed": args.seed, "repeats": args.repeats,
        "calibration": [],
        "overhead": [],
        "cases": [],
    }
    if payload["cpu_sets"]:
        for where in ("pcore", "ecore"):
            row = _spawn(["--calibrate", where])
            payload["calibration"].append(row)
            print(f"calibration {where}: loop {row['python_loop_ms']} ms, sort {row['numpy_sort_ms']} ms", flush=True)
    for backend in args.backends:
        row = _spawn(["--overhead", "--backend", backend], {"PLANT_TOOLS_BACKEND": backend})
        payload["overhead"].append(row)
        for r in row["rows"]:
            print(f"overhead {backend:6s} n_jobs {r['n_jobs']:2d}: rf fit {r['rf_fit_4000x12_ms']} ms, "
                  f"predict 1 row {r['rf_predict_1_row_ms']} ms, 3000 rows {r['rf_predict_3000_rows_ms']} ms, "
                  f"knn {r['knn_fit_query_800_200_ms']} ms", flush=True)
    for case in args.cases:
        for backend in args.backends:
            extra = {k: "1" for k in THREAD_VARS} if case == "threads1" else {}
            extra["PLANT_TOOLS_BACKEND"] = backend
            row = _spawn(["--child", case, "--backend", backend, "--size", args.size, "--seed", str(args.seed),
                          "--repeats", str(args.repeats)], extra)
            payload["cases"].append(row)
            if "skipped" in row:
                print(f"{case:9s} {backend:6s} skipped: {row['skipped']}", flush=True)
                continue
            per = "  ".join(f"{n.split('_')[-1]} {row['tools'][n]['median_ms']:.0f}" for n in TOOL_ORDER)
            share = row["main_thread"]["pcore_share"]
            on_p = "" if share is None else f"  main thread on P-cores {share:.0%}"
            print(f"{case:9s} {backend:6s} answer {row['answer_ms']:8.0f} ms  ({per}){on_p}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
