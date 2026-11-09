"""Benchmark: stock scikit-learn vs oneDAL behind the agent's tools.

Each backend runs in its own fresh Python process, so thread pools and caches
never leak between them. Three measurements per plant size:

* on_demand: one tool call scoped to one line, as when someone asks in chat
  (1 warm-up, then --repeats timed calls; median reported)
* answer:    the four tool calls an agent typically chains for one question
* nightly:   all four tools over the whole plant; the first run in the fresh
  process is reported as "cold", the median of the next two as "warm";
  CPU-seconds come from process_time (all threads)

On-demand calls report the existing median plus p50 and p90 of the timed
repeats. ``--seeds`` pools those repeats across plants; the default is still
the single ``--seed`` (7). Nightly stays on the first seed.

The kernel sweep also runs at 1 thread, 2 threads, and all logical CPUs.
Those processes set the BLAS and TBB thread variables before importing NumPy.

On Windows every timed process opts out of power throttling (EcoQoS), which
otherwise tends to put the main thread on the E-cores of a hybrid CPU, and
records ``ecoqos_opt_out`` in its machine block. ``--keep-ecoqos`` leaves
EcoQoS to Windows, as runs before 2026-11-09 did. Each backend also records
the share of on-demand calls whose main thread was on a P-core (``main_thread``).

Usage (writes results/<machine-tag>/):
    python -m bench.run_bench                      # both backends, sizes S M L
    python -m bench.run_bench --sizes M --repeats 3
    python -m bench.run_bench --seeds 1 2 3        # on-demand spread across plants
    python -m bench.run_bench --backend onedal     # single backend, in-process
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

from bench.hybrid import ECOQOS_ENV, apply_ecoqos_policy, cpu_sets, current_cpu, placement

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _windows_cpu_mem() -> tuple[str, float | None]:
    """CPU name and RAM from CIM. PowerShell is the path that works on the plant PCs."""
    script = (
        "$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "$m = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
        "Write-Output $p.Name; Write-Output $m"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return "", None
    cpu = out[0].strip() if out else ""
    mem = None
    if len(out) > 1 and out[1].strip().isdigit():
        mem = round(int(out[1].strip()) / 1024 ** 3, 1)
    return cpu, mem


def _windows_power() -> dict:
    """AC or battery, and the active power plan. Laptop timings move with both (docs/findings/stock-gap.md)."""
    out = {"ac_power": None, "power_plan": None}
    try:
        ac = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue"
             " | Select-Object -First 1).PowerOnline"],
            capture_output=True, text=True, timeout=20, check=False).stdout.strip()
        plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=20,
                              check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return out
    # no battery (desktop, server) reports nothing: that is mains power
    out["ac_power"] = {"True": True, "False": False}.get(ac, True if ac == "" else None)
    m = re.search(r"\(([^)]+)\)\s*$", plan.strip())
    out["power_plan"] = m.group(1) if m else None
    return out


def machine_info() -> dict:
    cpu = platform.processor() or ""
    isa: list[str] = []
    mem_gb = None
    power: dict = {}
    if platform.system() == "Windows":
        wcpu, mem_gb = _windows_cpu_mem()
        cpu = wcpu or cpu
        power = _windows_power()
    else:
        try:
            txt = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", txt)
            cpu = m.group(1).strip() if m else cpu
            flags = re.search(r"flags\s*:\s*(.+)", txt)
            isa = [f for f in ("avx2", "avx512f", "avx512_vnni", "amx_tile") if flags and f in flags.group(1).split()]
        except FileNotFoundError:
            pass
        try:
            mem_gb = round(int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 1024 ** 2, 1)
        except (OSError, AttributeError, ValueError):
            pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "isa": isa, "mem_gb": mem_gb,
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(), **power}


def machine_tag(info: dict) -> str:
    s = re.sub(r"\(r\)|\(tm\)|cpu|processor|@.*$", "", info["cpu"].lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return f"{s}-{info['logical_cpus']}t"


def resolve_seeds(seed: int, seeds: list[int] | None) -> list[int]:
    """``--seeds`` wins when it is passed. Otherwise the single ``--seed`` is the whole run."""
    return [int(s) for s in seeds] if seeds else [int(seed)]


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear percentile. ``p`` is in [0, 1]. For an odd count, p=0.5 is the middle element."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def on_demand_scope(plant, tool: str) -> dict:
    n_lines = plant.machines["line_id"].nunique()
    line = "L3" if n_lines >= 3 else f"L{n_lines}"
    if tool == "cluster_stops":
        return {"line_id": line, "day_from": plant.days - 7}
    return {"line_id": line}


def run_backend(backend: str, sizes: list[str], repeats: int, seeds: list[int]) -> dict:
    ecoqos = apply_ecoqos_policy()  # before any timing; None off Windows
    t_import = time.perf_counter()
    from plantgen import generate_plant
    from tools import TOOLS, set_backend, versions
    set_backend(backend)
    from tools import estimators
    estimators()  # import the backend's estimator modules now, not inside the first timed call
    import_ms = (time.perf_counter() - t_import) * 1000

    machine = machine_info()
    if ecoqos is not None:
        machine["ecoqos_opt_out"] = ecoqos
    out = {"backend": backend, "versions": versions(backend), "machine": machine, "seed": seeds[0],
           "seeds": seeds, "repeats": repeats, "import_ms": round(import_ms, 1), "sizes": {}}
    where = []  # logical processor of the main thread around each timed on-demand call
    for size in sizes:
        # Nightly uses the first seed only, so a one-seed run does the same work as before.
        plants = [generate_plant(size, seed=s) for s in seeds]
        plant = plants[0]
        res = {"plant": plant.summary(), "nightly": {}, "on_demand": {}}

        # nightly first: the first pass in this process is the cold run
        runs = []
        for _ in range(3):
            c0, w0 = time.process_time(), time.perf_counter()
            per_tool = {}
            for name, fn in TOOLS.items():
                tc, tw = time.process_time(), time.perf_counter()
                env = fn(plant, None)
                per_tool[name] = {"wall_ms": round((time.perf_counter() - tw) * 1000, 1),
                                  "cpu_ms": round((time.process_time() - tc) * 1000, 1),
                                  "phases_ms": env["timings_ms"]}
                if name == "cluster_stops":
                    per_tool[name]["dbscan_path"] = env["params"].get("dbscan_path")
            runs.append({"wall_ms": round((time.perf_counter() - w0) * 1000, 1),
                         "cpu_ms": round((time.process_time() - c0) * 1000, 1), "tools": per_tool})
        warm = sorted(runs[1:], key=lambda r: r["wall_ms"])[0 if len(runs) == 2 else len(runs[1:]) // 2]
        res["nightly"] = {"cold": runs[0], "warm": warm, "all_wall_ms": [r["wall_ms"] for r in runs]}

        for name, fn in TOOLS.items():
            walls, cpus, phases = [], [], []
            env = None
            for p in plants:
                scope = on_demand_scope(p, name)
                fn(p, scope)  # warm-up, per plant
                for _ in range(repeats):
                    where.append(current_cpu())
                    tc, tw = time.process_time(), time.perf_counter()
                    env = fn(p, scope)
                    walls.append((time.perf_counter() - tw) * 1000)
                    cpus.append((time.process_time() - tc) * 1000)
                    where.append(current_cpu())
                    phases.append(env["timings_ms"])
            walls.sort()
            cpus.sort()
            ml = sorted(sum(v for k, v in phase.items() if k.startswith("ml_")) for phase in phases)
            mid = len(walls) // 2
            res["on_demand"][name] = {
                "scope": on_demand_scope(plant, name),
                "median_ms": round(walls[mid], 2),
                "p50_ms": round(percentile(walls, 0.5), 2),
                "p90_ms": round(percentile(walls, 0.9), 2),
                "n_calls": len(walls),
                "min_ms": round(walls[0], 2),
                "max_ms": round(walls[-1], 2),
                "ml_median_ms": round(ml[mid], 2),
                "cpu_median_ms": round(cpus[mid], 2),
                "rows": _rows(env), "dbscan_path": env["params"].get("dbscan_path"),
            }
        res["answer_ms"] = round(sum(v["median_ms"] for v in res["on_demand"].values()), 1)
        res["answer_cpu_ms"] = round(sum(v["cpu_median_ms"] for v in res["on_demand"].values()), 1)
        out["sizes"][size] = res
        print(f"[{backend}] {size}: nightly cold {runs[0]['wall_ms']/1000:.1f}s warm {warm['wall_ms']/1000:.1f}s "
              f"cpu {warm['cpu_ms']/1000:.1f}s | answer {res['answer_ms']:.0f} ms", flush=True)
    out["main_thread"] = placement(where, cpu_sets())
    return out


def _rows(env) -> int | None:
    r = env["result"]
    for k in ("n_stops", "hours"):
        if k in r:
            return int(r[k])
    if "machine_programs_checked" in r:
        return int(r["machine_programs_checked"])
    if "models" in r:
        return int(r["models"])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stock", "onedal"])
    ap.add_argument("--sizes", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="on-demand plants. Default: just --seed, which stays 7")
    ap.add_argument("--out", default=None, help="output dir (default results/<machine-tag>)")
    ap.add_argument("--skip-kernels", action="store_true")
    ap.add_argument("--skip-thread-sweep", action="store_true")
    ap.add_argument("--keep-ecoqos", action="store_true",
                    help="Windows: leave power throttling (EcoQoS) to Windows instead of opting out")
    args = ap.parse_args(argv)
    if args.keep_ecoqos:
        os.environ[ECOQOS_ENV] = "keep"  # read by this process and inherited by the children below
    seeds = resolve_seeds(args.seed, args.seeds)
    outdir = Path(args.out) if args.out else REPO / "results" / machine_tag(machine_info())
    outdir.mkdir(parents=True, exist_ok=True)

    if args.backend:
        data = run_backend(args.backend, args.sizes, args.repeats, seeds)
        (outdir / f"bench_{args.backend}.json").write_text(json.dumps(data, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    for be in ("stock", "onedal"):
        cmd = [sys.executable, "-m", "bench.run_bench", "--backend", be, "--sizes", *args.sizes,
               "--repeats", str(args.repeats), "--seeds", *[str(s) for s in seeds], "--out", str(outdir)]
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
        if not args.skip_kernels:
            subprocess.run([sys.executable, "-m", "bench.kernels", "--backend", be, "--out", str(outdir)],
                           check=True, cwd=REPO, env=env)
            if not args.skip_thread_sweep:
                for threads in ("1", "2", "all"):
                    subprocess.run([sys.executable, "-m", "bench.kernels", "--backend", be, "--threads", threads,
                                    "--out", str(outdir)], check=True, cwd=REPO, env=env)
    subprocess.run([sys.executable, "-m", "bench.report", str(outdir)], check=True, cwd=REPO, env=env)
    print("results in", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
