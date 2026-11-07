"""Benchmark: stock scikit-learn vs oneDAL behind the agent's tools.

Each backend runs in its own fresh Python process, so thread pools and caches
never leak between them. Three measurements per plant size:

* on_demand: one tool call scoped to one line, as when someone asks in chat
  (1 warm-up, then --repeats timed calls; median reported)
* answer:    the four tool calls an agent typically chains for one question
* nightly:   all four tools over the whole plant; the first run in the fresh
  process is reported as "cold", the median of the next two as "warm";
  CPU-seconds come from process_time (all threads)

Usage (writes results/<machine-tag>/):
    python -m bench.run_bench                      # both backends, sizes S M L
    python -m bench.run_bench --sizes M --repeats 3
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

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def machine_info() -> dict:
    cpu = platform.processor() or ""
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        cpu = m.group(1).strip() if m else cpu
        flags = re.search(r"flags\s*:\s*(.+)", txt)
        isa = [f for f in ("avx2", "avx512f", "avx512_vnni", "amx_tile") if flags and f in flags.group(1).split()]
    except FileNotFoundError:  # Windows / macOS
        isa = []
        if platform.system() == "Windows":
            try:
                cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                     capture_output=True, text=True, timeout=20).stdout.strip() or cpu
            except Exception:
                pass
    mem_gb = None
    try:
        mem_gb = round(int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 1024 ** 2, 1)
    except Exception:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "isa": isa, "mem_gb": mem_gb,
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def machine_tag(info: dict) -> str:
    s = re.sub(r"\(r\)|\(tm\)|cpu|processor|@.*$", "", info["cpu"].lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return f"{s}-{info['logical_cpus']}t"


def on_demand_scope(plant, tool: str) -> dict:
    n_lines = plant.machines["line_id"].nunique()
    line = "L3" if n_lines >= 3 else f"L{n_lines}"
    if tool == "cluster_stops":
        return {"line_id": line, "day_from": plant.days - 7}
    return {"line_id": line}


def run_backend(backend: str, sizes: list[str], repeats: int, seed: int) -> dict:
    t_import = time.perf_counter()
    from plantgen import generate_plant
    from tools import TOOLS, set_backend, versions
    set_backend(backend)
    from tools import estimators
    estimators()  # import the backend's estimator modules now, not inside the first timed call
    import_ms = (time.perf_counter() - t_import) * 1000

    out = {"backend": backend, "versions": versions(backend), "machine": machine_info(), "seed": seed,
           "repeats": repeats, "import_ms": round(import_ms, 1), "sizes": {}}
    for size in sizes:
        plant = generate_plant(size, seed=seed)
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
            scope = on_demand_scope(plant, name)
            fn(plant, scope)  # warm-up
            walls, cpus, phases = [], [], []
            for _ in range(repeats):
                tc, tw = time.process_time(), time.perf_counter()
                env = fn(plant, scope)
                walls.append((time.perf_counter() - tw) * 1000)
                cpus.append((time.process_time() - tc) * 1000)
                phases.append(env["timings_ms"])
            walls.sort()
            cpus.sort()
            ml = [sum(v for k, v in p.items() if k.startswith("ml_")) for p in phases]
            ml.sort()
            res["on_demand"][name] = {
                "scope": scope, "median_ms": round(walls[len(walls) // 2], 2), "min_ms": round(walls[0], 2),
                "max_ms": round(walls[-1], 2), "ml_median_ms": round(ml[len(ml) // 2], 2),
                "cpu_median_ms": round(cpus[len(cpus) // 2], 2),
                "rows": _rows(env), "dbscan_path": env["params"].get("dbscan_path"),
            }
        res["answer_ms"] = round(sum(v["median_ms"] for v in res["on_demand"].values()), 1)
        res["answer_cpu_ms"] = round(sum(v["cpu_median_ms"] for v in res["on_demand"].values()), 1)
        out["sizes"][size] = res
        print(f"[{backend}] {size}: nightly cold {runs[0]['wall_ms']/1000:.1f}s warm {warm['wall_ms']/1000:.1f}s "
              f"cpu {warm['cpu_ms']/1000:.1f}s | answer {res['answer_ms']:.0f} ms", flush=True)
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
    ap.add_argument("--out", default=None, help="output dir (default results/<machine-tag>)")
    ap.add_argument("--skip-kernels", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.out) if args.out else REPO / "results" / machine_tag(machine_info())
    outdir.mkdir(parents=True, exist_ok=True)

    if args.backend:
        data = run_backend(args.backend, args.sizes, args.repeats, args.seed)
        (outdir / f"bench_{args.backend}.json").write_text(json.dumps(data, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    for be in ("stock", "onedal"):
        cmd = [sys.executable, "-m", "bench.run_bench", "--backend", be, "--sizes", *args.sizes,
               "--repeats", str(args.repeats), "--seed", str(args.seed), "--out", str(outdir)]
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
        if not args.skip_kernels:
            subprocess.run([sys.executable, "-m", "bench.kernels", "--backend", be, "--out", str(outdir)],
                           check=True, cwd=REPO, env=env)
    subprocess.run([sys.executable, "-m", "bench.report", str(outdir)], check=True, cwd=REPO, env=env)
    print("results in", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
