"""Merge stock/oneDAL benchmark JSONs into summary.md, summary.json and charts.

    python -m bench.report results/<machine-tag> [--questions-per-day 40] [--usd-per-core-hour 0.04]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOL_LABEL = {"cluster_stops": "cluster_stops (k-means + DBSCAN)", "detect_cycle_drift": "detect_cycle_drift (kNN)",
              "rank_loss_drivers": "rank_loss_drivers (random forest)", "signal_health": "signal_health (PCA)"}


def load(d: Path, kind: str):
    out = {}
    for be in ("stock", "onedal"):
        f = d / f"{kind}_{be}.json"
        if f.exists():
            out[be] = json.loads(f.read_text())
    return out


def sp(a, b):
    return round(a / b, 2) if b else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--questions-per-day", type=int, default=40)
    ap.add_argument("--usd-per-core-hour", type=float, default=0.04)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    B = load(d, "bench")
    Kn = load(d, "kernels")
    if set(B) != {"stock", "onedal"}:
        sys.exit("need bench_stock.json and bench_onedal.json")
    s, o = B["stock"], B["onedal"]
    q = args.questions_per_day
    summary = {"machine": s["machine"], "versions": {"stock": s["versions"], "onedal": o["versions"]},
               "assumptions": {"questions_per_day": q, "usd_per_core_hour": args.usd_per_core_hour}, "sizes": {}}
    md = ["# Benchmark: stock scikit-learn vs oneDAL", "",
          f"**Machine:** {s['machine']['cpu']} · {s['machine']['logical_cpus']} logical CPUs · ISA {', '.join(s['machine']['isa']) or 'n/a'}"
          f" · {s['machine']['os']} · Python {s['machine']['python']}",
          f"**Versions:** scikit-learn {s['versions']['sklearn']} · scikit-learn-intelex {o['versions'].get('sklearnex')}"
          f" · oneDAL {o['versions'].get('onedal', '?')}", "",
          f"Synthetic plants, 30 days, seed {s['seed']}. On-demand = one line, median of {s['repeats']} warm calls. "
          "Nightly = whole plant, all four tools; cold = first run in a fresh process.", ""]
    for size in s["sizes"]:
        S, O = s["sizes"][size], o["sizes"][size]
        p = S["plant"]
        row = {"plant": p, "on_demand": {}, "nightly": {}}
        md += [f"## Plant {size}: {p['machines']} machines · {p['stops_per_day']:.0f} stops/day · {p['cycles_per_day']:.0f} cycles/day", "",
               "| On-demand tool call (one line) | Stock ms | oneDAL ms | Speed-up | ML part speed-up |", "|---|---:|---:|---:|---:|"]
        for t in S["on_demand"]:
            a, b = S["on_demand"][t], O["on_demand"][t]
            r = {"stock_ms": a["median_ms"], "onedal_ms": b["median_ms"], "speedup": sp(a["median_ms"], b["median_ms"]),
                 "ml_speedup": sp(a["ml_median_ms"], b["ml_median_ms"]), "onedal_path": b.get("dbscan_path")}
            row["on_demand"][t] = r
            note = f" ({b['dbscan_path']})" if b.get("dbscan_path") else ""
            md.append(f"| {TOOL_LABEL[t]}{note} | {r['stock_ms']:.0f} | {r['onedal_ms']:.0f} | {r['speedup']}× | {r['ml_speedup']}× |")
        row["answer"] = {"stock_ms": S["answer_ms"], "onedal_ms": O["answer_ms"], "speedup": sp(S["answer_ms"], O["answer_ms"])}
        md.append(f"| **One answer (4 calls)** | **{S['answer_ms']:.0f}** | **{O['answer_ms']:.0f}** | **{row['answer']['speedup']}×** | |")
        md += ["", "| Nightly batch (whole plant) | Stock | oneDAL | Speed-up |", "|---|---:|---:|---:|"]
        for k in ("cold", "warm"):
            a, b = S["nightly"][k]["wall_ms"] / 1000, O["nightly"][k]["wall_ms"] / 1000
            row["nightly"][k] = {"stock_s": round(a, 2), "onedal_s": round(b, 2), "speedup": sp(a, b)}
            md.append(f"| Wall time, {k} | {a:.1f} s | {b:.1f} s | {sp(a, b)}× |")
        for t in S["nightly"]["warm"]["tools"]:
            a, b = S["nightly"]["warm"]["tools"][t]["wall_ms"] / 1000, O["nightly"]["warm"]["tools"][t]["wall_ms"] / 1000
            row["nightly"].setdefault("tools", {})[t] = {"stock_s": round(a, 2), "onedal_s": round(b, 2), "speedup": sp(a, b)}
            md.append(f"| · {t} | {a:.2f} s | {b:.2f} s | {sp(a, b)}× |")
        cs = S["nightly"]["warm"]["cpu_ms"] / 1000 + q * S["answer_cpu_ms"] / 1000
        co = O["nightly"]["warm"]["cpu_ms"] / 1000 + q * O["answer_cpu_ms"] / 1000
        row["daily_cpu"] = {"stock_core_s": round(cs, 1), "onedal_core_s": round(co, 1), "ratio": sp(cs, co),
                            "stock_usd": round(cs / 3600 * args.usd_per_core_hour, 4),
                            "onedal_usd": round(co / 3600 * args.usd_per_core_hour, 4)}
        md += [f"| CPU per plant per day (nightly + {q} questions) | {cs:.0f} core-s | {co:.0f} core-s | {sp(cs, co)}× |",
               f"| …at ${args.usd_per_core_hour}/core-hour | ${row['daily_cpu']['stock_usd']:.4f} | ${row['daily_cpu']['onedal_usd']:.4f} | |", ""]
        summary["sizes"][size] = row

    if Kn.get("stock") and Kn.get("onedal"):
        md += ["## Kernel sweeps (single calls, large plant data)", ""]
        summary["kernels"] = {}
        for k, key in [("dbscan", "n"), ("kmeans", "n"), ("knn", "n_ref"), ("rf_fit", "n"), ("rf_predict", "n"), ("pca", "n")]:
            a, b = Kn["stock"]["kernels"][k], Kn["onedal"]["kernels"][k]
            rows = [{key: x[key], "stock_ms": x["ms"], "onedal_ms": y["ms"], "speedup": sp(x["ms"], y["ms"])} for x, y in zip(a, b)]
            summary["kernels"][k] = rows
            md += [f"**{k}**", "", f"| {key} | stock ms | oneDAL ms | speed-up |", "|---:|---:|---:|---:|"]
            md += [f"| {r[key]:,} | {r['stock_ms']:.1f} | {r['onedal_ms']:.1f} | {r['speedup']}× |" for r in rows] + [""]
        db = summary["kernels"]["dbscan"]
        brute_max = max((r["n"] for r in db if r["speedup"] and r["speedup"] >= 1.1), default=0)
        pc = summary["kernels"]["pca"]
        pca_min = next((r["n"] for r in pc if r["speedup"] and r["speedup"] >= 1.1), None)
        summary["suggested_routing"] = {"PLANT_TOOLS_DBSCAN_BRUTE_MAX": brute_max,
                                        "PLANT_TOOLS_PCA_MIN_ROWS": pca_min if pca_min is not None else 10 ** 9}
        md += ["## Suggested routing for this machine",
               "Use oneDAL where it is at least 1.1× faster in the sweeps above, stock elsewhere:", "", "```",
               *[f"{k}={v}" for k, v in summary["suggested_routing"].items()], "```",
               "Set these as environment variables before running the tools (defaults: 30000 and 8000).", ""]
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    (d / "summary.md").write_text("\n".join(md) + "\n")
    try:
        _charts(d, summary)
    except ImportError:
        print("matplotlib not installed; skipping charts")
    print("\n".join(md))


def _charts(d: Path, summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = list(summary["sizes"])
    tools = list(summary["sizes"][sizes[0]]["on_demand"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(tools))
    size = "M" if "M" in sizes else sizes[0]
    a = [summary["sizes"][size]["on_demand"][t]["stock_ms"] for t in tools]
    b = [summary["sizes"][size]["on_demand"][t]["onedal_ms"] for t in tools]
    ax.bar([i - 0.2 for i in x], a, 0.4, label="stock scikit-learn", color="#9aa0a6")
    ax.bar([i + 0.2 for i in x], b, 0.4, label="oneDAL (sklearnex)", color="#0068b5")
    ax.set_xticks(list(x), [t.replace("_", "\n") for t in tools])
    ax.set_ylabel("ms per call (median)")
    ax.set_title(f"On-demand tool call, one line, plant {size}")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "on_demand.png", dpi=160)
    plt.close(fig)
    if "kernels" in summary:
        fig, ax = plt.subplots(figsize=(6, 4))
        k = summary["kernels"]["dbscan"]
        ax.loglog([r["n"] for r in k], [r["stock_ms"] for r in k], "o-", color="#9aa0a6", label="stock (KD-tree)")
        ax.loglog([r["n"] for r in k], [r["onedal_ms"] for r in k], "o-", color="#0068b5", label="oneDAL (brute force)")
        ax.set_xlabel("stop events")
        ax.set_ylabel("ms")
        ax.set_title("DBSCAN on stop events")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(d / "dbscan_crossover.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
