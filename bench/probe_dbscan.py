"""How stable are the DBSCAN eps / min_samples heuristic's outputs?

Runs cluster_stops on seeds 1-10, sizes S and M, whole plant and line L1.
Does not change the heuristic. Writes a JSON file and a short markdown table.

    python -m bench.probe_dbscan --out results/findings/dbscan-heuristic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plantgen import generate_plant  # noqa: E402
from tools import TOOLS, set_backend  # noqa: E402
from tools.common import day_range, scope_machines, select  # noqa: E402


def _stops(plant, env) -> dict:
    fam = {f["family"]: f["suggested_reason"] for f in env["result"].get("families", [])}
    if not fam or "_labels" not in env:
        return {"n": int(env["result"].get("n_stops", 0)), "irregular_share": None,
                "coverage_count": None, "accuracy_count": None, "accuracy_minutes": None}
    suggested = np.array([fam.get(f"F{int(lab)}") for lab in env["_labels"]["family"]], dtype=object)
    machines = scope_machines(plant, env["scope"] or None)
    d0, d1 = day_range(plant, env["scope"] or None)
    frame = select(plant.stops, machines, d0, d1)
    unjust = frame["reason_reported"].to_numpy() == ""
    has = np.array([s is not None for s in suggested])
    correct = unjust & has & (suggested == frame["reason_true"].to_numpy())
    covered = unjust & has
    minutes = frame["duration_s"].to_numpy()
    n = len(frame)
    irr = int(env["result"]["patterns"]["irregular_stops"])
    def rate(num, den):
        return None if den == 0 else float(num) / float(den)
    return {
        "n": n,
        "eps": env["params"].get("eps"),
        "min_samples": env["params"].get("min_samples"),
        "irregular_share": rate(irr, n),
        "coverage_count": rate(int(covered.sum()), int(unjust.sum())),
        "accuracy_count": rate(int(correct.sum()), int(covered.sum())),
        "accuracy_minutes": rate(float(minutes[correct].sum()), float(minutes[covered].sum())),
    }


def _summ(vals: list[float]) -> dict:
    xs = [v for v in vals if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(xs), "mean": round(float(np.mean(xs)), 4),
            "min": round(float(np.min(xs)), 4), "max": round(float(np.max(xs)), 4)}


def run(seeds: list[int], sizes: list[str]) -> dict:
    set_backend("stock")
    rows = []
    for size in sizes:
        for seed in seeds:
            plant = generate_plant(size, seed=seed)
            for scope_name, scope in (("plant", None), ("L1", {"line_id": "L1"})):
                env = TOOLS["cluster_stops"](plant, scope)
                rows.append({"size": size, "seed": seed, "scope": scope_name, **_stops(plant, env)})
                r = rows[-1]
                print(f"[{size} seed {seed} {scope_name}] n={r['n']} eps={r.get('eps')} "
                      f"min_samples={r.get('min_samples')} irr={r['irregular_share']}", flush=True)
    groups = {}
    for size in sizes:
        for scope_name in ("plant", "L1"):
            chunk = [r for r in rows if r["size"] == size and r["scope"] == scope_name]
            groups[f"{size}-{scope_name}"] = {
                "irregular_share": _summ([r["irregular_share"] for r in chunk]),
                "coverage_count": _summ([r["coverage_count"] for r in chunk]),
                "accuracy_count": _summ([r["accuracy_count"] for r in chunk]),
                "accuracy_minutes": _summ([r["accuracy_minutes"] for r in chunk]),
                "eps": sorted({r.get("eps") for r in chunk}),
                "min_samples": sorted({r.get("min_samples") for r in chunk}),
            }
    return {"backend": "stock", "seeds": seeds, "sizes": sizes, "groups": groups, "runs": rows}


def render(payload: dict) -> str:
    lines = ["# DBSCAN heuristic stability", "",
             "Rule in `tools/stops.py`: `min_samples = max(10, n/1400)`, "
             "`eps = clip(0.5 * (40000/n)^(1/d), 0.5, 1.0)` with d = 7. "
             "Stock backend. Seeds and sizes below. No change to the rule.",
             "",
             "| Scope | Irregular share mean (min–max) | Coverage mean (min–max) | Accuracy mean (min–max) | Accuracy by minutes mean (min–max) | eps | min_samples |",
             "|---|---|---|---|---|---|---|"]
    def pct(block):
        if block["mean"] is None:
            return "n/a"
        return f"{100 * block['mean']:.1f}% ({100 * block['min']:.1f}–{100 * block['max']:.1f})"
    for name, g in payload["groups"].items():
        lines.append(
            f"| {name} | {pct(g['irregular_share'])} | {pct(g['coverage_count'])} | "
            f"{pct(g['accuracy_count'])} | {pct(g['accuracy_minutes'])} | `{g['eps']}` | `{g['min_samples']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--sizes", nargs="+", default=["S", "M"])
    ap.add_argument("--out", default=str(REPO / "results" / "findings" / "dbscan-heuristic.json"))
    args = ap.parse_args(argv)
    payload = run(args.seeds, args.sizes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render(payload), encoding="utf-8")
    print(render(payload))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
