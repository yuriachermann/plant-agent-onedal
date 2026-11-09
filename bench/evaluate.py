"""Score the four tools against the faults and drivers the generator planted.

The generator's ``truth`` is the only answer key. This module does not change
the tools or the plant. It runs each tool once per seed and size and writes
the rates below.

* Drift: precision and recall for wear and step faults together, recall per
  fault type, and the share of healthy machines flagged. A flagged machine
  that is not in the wear or step set, including a coolant fault, counts as a
  false positive. ``--drift-step-test off`` scores the envelope test alone.
* Signal health: recall on coolant faults, and the share of healthy machines
  that are flagged. Healthy means no wear, step, or coolant fault.
* Drivers: how often each planted driver is in the top 3 by importance, how
  often each planted driver is reported, and how often a null driver is
  reported (both by the tool's own reporting rule; ``--drivers-rule`` picks
  which one, and ``--drivers-holdout`` whether hours or whole days are held
  out).
* Stops: suggestion coverage and accuracy on unjustified stops, by count and
  by minutes. Accuracy is among the stops that received a suggestion.
  Minute weights are stop durations.

    python -m bench.evaluate
    python -m bench.evaluate --seeds 1 --sizes S --out results/_eval_smoke
    python -m bench.evaluate --drivers-rule shadow --out results/eval/drivers-shadow
    python -m bench.evaluate --drivers-holdout day --out results/eval/drivers-holdout-day
    python -m bench.evaluate --drift-step-test off --out results/eval/drift-envelope-only
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
from tools.drivers import HOLDOUT_BY, NULL_RULES  # noqa: E402


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def _drift(plant, env) -> dict:
    wear, step = set(plant.truth["drift_wear"]), set(plant.truth["drift_step"])
    planted = wear | step
    healthy = set(plant.machines["machine_id"]) - planted - set(plant.truth["coolant_fault"])
    flagged = {f["machine_id"] for f in env["result"]["flagged"]}
    tp = len(flagged & planted)
    return {
        "planted": sorted(planted),
        "flagged": sorted(flagged),
        "precision": _rate(tp, len(flagged)),
        "recall": _rate(tp, len(planted)),
        "wear_recall": _rate(len(flagged & wear), len(wear)),
        "step_recall": _rate(len(flagged & step), len(step)),
        "healthy_false_positives": sorted(flagged & healthy),
        "healthy_false_positive_rate": _rate(len(flagged & healthy), len(healthy)),
        "coolant_also_flagged": sorted(flagged & set(plant.truth["coolant_fault"])),
        "flagged_by": {f["machine_id"]: f.get("flagged_by", ["envelope"]) for f in env["result"]["flagged"]},
    }


def _health(plant, env) -> dict:
    coolant = set(plant.truth["coolant_fault"])
    faulty = coolant | set(plant.truth["drift_wear"]) | set(plant.truth["drift_step"])
    healthy = set(plant.machines["machine_id"]) - faulty
    flagged = {f["machine_id"] for f in env["result"]["flagged"]}
    fp = flagged & healthy
    return {
        "coolant_recall": _rate(len(flagged & coolant), len(coolant)),
        "healthy_machines": len(healthy),
        "healthy_false_positives": sorted(fp),
        "healthy_false_positive_rate": _rate(len(fp), len(healthy)),
        "flagged_other_faults": sorted(flagged & faulty - coolant),
    }


def _drivers(plant, env) -> dict:
    ranking = env["result"]["ranking"]
    top3 = [r["driver"] for r in ranking[:3]]
    reported = [r["driver"] for r in ranking if r.get("reported")]
    planted = list(plant.truth["loss_drivers"])
    nulls = list(plant.truth["null_drivers"])
    return {
        "top3": top3,
        "reported": reported,
        "planted_in_top3": {d: d in top3 for d in planted},
        "planted_reported": {d: d in reported for d in planted},
        "null_reported": {d: d in reported for d in nulls},
        "rule": env["result"].get("null_rule"),
        "holdout_by": env["result"].get("holdout_by"),
    }


def _stops(plant, env) -> dict:
    if "reason_true" not in plant.stops.columns or "_labels" not in env:
        return {"coverage_count": None, "accuracy_count": None, "coverage_minutes": None, "accuracy_minutes": None}
    fam = {f["family"]: f["suggested_reason"] for f in env["result"]["families"]}
    suggested = np.array([fam[f"F{int(lab)}"] for lab in env["_labels"]["family"]], dtype=object)
    stops = plant.stops
    unjust = stops["reason_reported"].to_numpy() == ""
    has = np.array([s is not None for s in suggested])
    correct = unjust & has & (suggested == stops["reason_true"].to_numpy())
    covered = unjust & has
    minutes = stops["duration_s"].to_numpy()
    return {
        "unjustified_stops": int(unjust.sum()),
        "coverage_count": _rate(int(covered.sum()), int(unjust.sum())),
        "accuracy_count": _rate(int(correct.sum()), int(covered.sum())),
        "coverage_minutes": _rate(float(minutes[covered].sum()), float(minutes[unjust].sum())),
        "accuracy_minutes": _rate(float(minutes[correct].sum()), float(minutes[covered].sum())),
    }


def score_plant(plant, backend: str = "stock", drivers_rule: str | None = None, drift_step_test: bool = True,
                drivers_holdout: str | None = None) -> dict:
    set_backend(backend)
    drift = _drift(plant, TOOLS["detect_cycle_drift"](plant, None, step_test=drift_step_test))
    health = _health(plant, TOOLS["signal_health"](plant, None))
    dparams = {"null_rule": drivers_rule} if drivers_rule else {}
    if drivers_holdout:
        dparams["holdout_by"] = drivers_holdout
    drivers = _drivers(plant, TOOLS["rank_loss_drivers"](plant, None, **dparams))
    stops = _stops(plant, TOOLS["cluster_stops"](plant, None))
    return {"drift": drift, "signal_health": health, "drivers": drivers, "stops": stops}


def _mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def _min(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return float(np.min(xs)) if xs else None


def _max(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return float(np.max(xs)) if xs else None


def _round(v, nd=4):
    return None if v is None else round(float(v), nd)


def aggregate(runs: list[dict]) -> dict:
    def col(path):
        out = []
        for run in runs:
            cur = run
            for key in path:
                cur = cur[key]
            out.append(cur)
        return out

    planted = sorted({d for run in runs for d in run["drivers"]["planted_in_top3"]})
    nulls = sorted({d for run in runs for d in run["drivers"]["null_reported"]})
    planted_rates, planted_reported = {}, {}
    for d in planted:
        hits = [1.0 if run["drivers"]["planted_in_top3"][d] else 0.0 for run in runs]
        planted_rates[d] = {"rate": _round(_mean(hits)), "runs": len(hits)}
        rep = [1.0 if run["drivers"]["planted_reported"][d] else 0.0 for run in runs]
        planted_reported[d] = {"rate": _round(_mean(rep)), "runs": len(rep)}
    null_rates = {}
    for d in nulls:
        hits = [1.0 if run["drivers"]["null_reported"][d] else 0.0 for run in runs]
        null_rates[d] = {"rate": _round(_mean(hits)), "runs": len(hits)}
    any_null = [1.0 if any(run["drivers"]["null_reported"].values()) else 0.0 for run in runs]
    in_top3 = [np.mean(list(run["drivers"]["planted_in_top3"].values())) for run in runs]
    reported = [np.mean(list(run["drivers"]["planted_reported"].values())) for run in runs]
    return {
        "n_runs": len(runs),
        "drift": {
            "precision_mean": _round(_mean(col(("drift", "precision")))),
            "precision_min": _round(_min(col(("drift", "precision")))),
            "recall_mean": _round(_mean(col(("drift", "recall")))),
            "recall_min": _round(_min(col(("drift", "recall")))),
            "wear_recall_mean": _round(_mean(col(("drift", "wear_recall")))),
            "step_recall_mean": _round(_mean(col(("drift", "step_recall")))),
            "healthy_false_positive_rate_mean": _round(_mean(col(("drift", "healthy_false_positive_rate")))),
            "healthy_false_positive_rate_max": _round(_max(col(("drift", "healthy_false_positive_rate")))),
        },
        "signal_health": {
            "coolant_recall_mean": _round(_mean(col(("signal_health", "coolant_recall")))),
            "coolant_recall_min": _round(_min(col(("signal_health", "coolant_recall")))),
            "healthy_false_positive_rate_mean": _round(_mean(col(("signal_health", "healthy_false_positive_rate")))),
            "healthy_false_positive_rate_max": _round(_max(col(("signal_health", "healthy_false_positive_rate")))),
        },
        "drivers": {
            "planted_in_top3_mean": _round(_mean(in_top3)),
            "planted_in_top3_min": _round(_min(in_top3)),
            "per_driver_top3_rate": planted_rates,
            "planted_reported_mean": _round(_mean(reported)),
            "per_driver_reported_rate": planted_reported,
            "per_null_reported_rate": null_rates,
            "any_null_reported_rate": _round(_mean(any_null)),
        },
        "stops": {
            "coverage_count_mean": _round(_mean(col(("stops", "coverage_count")))),
            "coverage_count_min": _round(_min(col(("stops", "coverage_count")))),
            "accuracy_count_mean": _round(_mean(col(("stops", "accuracy_count")))),
            "accuracy_count_min": _round(_min(col(("stops", "accuracy_count")))),
            "coverage_minutes_mean": _round(_mean(col(("stops", "coverage_minutes")))),
            "coverage_minutes_min": _round(_min(col(("stops", "coverage_minutes")))),
            "accuracy_minutes_mean": _round(_mean(col(("stops", "accuracy_minutes")))),
            "accuracy_minutes_min": _round(_min(col(("stops", "accuracy_minutes")))),
        },
    }


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_md(summary: dict) -> str:
    agg = summary["aggregate"]
    flagged_plants = sum(1 for r in summary["runs"] if r["drift"]["precision"] is not None)
    lines = [
        "# Detection quality against planted truth",
        "",
        f"Backend `{summary['backend']}`, seeds {summary['seeds']}, sizes {summary['sizes']}, "
        f"{summary['aggregate']['n_runs']} plants, driver rule `{summary.get('drivers_rule')}`, "
        f"driver holdout by `{summary.get('drivers_holdout') or 'hour'}`. "
        "Definitions are in `summary.json`.",
        "",
        "## Drift (wear + step)",
        "",
        f"Precision mean {_pct(agg['drift']['precision_mean'])} (min {_pct(agg['drift']['precision_min'])}) "
        f"on the {flagged_plants} plants that flagged at least one machine. "
        f"Recall mean {_pct(agg['drift']['recall_mean'])} (min {_pct(agg['drift']['recall_min'])}): "
        f"wear {_pct(agg['drift']['wear_recall_mean'])}, step {_pct(agg['drift']['step_recall_mean'])}. "
        f"Healthy-machine false-positive rate mean {_pct(agg['drift']['healthy_false_positive_rate_mean'])} "
        f"(max {_pct(agg['drift']['healthy_false_positive_rate_max'])}). "
        f"Step test {'on' if summary.get('drift_step_test', True) else 'off'}.",
        "",
        "## Signal health",
        "",
        f"Coolant recall mean {_pct(agg['signal_health']['coolant_recall_mean'])} "
        f"(min {_pct(agg['signal_health']['coolant_recall_min'])}). "
        f"Healthy-machine false-positive rate mean {_pct(agg['signal_health']['healthy_false_positive_rate_mean'])} "
        f"(max {_pct(agg['signal_health']['healthy_false_positive_rate_max'])}).",
        "",
        "## Loss drivers",
        "",
        f"Share of planted drivers in the top 3: mean {_pct(agg['drivers']['planted_in_top3_mean'])} "
        f"(min {_pct(agg['drivers']['planted_in_top3_min'])}). "
        f"Share of planted drivers reported: mean {_pct(agg['drivers']['planted_reported_mean'])}. "
        f"Runs with any null driver reported: {_pct(agg['drivers']['any_null_reported_rate'])}.",
        "",
        "| Planted driver | Share of runs in the top 3 | Share of runs reported |",
        "|---|---:|---:|",
    ]
    for name, rec in agg["drivers"]["per_driver_top3_rate"].items():
        rep = agg["drivers"]["per_driver_reported_rate"][name]["rate"]
        lines.append(f"| {name} | {_pct(rec['rate'])} | {_pct(rep)} |")
    lines += ["", "| Null driver | Share of runs reported |", "|---|---:|"]
    for name, rec in agg["drivers"]["per_null_reported_rate"].items():
        lines.append(f"| {name} | {_pct(rec['rate'])} |")
    st = agg["stops"]
    lines += [
        "",
        "## Stop suggestions",
        "",
        "| | By count | By minutes |",
        "|---|---:|---:|",
        f"| Coverage mean | {_pct(st['coverage_count_mean'])} | {_pct(st['coverage_minutes_mean'])} |",
        f"| Coverage min | {_pct(st['coverage_count_min'])} | {_pct(st['coverage_minutes_min'])} |",
        f"| Accuracy mean | {_pct(st['accuracy_count_mean'])} | {_pct(st['accuracy_minutes_mean'])} |",
        f"| Accuracy min | {_pct(st['accuracy_count_min'])} | {_pct(st['accuracy_minutes_min'])} |",
        "",
        "Accuracy is the share of suggestions that match `reason_true`, among unjustified stops that received one. "
        "Minute weights are stop durations.",
        "",
    ]
    if summary.get("by_size"):
        lines += ["## By plant size", "",
                  "| Size | Drift recall | Coolant recall | Planted drivers in top 3 | Stop coverage | Stop accuracy |",
                  "|---|---:|---:|---:|---:|---:|"]
        for size, block in summary["by_size"].items():
            lines.append(
                f"| {size} | {_pct(block['drift']['recall_mean'])} | {_pct(block['signal_health']['coolant_recall_mean'])} | "
                f"{_pct(block['drivers']['planted_in_top3_mean'])} | {_pct(block['stops']['coverage_count_mean'])} | "
                f"{_pct(block['stops']['accuracy_count_mean'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(seeds: list[int], sizes: list[str], backend: str, out: Path, drivers_rule: str | None = None,
        drift_step_test: bool = True, drivers_holdout: str | None = None) -> dict:
    runs = []
    for size in sizes:
        for seed in seeds:
            plant = generate_plant(size, seed=seed)
            scored = score_plant(plant, backend, drivers_rule, drift_step_test, drivers_holdout)
            runs.append({"size": size, "seed": seed, "plant_id": plant.plant_id, **scored})
            d, h, s = scored["drift"], scored["signal_health"], scored["stops"]
            print(
                f"[{size} seed {seed}] drift P {_pct(d['precision'])} R {_pct(d['recall'])} | "
                f"coolant R {_pct(h['coolant_recall'])} healthy FP {_pct(h['healthy_false_positive_rate'])} | "
                f"stops cov {_pct(s['coverage_count'])} acc {_pct(s['accuracy_count'])}",
                flush=True,
            )
    summary = {
        "backend": backend,
        "seeds": seeds,
        "sizes": sizes,
        "drivers_rule": runs[0]["drivers"]["rule"] if runs else drivers_rule,
        "drivers_holdout": runs[0]["drivers"]["holdout_by"] if runs else drivers_holdout,
        "drift_step_test": drift_step_test,
        "definitions": {
            "drift_positive": "drift_wear ∪ drift_step. Other flagged machines, including coolant faults, are false positives.",
            "signal_health_healthy": "Machines with no wear, step, or coolant fault.",
            "drivers_top3": "First three rows of ranking, ordered by permutation importance.",
            "drivers_reported": "ranking[].reported, the tool's own rule (drivers_rule).",
            "stops_unjustified": "reason_reported is blank.",
            "stops_accuracy": "suggested reason equals reason_true, among unjustified stops that got a suggestion.",
            "stops_minutes": "Weighted by duration_s.",
        },
        "runs": _public_runs(runs),
        "aggregate": aggregate(runs),
        "by_size": {size: aggregate([r for r in runs if r["size"] == size]) for size in sizes},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(render_md(summary), encoding="utf-8")
    print("wrote", out / "summary.md")
    return summary


def _public_runs(runs: list[dict]) -> list[dict]:
    """JSON-ready copy: rates rounded, numpy absent."""
    out = []
    for run in runs:
        item = {"size": run["size"], "seed": run["seed"], "plant_id": run["plant_id"], "drift": {}, "signal_health": {},
                "drivers": run["drivers"], "stops": {}}
        for key in ("precision", "recall", "wear_recall", "step_recall", "healthy_false_positive_rate"):
            item["drift"][key] = _round(run["drift"][key])
        for key in ("planted", "flagged", "healthy_false_positives", "coolant_also_flagged", "flagged_by"):
            item["drift"][key] = run["drift"][key]
        item["signal_health"] = {
            "coolant_recall": _round(run["signal_health"]["coolant_recall"]),
            "healthy_false_positives": run["signal_health"]["healthy_false_positives"],
            "healthy_false_positive_rate": _round(run["signal_health"]["healthy_false_positive_rate"]),
            "flagged_other_faults": run["signal_health"]["flagged_other_faults"],
        }
        item["stops"] = {k: _round(v) if isinstance(v, float) else v for k, v in run["stops"].items()}
        out.append(item)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--sizes", nargs="+", default=["S", "M"])
    ap.add_argument("--backend", choices=["stock", "onedal"], default="stock")
    ap.add_argument("--drivers-rule", choices=list(NULL_RULES), default=None,
                    help="rank_loss_drivers reporting rule; default is the tool's own default")
    ap.add_argument("--drivers-holdout", choices=list(HOLDOUT_BY), default=None,
                    help="rank_loss_drivers holdout split, random hours or whole days; default is the tool's own")
    ap.add_argument("--drift-step-test", choices=["on", "off"], default="on",
                    help="detect_cycle_drift step test; off scores the kNN envelope alone")
    ap.add_argument("--out", default=str(REPO / "results" / "eval"))
    args = ap.parse_args(argv)
    run(args.seeds, args.sizes, args.backend, Path(args.out), args.drivers_rule, args.drift_step_test == "on",
        args.drivers_holdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
