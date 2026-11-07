"""Parity gate: oneDAL results must match stock scikit-learn before the agent may cite them.

For every tool, run stock and oneDAL on the same plant and scopes and compare
with per-tool tolerances. Also:

* audit dispatch: which estimator calls actually ran on oneDAL, which fell back
* check determinism: the same call twice on oneDAL gives the same result digest

Writes results/parity.json (read by tools/common.py at call time) and
results/parity_report.md. Exit code 1 if any tool fails.

    python -m bench.parity --size M --seed 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plantgen import generate_plant  # noqa: E402
from tools import TOOLS, capture_dispatch, set_backend, versions  # noqa: E402
from tools.common import TOOL_VERSION, result_digest  # noqa: E402

TOL = {
    "cluster_stops": {"family_ari_min": 0.99, "noise_agreement_min": 0.99},
    "detect_cycle_drift": {"flagged_equal": True, "share_abs_diff_max": 0.01},
    "rank_loss_drivers": {"top3_equal": True, "spearman_min": 0.80, "r2_abs_diff_max": 0.02},
    "signal_health": {"flagged_equal": True, "share_abs_diff_max": 0.01},
}


def _run(name, plant, scope, backend):
    set_backend(backend)
    return TOOLS[name](plant, scope)


def compare(name, a, b) -> tuple[bool, dict]:
    t = TOL[name]
    if name == "cluster_stops":
        la, lb = a["_labels"], b["_labels"]
        ari = float(adjusted_rand_score(la["family"], lb["family"]))
        noise = float((la["noise"] == lb["noise"]).mean())
        m = {"family_ari": round(ari, 5), "noise_agreement": round(noise, 5),
             "dbscan_path": b["params"]["dbscan_path"]}
        return ari >= t["family_ari_min"] and noise >= t["noise_agreement_min"], m
    if name in ("detect_cycle_drift", "signal_health"):
        fa = sorted(f["machine_id"] for f in a["result"]["flagged"])
        fb = sorted(f["machine_id"] for f in b["result"]["flagged"])
        sa = a.get("_scores") or a.get("_shares")
        sb = b.get("_scores") or b.get("_shares")
        keys = set(sa) | set(sb)
        diff = max((abs(sa.get(k, -1) - sb.get(k, -1)) for k in keys), default=0.0)
        m = {"flagged_stock": fa, "flagged_onedal": fb, "max_share_abs_diff": round(float(diff), 5),
             "compared_units": len(keys)}
        return fa == fb and diff <= t["share_abs_diff_max"], m
    if name == "rank_loss_drivers":
        ra, rb = a["result"]["ranking"], b["result"]["ranking"]
        top_a, top_b = [r["driver"] for r in ra[:3]], [r["driver"] for r in rb[:3]]
        order = [r["driver"] for r in ra]
        ia = np.array([next(x["importance"] for x in ra if x["driver"] == d) for d in order])
        ib = np.array([next(x["importance"] for x in rb if x["driver"] == d) for d in order])
        rho = float(spearmanr(ia, ib).statistic)
        dr2 = abs(a["result"]["holdout_r2"] - b["result"]["holdout_r2"])
        m = {"top3_stock": top_a, "top3_onedal": top_b, "spearman": round(rho, 4), "r2_abs_diff": round(dr2, 4),
             "r2_stock": a["result"]["holdout_r2"], "r2_onedal": b["result"]["holdout_r2"]}
        return set(top_a) == set(top_b) and rho >= t["spearman_min"] and dr2 <= t["r2_abs_diff_max"], m
    raise KeyError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="M")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plant = generate_plant(args.size, seed=args.seed)
    scopes = {"plant": None, "line": {"line_id": "L2", "day_from": plant.days - 7}}
    v = versions("onedal")
    record = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "plant": plant.summary(), "versions": v, "tool_version": TOOL_VERSION, "tolerances": TOL, "tools": {}}
    all_ok = True
    for name in TOOLS:
        per_scope, ok_all = {}, True
        for sname, scope in scopes.items():
            if name in ("detect_cycle_drift", "signal_health", "rank_loss_drivers") and sname == "line":
                scope = {"line_id": "L2"}  # these tools pick their own time windows
            a = _run(name, plant, scope, "stock")
            with capture_dispatch() as log:
                b = _run(name, plant, scope, "onedal")
            b2 = _run(name, plant, scope, "onedal")
            ok, metrics = compare(name, a, b)
            deterministic = result_digest(b) == result_digest(b2)
            metrics["deterministic_onedal"] = deterministic
            metrics["dispatch"] = log.summary()
            ok = ok and deterministic
            per_scope[sname] = {"pass": ok, **metrics}
            ok_all &= ok
        record["tools"][name] = {"pass": ok_all, "tool_version": TOOL_VERSION, "sklearn": v["sklearn"],
                                 "sklearnex": v["sklearnex"], "checked_at": record["checked_at"], "scopes": per_scope}
        all_ok &= ok_all
        print(f"{name:20s} {'PASS' if ok_all else 'FAIL'}  " +
              "  ".join(f"{s}: " + json.dumps({k: v for k, v in m.items() if k not in ('dispatch', 'pass')}, default=str)
                        for s, m in per_scope.items()))
    (out / "parity.json").write_text(json.dumps(record, indent=2, default=str))
    (out / "parity_report.md").write_text(_report(record))
    print("wrote", out / "parity.json")
    return 0 if all_ok else 1


def _report(rec) -> str:
    lines = [f"# Parity gate: oneDAL vs stock scikit-learn", "",
             f"Checked {rec['checked_at']} · sklearn {rec['versions']['sklearn']} · sklearnex {rec['versions']['sklearnex']}"
             f" · tool version {rec['tool_version']} · plant {rec['plant']['machines']} machines, {rec['plant']['days']} days", "",
             "| Tool | Scope | Result | Key metrics | oneDAL dispatch |", "|---|---|---|---|---|"]
    for name, t in rec["tools"].items():
        for s, m in t["scopes"].items():
            keym = {k: v for k, v in m.items() if k not in ("dispatch", "pass")}
            disp = "; ".join(f"{c.split('.')[-2]}.{c.split('.')[-1]}: {d['accelerated']} acc / {d['fallback']} fb"
                             for c, d in m["dispatch"].items())
            lines.append(f"| {name} | {s} | {'PASS' if m['pass'] else 'FAIL'} | `{json.dumps(keym, default=str)}` | {disp} |")
    lines += ["", "Tolerances:", "```json", json.dumps(rec["tolerances"], indent=2), "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
