"""detect_cycle_drift: nearest-neighbour novelty detection on part cycles.

For every machine x program, the reference window (the weeks before) defines
what normal cycles look like in a 6-signal space. Each recent cycle is scored
by its mean distance to the k nearest reference cycles. A cycle is outside the
envelope when its score exceeds the 99th percentile of reference scores
(leave-one-out). Results are rolled up per machine across its programs, and a
machine is flagged when a large share of its recent cycles are outside.
Programs without enough reference cycles (new programs) are skipped and
listed, never scored against another program's baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backend as B
from .common import Timer, day_to_date, envelope, fingerprint, scope_machines, select

FEATS = ["rel_cycle", "spindle_load", "feed_override", "vibration_rms", "motor_current", "spindle_speed"]
# cycle time is what this tool is about: it counts double in the distance
WEIGHTS = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def detect_cycle_drift(plant, scope: dict | None = None, test_days: int = 7, ref_days: int = 21, k: int = 10,
                       min_cycles: int = 150, q: float = 0.99, flag_share: float = 0.25) -> dict:
    est = B.estimators()
    T = Timer()
    scope = scope or {}
    with T.phase("prep"):
        machines = scope_machines(plant, scope)
        d1 = int(scope.get("day_to", plant.days - 1))
        t0 = d1 - test_days + 1
        r0 = t0 - ref_days
        c = select(plant.cycles, machines, r0, d1)
        c = c.assign(rel_cycle=c["cycle_s"] / c["nominal_cycle_s"])
        groups = list(c.groupby(["machine_id", "program_id"], sort=True))
    T.ms["ml_knn"] = 0.0
    per_cycle, per_prog, skipped = [], [], []
    for (m, p), g in groups:
        with T.phase("prep"):
            is_ref = g["day"].to_numpy() < t0
            R = g.loc[is_ref, FEATS].to_numpy(np.float64)
            Q = g.loc[~is_ref, FEATS].to_numpy(np.float64)
            if len(Q) == 0:
                continue
            if len(R) < min_cycles:
                skipped.append({"machine_id": m, "program_id": p, "ref_cycles": int(len(R)), "recent_cycles": int(len(Q))})
                continue
            mu, sd = R.mean(0), R.std(0)
            sd[sd == 0] = 1.0
            R = np.ascontiguousarray((R - mu) / sd * WEIGHTS)
            Q = np.ascontiguousarray((Q - mu) / sd * WEIGHTS)
        with T.phase("ml_knn"):
            nn = est.NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(R)
            dr, _ = nn.kneighbors(R)
            dq, _ = nn.kneighbors(Q, n_neighbors=k)
        with T.phase("post"):
            thr = float(np.quantile(dr[:, 1:].mean(1), q))
            score = dq.mean(1)
            gq = g.loc[~is_ref]
            per_cycle.append(pd.DataFrame({
                "machine_id": m, "program_id": p, "day": gq["day"].to_numpy(), "cycle_id": gq["cycle_id"].to_numpy(),
                "out": score > thr, "excess": score / thr,
                **{f: Q[:, i] / WEIGHTS[i] for i, f in enumerate(FEATS)},
            }))
            per_prog.append({"machine_id": m, "program_id": p, "ref_cycles": int(len(R)), "recent_cycles": int(len(Q)),
                             "outside_share": round(float((score > thr).mean()), 3),
                             "cycle_time_change_pct": round(100 * (np.median(gq["cycle_s"]) /
                                                                   np.median(g.loc[is_ref, "cycle_s"]) - 1), 2)})
    with T.phase("post"):
        flagged, facts = [], {}
        machines_checked = 0
        if per_cycle:
            pc = pd.concat(per_cycle, ignore_index=True)
            pp = pd.DataFrame(per_prog)
            machines_checked = pc["machine_id"].nunique()
            rows = []
            for m, g in pc.groupby("machine_id", sort=True):
                daily = g.groupby("day")["out"].mean()
                onset = next((int(d) for d, v in daily.items() if v >= 0.3 and (daily.loc[d:] >= 0.3).mean() >= 0.75), None)
                w = pp[pp["machine_id"] == m]
                moved = g.loc[g["out"], FEATS].median() if g["out"].any() else g[FEATS].median()
                moved = sorted(moved.items(), key=lambda t: -abs(t[1]))[:3]
                rows.append({
                    "machine_id": m, "programs": int(len(w)), "recent_cycles": int(len(g)),
                    "outside_share": round(float(g["out"].mean()), 3),
                    "cycle_time_change_pct": round(float(np.average(w["cycle_time_change_pct"], weights=w["recent_cycles"])), 2),
                    "onset_day": onset,
                    "moved": [{"signal": f, "shift_sigma": round(float(z), 2)} for f, z in moved],
                    "worst_cycles": g.sort_values("excess", ascending=False)["cycle_id"].head(5).astype(int).tolist(),
                })
            res = pd.DataFrame(rows).sort_values(["outside_share", "machine_id"], ascending=[False, True])
            for i, item in enumerate(res[res["outside_share"] >= flag_share].to_dict("records"), 1):
                item["onset_date"] = day_to_date(plant, item["onset_day"]) if item["onset_day"] is not None else None
                flagged.append(item)
                mv = ", ".join(f"{d['signal']} {d['shift_sigma']:+.1f}σ" for d in item["moved"])
                facts[f"D{i}"] = (
                    f"Drift D{i}: {item['machine_id']} cycle time {item['cycle_time_change_pct']:+.1f}% vs the previous "
                    f"{ref_days} days; {item['outside_share']:.0%} of its cycles in the last {test_days} days are outside "
                    f"the normal envelope" + (f", starting about {item['onset_date']}" if item["onset_date"] else "")
                    + f". Biggest moves: {mv}."
                )
        result = {
            "window": {"reference_days": [r0, t0 - 1], "test_days": [t0, d1]},
            "machine_programs_checked": int(len(per_prog)),
            "machines_checked": int(machines_checked),
            "flagged": flagged,
            "skipped_new_programs": skipped,
        }
        pl = lambda k, w: f"{k} {w}" + ("" if k == 1 else "s")  # noqa: E731
        facts["N"] = (f"Checked {pl(len(per_prog), 'machine-program')} on {pl(machines_checked, 'machine')}; "
                      f"{pl(len(flagged), 'machine')} with cycle-time drift; {pl(len(skipped), 'new program')} "
                      "skipped for lack of a baseline.")
    params = {"test_days": test_days, "ref_days": ref_days, "k": k, "q": q, "flag_share": flag_share, "features": FEATS}
    return envelope("detect_cycle_drift", scope, params, fingerprint(c["cycle_id"].to_numpy(), c["cycle_s"].to_numpy()),
                    result, facts, T,
                    extra={"_scores": {(r["machine_id"], r["program_id"]): r["outside_share"] for r in per_prog}})
