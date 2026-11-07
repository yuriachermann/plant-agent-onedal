"""signal_health: PCA monitoring of the 12 machine signals of each cycle.

Per machine x program, PCA learns how the signals normally move together in
the reference window. Two standard statistics score each recent cycle:

* SPE (Q): squared reconstruction error, i.e. a new pattern between signals
* Hotelling T²: an unusually large move inside the usual pattern

A cycle is abnormal when either exceeds the 99th percentile of the reference.
The signals with the largest standardised deviations in abnormal cycles say
what changed. This catches faults that do not change cycle time, such as a
weakening coolant pump. New programs (no baseline) are skipped and listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plantgen import SIGNALS

from . import backend as B
from .common import Timer, day_to_date, envelope, fingerprint, scope_machines, select


def signal_health(plant, scope: dict | None = None, test_days: int = 7, ref_days: int = 21, var: float = 0.90,
                  q: float = 0.99, flag_share: float = 0.25, min_cycles: int = 150) -> dict:
    est = B.estimators()
    T = Timer()
    scope = scope or {}
    with T.phase("prep"):
        machines = scope_machines(plant, scope)
        d1 = int(scope.get("day_to", plant.days - 1))
        t0 = d1 - test_days + 1
        r0 = t0 - ref_days
        c = select(plant.cycles, machines, r0, d1)
        groups = list(c.groupby(["machine_id", "program_id"], sort=True))
    T.ms["ml_pca"] = 0.0
    per_cycle, skipped, comps = [], [], []
    paths = {"onedal": 0, "stock": 0}
    for (m, p), g in groups:
        with T.phase("prep"):
            is_ref = g["day"].to_numpy() < t0
            R = g.loc[is_ref, SIGNALS].to_numpy(np.float64)
            Q = g.loc[~is_ref, SIGNALS].to_numpy(np.float64)
            if len(Q) == 0:
                continue
            if len(R) < min_cycles:
                skipped.append({"machine_id": m, "program_id": p, "ref_cycles": int(len(R))})
                continue
            mu, sd = R.mean(0), R.std(0)
            sd[sd == 0] = 1.0
            R = np.ascontiguousarray((R - mu) / sd)
            Q = np.ascontiguousarray((Q - mu) / sd)
        # route small blocks to stock PCA: per-call overhead dominates below a few thousand rows
        onedal = est.name == "onedal" and len(R) >= B.ROUTING["pca_onedal_min_rows"]
        PCA = est.PCA if onedal else est.stock.PCA
        paths["onedal" if onedal else "stock"] += 1
        with T.phase("ml_pca"):
            pca = PCA(n_components=var, svd_solver="full").fit(R)
            Zr, Zq = pca.transform(R), pca.transform(Q)
            Rr = R - pca.inverse_transform(Zr)
            Qr = Q - pca.inverse_transform(Zq)
        with T.phase("post"):
            lam = pca.explained_variance_
            spe_r, spe_q = (Rr ** 2).sum(1), (Qr ** 2).sum(1)
            t2_r, t2_q = (Zr ** 2 / lam).sum(1), (Zq ** 2 / lam).sum(1)
            out = (spe_q > np.quantile(spe_r, q)) | (t2_q > np.quantile(t2_r, q))
            comps.append(int(pca.n_components_))
            per_cycle.append(pd.DataFrame({"machine_id": m, "program_id": p, "day": g.loc[~is_ref, "day"].to_numpy(),
                                           "out": out, **{s: Q[:, i] for i, s in enumerate(SIGNALS)}}))
    with T.phase("post"):
        rows = []
        if per_cycle:
            pc = pd.concat(per_cycle, ignore_index=True)
            for m, g in pc.groupby("machine_id", sort=True):
                ab = g[g["out"]]
                dev = ab[SIGNALS].median() if len(ab) else g[SIGNALS].median()
                top = sorted(dev.items(), key=lambda t: -abs(t[1]))[:3]
                daily = g.groupby("day")["out"].mean()
                first = next((int(d) for d, v in daily.items() if v >= 0.3 and (daily.loc[d:] >= 0.3).mean() >= 0.75), None)
                rows.append({"machine_id": m, "programs": int(g["program_id"].nunique()), "recent_cycles": int(len(g)),
                             "abnormal_share": round(float(g["out"].mean()), 3), "first_abnormal_day": first,
                             "top_signals": [{"signal": s, "shift_sigma": round(float(z), 2)} for s, z in top]})
        rows.sort(key=lambda r: (-r["abnormal_share"], r["machine_id"]))
        flagged = [r for r in rows if r["abnormal_share"] >= flag_share]
        facts = {}
        for i, r in enumerate(flagged, 1):
            r["first_abnormal_date"] = day_to_date(plant, r["first_abnormal_day"]) if r["first_abnormal_day"] is not None else None
            sig = ", ".join(f"{t['signal']} {t['shift_sigma']:+.1f}σ" for t in r["top_signals"])
            facts[f"H{i}"] = (f"Signal H{i}: {r['machine_id']} signals left their usual pattern in {r['abnormal_share']:.0%} of "
                              f"its cycles in the last {test_days} days"
                              + (f", starting about {r['first_abnormal_date']}" if r["first_abnormal_date"] else "")
                              + f". Largest deviations: {sig}.")
        result = {"window": {"reference_days": [r0, t0 - 1], "test_days": [t0, d1]},
                  "models": len(comps), "median_components": float(np.median(comps)) if comps else None,
                  "machines_checked": len(rows), "flagged": flagged, "skipped_new_programs": skipped}
        facts["N"] = (f"Checked signal patterns on {len(rows)} machines ({len(comps)} machine-program models); "
                      f"{len(flagged)} abnormal.")
    params = {"test_days": test_days, "ref_days": ref_days, "var": var, "q": q, "flag_share": flag_share,
              "signals": SIGNALS, "pca_path": paths}
    return envelope("signal_health", scope, params, fingerprint(c["cycle_id"].to_numpy(), c["coolant_pressure"].to_numpy()),
                    result, facts, T, extra={"_shares": {r["machine_id"]: r["abnormal_share"] for r in rows}})
