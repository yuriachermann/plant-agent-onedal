"""cluster_stops: group stop events into families and find irregular stops.

* k-means puts every stop in one of ``k`` families, so the briefing can total
  minutes by family and suggest a reason for stops nobody justified.
* DBSCAN separates recurring, dense patterns (meals, setups at shift start,
  micro-stop bursts) from irregular stops (DBSCAN noise), which are the ones
  a person has to explain.

DBSCAN path: oneDAL's DBSCAN is brute force. On low-dimensional data with many
points the stock KD-tree is faster, so with the oneDAL backend the tool uses
oneDAL up to ``ROUTING['dbscan_brute_max']`` points and calls the stock
KD-tree directly above it (going through sklearnex's fallback costs ~8%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import kmeans_plusplus

from . import backend as B
from .common import Timer, day_range, envelope, fingerprint, scope_machines, select, ts_to_str

SHIFT_START_H = {"A": 6.0, "B": 14.0, "C": 22.0}
# feature weights after standardisation (duration matters most)
WEIGHTS = np.array([2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
FEATURE_NAMES = ["log_duration", "hour_sin", "hour_cos", "log_cycles_since_prev",
                 "spindle_load_before", "tool_age_before", "log_hours_into_shift"]


def stop_features(s: pd.DataFrame) -> np.ndarray:
    h = s["hour_of_day"].to_numpy()
    into = (h - s["shift"].map(SHIFT_START_H).to_numpy()) % 24
    X = np.column_stack([
        np.log1p(s["duration_s"].to_numpy()),
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.log1p(s["cycles_since_prev"].to_numpy()),
        s["spindle_load_before"].to_numpy() / 100,
        s["tool_age_before"].to_numpy() / 100,
        np.log1p(into),
    ]).astype(np.float64)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return np.ascontiguousarray((X - X.mean(0)) / sd * WEIGHTS)


def _circ_mean_hour(h: np.ndarray) -> float:
    a = 2 * np.pi * h / 24
    return float((np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * 24 / (2 * np.pi)) % 24)


def cluster_stops(plant, scope: dict | None = None, k: int = 10, eps: float | None = None, min_samples: int | None = None,
                  dbscan_policy: str = "auto", min_confidence: float = 0.6, seed: int = 0) -> dict:
    est = B.estimators()
    T = Timer()
    with T.phase("prep"):
        machines = scope_machines(plant, scope)
        d0, d1 = day_range(plant, scope)
        s = select(plant.stops, machines, d0, d1)
        X = stop_features(s)
        n = len(X)
    if min_samples is None:
        # density threshold scales with the number of stops in scope (30 at ~40k stops)
        min_samples = max(10, int(round(n / 1400)))
    if eps is None:
        # calibrated at eps=0.5 for ~40k stops; nearest-neighbour distances grow as n^(-1/d) when n shrinks
        eps = float(np.clip(0.5 * (40_000 / max(n, 1)) ** (1 / X.shape[1]), 0.5, 1.0))
    if n < max(k, min_samples) * 2:
        return envelope("cluster_stops", scope, {"k": k}, fingerprint(s["stop_id"].to_numpy()),
                        {"n_stops": int(n), "note": "too few stops in scope"}, {}, T)

    use_onedal = est.name == "onedal" and (
        dbscan_policy == "brute" or (dbscan_policy == "auto" and n <= B.ROUTING["dbscan_brute_max"]))
    DBSCAN, algorithm = (est.DBSCAN, "brute") if use_onedal else (est.stock.DBSCAN, "auto")
    path = "onedal-brute" if use_onedal else "stock-tree"

    with T.phase("init"):
        # Initial centers are computed once, outside the backend, so stock and oneDAL start from the
        # same point and converge to the same families. Library-internal k-means++ uses different RNGs.
        centers, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
    with T.phase("ml_kmeans"):
        km = est.KMeans(n_clusters=k, init=centers, n_init=1).fit(X)
    with T.phase("ml_dbscan"):
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, n_jobs=-1).fit(X)

    with T.phase("post"):
        df = pd.DataFrame({
            "fam": km.labels_, "noise": db.labels_ == -1, "dur": s["duration_s"].to_numpy(),
            "h": s["hour_of_day"].to_numpy(), "rep": s["reason_reported"].to_numpy(),
            "m": s["machine_id"].to_numpy(),
        })
        df["unjust"] = df["rep"] == ""
        # order families by lost minutes so labels are stable and comparable across backends
        order = df.groupby("fam")["dur"].sum().sort_values(ascending=False).index
        relabel = {f: i + 1 for i, f in enumerate(order)}
        df["F"] = df["fam"].map(relabel)
        fams, facts = [], {}
        sug_n = sug_min = 0.0
        for F, g in df.groupby("F", sort=True):
            just = g.loc[~g["unjust"], "rep"]
            if len(just):
                vc = just.value_counts()
                reason, conf = str(vc.index[0]), float(vc.iloc[0] / len(just))
            else:
                reason, conf = None, 0.0
            suggested = reason if conf >= min_confidence else None
            if suggested:
                sug_n += g["unjust"].sum()
                sug_min += g.loc[g["unjust"], "dur"].sum() / 60
            top_m = g.groupby("m")["dur"].sum().sort_values(ascending=False).head(3)
            fam = {
                "family": f"F{F}", "stops": int(len(g)), "hours": round(g["dur"].sum() / 3600, 2),
                "median_min": round(g["dur"].median() / 60, 2), "typical_hour": round(_circ_mean_hour(g["h"].to_numpy()), 1),
                "unjustified_share": round(g["unjust"].mean(), 3), "suggested_reason": suggested,
                "confidence": round(conf, 3), "irregular_share": round(g["noise"].mean(), 3),
                "top_machines": [{"machine_id": m, "hours": round(v / 3600, 2)} for m, v in top_m.items()],
            }
            fams.append(fam)
            hh = int(fam["typical_hour"])
            mm = int(round((fam["typical_hour"] - hh) * 60)) % 60
            facts[f"F{F}"] = (
                f"Family F{F}: {fam['stops']} stops, {fam['hours']} h stopped, median {fam['median_min']} min, "
                f"typically around {hh:02d}:{mm:02d}, {fam['unjustified_share']:.0%} without a reason"
                + (f"; likely '{suggested}' (confidence {conf:.0%})" if suggested else "; no reliable reason suggestion")
                + f"; most time on {', '.join(top_m.index[:3])}."
            )

        irr = s.loc[df["noise"].to_numpy()].sort_values("duration_s", ascending=False)
        irregular_top = []
        for i, (_, r) in enumerate(irr.head(10).iterrows(), 1):
            item = {"stop_id": int(r["stop_id"]), "machine_id": r["machine_id"], "start": ts_to_str(plant, r["t_start"]),
                    "minutes": round(r["duration_s"] / 60, 1), "reason_reported": r["reason_reported"] or None}
            irregular_top.append(item)
            facts[f"X{i}"] = (f"Irregular stop X{i}: {item['machine_id']} stopped {item['minutes']} min from {item['start']}"
                              + (f", logged as '{item['reason_reported']}'." if item["reason_reported"] else ", no reason logged."))
        n_unjust = int(df["unjust"].sum())
        result = {
            "n_stops": int(n),
            "days": [d0, d1],
            "machines": int(len(machines)),
            "families": fams,
            "patterns": {
                "dense_patterns": int(db.labels_.max() + 1),
                "regular_hours": round(df.loc[~df["noise"], "dur"].sum() / 3600, 2),
                "irregular_hours": round(df.loc[df["noise"], "dur"].sum() / 3600, 2),
                "irregular_stops": int(df["noise"].sum()),
            },
            "irregular_top": irregular_top,
            "suggestions": {
                "unjustified_stops": n_unjust,
                "with_suggestion": int(sug_n),
                "coverage": round(sug_n / max(n_unjust, 1), 3),
                "hours_covered": round(sug_min / 60, 2),
            },
        }
        facts["P"] = (f"{result['patterns']['irregular_stops']} of {n} stops are irregular (no recurring pattern), "
                      f"{result['patterns']['irregular_hours']} h of {round(df['dur'].sum() / 3600, 2)} h stopped.")
        facts["S"] = (f"{int(sug_n)} of {n_unjust} stops without a reason ({result['suggestions']['coverage']:.0%}) "
                      f"get a suggested reason from their family.")

    params = {"k": k, "eps": round(eps, 4), "min_samples": min_samples, "dbscan_policy": dbscan_policy,
              "dbscan_path": path, "features": FEATURE_NAMES}
    return envelope("cluster_stops", scope, params, fingerprint(s["stop_id"].to_numpy(), s["duration_s"].to_numpy()),
                    result, facts, T, extra={"_labels": {"family": df["F"].to_numpy(), "noise": df["noise"].to_numpy()}})
