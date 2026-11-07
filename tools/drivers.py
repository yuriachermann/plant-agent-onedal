"""rank_loss_drivers: which conditions go with lost production?

A random forest predicts lost minutes per machine-hour from the conditions of
that hour (shift, supplier batch, tool age, days since maintenance, ...).
Drivers are ranked by permutation importance on held-out hours: how much
worse the forest predicts when that one column is shuffled. Permutation
importance is used instead of impurity importance because impurity importance
favours high-cardinality columns such as operator IDs.

The ranking says what *goes with* losses, not what *causes* them. Shift and
line are confounded when only one line runs a night shift; the result keeps
both and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import backend as B
from .common import Timer, day_range, envelope, fingerprint, scope_machines, select

CATS = ["line_id", "machine_type", "shift", "supplier", "operator_id"]
NUMS = ["hour_of_day", "weekday", "program_new", "tool_age", "days_since_pm", "ambient_temp", "coolant_conc"]
FEATURES = CATS + NUMS
LABEL = {"line_id": "line", "machine_type": "machine type", "shift": "shift", "supplier": "material supplier",
         "operator_id": "operator", "hour_of_day": "hour of day", "weekday": "weekday",
         "program_new": "new program (first 3 days)", "tool_age": "tool age (cycles)",
         "days_since_pm": "days since preventive maintenance", "ambient_temp": "ambient temperature",
         "coolant_conc": "coolant concentration"}


def _effect(h: pd.DataFrame, f: str, target: str) -> dict:
    y = h[target]
    base = float(y.mean())
    if f in CATS or f == "program_new":
        means = y.groupby(h[f]).mean()
        lvl = means.idxmax()
        return {"kind": "level", "worst": str(lvl), "worst_mean": round(float(means.max()), 2), "overall_mean": round(base, 2)}
    q = pd.qcut(h[f].rank(method="first"), 4, labels=False)
    means = y.groupby(q).mean()
    return {"kind": "quartiles", "top_quartile_mean": round(float(means.iloc[-1]), 2),
            "bottom_quartile_mean": round(float(means.iloc[0]), 2), "overall_mean": round(base, 2)}


def rank_loss_drivers(plant, scope: dict | None = None, target: str = "unplanned_rate", n_estimators: int = 200,
                      max_depth: int = 10, min_samples_leaf: int = 30, n_repeats: int = 5, holdout: float = 0.25,
                      perm_rows: int = 3000, seed: int = 0, top: int = 6, relevance: float = 0.20) -> dict:
    est = B.estimators()
    T = Timer()
    with T.phase("prep"):
        machines = scope_machines(plant, scope)
        d0, d1 = day_range(plant, scope)
        h = select(plant.hours, machines, d0, d1)
        h = h[h[target].notna()].reset_index(drop=True)
        X = np.column_stack([
            *[pd.Categorical(h[c], categories=sorted(h[c].unique())).codes for c in CATS],
            *[h[c].to_numpy(np.float64) for c in NUMS],
        ]).astype(np.float64)
        y = h[target].to_numpy(np.float64)
        rng = np.random.default_rng(seed)
        te = rng.random(len(y)) < holdout
        Xtr, ytr, Xte, yte = (np.ascontiguousarray(a) for a in (X[~te], y[~te], X[te], y[te]))
        # permutation importance on a fixed-size sample of the holdout keeps the cost flat as plants grow
        pr = np.sort(rng.permutation(len(yte))[:perm_rows])
        Xp, yp = np.ascontiguousarray(Xte[pr]), yte[pr]
    with T.phase("ml_rf_fit"):
        rf = est.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                       max_features=0.5, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    with T.phase("ml_rf_permutation"):
        r2 = float(rf.score(Xte, yte))
        pi = permutation_importance(rf, Xp, yp, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    with T.phase("post"):
        order = np.argsort(-pi.importances_mean, kind="stable")
        ranking, facts = [], {}
        for rank, j in enumerate(order, 1):
            f = FEATURES[j]
            ranking.append({"rank": rank, "driver": f, "label": LABEL[f],
                            "importance": round(float(pi.importances_mean[j]), 4),
                            "importance_std": round(float(pi.importances_std[j]), 4)})
        best = max(ranking[0]["importance"], 1e-12)
        for item in ranking[:top]:
            # reported only if clearly above permutation noise AND a meaningful fraction of the top driver
            item["reported"] = bool(item["importance"] > 2 * item["importance_std"] and item["importance"] >= relevance * best)
            if not item["reported"]:
                continue
            e = _effect(h, item["driver"], target)
            item["effect"] = e
            if e["kind"] == "level":
                eff = (f"worst level '{e['worst']}' averages {e['worst_mean']} min/h vs {e['overall_mean']} min/h overall")
            else:
                eff = (f"highest quarter averages {e['top_quartile_mean']} min/h vs lowest quarter "
                       f"{e['bottom_quartile_mean']} min/h")
            facts[f"R{item['rank']}"] = (f"Driver #{item['rank']}: {item['label']} (importance {item['importance']:.3f} "
                                         f"± {item['importance_std']:.3f} R² drop); {eff}.")
        conf = []
        if "shift" in [r["driver"] for r in ranking[:top]] and plant.machines["night_shift"].sum() < len(plant.machines):
            conf.append("Only some lines run shift C, so 'shift' and 'line' are partly confounded.")
        result = {
            "target": target, "hours": int(len(y)), "train_hours": int((~te).sum()), "holdout_hours": int(te.sum()),
            "permutation_rows": int(len(yp)),
            "holdout_r2": round(r2, 4), "ranking": ranking, "caveats": conf,
            "target_hours_total": round(float(y.sum()) / 60, 1),
        }
        what = {"unplanned_rate": "unplanned stop minutes per available hour",  # noqa: E501
                "unplanned_minutes": "unplanned stop minutes"}.get(target, target.replace("_", " "))
        facts["M"] = (f"Model of {what} on {len(y)} machine-hours over {d1 - d0 + 1} days; holdout R² {r2:.2f}. "
                      "Rankings show association, not cause." + (" " + conf[0] if conf else ""))
    params = {"target": target, "n_estimators": n_estimators, "max_depth": max_depth,
              "min_samples_leaf": min_samples_leaf, "n_repeats": n_repeats, "holdout": holdout,
              "perm_rows": perm_rows, "relevance": relevance,
              "seed": seed, "features": FEATURES}
    return envelope("rank_loss_drivers", scope, params, fingerprint(y, X[:, -3]), result, facts, T,
                    extra={"_importances": pi.importances_mean})
