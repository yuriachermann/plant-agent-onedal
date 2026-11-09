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

Two rules decide which ranked columns are reported as drivers:

* ``sigma``: importance above 2 standard deviations of its own permutation
  repeats and at least ``relevance`` times the top driver's importance.
* ``shadow``: every feature also gets a shadow copy, the same column with its
  rows shuffled, so it keeps the distribution but loses any link to the
  target. The forest is fitted on features and shadows together, and a column
  is reported only if its importance is above zero and above the strongest
  shadow's.

Two ways to pick the held-out hours:

* ``hour`` (default): each machine-hour goes to the holdout with probability
  ``holdout``, so most days have hours on both sides of the split.
* ``day``: a ``holdout`` share of whole days is held out, so no day is on
  both sides. Columns drawn once per day (operator, weekday) can then no
  longer stand in for whatever else happened on that day.
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
NULL_RULES = ("sigma", "shadow")
HOLDOUT_BY = ("hour", "day")


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
                      perm_rows: int = 3000, seed: int = 0, top: int = 6, relevance: float = 0.20,
                      null_rule: str = "sigma", holdout_by: str = "hour") -> dict:
    if null_rule not in NULL_RULES:
        raise ValueError(f"rank_loss_drivers: null_rule must be one of {NULL_RULES}")
    if holdout_by not in HOLDOUT_BY:
        raise ValueError(f"rank_loss_drivers: holdout_by must be one of {HOLDOUT_BY}")
    est = B.estimators()
    T = Timer()
    with T.phase("prep"):
        machines = scope_machines(plant, scope)
        d0, d1 = day_range(plant, scope)
        h = select(plant.hours, machines, d0, d1)
        if target not in h.columns:
            raise ValueError(f"rank_loss_drivers: hours table has no '{target}' column")
        missing = [c for c in FEATURES if c not in h.columns]
        cats = [c for c in CATS if c in h.columns]
        nums = [c for c in NUMS if c in h.columns]
        if not cats and not nums:
            raise ValueError(f"rank_loss_drivers: no usable features; missing {missing}")
        h = h[h[target].notna()].reset_index(drop=True)
        # Drop incomplete rows only when a present feature has gaps. A full table takes the same path as before.
        if h[cats + nums].isna().any().any():
            h = h.dropna(subset=cats + nums).reset_index(drop=True)
        X = np.column_stack([
            *[pd.Categorical(h[c], categories=sorted(h[c].dropna().unique())).codes for c in cats],
            *[h[c].to_numpy(np.float64) for c in nums],
        ]).astype(np.float64)
        y = h[target].to_numpy(np.float64)
        n_feat = X.shape[1]
        Xm = X
        if null_rule == "shadow":
            # own random stream, so the holdout split and permutation sample match the sigma rule's
            srng = np.random.default_rng([seed, 1])
            Xm = np.hstack([X, np.column_stack([srng.permutation(X[:, j]) for j in range(n_feat)])])
        rng = np.random.default_rng(seed)
        hold_days, n_days = None, None
        if holdout_by == "day":
            if "day" not in h.columns:
                raise ValueError("rank_loss_drivers: holdout_by='day' needs a 'day' column in the hours table")
            day = h["day"].to_numpy()
            days = np.unique(day)
            n_days = len(days)
            if n_days < 2:
                raise ValueError("rank_loss_drivers: holdout_by='day' needs at least 2 days in scope")
            k = min(n_days - 1, max(1, int(round(holdout * n_days))))
            hold_days = np.sort(rng.choice(days, size=k, replace=False))
            te = np.isin(day, hold_days)
        else:
            te = rng.random(len(y)) < holdout
        Xtr, ytr, Xte, yte = (np.ascontiguousarray(a) for a in (Xm[~te], y[~te], Xm[te], y[te]))
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
        imp_mean, imp_std = pi.importances_mean[:n_feat], pi.importances_std[:n_feat]
        shadow_max = float(pi.importances_mean[n_feat:].max()) if null_rule == "shadow" else None
        order = np.argsort(-imp_mean, kind="stable")
        used = cats + nums
        ranking, facts = [], {}
        for rank, j in enumerate(order, 1):
            f = used[j]
            ranking.append({"rank": rank, "driver": f, "label": LABEL[f],
                            "importance": round(float(imp_mean[j]), 4),
                            "importance_std": round(float(imp_std[j]), 4)})
        best = max(ranking[0]["importance"], 1e-12)
        for item, j in zip(ranking[:top], order[:top]):
            if null_rule == "shadow":
                # reported only if shuffling it hurts more than shuffling any shadow column
                item["reported"] = bool(imp_mean[j] > max(shadow_max, 0.0))
            else:
                # reported only if clearly above permutation noise AND a meaningful fraction of the top driver
                item["reported"] = bool(item["importance"] > 2 * item["importance_std"]
                                        and item["importance"] >= relevance * best)
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
            "missing_features": missing,
            "target_hours_total": round(float(y.sum()) / 60, 1),
            "null_rule": null_rule,
            "holdout_by": holdout_by,
        }
        if null_rule == "shadow":
            result["shadow_max_importance"] = round(shadow_max, 4)
        if hold_days is not None:
            result["holdout_days"] = [int(d) for d in hold_days]
        what = {"unplanned_rate": "unplanned stop minutes per available hour",  # noqa: E501
                "unplanned_minutes": "unplanned stop minutes"}.get(target, target.replace("_", " "))
        facts["M"] = (f"Model of {what} on {len(y)} machine-hours over {d1 - d0 + 1} days; holdout R² {r2:.2f}. "
                      "Rankings show association, not cause." + (" " + conf[0] if conf else ""))
        if hold_days is not None:
            facts["M"] += f" The holdout is {len(hold_days)} whole days of the {n_days} in scope."
        if null_rule == "shadow":
            facts["M"] += (f" A driver is reported only if it beats the strongest of {n_feat} shuffled copies of the "
                           f"features (importance {shadow_max:.3f}).")
        if missing:
            facts["M"] += " Missing features, left out rather than filled in: " + ", ".join(missing) + "."
    params = {"target": target, "n_estimators": n_estimators, "max_depth": max_depth,
              "min_samples_leaf": min_samples_leaf, "n_repeats": n_repeats, "holdout": holdout,
              "perm_rows": perm_rows, "relevance": relevance, "null_rule": null_rule, "holdout_by": holdout_by,
              "seed": seed, "features": cats + nums, "missing_features": missing}
    return envelope("rank_loss_drivers", scope, params, fingerprint(y, X[:, -3]), result, facts, T,
                    extra={"_importances": imp_mean})
