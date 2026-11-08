"""Shared pieces for every tool: scope selection, timing, and the result envelope.

Every tool returns the same envelope. The agent may only quote numbers that
appear in ``facts`` and must cite them by ``fact id`` (``<call_id>#<key>``).
``citable`` is false when the backend/version combination has not passed
the parity gate (see ``bench/parity.py``); the agent then re-runs the call
on the stock backend or answers without the number.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import backend as B

TOOL_VERSION = "0.3.0"
DAY = 86_400
REPO = Path(__file__).resolve().parents[1]
PARITY_FILE = Path(os.environ.get("PLANT_TOOLS_PARITY", REPO / "results" / "parity.json"))


# ------------------------------------------------------------------ scope
def scope_machines(plant, scope: dict | None) -> np.ndarray:
    scope = scope or {}
    m = plant.machines
    keep = np.ones(len(m), dtype=bool)
    if scope.get("line_id"):
        keep &= (m["line_id"] == scope["line_id"]).to_numpy()
    if scope.get("machine_ids"):
        keep &= m["machine_id"].isin(scope["machine_ids"]).to_numpy()
    return m["machine_id"].to_numpy()[keep]


def day_range(plant, scope: dict | None) -> tuple[int, int]:
    scope = scope or {}
    d1 = int(scope.get("day_to", plant.days - 1))
    d0 = int(scope.get("day_from", 0))
    return d0, d1


def select(df: pd.DataFrame, machines: np.ndarray, d0: int, d1: int) -> pd.DataFrame:
    mask = df["machine_id"].isin(machines).to_numpy() & (df["day"].to_numpy() >= d0) & (df["day"].to_numpy() <= d1)
    return df.loc[mask]


def day_to_date(plant, day: int) -> str:
    return (plant.start + pd.Timedelta(days=int(day))).date().isoformat()


def ts_to_str(plant, t: float) -> str:
    return (plant.start + pd.Timedelta(seconds=float(t))).strftime("%Y-%m-%d %H:%M")


def fingerprint(*cols: np.ndarray) -> str:
    """Cheap, deterministic fingerprint of the input slice (not a full content hash)."""
    h = hashlib.sha1()
    for c in cols:
        c = np.asarray(c)
        h.update(np.array([len(c)], dtype=np.int64).tobytes())
        if len(c):
            if c.dtype.kind in "fiu":
                h.update(np.array([np.nansum(c), c[0], c[-1]], dtype=np.float64).tobytes())
            else:
                h.update(str(c[0]).encode() + str(c[-1]).encode())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ timing
class Timer:
    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t) * 1000

    def total(self) -> float:
        return sum(self.ms.values())


# ------------------------------------------------------------------ parity / envelope
def _parity_status(tool: str, v: dict) -> dict:
    if v["backend"] == "stock":
        return {"citable": True, "reason": "reference backend"}
    try:
        reg = json.loads(PARITY_FILE.read_text())
    except FileNotFoundError:
        return {"citable": False, "reason": "no parity record; run bench/parity.py"}
    rec = reg.get("tools", {}).get(tool)
    if not rec:
        return {"citable": False, "reason": f"tool {tool} not in parity record"}
    same = (rec.get("sklearn") == v.get("sklearn") and rec.get("sklearnex") == v.get("sklearnex")
            and rec.get("tool_version") == TOOL_VERSION)
    if not same:
        return {"citable": False, "reason": "parity record is for other library or tool versions"}
    return {"citable": bool(rec.get("pass")), "reason": "parity gate " + ("passed" if rec.get("pass") else "FAILED"),
            "checked_at": rec.get("checked_at")}


def envelope(tool: str, scope: dict | None, params: dict, data_fp: str, result: dict,
             facts: dict[str, str], timer: Timer, extra: dict | None = None) -> dict:
    v = B.versions()
    key = json.dumps([tool, TOOL_VERSION, scope or {}, params, data_fp, v["backend"]], sort_keys=True, default=str)
    short = {"cluster_stops": "stops", "detect_cycle_drift": "drift",
             "rank_loss_drivers": "drivers", "signal_health": "health"}.get(tool, tool)
    call_id = f"{short}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
    out = {
        "tool": tool,
        "tool_version": TOOL_VERSION,
        "call_id": call_id,
        "scope": scope or {},
        "params": params,
        "input_fingerprint": data_fp,
        "backend": v,
        "parity": _parity_status(tool, v),
        "timings_ms": {k: round(x, 2) for k, x in timer.ms.items()} | {"total": round(timer.total(), 2)},
        "result": result,
        "facts": [{"id": f"{call_id}#{k}", "text": t} for k, t in facts.items()],
    }
    if extra:
        out.update(extra)
    return out


def result_digest(env: dict) -> str:
    """Hash of the result payload only (no timings), used to prove determinism."""
    return hashlib.sha1(json.dumps(env["result"], sort_keys=True, default=str).encode()).hexdigest()[:12]
