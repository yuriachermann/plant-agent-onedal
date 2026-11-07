"""Agent tool layer. Every tool takes (plant, scope, **params) and returns an envelope."""
from .backend import capture_dispatch, estimators, get_backend, set_backend, versions
from .drift import detect_cycle_drift
from .drivers import rank_loss_drivers
from .signals import signal_health
from .stops import cluster_stops

TOOLS = {
    "cluster_stops": cluster_stops,
    "detect_cycle_drift": detect_cycle_drift,
    "rank_loss_drivers": rank_loss_drivers,
    "signal_health": signal_health,
}


def public(env: dict) -> dict:
    """Strip internal fields (numpy arrays kept for the parity gate) before handing to the agent."""
    return {k: v for k, v in env.items() if not k.startswith("_")}


def call_tool(name: str, plant, scope: dict | None = None, **params) -> dict:
    return public(TOOLS[name](plant, scope, **params))


__all__ = ["TOOLS", "call_tool", "public", "set_backend", "get_backend", "estimators", "versions", "capture_dispatch",
           "cluster_stops", "detect_cycle_drift", "rank_loss_drivers", "signal_health"]
