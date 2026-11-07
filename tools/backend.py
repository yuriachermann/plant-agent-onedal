"""Backend switch: the only place the tools touch scikit-learn imports.

The tool code is plain scikit-learn. Switching to oneDAL is one import
change, done here, so stock and oneDAL can run side by side in the same
process (needed for the parity gate). In production you can instead call
``sklearnex.patch_sklearn()`` once at start-up and keep stock imports
everywhere: both routes land on the same oneDAL kernels.

Select with ``PLANT_TOOLS_BACKEND=stock|onedal`` or ``set_backend()``.
"""
from __future__ import annotations

import contextlib
import logging
import os
from types import SimpleNamespace

import sklearn

_BACKEND = os.environ.get("PLANT_TOOLS_BACKEND", "stock")
VALID = ("stock", "onedal")

# Size-based routing for the oneDAL backend: use oneDAL where it was measured faster, stock where not.
# Defaults measured with `python -m bench.kernels` on the reference machine; the benchmark report
# prints suggested values for whatever CPU it runs on. Override with environment variables.
ROUTING = {
    # oneDAL DBSCAN is brute force: fastest for small inputs, the stock KD-tree wins on large low-dim inputs
    "dbscan_brute_max": int(os.environ.get("PLANT_TOOLS_DBSCAN_BRUTE_MAX", 30_000)),
    # tiny PCA fits are dominated by per-call overhead; oneDAL wins from a few thousand rows up
    "pca_onedal_min_rows": int(os.environ.get("PLANT_TOOLS_PCA_MIN_ROWS", 8_000)),
}


def set_backend(name: str) -> None:
    global _BACKEND
    if name not in VALID:
        raise ValueError(f"backend must be one of {VALID}")
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def _stock() -> SimpleNamespace:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import NearestNeighbors
    return SimpleNamespace(name="stock", DBSCAN=DBSCAN, KMeans=KMeans, NearestNeighbors=NearestNeighbors,
                           RandomForestRegressor=RandomForestRegressor, PCA=PCA)


def estimators(name: str | None = None) -> SimpleNamespace:
    """Estimator classes for the backend. ``.stock`` always holds the stock classes for routed calls."""
    name = name or _BACKEND
    if name != "onedal":
        ns = _stock()
        ns.stock = ns
        return ns
    from sklearnex.cluster import DBSCAN, KMeans
    from sklearnex.decomposition import PCA
    from sklearnex.ensemble import RandomForestRegressor
    from sklearnex.neighbors import NearestNeighbors
    return SimpleNamespace(name=name, DBSCAN=DBSCAN, KMeans=KMeans, NearestNeighbors=NearestNeighbors,
                           RandomForestRegressor=RandomForestRegressor, PCA=PCA, stock=_stock())


def versions(name: str | None = None) -> dict:
    name = name or _BACKEND
    v = {"backend": name, "sklearn": sklearn.__version__}
    if name == "onedal":
        from importlib.metadata import version
        v["sklearnex"] = version("scikit-learn-intelex")
        try:
            v["onedal"] = version("daal")
        except Exception:  # pragma: no cover
            pass
    return v


class DispatchLog(logging.Handler):
    """Collects sklearnex dispatch messages: which calls ran on oneDAL, which fell back."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: list[dict] = []

    def emit(self, record):
        msg = record.getMessage()
        if ":" not in msg:
            return
        call, outcome = msg.split(":", 1)
        self.records.append({
            "call": call.strip().replace("sklearn.", ""),
            "accelerated": "running accelerated" in outcome,
            "fallback": "fallback" in outcome,
        })

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for r in self.records:
            d = out.setdefault(r["call"], {"accelerated": 0, "fallback": 0})
            d["accelerated"] += r["accelerated"]
            d["fallback"] += r["fallback"]
        return out


@contextlib.contextmanager
def capture_dispatch():
    """Context manager: ``with capture_dispatch() as log: ...; log.summary()``.

    Only use it for audits, not timing runs: INFO logging adds overhead per call.
    """
    lg = logging.getLogger("sklearnex")
    h = DispatchLog()
    old_level, old_prop, old_handlers = lg.level, lg.propagate, list(lg.handlers)
    lg.handlers = [h]  # sklearnex may attach its own console handler; silence it during the audit
    lg.setLevel(logging.INFO)
    lg.propagate = False
    try:
        yield h
    finally:
        lg.handlers = old_handlers
        lg.setLevel(old_level)
        lg.propagate = old_prop
