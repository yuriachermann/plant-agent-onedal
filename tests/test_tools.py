"""Fast checks (small plant): parity, determinism, planted faults found, citation check, CSV round trip.

    pytest -q
"""
import warnings

import pytest

warnings.filterwarnings("ignore")

from bench.parity import compare  # noqa: E402
from demo.cite_check import check_answer  # noqa: E402
from plantgen import generate_plant, load_plant, save_plant  # noqa: E402
from tools import TOOLS, set_backend  # noqa: E402
from tools.common import result_digest  # noqa: E402


@pytest.fixture(scope="module")
def plant():
    return generate_plant("S", seed=3)


def _both(name, plant, scope=None):
    set_backend("stock")
    a = TOOLS[name](plant, scope)
    set_backend("onedal")
    b = TOOLS[name](plant, scope)
    return a, b


@pytest.mark.parametrize("name", list(TOOLS))
def test_parity(name, plant):
    a, b = _both(name, plant)
    ok, metrics = compare(name, a, b)
    assert ok, metrics


@pytest.mark.parametrize("name", list(TOOLS))
def test_deterministic(name, plant):
    set_backend("onedal")
    assert result_digest(TOOLS[name](plant, None)) == result_digest(TOOLS[name](plant, None))


def test_planted_faults_found(plant):
    set_backend("onedal")
    health = {f["machine_id"] for f in TOOLS["signal_health"](plant, None)["result"]["flagged"]}
    assert set(plant.truth["coolant_fault"]) <= health
    flagged = {f["machine_id"] for f in TOOLS["detect_cycle_drift"](plant, None)["result"]["flagged"]}
    planted = set(plant.truth["drift_wear"]) | set(plant.truth["drift_step"])
    assert flagged <= planted | set(plant.truth["coolant_fault"]), "drift tool flagged a healthy machine"


def test_citation_check_blocks_invented_numbers(plant):
    set_backend("onedal")
    env = TOOLS["cluster_stops"](plant, None)
    f = env["facts"][0]
    assert check_answer(f"{f['text']} [{f['id']}]", [env]) == []
    bad = check_answer(f"M001 lost 37.5% more [{f['id']}].", [env])
    assert any("37.5" in p["problem"] for p in bad)
    assert check_answer("Night shift is 2.4 times worse.", [env])
    assert check_answer("See [stops-deadbeef#F1].", [env])


def test_csv_round_trip(plant, tmp_path):
    save_plant(plant, tmp_path)
    p2 = load_plant(tmp_path)
    set_backend("stock")
    a = TOOLS["detect_cycle_drift"](plant, None)
    b = TOOLS["detect_cycle_drift"](p2, None)
    assert a["result"]["flagged"] == b["result"]["flagged"]
