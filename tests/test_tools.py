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


def test_driver_null_rules(plant):
    set_backend("stock")
    sigma = TOOLS["rank_loss_drivers"](plant, None)["result"]
    shadow = TOOLS["rank_loss_drivers"](plant, None, null_rule="shadow")["result"]
    assert sigma["null_rule"] == "sigma" and "shadow_max_importance" not in sigma
    assert shadow["null_rule"] == "shadow"
    floor = max(shadow["shadow_max_importance"], 0.0)
    for item in shadow["ranking"][:6]:
        # rounded to 4 decimals in the result, so only check clear cases
        if item["importance"] > floor + 1e-4:
            assert item["reported"]
        elif item["importance"] < floor - 1e-4:
            assert not item["reported"]
    assert {r["driver"] for r in shadow["ranking"]} == {r["driver"] for r in sigma["ranking"]}
    with pytest.raises(ValueError):
        TOOLS["rank_loss_drivers"](plant, None, null_rule="nope")


def test_driver_holdout_by_day(plant):
    set_backend("stock")
    by_hour = TOOLS["rank_loss_drivers"](plant, None)["result"]
    env = TOOLS["rank_loss_drivers"](plant, None, holdout_by="day")
    by_day = env["result"]
    assert by_hour["holdout_by"] == "hour" and "holdout_days" not in by_hour
    assert by_day["holdout_by"] == "day" and env["params"]["holdout_by"] == "day"
    hours = plant.hours[plant.hours["unplanned_rate"].notna()]
    held = hours["day"].isin(by_day["holdout_days"])
    # whole days: every hour of a held-out day is in the holdout, every other hour trains
    assert by_day["holdout_hours"] == int(held.sum())
    assert by_day["train_hours"] == int((~held).sum())
    assert 0 < len(by_day["holdout_days"]) < hours["day"].nunique()
    model_fact = next(f["text"] for f in env["facts"] if f["id"].endswith("#M"))
    assert f"{len(by_day['holdout_days'])} whole days" in model_fact
    assert {r["driver"] for r in by_day["ranking"]} == {r["driver"] for r in by_hour["ranking"]}
    with pytest.raises(ValueError):
        TOOLS["rank_loss_drivers"](plant, None, holdout_by="week")
    last = int(hours["day"].max())
    with pytest.raises(ValueError):  # one day in scope cannot be split by day
        TOOLS["rank_loss_drivers"](plant, {"day_from": last, "day_to": last}, holdout_by="day")


def test_drift_step_test_flags_planted_step(plant):
    set_backend("stock")
    env = TOOLS["detect_cycle_drift"](plant, None)
    flagged = {f["machine_id"]: f for f in env["result"]["flagged"]}
    for m in plant.truth["drift_step"]:
        assert m in flagged and "step" in flagged[m]["flagged_by"]
        assert flagged[m]["step"]["change_pct"] >= env["params"]["step_min_pct"]
    off = TOOLS["detect_cycle_drift"](plant, None, step_test=False)
    assert all(f["flagged_by"] == ["envelope"] for f in off["result"]["flagged"])
    assert {f["machine_id"] for f in off["result"]["flagged"]} <= set(flagged)
