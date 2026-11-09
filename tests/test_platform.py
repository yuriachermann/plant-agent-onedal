"""Windows machine info and plain demo output when stdout is not a terminal."""
import os
import sys

from bench import hybrid
from bench.hybrid import ECOQOS_ENV, apply_ecoqos_policy, current_cpu, placement
from bench.run_bench import machine_info
from demo.agent_demo import BOLD, color


def test_machine_info_reports_cpu_and_memory():
    info = machine_info()
    assert info["cpu"].strip()
    assert info["logical_cpus"] >= 1
    assert info["mem_gb"] > 0.5


def test_demo_color_is_plain_when_stdout_is_not_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color(BOLD) == ""


def test_demo_color_on_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color(BOLD) == BOLD
    monkeypatch.setenv("NO_COLOR", "1")
    assert color(BOLD) == ""


def test_probe_placement_share():
    sets = [{"logical": 0, "core": 0, "efficiency_class": 1},
            {"logical": 1, "core": 0, "efficiency_class": 1},
            {"logical": 2, "core": 1, "efficiency_class": 0}]
    assert placement([0, 1, 2, 2, None], sets) == {"samples": 4, "pcore_share": 0.5}
    assert placement([], sets) == {"samples": 0, "pcore_share": None}
    one_class = [dict(s, efficiency_class=0) for s in sets]
    assert placement([0, 2], one_class) == {"samples": 2, "pcore_share": None}
    cpu = current_cpu()
    if sys.platform == "win32":
        assert 0 <= cpu < (os.cpu_count() or 1)
    else:
        assert cpu is None


def test_bench_opts_out_of_ecoqos_unless_asked_to_keep_it(monkeypatch):
    calls = []
    monkeypatch.setattr(hybrid, "opt_out_of_ecoqos", lambda: calls.append(1))
    monkeypatch.setattr(hybrid.platform, "system", lambda: "Windows")
    monkeypatch.delenv(ECOQOS_ENV, raising=False)
    assert apply_ecoqos_policy() is True and calls == [1]
    monkeypatch.setenv(ECOQOS_ENV, "keep")
    assert apply_ecoqos_policy() is False and calls == [1]
    monkeypatch.delenv(ECOQOS_ENV)

    def fail():
        raise OSError(87, "SetProcessInformation(ProcessPowerThrottling) failed")
    monkeypatch.setattr(hybrid, "opt_out_of_ecoqos", fail)
    assert apply_ecoqos_policy() is False  # a failed call is recorded, not hidden
    monkeypatch.setattr(hybrid.platform, "system", lambda: "Linux")
    assert apply_ecoqos_policy() is None


def test_explicit_keep_wins_over_the_bench_variable(monkeypatch):
    calls = []
    monkeypatch.setattr(hybrid, "opt_out_of_ecoqos", lambda: calls.append(1))
    monkeypatch.setattr(hybrid.platform, "system", lambda: "Windows")
    monkeypatch.setenv(ECOQOS_ENV, "keep")
    assert apply_ecoqos_policy(keep=False) is True and calls == [1]
    monkeypatch.delenv(ECOQOS_ENV)
    assert apply_ecoqos_policy(keep=True) is False and calls == [1]


def test_mcp_server_opts_out_unless_asked_to_keep(monkeypatch):
    # runs without the mcp SDK: the server module imports mcp only inside its functions
    from agent.mcp_server import set_power_policy
    calls = []
    monkeypatch.setattr(hybrid, "opt_out_of_ecoqos", lambda: calls.append(1))
    monkeypatch.setattr(hybrid.platform, "system", lambda: "Windows")
    monkeypatch.setenv(ECOQOS_ENV, "keep")  # the benchmark's switch does not reach the server
    assert set_power_policy(keep_ecoqos=False) == ", EcoQoS opt-out" and calls == [1]
    assert set_power_policy(keep_ecoqos=True) == ", EcoQoS left to Windows" and calls == [1]
    monkeypatch.setattr(hybrid.platform, "system", lambda: "Linux")
    assert set_power_policy(keep_ecoqos=False) == ""


def test_machine_info_records_power_on_windows():
    info = machine_info()
    if sys.platform != "win32":
        assert "ac_power" not in info
        return
    assert info["ac_power"] in (True, False, None)
    assert "power_plan" in info
