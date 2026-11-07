"""Windows machine info and plain demo output when stdout is not a terminal."""
import sys

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
