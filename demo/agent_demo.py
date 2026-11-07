"""Scripted agent demo: one question, four tool calls, a cited briefing.

The planner and writer here are deterministic templates standing in for the
language model, so the demo runs offline and gives the same output every
time. The parts that matter for the talk are real: the tool calls, their
timings on each backend, the envelope with provenance and parity status, and
the citation check that blocks any number a tool did not return.

    python -m demo.agent_demo                       # oneDAL backend, Line 3, plant M
    python -m demo.agent_demo --backend stock
    python -m demo.agent_demo --compare             # both backends in fresh processes, side by side
    python -m demo.agent_demo --inject-error        # show the citation check catching a made-up number
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

QUESTION = "Why did Line {line} lose output this week?"
BOLD, DIM, GRN, RED, BLU, END = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[34m", "\033[0m"


def plan(plant, line: str) -> list[tuple[str, dict]]:
    """What the model would decide to call for a 'why did line X lose output' question."""
    return [
        ("cluster_stops", {"line_id": line, "day_from": plant.days - 7}),
        ("detect_cycle_drift", {"line_id": line}),
        ("signal_health", {"line_id": line}),
        ("rank_loss_drivers", {"line_id": line}),
    ]


def write_briefing(plant, line: str, envs: dict) -> str:
    """Template writer: quotes fact text verbatim and cites it. A real LLM paraphrases, then cite_check runs."""
    def cite(env, key):
        f = next((f for f in env["facts"] if f["id"].endswith("#" + key)), None)
        return f"{f['text']} [{f['id']}]" if f else None

    st, dr, he, rk = (envs[k] for k in ("cluster_stops", "detect_cycle_drift", "signal_health", "rank_loss_drivers"))
    out = [f"Line {line[1:] if line.startswith('L') else line}, this week."]
    out.append("Where the time went:")
    fams = [f for f in st["result"].get("families", []) if f.get("suggested_reason") != "meal"][:3]
    for f in fams:
        out.append("- " + cite(st, f["family"]))
    if (p := cite(st, "P")):
        out.append("- " + p)
    if dr["result"]["flagged"]:
        out.append("Getting slower:")
        for i in range(1, len(dr["result"]["flagged"]) + 1):
            out.append("- " + cite(dr, f"D{i}"))
    skipped = dr["result"].get("skipped_new_programs", [])
    if skipped:
        out.append("- " + cite(dr, "N"))
    if he["result"]["flagged"]:
        out.append("Machine health:")
        for i in range(1, len(he["result"]["flagged"]) + 1):
            out.append("- " + cite(he, f"H{i}"))
    out.append("What goes with unplanned stops on this line:")
    for r in rk["result"]["ranking"][:3]:
        if r.get("reported") and (c := cite(rk, f"R{r['rank']}")):
            out.append("- " + c)
    out.append("- " + cite(rk, "M"))
    return "\n".join(out)


def run(backend: str, size: str, seed: int, line: str, inject_error: bool, quiet: bool) -> dict:
    os.environ["PLANT_TOOLS_BACKEND"] = backend
    from demo.cite_check import check_answer
    from plantgen import generate_plant
    from tools import TOOLS, public, set_backend
    set_backend(backend)
    plant = generate_plant(size, seed=seed)
    q = QUESTION.format(line=line[1:] if line.startswith("L") else line)
    say = (lambda *a: None) if quiet else print
    say(f"\n{BOLD}User:{END} {q}\n{DIM}backend: {backend} · plant {plant.summary()['machines']} machines · "
        f"{plant.summary()['stops_per_day']:.0f} stops/day{END}\n")
    envs, trace = {}, []
    t_all = time.perf_counter()
    for name, scope in plan(plant, line):
        t = time.perf_counter()
        env = public(TOOLS[name](plant, scope))
        ms = (time.perf_counter() - t) * 1000
        envs[name] = env
        trace.append({"tool": name, "scope": scope, "ms": round(ms, 1), "call_id": env["call_id"],
                      "citable": env["parity"]["citable"], "facts": len(env["facts"])})
        mark = f"{GRN}citable{END}" if env["parity"]["citable"] else f"{RED}NOT citable: {env['parity']['reason']}{END}"
        say(f"  {BLU}→ {name}{END}({json.dumps(scope)})  {BOLD}{ms:7.0f} ms{END}  {env['call_id']}  {len(env['facts'])} facts  {mark}")
    tools_ms = (time.perf_counter() - t_all) * 1000
    answer = write_briefing(plant, line, envs)
    if inject_error:
        f1 = next(f["id"] for f in envs["cluster_stops"]["facts"] if f["id"].endswith("#F1"))
        answer += (f"\nM021 lost 37% more time than last week [{f1}]."
                   "\nNight shift is 2.4 times worse than day shift.")
    problems = check_answer(answer, list(envs.values()))
    say(f"\n  {DIM}tool time for this answer: {tools_ms:.0f} ms{END}\n")
    say(f"{BOLD}Agent:{END}\n{answer}\n")
    if problems:
        say(f"{RED}{BOLD}Citation check: {len(problems)} problem(s), answer blocked{END}")
        for p in problems:
            say(f"  {RED}✗{END} {p['problem']}  {DIM}← {p['sentence'][:90]}{END}")
    else:
        say(f"{GRN}{BOLD}Citation check: every number traces to a cited, citable fact ✓{END}")
    rec = {"backend": backend, "question": q, "tools_ms": round(tools_ms, 1), "trace": trace, "answer": answer,
           "citation_problems": problems, "envelopes": envs}
    out = REPO / "results" / f"demo_trace_{backend}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stock", "onedal"], default="onedal")
    ap.add_argument("--size", default="M")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--line", default="L3")
    ap.add_argument("--inject-error", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.compare:
        run(args.backend, args.size, args.seed, args.line, args.inject_error, args.quiet)
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    recs = {}
    for be in ("stock", "onedal"):
        subprocess.run([sys.executable, "-m", "demo.agent_demo", "--backend", be, "--size", args.size, "--seed",
                        str(args.seed), "--line", args.line, "--quiet"], check=True, cwd=REPO, env=env)
        recs[be] = json.loads((REPO / "results" / f"demo_trace_{be}.json").read_text())
    print(f"\n{BOLD}{recs['stock']['question']}{END}  (fresh process per backend, first call included)\n")
    print(f"  {'tool call':22s} {'stock':>9s} {'oneDAL':>9s} {'speed-up':>9s}")
    for a, b in zip(recs["stock"]["trace"], recs["onedal"]["trace"]):
        print(f"  {a['tool']:22s} {a['ms']:7.0f} ms {b['ms']:7.0f} ms {a['ms'] / b['ms']:8.1f}×")
    s, o = recs["stock"]["tools_ms"], recs["onedal"]["tools_ms"]
    print(f"  {BOLD}{'whole answer':22s} {s:7.0f} ms {o:7.0f} ms {s / o:8.1f}×{END}")
    print(f"\n  Same facts from both backends (ignoring call ids): "
          f"{_same_facts(recs['stock'], recs['onedal'])}")


def _same_facts(a, b) -> str:
    import re
    strip = lambda t: re.sub(r"\[[a-z]+-[0-9a-f]{8}#", "[", t)  # noqa: E731
    sa, sb = strip(a["answer"]).splitlines(), strip(b["answer"]).splitlines()
    diff = [(x, y) for x, y in zip(sa, sb) if x != y]
    if not diff and len(sa) == len(sb):
        return "yes, identical briefing"
    return f"{len(diff)} line(s) differ (expected only in random-forest importances): " + "; ".join(
        y[:80] for _, y in diff[:2])


if __name__ == "__main__":
    main()
