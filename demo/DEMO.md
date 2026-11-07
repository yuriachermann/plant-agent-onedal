# Agent demo plan (≈5 minutes of the 40-minute slot)

**What the demo has to prove, in this order**

1. The agent's numbers come from tools, not from the model.
2. With oneDAL, the whole tool chain for one question fits a chat turn on an edge-class CPU.
3. The parity gate and the citation check keep that honest: a number nobody computed cannot reach the user.

Everything runs offline from this repo. The planner and writer in `agent_demo.py` are deterministic
templates standing in for the language model, so the output is the same on every run. That is a
feature on stage: no API key, no network, no surprises. Say so plainly when you start the demo.

## Setup (the day before and 30 minutes before)

| When | Do | Why |
|---|---|---|
| Day before | Run on the NUC: `python -m bench.parity` then `python -m bench.run_bench` | The parity record must match the installed library versions or every oneDAL call shows *NOT citable*. The benchmark gives you the slide numbers from the same machine you demo on. |
| Day before | Record the whole demo once (OBS, 1080p, terminal at 22 pt) | The event is virtual; talks are often pre-recorded. Even if live, this is your fallback. |
| 30 min before | `python -m demo.agent_demo --backend onedal --quiet` and the same with `--backend stock` | Warms the disk cache and imports; the first call in a cold process is slower. |
| 30 min before | Terminal: dark theme, 22 pt, ~100×30, Windows Terminal (ANSI colours) | Readable on a shared screen. |

## Run of show

| Time | Command | What the audience sees | What you say |
|---|---|---|---|
| 0:00 | *(slide: the question)* | "Why did Line 3 lose output this week?" | "This is the Monday question. The model will not compute a single number." |
| 0:20 | `python -m demo.agent_demo --backend stock` | Four tool calls with timings, then a cited briefing | "Stock scikit-learn. Watch the right-hand column: that's how long the user waits before the model can even start writing." |
| 1:20 | `python -m demo.agent_demo --backend onedal` | Same four calls, same facts, a fraction of the time | "Same code, one import changed. Same answer. The difference is the wait." |
| 2:10 | `python -c "import json;e=json.load(open('results/demo_trace_onedal.json'))['envelopes']['detect_cycle_drift'];print(json.dumps({k:e[k] for k in ['call_id','backend','parity','timings_ms']},indent=1));print(e['facts'][0])"` | One envelope: call id, versions, `citable: true`, a fact with its id | "Every number the agent may say lives in a fact with an id. And `citable` is only true because this exact library version passed the parity gate against stock." |
| 3:00 | `python -m demo.agent_demo --inject-error` | Briefing plus two invented sentences; the check blocks both | "Here I made the model claim 37% and 2.4 times. Neither number came from a tool, so the answer is blocked before anyone sees it." |
| 3:50 | `python -m demo.agent_demo --compare` | Side-by-side table, stock vs oneDAL, per call and per answer | "Fresh process per backend, first call included. This is the number that changed how we designed the agent." |
| 4:40 | back to slides | | |

## What to say if something goes wrong

* **A call shows *NOT citable*:** the parity record does not match the installed versions. That is the
  gate working. Say so, run `python -m bench.parity` (≈30 s), and re-run. It makes a good point live.
* **Timings look different from the slides:** they will vary a few percent. The slides are medians of
  5 warm runs; the live demo is single runs. Say that once.
* **Anything else:** switch to the recording.

## Wiring the same tools into the real agent

1. Register the four tools from `tools/schemas.json` (Anthropic tool use `input_schema` format; the
   same shape works for an MCP server).
2. System-prompt rules for the model:
   * Never compute, estimate or round a number yourself. Quote only `facts[].text` and cite as `[id]`.
   * If `parity.citable` is false, call the tool again with the stock backend or say the number is unavailable.
   * Say "associated with", never "caused by", for `rank_loss_drivers`.
3. Run `demo.cite_check.check_answer(answer, envelopes)` on every answer. On problems: ask the model to
   revise once with the problem list; if it still fails, show the answer with the offending sentences removed.
4. Nightly job: run the four tools plant-wide, store the envelopes, and generate the morning briefing
   from them. Chat questions call the tools live with a narrower scope.
