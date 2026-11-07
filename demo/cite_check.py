"""Citation check: every number in the agent's answer must come from a cited, citable fact.

The language model writes the answer; this check runs before the answer is
shown. Citations work like footnotes: a citation (or a group of adjacent
citations) covers the text back to the previous citation or the start of the
line. A claim fails when it:

* cites a fact id that no tool call returned,
* cites a fact from a call whose backend has not passed the parity gate, or
* contains a number that none of its cited facts contains (uncited text with
  a number always fails).

Dates and times count as numbers. Small integers (0-10) are allowed without a
citation, since they are list markers, counts or ranks.
"""
from __future__ import annotations

import re

CITE_ID = r"[a-z]+-[0-9a-f]{8}#[A-Z]\d*"
GROUP = re.compile(rf"((?:\s*\[{CITE_ID}\])+)")
ONE = re.compile(rf"\[({CITE_ID})\]")
NUM = re.compile(r"(?<![\w#-])[-+]?\d[\d,]*(?:\.\d+)?%?")


def _norm(n: str) -> str:
    return n.replace(",", "").lstrip("+").rstrip("%").rstrip(".")


def _nums(text: str) -> list[str]:
    return [n for n in NUM.findall(text) if _norm(n)]


def check_answer(answer: str, envelopes: list[dict]) -> list[dict]:
    facts = {}
    for env in envelopes:
        for f in env["facts"]:
            facts[f["id"]] = {"text": f["text"], "citable": env["parity"]["citable"]}
    problems = []

    def claim(text: str, ids: list[str]):
        for i in ids:
            if i not in facts:
                problems.append({"sentence": text.strip(), "problem": f"cites unknown fact {i}"})
            elif not facts[i]["citable"]:
                problems.append({"sentence": text.strip(), "problem": f"cites {i} from a backend that has not passed parity"})
        pool = {_norm(x) for i in ids if i in facts for x in _nums(facts[i]["text"])}
        for n in _nums(text):
            v = _norm(n)
            if v.isdigit() and int(v) <= 10:
                continue
            if v not in pool:
                where = "any cited fact" if ids else "a citation (uncited text)"
                problems.append({"sentence": text.strip(), "problem": f"number '{n}' is not in {where}"})

    for line in answer.splitlines():
        parts = GROUP.split(line)  # [text, cites, text, cites, ..., tail]
        for k in range(0, len(parts) - 1, 2):
            claim(parts[k], ONE.findall(parts[k + 1]))
        tail = parts[-1]
        if tail.strip():
            claim(tail, [])
    return problems
