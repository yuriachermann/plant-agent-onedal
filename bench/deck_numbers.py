"""Slide figures for one benchmark directory, each with the JSON key it came from.

    python -m bench.deck_numbers results/intel-xeon-2t

Writes ``results/<tag>/deck_numbers.md``. Timings come from that directory's
``summary.json``. Parity comes from ``results/parity.json`` when the tag
directory has none. Stop suggestions for plant M seed 7 come from
``results/eval/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _ms(v: float) -> str:
    return f"{v:,.0f}"


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def build(tag: Path) -> str:
    tag = tag.resolve()
    summary = _load(tag / "summary.json")
    rows: list[tuple[str, str, str]] = []

    def add(figure: str, value: str, source: str) -> None:
        rows.append((figure, value, source))

    for size, block in summary["sizes"].items():
        ans = block["answer"]
        add(f"One answer, plant {size}", f"{_ms(ans['stock_ms'])} → {_ms(ans['onedal_ms'])} ms",
            f"summary.json sizes.{size}.answer.stock_ms / onedal_ms")
    if "M" in summary["sizes"]:
        for tool, block in summary["sizes"]["M"]["on_demand"].items():
            add(f"{tool}, plant M", f"{_ms(block['stock_ms'])} → {_ms(block['onedal_ms'])} ms",
                f"summary.json sizes.M.on_demand.{tool}.stock_ms / onedal_ms")
    for size, block in summary["sizes"].items():
        for when in ("cold", "warm"):
            night = block["nightly"][when]
            add(f"Nightly {when}, plant {size}", f"{night['stock_s']:.1f} → {night['onedal_s']:.1f} s",
                f"summary.json sizes.{size}.nightly.{when}.stock_s / onedal_s")
        cpu = block["daily_cpu"]
        add(f"CPU per day, plant {size}", f"{cpu['stock_core_s']:.0f} → {cpu['onedal_core_s']:.0f} core-s",
            f"summary.json sizes.{size}.daily_cpu.stock_core_s / onedal_core_s")
        add(f"Cost per day, plant {size}", f"${cpu['stock_usd']:.4f} → ${cpu['onedal_usd']:.4f}",
            f"summary.json sizes.{size}.daily_cpu.stock_usd / onedal_usd")
    for kernel, points in summary.get("kernels", {}).items():
        key = "n_ref" if points and "n_ref" in points[0] else "n"
        for point in points:
            add(f"Kernel {kernel} n={point[key]:,}", f"{point['speedup']}×",
                f"summary.json kernels.{kernel}[n={point[key]}].speedup")
    _parity_rows(tag, add)
    _stop_rows(add)

    lines = ["# Deck numbers", "",
             f"Source directory: `{tag.name}`. Stock → oneDAL. Regenerate with `python -m bench.deck_numbers {tag}`.",
             "",
             "| Figure | Value | Source |",
             "|---|---|---|"]
    lines += [f"| {fig} | {val} | `{src}` |" for fig, val, src in rows]
    lines.append("")
    return "\n".join(lines)


def _parity_rows(tag: Path, add) -> None:
    path = tag / "parity.json"
    rel = f"{tag.name}/parity.json"
    if not path.exists():
        path = REPO / "results" / "parity.json"
        rel = "results/parity.json"
    if not path.exists():
        add("Parity", "no parity.json", rel)
        return
    rec = _load(path)
    add("Parity record", rec.get("checked_at", ""), f"{rel} checked_at")
    for tool, body in rec["tools"].items():
        for scope, metrics in body["scopes"].items():
            if tool == "cluster_stops":
                value = f"ARI {metrics['family_ari']}, noise agreement {metrics['noise_agreement']}"
                source = f"{rel} tools.{tool}.scopes.{scope}.family_ari / noise_agreement"
            elif tool == "rank_loss_drivers":
                value = (f"Spearman {metrics['spearman']}, ΔR² {metrics['r2_abs_diff']}, "
                         f"top3 {metrics['top3_stock']}")
                source = f"{rel} tools.{tool}.scopes.{scope}.spearman / r2_abs_diff / top3_stock"
            else:
                value = (f"flagged {metrics['flagged_stock']}, max share diff {metrics['max_share_abs_diff']}")
                source = f"{rel} tools.{tool}.scopes.{scope}.flagged_stock / max_share_abs_diff"
            add(f"Parity {tool} ({scope})", value, source)


def _stop_rows(add) -> None:
    path = REPO / "results" / "eval" / "summary.json"
    rel = "results/eval/summary.json"
    if not path.exists():
        add("Stop suggestions, M seed 7", "no eval summary", rel)
        return
    runs = _load(path)["runs"]
    run = next((r for r in runs if r["size"] == "M" and r["seed"] == 7), None)
    if run is None:
        add("Stop suggestions, M seed 7", "run not in eval summary", rel)
        return
    stops = run["stops"]
    base = f"{rel} runs[size=M, seed=7].stops"
    add("Stop coverage, M seed 7", _pct(stops["coverage_count"]), f"{base}.coverage_count")
    add("Stop accuracy, M seed 7", _pct(stops["accuracy_count"]), f"{base}.accuracy_count")
    add("Stop accuracy by minutes, M seed 7", _pct(stops["accuracy_minutes"]), f"{base}.accuracy_minutes")


def _configure_stdio() -> None:
    """Windows consoles default to cp1252, which cannot print the table's arrows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv=None):
    _configure_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    args = ap.parse_args(argv)
    text = build(args.dir)
    out = args.dir / "deck_numbers.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
