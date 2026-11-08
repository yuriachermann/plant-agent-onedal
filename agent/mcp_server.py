"""Stdio MCP server for the four plant tools.

Exposes the tools in ``tools/schemas.json`` over the Model Context Protocol
with the official ``mcp`` SDK (v2 low-level ``Server``). Each call returns
``tools.public(envelope)``: the same envelope the agent sees in-process,
with call id, parity status and the facts it may quote. The JSON goes out
as text content and as structured content.

``mcp`` is optional (``pip install -e ".[mcp]"``). Nothing else in the repo
imports this module.

    python -m agent.mcp_server                       # plant M, seed 7, stock
    python -m agent.mcp_server --backend onedal
    python -m agent.mcp_server --plant-dir data/plant-M

Logs go to stderr. stdout carries the protocol only.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "tools" / "schemas.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def tool_specs() -> list[dict]:
    """Tool name, description and input schema from ``tools/schemas.json``.

    MCP input schemas must stand alone, so ``$ref`` to the shared scope
    definition is replaced by a copy of it.
    """
    spec = json.loads(SCHEMAS.read_text(encoding="utf-8"))
    defs = spec["definitions"]

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                return inline(copy.deepcopy(defs[ref.rsplit("/", 1)[-1]]))
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return [{"name": t["name"], "description": t["description"], "input_schema": inline(t["input_schema"])}
            for t in spec["tools"]]


def _jsonable(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_tool(plant, name: str, arguments: dict | None, specs: dict[str, dict]) -> dict:
    """Call one tool and return its public envelope as plain JSON types. Raises ValueError on bad arguments."""
    from tools import TOOLS, public
    if name not in specs:
        raise ValueError(f"unknown tool {name!r}; available: {sorted(specs)}")
    args = dict(arguments or {})
    allowed = set(specs[name]["input_schema"].get("properties", {}))
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"{name}: unknown arguments {unknown}; allowed: {sorted(allowed)}")
    scope = args.pop("scope", None)
    env = public(TOOLS[name](plant, scope, **args))
    return json.loads(json.dumps(env, default=_jsonable))


def build_server(plant, name: str = "plant-agent-onedal"):
    """Low-level MCP server bound to one plant. The backend is whatever ``tools.set_backend`` holds."""
    import anyio
    from mcp.server import Server, ServerRequestContext
    from mcp.types import (
        CallToolRequestParams,
        CallToolResult,
        ListToolsResult,
        PaginatedRequestParams,
        TextContent,
        Tool,
    )

    specs = {s["name"]: s for s in tool_specs()}
    tools = [Tool(name=s["name"], description=s["description"], input_schema=s["input_schema"]) for s in specs.values()]

    async def list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
        return ListToolsResult(tools=tools)

    async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        try:
            # the tools are CPU-bound; a worker thread keeps the session responsive
            env = await anyio.to_thread.run_sync(run_tool, plant, params.name, params.arguments, specs)
        except (ValueError, TypeError) as err:
            return CallToolResult(content=[TextContent(type="text", text=str(err))], is_error=True)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(env, ensure_ascii=False))],
                              structured_content=env)

    return Server(name, on_list_tools=list_tools, on_call_tool=call_tool)


def load(args) -> Any:
    from plantgen import generate_plant, load_plant
    from tools import set_backend
    set_backend(args.backend)
    if args.plant_dir:
        return load_plant(Path(args.plant_dir))
    return generate_plant(args.size, seed=args.seed)


async def serve(server) -> None:
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["stock", "onedal"], default="stock")
    ap.add_argument("--plant-dir", help="directory written by `python -m plantgen` or a real export")
    ap.add_argument("--size", default="M", help="synthetic plant size when --plant-dir is not given")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    try:
        import anyio
        import mcp  # noqa: F401
    except ImportError:
        print("agent.mcp_server needs the optional mcp SDK: pip install -e \".[mcp]\"", file=sys.stderr)
        return 2
    import warnings
    warnings.filterwarnings("ignore")
    plant = load(args)
    print(f"plant-agent MCP server: {plant.plant_id}, backend {args.backend}", file=sys.stderr)
    anyio.run(serve, build_server(plant))
    return 0


if __name__ == "__main__":
    sys.exit(main())
