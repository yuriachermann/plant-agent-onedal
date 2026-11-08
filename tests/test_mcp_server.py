"""MCP server, driven by an in-process client. Skipped when the optional mcp SDK is not installed."""
import json

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402

from agent.mcp_server import build_server, tool_specs  # noqa: E402
from plantgen import generate_plant  # noqa: E402
from tools import TOOLS, public, set_backend  # noqa: E402


@pytest.fixture(scope="module")
def plant():
    return generate_plant("S", seed=3)


def test_specs_match_tools():
    specs = tool_specs()
    assert [s["name"] for s in specs] == list(TOOLS)
    for s in specs:
        assert "$ref" not in json.dumps(s["input_schema"])
        assert s["input_schema"]["properties"]["scope"]["type"] == "object"


def test_in_process_client(plant):
    set_backend("stock")
    scope = {"line_id": "L1"}

    async def session():
        async with Client(build_server(plant)) as client:
            listed = await client.list_tools()
            calls = {name: await client.call_tool(name, {"scope": scope}) for name in TOOLS}
            bad = await client.call_tool("detect_cycle_drift", {"scope": scope, "nope": 1})
            return listed, calls, bad

    listed, calls, bad = anyio.run(session)
    assert [t.name for t in listed.tools] == list(TOOLS)
    for name, res in calls.items():
        assert not res.is_error, res.content
        env = json.loads(res.content[0].text)
        assert res.structured_content == env
        direct = public(TOOLS[name](plant, scope))
        assert "_labels" not in env and not any(k.startswith("_") for k in env)
        assert env["call_id"] == direct["call_id"]
        assert env["facts"] == direct["facts"]
        assert env["parity"] == direct["parity"]
    assert bad.is_error
    assert "nope" in bad.content[0].text
