from __future__ import annotations

from mcp import Client
import pytest

from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import Settings
from laya_mcp.server import create_server
from tests.fakes import FakeAgent


@pytest.mark.asyncio
async def test_mcp_lifespan_loads_once_and_exposes_structured_tools() -> None:
    agent = FakeAgent(max_length=256)
    load_count = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal load_count
        load_count += 1
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    server = create_server(Settings(model="fake/model"), backend_factory=factory)

    async with Client(server, mode="legacy") as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}
        assert tool_names == {"classify", "info"}
        assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools.tools)

        info = await client.call_tool("info")
        assert info.is_error is False
        assert info.structured_content is not None
        assert info.structured_content["initialization_state"] == "ready"
        assert info.structured_content["capabilities"]["max_total_tokens"] == 256

        decision = await client.call_tool(
            "classify",
            {
                "context": "movement and jumping",
                "question": "Is this relevant?",
                "decision_type": "noul",
            },
        )
        assert decision.is_error is False
        assert decision.structured_content is not None
        assert decision.structured_content["result"] is True
        assert decision.structured_content["usage"] == {
            "input_tokens": 42,
            "output_tokens": 0,
        }

    assert load_count == 1
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_mcp_returns_useful_capacity_error() -> None:
    agent = FakeAgent(max_length=24)

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=1.0)

    server = create_server(Settings(model="fake/model"), backend_factory=factory)
    async with Client(server, mode="legacy") as client:
        result = await client.call_tool(
            "classify",
            {
                "context": " ".join(["important"] * 50),
                "question": "Is this relevant?",
                "decision_type": "noul",
            },
        )

    assert result.is_error is True
    assert "input was not sent and was not truncated" in result.content[0].text
    assert agent.predict_calls == 0
