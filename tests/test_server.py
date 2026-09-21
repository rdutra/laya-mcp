from __future__ import annotations

import json
from typing import Any

from mcp import Client
import pytest

from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import DEFAULT_MODEL, Settings
from laya_mcp.lifecycle import LazyBackendManager
from laya_mcp.server import create_server
from tests.fakes import FakeAgent


def server_for(agent: FakeAgent) -> Any:
    async def factory(settings: Settings) -> LayaCoreMLBackend:
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    return create_server(Settings(model="fake/model"), backend_factory=factory)


@pytest.mark.asyncio
async def test_public_tools_and_binary_decide_load_once() -> None:
    agent = FakeAgent(max_length=256)
    load_count = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal load_count
        load_count += 1
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    server = create_server(Settings(model="fake/model"), backend_factory=factory)
    async with Client(server, mode="legacy") as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {
            "batch_decide",
            "decide",
            "filter",
            "info",
        }
        assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools.tools)
        result = await client.call_tool(
            "decide",
            {
                "context": "movement and jumping",
                "question": "Is this relevant?",
                "request_id": "request-7",
                "confidence_threshold": 0.9,
            },
        )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["request_id"] == "request-7"
    assert result.structured_content["decision_type"] == "binary"
    assert result.structured_content["result"] is True
    assert result.structured_content["needs_escalation"] is True
    assert result.structured_content["details"]["probabilities"] == {
        "false": 0.1875,
        "true": 0.8125,
    }
    assert load_count == 1
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_info_is_available_without_loading_the_model() -> None:
    agent = FakeAgent(max_length=96)
    load_count = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal load_count
        load_count += 1
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    server = create_server(Settings(model=DEFAULT_MODEL), backend_factory=factory)
    async with Client(server, mode="legacy") as client:
        result = await client.call_tool("info", {})

    assert result.is_error is False
    assert result.structured_content is not None
    body = result.structured_content
    assert body["model"] == DEFAULT_MODEL
    assert body["initialization_state"] == "unloaded"
    assert body["capabilities"]["max_total_tokens"] == 96
    assert body["metrics"]["initialization_attempts"] == 0
    assert load_count == 0


@pytest.mark.asyncio
async def test_failed_initialization_is_reported_and_next_request_retries() -> None:
    agent = FakeAgent(max_length=96)
    attempts = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("model cache is unavailable")
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    server = create_server(Settings(model="fake/model"), backend_factory=factory)
    async with Client(server, mode="legacy") as client:
        first = await client.call_tool(
            "decide", {"context": "x", "question": "Is this valid?"}
        )
        failed_info = await client.call_tool("info", {})
        second = await client.call_tool(
            "decide", {"context": "x", "question": "Is this valid?"}
        )
        ready_info = await client.call_tool("info", {})

    assert first.is_error is True
    assert "unable to initialize model" in first.content[0].text
    assert failed_info.structured_content is not None
    assert failed_info.structured_content["initialization_state"] == "failed"
    assert failed_info.structured_content["capabilities"] is None
    assert failed_info.structured_content["metrics"]["initialization_attempts"] == 1
    assert "model cache is unavailable" in failed_info.structured_content["metrics"][
        "last_initialization_error"
    ]
    assert second.is_error is False
    assert ready_info.structured_content is not None
    assert ready_info.structured_content["initialization_state"] == "ready"
    assert ready_info.structured_content["metrics"]["initialization_attempts"] == 2
    assert ready_info.structured_content["metrics"]["initialization_count"] == 1
    assert attempts == 2


@pytest.mark.asyncio
async def test_simultaneous_first_use_initializes_once() -> None:
    import asyncio

    agent = FakeAgent(max_length=96)
    attempts = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.01)
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    manager = LazyBackendManager(Settings(model="fake/model"), factory)
    loaded = await asyncio.gather(manager.ensure_loaded(), manager.ensure_loaded())

    assert attempts == 1
    assert loaded[0] is loaded[1]
    assert manager.info().initialization_state == "ready"
    assert manager.info().metrics.initialization_attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            "batch_decide",
            {
                "shared_context": "movement bug",
                "items": [{"id": "movement", "question": "Relevant?"}],
            },
        ),
        (
            "filter",
            {
                "criterion": "Relevant to movement",
                "candidates": [{"id": "movement", "text": "movement controller"}],
            },
        ),
    ],
)
async def test_batch_and_filter_initialize_on_first_use_and_then_reuse(
    tool: str, arguments: dict[str, Any]
) -> None:
    agent = FakeAgent(max_length=96)
    load_count = 0

    async def factory(settings: Settings) -> LayaCoreMLBackend:
        nonlocal load_count
        load_count += 1
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=7.5)

    server = create_server(Settings(model="fake/model"), backend_factory=factory)
    async with Client(server, mode="legacy") as client:
        before = await client.call_tool("info", {})
        first = await client.call_tool(tool, arguments)
        second = await client.call_tool(tool, arguments)

    assert before.structured_content is not None
    assert before.structured_content["initialization_state"] == "unloaded"
    assert first.is_error is False
    assert second.is_error is False
    assert load_count == 1
    assert agent.predict_calls == 2


@pytest.mark.asyncio
async def test_decide_choice_and_ordered_score() -> None:
    agent = FakeAgent(max_length=256)
    async with Client(server_for(agent), mode="legacy") as client:
        choice = await client.call_tool(
            "decide",
            {
                "context": "landing bug",
                "question": "Which subsystem?",
                "decision": {"kind": "choice", "options": ["movement", "animation"]},
            },
        )
        score = await client.call_tool(
            "decide",
            {
                "context": "moderate impact",
                "question": "How severe?",
                "decision": {
                    "kind": "ordered_score",
                    "options": ["low", "medium", "high"],
                },
            },
        )

    assert choice.structured_content is not None
    assert choice.structured_content["result"] == "movement"
    assert choice.structured_content["details"]["probabilities"] == {
        "movement": 0.75,
        "animation": 0.25,
    }
    assert score.structured_content is not None
    assert score.structured_content["decision_type"] == "ordered_score"
    assert score.structured_content["result"] == 1.25
    assert set(score.structured_content["details"]["probabilities"]) == {
        "low",
        "medium",
        "high",
    }


@pytest.mark.asyncio
async def test_decide_rejects_capacity_malformed_input_and_backend_failure() -> None:
    capacity_agent = FakeAgent(max_length=24)
    async with Client(server_for(capacity_agent), mode="legacy") as client:
        capacity = await client.call_tool(
            "decide",
            {
                "context": " ".join(["important"] * 50),
                "question": "Is this relevant?",
            },
        )
        malformed = await client.call_tool(
            "decide",
            {
                "context": "x",
                "question": "Choose",
                "decision": {"kind": "choice", "options": ["only"]},
            },
        )
    assert capacity.is_error is True
    assert "not truncated" in capacity.content[0].text
    assert malformed.is_error is True
    assert capacity_agent.predict_calls == 0

    failing_agent = FakeAgent(max_length=256, fail_on_calls={1})
    async with Client(server_for(failing_agent), mode="legacy") as client:
        failed = await client.call_tool(
            "decide", {"context": "x", "question": "Is this valid?"}
        )
    assert failed.is_error is True
    assert "decision backend failure" in failed.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 10, 100])
async def test_batch_shared_context_uses_one_backend_call(count: int) -> None:
    agent = FakeAgent(max_length=256)
    items = [{"id": str(index), "question": "Is this relevant?"} for index in range(count)]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide", {"shared_context": "movement bug", "items": items}
        )

    assert result.is_error is False
    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["items"]] == [str(index) for index in range(count)]
    assert body["metrics"]["decision_count"] == count
    assert body["metrics"]["successful_count"] == count
    assert body["metrics"]["backend_calls"] == 1
    assert body["metrics"]["model_evaluations"] == count
    assert body["metrics"]["total_input_tokens"] == 42 * count
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_batch_independent_contexts_preserve_order_and_usage() -> None:
    agent = FakeAgent(max_length=256)
    items = [
        {"id": "third", "context": "context 3", "question": "Relevant?"},
        {"id": "first", "context": "context 1", "question": "Relevant?"},
        {"id": "second", "context": "context 2", "question": "Relevant?"},
    ]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool("batch_decide", {"items": items})

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["items"]] == ["third", "first", "second"]
    assert body["metrics"]["backend_calls"] == 3
    assert body["metrics"]["model_evaluations"] == 3
    assert body["metrics"]["total_input_tokens"] == 126


@pytest.mark.asyncio
async def test_batch_mixed_confidence_and_detailed_metadata() -> None:
    agent = FakeAgent(
        max_length=256,
        confidence_by_id={"high": 0.95, "low": 0.4, "override": 0.7},
    )
    items = [
        {"id": "high", "question": "Relevant?", "confidence_threshold": 0.95},
        {"id": "low", "question": "Relevant?"},
        {"id": "override", "question": "Relevant?", "confidence_threshold": 0.6},
    ]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide",
            {
                "shared_context": "movement bug",
                "items": items,
                "confidence_threshold": 0.8,
                "response_detail": "detailed",
            },
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["needs_escalation"] for item in body["items"]] == [False, True, False]
    assert body["metrics"]["escalation_count"] == 1
    assert all(item["details"]["probabilities"] for item in body["items"])


@pytest.mark.asyncio
async def test_batch_partial_policy_reports_oversized_item_without_omission() -> None:
    agent = FakeAgent(max_length=40)
    items = [
        {"id": "good", "question": "Relevant?"},
        {"id": "too-large", "question": " ".join(["long"] * 80)},
    ]
    async with Client(server_for(agent), mode="legacy") as client:
        partial = await client.call_tool(
            "batch_decide",
            {
                "shared_context": "state",
                "items": items,
                "failure_policy": "partial",
            },
        )

    assert partial.is_error is False
    assert partial.structured_content is not None
    body = partial.structured_content
    assert [item["status"] for item in body["items"]] == ["ok", "error"]
    assert body["items"][1]["error"]["code"] == "token_budget_exceeded"
    assert body["metrics"]["successful_count"] == 1
    assert body["metrics"]["failed_count"] == 1
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_batch_fail_fast_preflights_all_items_before_inference() -> None:
    agent = FakeAgent(max_length=40)
    items = [
        {"id": "good", "question": "Relevant?"},
        {"id": "too-large", "question": " ".join(["long"] * 80)},
    ]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide", {"shared_context": "state", "items": items}
        )
    assert result.is_error is True
    assert agent.predict_calls == 0


@pytest.mark.asyncio
async def test_batch_partial_policy_reports_chunk_failure_and_continues() -> None:
    agent = FakeAgent(max_length=256, fail_on_calls={2})
    items = [{"id": str(index), "question": "Relevant?"} for index in range(101)]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide",
            {
                "shared_context": "state",
                "items": items,
                "failure_policy": "partial",
            },
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert body["metrics"]["backend_calls"] == 2
    assert body["metrics"]["successful_count"] == 100
    assert body["metrics"]["failed_count"] == 1
    assert body["items"][-1]["id"] == "100"
    assert body["items"][-1]["error"]["code"] == "backend_failure"


@pytest.mark.asyncio
async def test_batch_fail_fast_reports_chunk_failure() -> None:
    agent = FakeAgent(max_length=256, fail_on_calls={2})
    items = [{"id": str(index), "question": "Relevant?"} for index in range(101)]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide", {"shared_context": "state", "items": items}
        )

    assert result.is_error is True
    assert "chunk 1 failed" in result.content[0].text
    assert "100" in result.content[0].text
    assert agent.predict_calls == 2


@pytest.mark.asyncio
async def test_batch_rejects_empty_duplicate_and_ambiguous_contexts() -> None:
    agent = FakeAgent(max_length=256)
    async with Client(server_for(agent), mode="legacy") as client:
        empty = await client.call_tool("batch_decide", {"items": []})
        duplicate = await client.call_tool(
            "batch_decide",
            {
                "shared_context": "state",
                "items": [
                    {"id": "same", "question": "One?"},
                    {"id": "same", "question": "Two?"},
                ],
            },
        )
        ambiguous = await client.call_tool(
            "batch_decide",
            {
                "shared_context": "state",
                "items": [{"id": "x", "context": "other", "question": "One?"}],
            },
        )
    assert empty.is_error is True
    assert duplicate.is_error is True
    assert ambiguous.is_error is True
    assert agent.predict_calls == 0


@pytest.mark.asyncio
async def test_compact_batch_response_omits_probability_maps() -> None:
    agent = FakeAgent(max_length=256)
    items = [{"id": str(index), "question": "Relevant?"} for index in range(100)]
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "batch_decide", {"shared_context": "state", "items": items}
        )

    assert result.structured_content is not None
    rendered = json.dumps(result.structured_content, separators=(",", ":"))
    assert '"probabilities"' not in rendered
    assert rendered.count('"backend"') == 1
    assert rendered.count('"model"') == 1
    assert len(rendered.encode("utf-8")) < 20_000
    assert "details" not in result.structured_content["items"][0]
    assert "inference_latency_ms" not in result.structured_content["items"][0]
