from __future__ import annotations

import json
from typing import Any

from mcp import Client
import pytest

from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import Settings
from laya_mcp.server import create_server
from tests.fakes import FakeAgent


def server_for(agent: FakeAgent) -> Any:
    async def factory(settings: Settings) -> LayaCoreMLBackend:
        return LayaCoreMLBackend(agent=agent, settings=settings, initialization_ms=1.0)

    return create_server(Settings(model="fake/model"), backend_factory=factory)


def candidates(count: int) -> list[dict[str, str]]:
    return [
        {"id": f"candidate-{index}", "text": f"candidate text {index}"}
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_filter_retains_relevant_and_rejects_confidently_irrelevant() -> None:
    agent = FakeAgent(
        max_length=256,
        noul_by_id={"keep": 0.9, "drop": 0.1},
        confidence_by_id={"keep": 0.95, "drop": 0.95},
    )
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {
                "criterion": "Relevant to the movement change",
                "candidates": [
                    {"id": "keep", "text": "movement controller"},
                    {"id": "drop", "text": "unrelated save system"},
                ],
            },
        )

    assert result.is_error is False
    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["selected"]] == ["keep"]
    assert body["summary"] == {
        "input_count": 2,
        "selected_count": 1,
        "rejected_count": 1,
        "uncertain_retained": 0,
        "failed_retained": 0,
        "oversized_retained": 0,
    }
    assert body["failures"] == []
    assert body["details"] is None
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_filter_retains_uncertain_and_respects_threshold_boundary() -> None:
    agent = FakeAgent(
        max_length=256,
        noul_by_id={"boundary": 0.1, "uncertain": 0.1},
        confidence_by_id={"boundary": 0.8, "uncertain": 0.79},
    )
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {
                "criterion": "Relevant",
                "rejection_threshold": 0.8,
                "response_detail": "detailed",
                "candidates": [
                    {"id": "boundary", "text": "candidate"},
                    {"id": "uncertain", "text": "candidate"},
                ],
            },
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["selected"]] == ["uncertain"]
    assert [item["status"] for item in body["details"]] == [
        "rejected",
        "uncertain_retained",
    ]
    assert body["summary"]["uncertain_retained"] == 1


@pytest.mark.asyncio
async def test_filter_oversized_candidate_is_retained_without_truncation() -> None:
    agent = FakeAgent(max_length=40)
    oversized_text = " ".join(["long"] * 80)
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {
                "criterion": "Relevant",
                "candidates": [
                    {"id": "large", "text": oversized_text},
                    {"id": "small", "text": "short"},
                ],
            },
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["selected"]] == ["large", "small"]
    assert body["summary"]["oversized_retained"] == 1
    assert len(body["failures"]) == 1
    assert body["failures"][0]["id"] == "large"
    assert body["failures"][0]["code"] == "token_budget_exceeded"
    assert "not truncated" in body["failures"][0]["message"]
    assert "long long" in oversized_text
    assert agent.predict_calls == 1


@pytest.mark.asyncio
async def test_filter_backend_failure_is_visible_and_retained() -> None:
    agent = FakeAgent(max_length=256, fail_on_calls={1})
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {"criterion": "Relevant", "candidates": candidates(3)},
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["selected"]] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
    ]
    assert body["summary"]["failed_retained"] == 3
    assert [failure["id"] for failure in body["failures"]] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
    ]
    assert all(failure["code"] == "backend_failure" for failure in body["failures"])


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 10, 100])
async def test_filter_preserves_order_and_aggregate_usage(count: int) -> None:
    agent = FakeAgent(max_length=256)
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {"criterion": "Relevant", "candidates": candidates(count)},
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert [item["id"] for item in body["selected"]] == [
        f"candidate-{index}" for index in range(count)
    ]
    assert body["metrics"]["candidate_count"] == count
    assert body["metrics"]["model_evaluations"] == count
    assert body["metrics"]["backend_calls"] == (1 if count <= 100 else 2)
    assert body["metrics"]["total_input_tokens"] == 42 * count


@pytest.mark.asyncio
async def test_filter_partial_chunk_failure_retains_only_failed_chunk_as_failed() -> None:
    agent = FakeAgent(max_length=256, fail_on_calls={2})
    async with Client(server_for(agent), mode="legacy") as client:
        result = await client.call_tool(
            "filter",
            {"criterion": "Relevant", "candidates": candidates(101)},
        )

    assert result.structured_content is not None
    body = result.structured_content
    assert body["metrics"]["backend_calls"] == 2
    assert body["metrics"]["model_evaluations"] == 100
    assert body["summary"]["failed_retained"] == 1
    assert body["failures"][-1]["id"] == "candidate-100"


@pytest.mark.asyncio
async def test_filter_detailed_mode_contains_diagnostics_but_compact_does_not() -> None:
    agent = FakeAgent(
        max_length=256,
        noul_by_id={"drop": 0.1},
        confidence_by_id={"drop": 0.95},
    )
    items = {
        "criterion": "Relevant",
        "candidates": [
            {"id": "keep", "text": "keep text"},
            {"id": "drop", "text": "secret rejected text"},
        ],
    }
    async with Client(server_for(agent), mode="legacy") as client:
        compact = await client.call_tool("filter", items)
        detailed = await client.call_tool(
            "filter", {**items, "response_detail": "detailed"}
        )

    assert compact.structured_content is not None
    assert detailed.structured_content is not None
    compact_body = compact.structured_content
    detailed_body = detailed.structured_content
    compact_text = json.dumps(compact_body, separators=(",", ":"))
    assert "secret rejected text" not in compact_text
    assert compact_body["details"] is None
    assert [detail["id"] for detail in detailed_body["details"]] == ["keep", "drop"]
    assert detailed_body["details"][1]["status"] == "rejected"
    assert detailed_body["details"][1]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_filter_rejects_empty_and_duplicate_candidates() -> None:
    agent = FakeAgent(max_length=256)
    async with Client(server_for(agent), mode="legacy") as client:
        empty = await client.call_tool(
            "filter", {"criterion": "Relevant", "candidates": []}
        )
        duplicate = await client.call_tool(
            "filter",
            {
                "criterion": "Relevant",
                "candidates": [
                    {"id": "same", "text": "one"},
                    {"id": "same", "text": "two"},
                ],
            },
        )

    assert empty.is_error is True
    assert duplicate.is_error is True
    assert agent.predict_calls == 0
