#!/usr/bin/env python3
"""Measure filter scale and MCP response-size reduction on a real local model."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import DEFAULT_MODEL, Settings
from laya_mcp.schemas import (
    BatchDecisionInput,
    FailurePolicy,
    FilterCandidate,
    FilterInput,
    ResponseDetail,
)
from laya_mcp.service import DecisionService, FilterService


COUNTS = (10, 50, 100, 500)


def candidate_items(count: int) -> list[FilterCandidate]:
    templates = (
        "player movement controller handles acceleration and jumping",
        "player animation state transitions follow locomotion",
        "enemy navigation computes paths around obstacles",
        "save system serializes inventory and progress",
        "audio mixer loads music and sound effects",
    )
    return [
        FilterCandidate(id=f"candidate-{index}", text=templates[index % len(templates)])
        for index in range(count)
    ]


def serialized_bytes(value: object) -> int:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


async def measure_filter(
    service: FilterService,
    criterion: str,
    items: list[FilterCandidate],
    detail: ResponseDetail,
) -> dict[str, Any]:
    request = FilterInput(criterion=criterion, candidates=items, response_detail=detail)
    started = perf_counter()
    response = await service.filter(request)
    operation_ms = (perf_counter() - started) * 1000
    metrics = response.metrics.model_copy(update={"total_operation_latency_ms": round(operation_ms, 3)})
    response = response.model_copy(update={"metrics": metrics})
    return {
        "strategy": f"filter-{detail.value}",
        "candidate_count": len(items),
        "selected_count": response.summary.selected_count,
        "rejected_count": response.summary.rejected_count,
        "uncertain_retained": response.summary.uncertain_retained,
        "failed_retained": response.summary.failed_retained,
        "model_evaluations": metrics.model_evaluations,
        "backend_calls": metrics.backend_calls,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "total_inference_latency_ms": metrics.total_inference_latency_ms,
        "total_operation_latency_ms": operation_ms,
        "input_payload_bytes": metrics.input_payload_bytes,
        "output_payload_bytes": serialized_bytes(response),
        "output_reduction_percent": metrics.output_reduction_percent,
    }


async def measure_batch(
    service: DecisionService,
    criterion: str,
    items: list[FilterCandidate],
) -> dict[str, Any]:
    batch_items = [
        BatchDecisionInput(
            id=item.id,
            question=f"Is this candidate relevant? Candidate: {item.text}",
        )
        for item in items
    ]
    request = {
        "shared_context": f"Criterion: {criterion}",
        "items": [item.model_dump(mode="json") for item in batch_items],
        "response_detail": ResponseDetail.COMPACT.value,
    }
    started = perf_counter()
    response = await service.batch_decide(
        batch_items,
        shared_context=f"Criterion: {criterion}",
        confidence_threshold=0.8,
        failure_policy=FailurePolicy.PARTIAL,
        response_detail=ResponseDetail.COMPACT,
    )
    operation_ms = (perf_counter() - started) * 1000
    return {
        "strategy": "batch_decide-compact",
        "candidate_count": len(items),
        "selected_count": len(items),
        "rejected_count": 0,
        "uncertain_retained": 0,
        "failed_retained": response.metrics.failed_count,
        "model_evaluations": response.metrics.model_evaluations,
        "backend_calls": response.metrics.backend_calls,
        "total_input_tokens": response.metrics.total_input_tokens,
        "total_output_tokens": response.metrics.total_output_tokens,
        "total_inference_latency_ms": response.metrics.total_inference_latency_ms,
        "total_operation_latency_ms": operation_ms,
        "input_payload_bytes": serialized_bytes(request),
        "output_payload_bytes": serialized_bytes(response),
        "output_reduction_percent": round(
            (1 - serialized_bytes(response) / serialized_bytes(request)) * 100, 2
        ),
    }


async def run(model: str, local_files_only: bool, counts: tuple[int, ...]) -> dict[str, Any]:
    settings = Settings(model=model, local_files_only=local_files_only)
    started = perf_counter()
    backend = await LayaCoreMLBackend.create(settings)
    initialization_ms = (perf_counter() - started) * 1000
    decisions = DecisionService(backend)
    filters = FilterService(backend)
    criterion = "Relevant to fixing player acceleration and jumping"
    warmup = candidate_items(1)
    await filters.filter(FilterInput(criterion=criterion, candidates=warmup))

    rows: list[dict[str, Any]] = []
    for count in counts:
        items = candidate_items(count)
        rows.append({"count": count, **await measure_batch(decisions, criterion, items)})
        rows.append({"count": count, **await measure_filter(filters, criterion, items, ResponseDetail.COMPACT)})
        if count == 100:
            rows.append({"count": count, **await measure_filter(filters, criterion, items, ResponseDetail.DETAILED)})
    capabilities = backend.info().capabilities
    assert capabilities is not None
    return {
        "model": model,
        "initialization_ms": round(initialization_ms, 3),
        "max_total_tokens": capabilities.max_total_tokens,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(
        run(args.model, args.local_files_only, tuple(int(value) for value in args.counts.split(",")))
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
