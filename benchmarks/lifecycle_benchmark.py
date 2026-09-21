"""Measure lazy MCP startup, cold first use, and warm inference locally.

This benchmark requires a compatible Apple Silicon machine and a locally usable
Laya-CoreML installation. It intentionally uses a real stdio MCP subprocess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def run(model: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["LAYA_MCP_MODEL"] = model
    environment.setdefault("PYTHONUNBUFFERED", "1")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp"],
        env=environment,
    )

    process_started = perf_counter()
    async with stdio_client(parameters) as streams:
        stdio_ready = perf_counter()
        async with ClientSession(*streams) as session:
            await session.initialize()
            await session.list_tools()
            info_before = await session.call_tool("info", {})
            discovery_finished = perf_counter()

            cold_started = perf_counter()
            await session.call_tool(
                "decide",
                {
                    "context": "player movement and acceleration",
                    "question": "Is this context relevant to the task?",
                },
            )
            cold_finished = perf_counter()

            warm_started = perf_counter()
            await session.call_tool(
                "decide",
                {
                    "context": "player movement and acceleration",
                    "question": "Is this context relevant to the task?",
                },
            )
            warm_finished = perf_counter()
            info_after = await session.call_tool("info", {})

    return {
        "model": model,
        "process_to_stdio_ready_ms": round((stdio_ready - process_started) * 1000, 3),
        "tool_discovery_and_unloaded_info_ms": round(
            (discovery_finished - process_started) * 1000, 3
        ),
        "first_inference_cold_operation_ms": round((cold_finished - cold_started) * 1000, 3),
        "subsequent_warm_operation_ms": round((warm_finished - warm_started) * 1000, 3),
        "info_before": info_before.structured_content,
        "info_after": info_after.structured_content,
    }


def main() -> None:
    arguments = _arguments()
    result = asyncio.run(run(arguments.model))
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
