"""Exercise every public decision kind through a real MCP subprocess."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def run(model: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["LAYA_MCP_MODEL"] = model
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp"],
        env=environment,
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            responses = {
                "info_before": await session.call_tool("info", {}),
                "binary": await session.call_tool(
                    "decide",
                    {
                        "context": "The integer 2 is even.",
                        "question": "Is 2 even?",
                        "decision": {"kind": "binary"},
                    },
                ),
                "choice": await session.call_tool(
                    "decide",
                    {
                        "context": "The incident is an authentication timeout.",
                        "question": "Which subsystem should be inspected first?",
                        "decision": {"kind": "choice", "options": ["auth", "audio"]},
                    },
                ),
                "ordered_score": await session.call_tool(
                    "decide",
                    {
                        "context": "The issue is reproducible but has a workaround.",
                        "question": "How severe is the issue?",
                        "decision": {
                            "kind": "ordered_score",
                            "options": ["low", "medium", "high"],
                        },
                    },
                ),
                "info_after": await session.call_tool("info", {}),
            }
    return {
        key: value.structured_content
        for key, value in responses.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(asyncio.run(run(args.model)), indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

