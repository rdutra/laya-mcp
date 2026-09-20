"""Backend boundary kept independent from MCP transport."""

from __future__ import annotations

from typing import Protocol

from laya_mcp.schemas import DecisionRequest, DecisionResult, ServerInfo


class InferenceBackend(Protocol):
    async def classify(self, request: DecisionRequest) -> DecisionResult: ...

    def info(self) -> ServerInfo: ...

