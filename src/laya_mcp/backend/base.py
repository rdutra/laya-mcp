"""Backend boundary kept independent from MCP transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from laya_mcp.schemas import BackendBatchResult, DecisionRequest, DecisionResult, ServerInfo


class InferenceBackend(Protocol):
    def validate(self, request: DecisionRequest) -> int: ...

    async def classify(self, request: DecisionRequest) -> DecisionResult: ...

    async def classify_many(
        self,
        context: str,
        requests: Sequence[tuple[str, DecisionRequest]],
    ) -> BackendBatchResult: ...

    def info(self) -> ServerInfo: ...
