"""Client-agnostic MCP transport and tool registration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from laya_mcp import __version__
from laya_mcp.backend.base import InferenceBackend
from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import Settings
from laya_mcp.schemas import (
    BatchDecideResponse,
    BatchDecisionInput,
    ConfidenceThreshold,
    DecideInput,
    DecideResponse,
    DecisionSpec,
    FailurePolicy,
    NonEmptyText,
    ResponseDetail,
    ServerInfo,
)
from laya_mcp.service import DecisionService

BackendFactory = Callable[[Settings], Awaitable[InferenceBackend]]


@dataclass(slots=True)
class AppContext:
    backend: InferenceBackend


def create_server(
    settings: Settings | None = None,
    *,
    backend_factory: BackendFactory | None = None,
) -> MCPServer[AppContext]:
    active_settings = settings or Settings.from_env()
    factory = backend_factory or LayaCoreMLBackend.create

    @asynccontextmanager
    async def lifespan(_: MCPServer[AppContext]) -> AsyncIterator[AppContext]:
        backend = await factory(active_settings)
        yield AppContext(backend=backend)

    server: MCPServer[AppContext] = MCPServer(
        "laya-mcp",
        version=__version__,
        description="Persistent local typed decisions powered by Laya-CoreML",
        instructions=(
            "Use decide for one bounded binary, choice, or ordered-score decision, and "
            "batch_decide for multiple decisions. The server rejects any question that "
            "exceeds the resident model's per-question token limit. Call info to inspect "
            "the active model and limits."
        ),
        lifespan=lifespan,
    )

    @server.tool(
        structured_output=True,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def info(ctx: Context[AppContext]) -> ServerInfo:
        """Report server, resident backend, model, platform, limits, and metrics."""
        return ctx.request_context.lifespan_context.backend.info()

    @server.tool(
        structured_output=True,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def decide(
        context: NonEmptyText,
        question: NonEmptyText,
        ctx: Context[AppContext],
        decision: DecisionSpec | None = None,
        confidence_threshold: ConfidenceThreshold | None = None,
        request_id: NonEmptyText | None = None,
    ) -> DecideResponse:
        """Make one bounded decision using the resident local model.

        Decision kinds are `binary`, `choice`, and `ordered_score`. The server only
        signals `needs_escalation`; the caller remains responsible for any escalation.
        """
        try:
            request = DecideInput(
                context=context,
                question=question,
                decision=decision or DecisionSpec(),
                confidence_threshold=confidence_threshold,
                request_id=request_id,
            )
            service = DecisionService(ctx.request_context.lifespan_context.backend)
            return await service.decide(request)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise ToolError(f"decision backend failure: {exc}") from exc

    @server.tool(
        structured_output=True,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def batch_decide(
        items: list[BatchDecisionInput],
        ctx: Context[AppContext],
        shared_context: NonEmptyText | None = None,
        confidence_threshold: ConfidenceThreshold | None = None,
        failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST,
        response_detail: ResponseDetail = ResponseDetail.COMPACT,
    ) -> BatchDecideResponse:
        """Make ordered bounded decisions in one MCP request.

        Supply `shared_context` and omit item contexts, or omit `shared_context` and
        give every item its own context. This is API batching, not parallel inference.
        """
        try:
            service = DecisionService(ctx.request_context.lifespan_context.backend)
            return await service.batch_decide(
                items,
                shared_context=shared_context,
                confidence_threshold=confidence_threshold,
                failure_policy=failure_policy,
                response_detail=response_detail,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise ToolError(f"batch decision backend failure: {exc}") from exc

    return server


mcp = create_server()
