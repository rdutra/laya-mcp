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
from laya_mcp.schemas import DecisionRequest, DecisionResult, DecisionType, ServerInfo

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
            "Use classify for small, bounded noul, choice, or score decisions. "
            "The server rejects requests that exceed the resident model's token limit; "
            "call info to inspect the active model and limits."
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
    async def classify(
        context: str,
        question: str,
        ctx: Context[AppContext],
        decision_type: DecisionType = DecisionType.NOUL,
        options: list[str] | None = None,
    ) -> DecisionResult:
        """Make one bounded typed decision over context using the local resident model.

        Use `noul` for a boolean proposition (no options), `choice` for one label from
        two or more options, or `score` for an ordered rubric from low to high.
        """
        try:
            request = DecisionRequest(
                context=context,
                question=question,
                decision_type=decision_type,
                options=options,
            )
            return await ctx.request_context.lifespan_context.backend.classify(request)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return server


mcp = create_server()
