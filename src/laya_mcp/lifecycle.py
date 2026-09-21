"""Lazy, process-resident backend lifecycle for the MCP transport."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import platform
from time import perf_counter

from laya_mcp import __version__
from laya_mcp.backend.base import InferenceBackend
from laya_mcp.config import Settings
from laya_mcp.model_catalog import metadata_for
from laya_mcp.schemas import (
    BackendBatchResult,
    DecisionRequest,
    DecisionResult,
    ModelCapabilities,
    RuntimeMetrics,
    ServerInfo,
)


class BackendInitializationError(RuntimeError):
    """A model could not be initialized for an inference request."""


BackendFactory = Callable[[Settings], Awaitable[InferenceBackend]]


def _backend_version() -> str:
    try:
        return package_version("laya-coreml")
    except PackageNotFoundError:
        return "unknown"


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or exc.__class__.__name__
    return message[:256]


@dataclass(slots=True)
class _LifecycleSnapshot:
    state: str = "unloaded"
    initialization_attempts: int = 0
    initialization_count: int = 0
    initialization_ms: float = 0.0
    last_initialization_error: str | None = None


class LazyBackendManager:
    """Load one backend on first use and keep it resident thereafter."""

    def __init__(self, settings: Settings, factory: BackendFactory) -> None:
        self.settings = settings
        self._factory = factory
        self._backend: InferenceBackend | None = None
        self._lock = asyncio.Lock()
        self._snapshot = _LifecycleSnapshot()

    @property
    def loaded_backend(self) -> InferenceBackend | None:
        """Return the loaded backend without triggering initialization."""

        return self._backend

    async def ensure_loaded(self) -> InferenceBackend:
        backend = self._backend
        if backend is not None:
            return backend

        async with self._lock:
            backend = self._backend
            if backend is not None:
                return backend

            self._snapshot.state = "loading"
            self._snapshot.initialization_attempts += 1
            started = perf_counter()
            try:
                backend = await self._factory(self.settings)
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000
                self._snapshot.state = "failed"
                self._snapshot.initialization_ms = round(elapsed_ms, 3)
                self._snapshot.last_initialization_error = _safe_error(exc)
                raise BackendInitializationError(
                    f"unable to initialize model {self.settings.model!r}: "
                    f"{self._snapshot.last_initialization_error}"
                ) from exc

            self._backend = backend
            self._snapshot.state = "ready"
            self._snapshot.initialization_count += 1
            self._snapshot.initialization_ms = round((perf_counter() - started) * 1000, 3)
            self._snapshot.last_initialization_error = None
            return backend

    def info(self) -> ServerInfo:
        """Return lifecycle information without loading an unloaded model."""

        backend = self._backend
        if backend is not None:
            info = backend.info()
            runtime = info.metrics.model_copy(
                update={
                    "initialization_ms": self._snapshot.initialization_ms,
                    "initialization_attempts": self._snapshot.initialization_attempts,
                    "initialization_count": self._snapshot.initialization_count,
                    "last_initialization_error": self._snapshot.last_initialization_error,
                }
            )
            return info.model_copy(
                update={
                    "initialization_state": self._snapshot.state,
                    "metrics": runtime,
                }
            )

        metadata = metadata_for(self.settings.model)
        capabilities: ModelCapabilities | None = (
            metadata.capabilities.model_copy() if metadata is not None else None
        )
        return ServerInfo(
            version=__version__,
            backend=metadata.backend if metadata is not None else "laya-coreml",
            backend_version=_backend_version(),
            model=self.settings.model,
            initialization_state=self._snapshot.state,
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python_version=platform.python_version(),
            compute_units=(
                str(self.settings.compute_units)
                if self.settings.compute_units is not None
                else (metadata.compute_units if metadata is not None else "unknown")
            ),
            capabilities=capabilities,
            metrics=RuntimeMetrics(
                initialization_ms=self._snapshot.initialization_ms,
                initialization_attempts=self._snapshot.initialization_attempts,
                initialization_count=self._snapshot.initialization_count,
                last_initialization_error=self._snapshot.last_initialization_error,
                inference_count=0,
                inference_error_count=0,
                last_inference_ms=None,
            ),
        )


class LazyInferenceBackend:
    """InferenceBackend facade that preserves the existing service boundary."""

    def __init__(self, manager: LazyBackendManager) -> None:
        self._manager = manager

    async def ensure_loaded(self) -> InferenceBackend:
        return await self._manager.ensure_loaded()

    def validate(self, request: DecisionRequest) -> int:
        backend = self._manager.loaded_backend
        if backend is None:
            raise RuntimeError("backend is not loaded; await ensure_loaded before validation")
        return backend.validate(request)

    async def classify(self, request: DecisionRequest) -> DecisionResult:
        backend = await self._manager.ensure_loaded()
        return await backend.classify(request)

    async def classify_many(
        self,
        context: str,
        requests: Sequence[tuple[str, DecisionRequest]],
    ) -> BackendBatchResult:
        backend = await self._manager.ensure_loaded()
        return await backend.classify_many(context, requests)

    def info(self) -> ServerInfo:
        return self._manager.info()

