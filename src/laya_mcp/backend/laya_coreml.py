"""Persistent Laya-CoreML backend."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from importlib.metadata import version as package_version
import logging
import platform
from time import perf_counter
from typing import Any

from laya_mcp import __version__
from laya_mcp.config import Settings
from laya_mcp.metrics import InferenceMetrics
from laya_mcp.token_budget import (
    DecisionItem,
    InputCapacityError,
    TokenAwarePlanner,
    TokenBudgetEstimator,
    serialize_state,
)
from laya_mcp.schemas import (
    BackendBatchResult,
    DecisionRequest,
    DecisionResult,
    DecisionKind,
    DecisionType,
    IdentifiedDecisionResult,
    ModelCapabilities,
    RuntimeMetrics,
    ServerInfo,
    TokenUsage,
)

logger = logging.getLogger(__name__)
_QUESTION_ID = "decision"

__all__ = ["InputCapacityError", "LayaCoreMLBackend"]


def _default_loader(model: str, **kwargs: Any) -> Any:
    import laya_coreml as laya  # type: ignore[import-untyped]

    return laya.load(model, **kwargs)


class LayaCoreMLBackend:
    """One resident Laya model, shared and serialized for the process lifetime."""

    name = "laya-coreml"

    def __init__(self, *, agent: Any, settings: Settings, initialization_ms: float) -> None:
        self._agent = agent
        self._settings = settings
        self._inference_lock = asyncio.Lock()
        self._metrics = InferenceMetrics(initialization_ms)
        self._budget = TokenBudgetEstimator(agent)
        self._state = "ready"

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        loader: Callable[..., Any] = _default_loader,
    ) -> LayaCoreMLBackend:
        kwargs: dict[str, Any] = {
            "revision": settings.revision,
            "local_files_only": settings.local_files_only,
        }
        if settings.compute_units is not None:
            kwargs["compute_units"] = settings.compute_units

        logger.info("loading_model", extra={"model": settings.model})
        started = perf_counter()
        agent = await asyncio.to_thread(loader, settings.model, **kwargs)
        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "model_ready",
            extra={"model": settings.model, "initialization_ms": round(elapsed_ms, 3)},
        )
        return cls(agent=agent, settings=settings, initialization_ms=elapsed_ms)

    @property
    def agent(self) -> Any:
        """Exposed read-only for diagnostics and tests; never replace after startup."""
        return self._agent

    def _question_definition(self, request: DecisionRequest) -> dict[str, Any]:
        if request.decision_type is DecisionType.NOUL:
            if request.options is not None:
                raise ValueError("options must be omitted for a noul decision")
            return {"type": "noul", "instructions": request.question}

        if request.options is None or len(request.options) < 2:
            raise ValueError("choice and score decisions require at least two options")
        if request.decision_type is DecisionType.CHOICE and len(set(request.options)) != len(
            request.options
        ):
            raise ValueError("choice options must be unique")
        max_options = int(self._agent.shape["max_options"])
        if len(request.options) > max_options:
            raise ValueError(f"this model supports at most {max_options} options")
        return {
            "type": request.decision_type.value,
            "instructions": request.question,
            "criteria": request.options,
        }

    def _render_options(self, request: DecisionRequest) -> list[str]:
        if request.decision_type is DecisionType.NOUL:
            return [
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ]
        assert request.options is not None
        if request.decision_type is DecisionType.CHOICE:
            return request.options
        return [f"level {index}: {label}" for index, label in enumerate(request.options)]

    def _token_ids(self, text: str) -> list[int]:
        return list(self._agent.tok(text, add_special_tokens=False)["input_ids"])

    def _validate_no_truncation(self, request: DecisionRequest) -> int:
        """Reject every lossy path before calling Laya."""
        item = DecisionItem.from_request(_QUESTION_ID, request)
        return self._budget.estimate(serialize_state(request.context), item).total_tokens

    def validate(self, request: DecisionRequest) -> int:
        """Validate decision semantics and return the exact input token count."""
        self._question_definition(request)
        return self._validate_no_truncation(request)

    def _map_answer(
        self,
        request: DecisionRequest,
        answer: dict[str, Any],
        *,
        laya_model: str,
        input_tokens: int,
        output_tokens: int,
        inference_ms: float | None,
    ) -> DecisionResult:
        if request.decision_type is DecisionType.NOUL:
            noul_probability = float(answer["noul"])
            result: bool | str | float = noul_probability >= 0.5
        elif request.decision_type is DecisionType.CHOICE:
            noul_probability = None
            result = str(answer["choice"])
        else:
            noul_probability = None
            result = float(answer["score"])

        probabilities = answer.get("probabilities")
        return DecisionResult(
            backend=self.name,
            model=self._settings.model,
            laya_model=laya_model,
            decision_type=request.decision_type,
            result=result,
            confidence=float(answer["confidence"]),
            probabilities=(
                {str(key): float(value) for key, value in probabilities.items()}
                if probabilities is not None
                else None
            ),
            noul_probability=noul_probability,
            action_probability=(
                float(answer["action"]["act_probability"])
                if answer.get("action", {}).get("act_probability") is not None
                else None
            ),
            legend=(
                {str(key): str(value) for key, value in answer["legend"].items()}
                if "legend" in answer
                else None
            ),
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            inference_ms=round(inference_ms, 3) if inference_ms is not None else None,
        )

    def _predict_many(
        self,
        context: str,
        prepared: Sequence[tuple[str, DecisionRequest, dict[str, Any], int]],
    ) -> BackendBatchResult:
        started = perf_counter()
        raw = self._agent.predict(
            context,
            {item_id: question for item_id, _, question, _ in prepared},
        )
        inference_ms = (perf_counter() - started) * 1000
        usage = raw.get("usage", {})
        total_input_tokens = int(usage.get("input_tokens", 0))
        total_output_tokens = int(usage.get("output_tokens", 0))
        if total_output_tokens and len(prepared) > 1:
            raise RuntimeError("backend returned output tokens that cannot be attributed per item")
        per_item_output_tokens = 0 if total_output_tokens == 0 else total_output_tokens
        results = [
            IdentifiedDecisionResult(
                id=item_id,
                result=self._map_answer(
                    request,
                    raw["answers"][item_id],
                    laya_model=str(raw.get("model", "unknown")),
                    input_tokens=(total_input_tokens if len(prepared) == 1 else input_tokens),
                    output_tokens=per_item_output_tokens if len(prepared) == 1 else 0,
                    inference_ms=inference_ms if len(prepared) == 1 else None,
                ),
            )
            for item_id, request, _, input_tokens in prepared
        ]
        self._metrics.record_success(inference_ms, count=len(prepared))
        logger.info(
            "inference_complete",
            extra={
                "decision_count": len(prepared),
                "input_tokens": total_input_tokens,
                "inference_ms": round(inference_ms, 3),
            },
        )
        return BackendBatchResult(
            results=results,
            usage=TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
            inference_ms=round(inference_ms, 3),
            model_evaluations=len(prepared),
        )

    async def classify(self, request: DecisionRequest) -> DecisionResult:
        batch = await self.classify_many(request.context, ((_QUESTION_ID, request),))
        return batch.results[0].result

    async def classify_many(
        self,
        context: str,
        requests: Sequence[tuple[str, DecisionRequest]],
    ) -> BackendBatchResult:
        if not requests:
            raise ValueError("requests must not be empty")
        try:
            ids = [item_id for item_id, _ in requests]
            if len(set(ids)) != len(ids):
                raise ValueError("decision IDs must be unique within a backend call")
            prepared: list[tuple[str, DecisionRequest, dict[str, Any], int]] = []
            planning_items: list[DecisionItem] = []
            for item_id, request in requests:
                if request.context != context:
                    raise ValueError("all decisions in one backend call must share context")
                question = self._question_definition(request)
                planning_item = DecisionItem.from_request(item_id, request)
                planning_items.append(planning_item)
                prepared.append((item_id, request, question, 0))
            planned = TokenAwarePlanner(self._budget).plan(context, planning_items)
            estimates = {
                estimate.item_id: estimate.total_tokens
                for chunk in planned
                for estimate in chunk.estimates
            }
            prepared = [
                (item_id, request, question, estimates[item_id])
                for item_id, request, question, _ in prepared
            ]
            async with self._inference_lock:
                return await asyncio.to_thread(self._predict_many, context, prepared)
        except ValueError as exc:
            self._metrics.record_error()
            logger.warning(
                "inference_rejected",
                extra={"decision_count": len(requests), "reason": str(exc)},
            )
            raise
        except Exception:
            self._metrics.record_error()
            logger.exception(
                "inference_failed",
                extra={"decision_count": len(requests)},
            )
            raise

    def info(self) -> ServerInfo:
        shape = self._agent.shape
        metrics = self._metrics.snapshot()
        return ServerInfo(
            version=__version__,
            backend=self.name,
            backend_version=package_version("laya-coreml"),
            model=self._settings.model,
            initialization_state=self._state,
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            python_version=platform.python_version(),
            compute_units=str(getattr(self._agent, "compute_units", "unknown")),
            capabilities=ModelCapabilities(
                decision_types=list(DecisionKind),
                max_total_tokens=min(
                    int(self._agent.cfg.get("max_len", 512)),
                    int(shape["max_length"]),
                ),
                max_options=int(shape["max_options"]),
                batch_size=int(shape["batch_size"]),
                generated_output_tokens=0,
                overflow_behavior="reject_without_inference",
                local_inference=True,
            ),
            metrics=RuntimeMetrics(
                initialization_ms=metrics.initialization_ms,
                inference_count=metrics.inference_count,
                inference_error_count=metrics.inference_error_count,
                last_inference_ms=metrics.last_inference_ms,
            ),
        )
