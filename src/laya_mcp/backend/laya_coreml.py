"""Persistent Laya-CoreML backend."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from importlib.metadata import version as package_version
import json
import logging
import platform
from time import perf_counter
from typing import Any

from laya_mcp import __version__
from laya_mcp.config import Settings
from laya_mcp.metrics import InferenceMetrics
from laya_mcp.schemas import (
    DecisionRequest,
    DecisionResult,
    DecisionType,
    ModelCapabilities,
    RuntimeMetrics,
    ServerInfo,
    TokenUsage,
)

logger = logging.getLogger(__name__)
_QUESTION_ID = "decision"


class InputCapacityError(ValueError):
    """Raised instead of allowing a backend to truncate request content."""


def _default_loader(model: str, **kwargs: Any) -> Any:
    import laya_coreml as laya  # type: ignore[import-untyped]

    return laya.load(model, **kwargs)


def _serialize_state(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


class LayaCoreMLBackend:
    """One resident Laya model, shared and serialized for the process lifetime."""

    name = "laya-coreml"

    def __init__(self, *, agent: Any, settings: Settings, initialization_ms: float) -> None:
        self._agent = agent
        self._settings = settings
        self._inference_lock = asyncio.Lock()
        self._metrics = InferenceMetrics(initialization_ms)
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
        """Mirror Laya 0.1 prompt budgets and reject every lossy path."""
        tokenizer = self._agent.tok
        mask_token = tokenizer.mask_token
        head_limit = int(self._agent.cfg.get("head_max_len", 192))
        max_total = min(
            int(self._agent.cfg.get("max_len", 512)),
            int(self._agent.shape["max_length"]),
        )

        head_text = (
            f"{request.decision_type.value} question: "
            f"{request.question.replace(mask_token, ' ')}"
        )
        head_ids = self._token_ids(head_text)
        option_ids = [
            [tokenizer.mask_token_id]
            + self._token_ids(" " + option.replace(mask_token, " "))
            for option in self._render_options(request)
        ]

        if any(len(ids) > 49 for ids in option_ids):
            raise InputCapacityError(
                "an option exceeds Laya's 48-token option budget; input was not sent"
            )

        option_budget = head_limit - sum(len(ids) for ids in option_ids)
        if option_budget < 16:
            per_option = max(4, (head_limit - 16) // max(1, len(option_ids)))
            if any(len(ids) > per_option for ids in option_ids):
                raise InputCapacityError(
                    "the options exceed Laya's shared question budget; input was not sent"
                )
            option_budget = head_limit - sum(len(ids) for ids in option_ids)

        head_capacity = max(8, option_budget)
        if len(head_ids) > head_capacity:
            raise InputCapacityError(
                "the question exceeds Laya's instruction budget; input was not sent"
            )

        prefix_tokens = 1 + len(head_ids) + 1 + sum(map(len, option_ids)) + 1
        state_text = _serialize_state(request.context).replace(mask_token, " ")
        state_tokens = len(self._token_ids(state_text))
        required = prefix_tokens + state_tokens + 1
        if required > max_total:
            raise InputCapacityError(
                f"request requires {required} tokens but this model supports {max_total}; "
                "input was not sent and was not truncated"
            )
        return required

    def _predict(self, request: DecisionRequest, question: dict[str, Any]) -> DecisionResult:
        self._validate_no_truncation(request)
        started = perf_counter()
        raw = self._agent.predict(request.context, {_QUESTION_ID: question})
        inference_ms = (perf_counter() - started) * 1000
        answer = raw["answers"][_QUESTION_ID]

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
        usage = raw.get("usage", {})
        output = DecisionResult(
            backend=self.name,
            model=self._settings.model,
            laya_model=str(raw.get("model", "unknown")),
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
            usage=TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            inference_ms=round(inference_ms, 3),
        )
        self._metrics.record_success(inference_ms)
        logger.info(
            "inference_complete",
            extra={
                "decision_type": request.decision_type.value,
                "input_tokens": output.usage.input_tokens,
                "inference_ms": output.inference_ms,
            },
        )
        return output

    async def classify(self, request: DecisionRequest) -> DecisionResult:
        try:
            question = self._question_definition(request)
            async with self._inference_lock:
                return await asyncio.to_thread(self._predict, request, question)
        except ValueError as exc:
            self._metrics.record_error()
            logger.warning(
                "inference_rejected",
                extra={"decision_type": request.decision_type.value, "reason": str(exc)},
            )
            raise
        except Exception:
            self._metrics.record_error()
            logger.exception(
                "inference_failed",
                extra={"decision_type": request.decision_type.value},
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
                decision_types=list(DecisionType),
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
