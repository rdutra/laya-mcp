"""Backend-neutral decision orchestration used by the MCP transport."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from laya_mcp.backend.base import InferenceBackend
from laya_mcp.schemas import (
    BatchDecideResponse,
    BatchDecisionInput,
    BatchItemDetailedSuccess,
    BatchItemError,
    BatchItemErrorDetail,
    BatchItemResult,
    BatchItemSuccess,
    BatchMetrics,
    DecideInput,
    DecideResponse,
    DecisionDetails,
    DecisionKind,
    DecisionRequest,
    DecisionResult,
    DecisionType,
    FailurePolicy,
    ResponseDetail,
)
from laya_mcp.token_budget import ChunkExecutionError, InputCapacityError

_MAX_QUESTIONS_PER_BACKEND_CALL = 100


@dataclass(frozen=True, slots=True)
class _PreparedDecision:
    item: BatchDecisionInput
    request: DecisionRequest
    threshold: float | None


def _internal_type(kind: DecisionKind) -> DecisionType:
    return {
        DecisionKind.BINARY: DecisionType.NOUL,
        DecisionKind.CHOICE: DecisionType.CHOICE,
        DecisionKind.ORDERED_SCORE: DecisionType.SCORE,
    }[kind]


def _public_type(kind: DecisionType) -> DecisionKind:
    return {
        DecisionType.NOUL: DecisionKind.BINARY,
        DecisionType.CHOICE: DecisionKind.CHOICE,
        DecisionType.SCORE: DecisionKind.ORDERED_SCORE,
    }[kind]


def _request(value: DecideInput | BatchDecisionInput, context: str) -> DecisionRequest:
    return DecisionRequest(
        context=context,
        question=value.question,
        decision_type=_internal_type(value.decision.kind),
        options=value.decision.options,
    )


def _details(result: DecisionResult) -> DecisionDetails | None:
    if result.decision_type is DecisionType.NOUL and result.noul_probability is not None:
        return DecisionDetails(
            probabilities={
                "false": round(1.0 - result.noul_probability, 10),
                "true": result.noul_probability,
            }
        )
    if result.probabilities is None:
        return None
    if result.decision_type is DecisionType.SCORE and result.legend:
        return DecisionDetails(
            probabilities={
                result.legend.get(key, key): probability
                for key, probability in result.probabilities.items()
            }
        )
    return DecisionDetails(probabilities=result.probabilities)


def _needs_escalation(confidence: float | None, threshold: float | None) -> bool:
    if threshold is None:
        return False
    if confidence is None:
        return True
    return confidence < threshold


class DecisionService:
    """Stable decision semantics over an interchangeable local backend."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    async def decide(self, value: DecideInput) -> DecideResponse:
        request = _request(value, value.context)
        result = await self._backend.classify(request)
        return DecideResponse(
            request_id=value.request_id,
            result=result.result,
            decision_type=_public_type(result.decision_type),
            confidence=result.confidence,
            confidence_threshold=value.confidence_threshold,
            needs_escalation=_needs_escalation(result.confidence, value.confidence_threshold),
            details=_details(result),
            backend=result.backend,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            inference_latency_ms=result.inference_ms,
        )

    def _prepare(
        self,
        items: Sequence[BatchDecisionInput],
        shared_context: str | None,
        default_threshold: float | None,
    ) -> list[_PreparedDecision]:
        if not items:
            raise ValueError("items must not be empty")
        ids = [item.id for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("item IDs must be unique")
        prepared: list[_PreparedDecision] = []
        for item in items:
            if shared_context is None and item.context is None:
                raise ValueError(f"item {item.id!r} requires context when shared_context is omitted")
            if shared_context is not None and item.context is not None:
                raise ValueError(
                    f"item {item.id!r} must omit context when shared_context is supplied"
                )
            context = shared_context if shared_context is not None else item.context
            assert context is not None
            prepared.append(
                _PreparedDecision(
                    item=item,
                    request=_request(item, context),
                    threshold=(
                        item.confidence_threshold
                        if item.confidence_threshold is not None
                        else default_threshold
                    ),
                )
            )
        return prepared

    @staticmethod
    def _chunks(
        prepared: Sequence[_PreparedDecision], shared_context: str | None
    ) -> list[list[_PreparedDecision]]:
        if shared_context is None:
            return [[item] for item in prepared]
        return [
            list(prepared[start : start + _MAX_QUESTIONS_PER_BACKEND_CALL])
            for start in range(0, len(prepared), _MAX_QUESTIONS_PER_BACKEND_CALL)
        ]

    @staticmethod
    def _error(item_id: str, exc: Exception, *, code: str, retryable: bool) -> BatchItemError:
        return BatchItemError(
            id=item_id,
            error=BatchItemErrorDetail(code=code, message=str(exc), retryable=retryable),
        )

    async def batch_decide(
        self,
        items: Sequence[BatchDecisionInput],
        *,
        shared_context: str | None,
        confidence_threshold: float | None,
        failure_policy: FailurePolicy,
        response_detail: ResponseDetail,
    ) -> BatchDecideResponse:
        started = perf_counter()
        prepared = self._prepare(items, shared_context, confidence_threshold)
        results_by_id: dict[str, BatchItemResult] = {}
        valid: list[_PreparedDecision] = []

        for item in prepared:
            try:
                self._backend.validate(item.request)
                valid.append(item)
            except ValueError as exc:
                if failure_policy is FailurePolicy.FAIL_FAST:
                    raise
                code = (
                    "token_budget_exceeded"
                    if isinstance(exc, InputCapacityError)
                    else "invalid_decision"
                )
                results_by_id[item.item.id] = self._error(
                    item.item.id, exc, code=code, retryable=False
                )

        backend_calls = 0
        model_evaluations = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_inference_ms = 0.0
        for chunk_index, chunk in enumerate(self._chunks(valid, shared_context)):
            context = chunk[0].request.context
            backend_calls += 1
            try:
                backend_result = await self._backend.classify_many(
                    context,
                    [(item.item.id, item.request) for item in chunk],
                )
                returned_ids = [item.id for item in backend_result.results]
                expected_ids = [item.item.id for item in chunk]
                if returned_ids != expected_ids:
                    raise RuntimeError(
                        "backend returned IDs out of order or omitted decisions"
                    )
            except Exception as exc:
                if failure_policy is FailurePolicy.FAIL_FAST:
                    raise ChunkExecutionError(
                        chunk_index, tuple(item.item.id for item in chunk), exc
                    ) from exc
                for item in chunk:
                    results_by_id[item.item.id] = self._error(
                        item.item.id, exc, code="backend_failure", retryable=True
                    )
                continue

            model_evaluations += backend_result.model_evaluations
            total_input_tokens += backend_result.usage.input_tokens
            total_output_tokens += backend_result.usage.output_tokens
            total_inference_ms += backend_result.inference_ms
            thresholds = {item.item.id: item.threshold for item in chunk}
            for identified in backend_result.results:
                result = identified.result
                threshold = thresholds[identified.id]
                if response_detail is ResponseDetail.DETAILED:
                    results_by_id[identified.id] = BatchItemDetailedSuccess(
                        id=identified.id,
                        result=result.result,
                        decision_type=_public_type(result.decision_type),
                        confidence=result.confidence,
                        needs_escalation=_needs_escalation(result.confidence, threshold),
                        input_tokens=result.usage.input_tokens,
                        output_tokens=result.usage.output_tokens,
                        inference_latency_ms=result.inference_ms,
                        details=_details(result),
                    )
                else:
                    results_by_id[identified.id] = BatchItemSuccess(
                        id=identified.id,
                        result=result.result,
                        decision_type=_public_type(result.decision_type),
                        confidence=result.confidence,
                        needs_escalation=_needs_escalation(result.confidence, threshold),
                        input_tokens=result.usage.input_tokens,
                        output_tokens=result.usage.output_tokens,
                    )

        ordered = [results_by_id[item.item.id] for item in prepared]
        successful = [item for item in ordered if isinstance(item, BatchItemSuccess)]
        failed_count = len(ordered) - len(successful)
        info = self._backend.info()
        return BatchDecideResponse(
            backend=info.backend,
            model=info.model,
            failure_policy=failure_policy,
            response_detail=response_detail,
            items=ordered,
            metrics=BatchMetrics(
                decision_count=len(ordered),
                successful_count=len(successful),
                failed_count=failed_count,
                escalation_count=sum(item.needs_escalation for item in successful),
                backend_calls=backend_calls,
                model_evaluations=model_evaluations,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_inference_latency_ms=round(total_inference_ms, 3),
                total_operation_latency_ms=round((perf_counter() - started) * 1000, 3),
            ),
        )
