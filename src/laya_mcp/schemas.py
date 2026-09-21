"""Public decision contracts and internal backend result types."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionType(str, Enum):
    """Internal Laya decision names; not exposed by the stable MCP API."""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class DecisionKind(str, Enum):
    """Backend-neutral public decision semantics."""

    BINARY = "binary"
    CHOICE = "choice"
    ORDERED_SCORE = "ordered_score"


class FailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    PARTIAL = "partial"


class ResponseDetail(str, Enum):
    COMPACT = "compact"
    DETAILED = "detailed"


NonEmptyText = Annotated[str, Field(min_length=1)]
ConfidenceThreshold = Annotated[float, Field(ge=0.0, le=1.0)]


class DecisionSpec(BaseModel):
    """Public description of the bounded result space."""

    model_config = ConfigDict(extra="forbid")

    kind: DecisionKind = DecisionKind.BINARY
    options: list[NonEmptyText] | None = None

    @model_validator(mode="after")
    def validate_options(self) -> DecisionSpec:
        if self.kind is DecisionKind.BINARY:
            if self.options is not None:
                raise ValueError("options must be omitted for a binary decision")
            return self
        if self.options is None or len(self.options) < 2:
            raise ValueError("choice and ordered_score decisions require at least two options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("decision options must be unique")
        return self


class DecideInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: NonEmptyText
    question: NonEmptyText
    decision: DecisionSpec = Field(default_factory=DecisionSpec)
    confidence_threshold: ConfidenceThreshold | None = None
    request_id: NonEmptyText | None = None


class BatchDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    question: NonEmptyText
    context: NonEmptyText | None = None
    decision: DecisionSpec = Field(default_factory=DecisionSpec)
    confidence_threshold: ConfidenceThreshold | None = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: NonEmptyText
    question: NonEmptyText
    decision_type: DecisionType = DecisionType.NOUL
    options: list[NonEmptyText] | None = None


class TokenUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class DecisionResult(BaseModel):
    backend: str
    model: str
    laya_model: str
    decision_type: DecisionType
    result: bool | str | float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    noul_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    action_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    legend: dict[str, str] | None = None
    usage: TokenUsage
    inference_ms: float | None = Field(default=None, ge=0.0)


class IdentifiedDecisionResult(BaseModel):
    id: str
    result: DecisionResult


class BackendBatchResult(BaseModel):
    results: list[IdentifiedDecisionResult]
    usage: TokenUsage
    inference_ms: float = Field(ge=0.0)
    model_evaluations: int = Field(ge=0)


class DecisionDetails(BaseModel):
    """Optional distribution details normalized away from backend-specific keys."""

    probabilities: dict[str, float]


class DecideResponse(BaseModel):
    request_id: str | None = None
    result: bool | str | float
    decision_type: DecisionKind
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_escalation: bool
    details: DecisionDetails | None = None
    backend: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    inference_latency_ms: float | None = Field(default=None, ge=0.0)


class BatchItemSuccess(BaseModel):
    id: str
    status: Literal["ok"] = "ok"
    result: bool | str | float
    decision_type: DecisionKind
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_escalation: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class BatchItemDetailedSuccess(BatchItemSuccess):
    inference_latency_ms: float | None = Field(default=None, ge=0.0)
    details: DecisionDetails | None = None


class BatchItemErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool


class BatchItemError(BaseModel):
    id: str
    status: Literal["error"] = "error"
    error: BatchItemErrorDetail


BatchItemResult = BatchItemSuccess | BatchItemDetailedSuccess | BatchItemError


class BatchMetrics(BaseModel):
    decision_count: int = Field(ge=0)
    successful_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    escalation_count: int = Field(ge=0)
    backend_calls: int = Field(ge=0)
    model_evaluations: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_inference_latency_ms: float = Field(ge=0.0)
    total_operation_latency_ms: float = Field(ge=0.0)


class BatchDecideResponse(BaseModel):
    backend: str
    model: str
    failure_policy: FailurePolicy
    response_detail: ResponseDetail
    items: list[BatchItemResult]
    metrics: BatchMetrics


class FilterCandidate(BaseModel):
    """Opaque textual candidate supplied by the caller."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    text: NonEmptyText


class FilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: NonEmptyText
    candidates: list[FilterCandidate]
    rejection_threshold: ConfidenceThreshold = 0.9
    response_detail: ResponseDetail = ResponseDetail.COMPACT

    @model_validator(mode="after")
    def validate_candidates(self) -> FilterInput:
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        ids = [candidate.id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        return self


class FilterStatus(str, Enum):
    RETAINED = "retained"
    UNCERTAIN_RETAINED = "uncertain_retained"
    REJECTED = "rejected"
    FAILED_RETAINED = "failed_retained"
    OVERSIZED_RETAINED = "oversized_retained"


class FilterFailure(BaseModel):
    id: str
    code: str
    message: str


class FilterItemDetail(BaseModel):
    id: str
    status: FilterStatus
    relevant: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    needs_escalation: bool = False
    reason: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    inference_latency_ms: float | None = Field(default=None, ge=0.0)


class FilterSummary(BaseModel):
    input_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    uncertain_retained: int = Field(ge=0)
    failed_retained: int = Field(ge=0)
    oversized_retained: int = Field(ge=0)


class FilterMetrics(BaseModel):
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    uncertain_retained: int = Field(ge=0)
    failed_retained: int = Field(ge=0)
    oversized_retained: int = Field(ge=0)
    model_evaluations: int = Field(ge=0)
    backend_calls: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_inference_latency_ms: float = Field(ge=0.0)
    total_operation_latency_ms: float = Field(ge=0.0)
    input_payload_bytes: int = Field(ge=0)
    output_payload_bytes: int = Field(ge=0)
    output_reduction_percent: float


class FilterResponse(BaseModel):
    response_detail: ResponseDetail
    selected: list[FilterCandidate]
    summary: FilterSummary
    failures: list[FilterFailure]
    metrics: FilterMetrics
    details: list[FilterItemDetail] | None = None


class ModelCapabilities(BaseModel):
    decision_types: list[DecisionKind]
    max_total_tokens: int = Field(gt=0)
    max_options: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    generated_output_tokens: int = Field(ge=0)
    overflow_behavior: str
    local_inference: bool


class RuntimeMetrics(BaseModel):
    initialization_ms: float = Field(ge=0.0)
    inference_count: int = Field(ge=0)
    inference_error_count: int = Field(ge=0)
    last_inference_ms: float | None = Field(default=None, ge=0.0)


class ServerInfo(BaseModel):
    version: str
    backend: str
    backend_version: str
    model: str
    initialization_state: str
    platform: str
    python_version: str
    compute_units: str
    capabilities: ModelCapabilities
    metrics: RuntimeMetrics
