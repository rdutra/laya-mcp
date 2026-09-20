"""Public typed contracts shared by the backend and MCP transport."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class DecisionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


NonEmptyText = Annotated[str, Field(min_length=1)]


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
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    noul_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    action_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    legend: dict[str, str] | None = None
    usage: TokenUsage
    inference_ms: float = Field(ge=0.0)


class ModelCapabilities(BaseModel):
    decision_types: list[DecisionType]
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
