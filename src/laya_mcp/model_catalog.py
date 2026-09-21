"""Static metadata for model information that is safe before loading a model."""

from __future__ import annotations

from dataclasses import dataclass

from laya_mcp.schemas import DecisionKind, ModelCapabilities


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model: str
    capabilities: ModelCapabilities
    backend: str = "laya-coreml"
    compute_units: str = "configured by Laya-CoreML"


_COMMON_96 = ModelCapabilities(
    decision_types=list(DecisionKind),
    max_total_tokens=96,
    max_options=32,
    batch_size=1,
    generated_output_tokens=0,
    overflow_behavior="reject_without_inference",
    local_inference=True,
)


MODEL_CATALOG: dict[str, ModelMetadata] = {
    "aac6fef/laya-multilingual-coreml-ane-w8": ModelMetadata(
        model="aac6fef/laya-multilingual-coreml-ane-w8",
        capabilities=_COMMON_96,
    ),
    "aac6fef/laya-multilingual-coreml-ane": ModelMetadata(
        model="aac6fef/laya-multilingual-coreml-ane",
        capabilities=_COMMON_96,
    ),
}


def metadata_for(model: str) -> ModelMetadata | None:
    """Return metadata only for exact known official model identifiers."""

    return MODEL_CATALOG.get(model)

