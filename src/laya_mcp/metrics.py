"""Small in-process metrics for model lifecycle and inference."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    initialization_ms: float
    inference_count: int
    inference_error_count: int
    last_inference_ms: float | None


class InferenceMetrics:
    def __init__(self, initialization_ms: float) -> None:
        self._initialization_ms = initialization_ms
        self._inference_count = 0
        self._inference_error_count = 0
        self._last_inference_ms: float | None = None
        self._lock = Lock()

    def record_success(self, latency_ms: float, *, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        with self._lock:
            self._inference_count += count
            self._last_inference_ms = latency_ms

    def record_error(self) -> None:
        with self._lock:
            self._inference_error_count += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                initialization_ms=round(self._initialization_ms, 3),
                inference_count=self._inference_count,
                inference_error_count=self._inference_error_count,
                last_inference_ms=(
                    round(self._last_inference_ms, 3)
                    if self._last_inference_ms is not None
                    else None
                ),
            )
