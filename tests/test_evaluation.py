from __future__ import annotations

from benchmarks.evaluation_dataset import (
    RELEVANCE_TASKS,
    TOTAL_RELEVANCE_CANDIDATES,
)
from benchmarks.evaluate_milestone5 import binary_metrics, bucket_for


def test_milestone5_dataset_has_held_out_split_and_expected_scale() -> None:
    assert len(RELEVANCE_TASKS) == 21
    assert TOTAL_RELEVANCE_CANDIDATES == 126
    assert {task.split for task in RELEVANCE_TASKS} == {"dev", "eval"}
    assert all(len(task.candidates) >= 6 for task in RELEVANCE_TASKS)


def test_confidence_buckets_include_low_confidence_predictions() -> None:
    assert bucket_for(0.21) == "0.00-0.50"
    assert bucket_for(0.59) == "0.50-0.60"
    assert bucket_for(0.95) == "0.90-1.00"


def test_filter_threshold_metrics_only_exclude_confident_negative_results() -> None:
    rows = [
        {"predicted_retain": False, "expected_retain": True, "confidence": 0.95},
        {"predicted_retain": False, "expected_retain": True, "confidence": 0.75},
        {"predicted_retain": False, "expected_retain": False, "confidence": 0.95},
        {"predicted_retain": True, "expected_retain": False, "confidence": 0.95},
    ]
    metrics = binary_metrics(rows, threshold=0.9)
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["uncertain_retained"] == 1
