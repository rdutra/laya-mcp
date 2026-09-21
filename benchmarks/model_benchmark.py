#!/usr/bin/env python3
"""Benchmark one official Laya-CoreML checkpoint on the local machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from statistics import median
from time import perf_counter
from typing import Any

from laya_mcp.config import DEFAULT_MODEL


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.  Keep the raw platform observable.
    return value


def questions(count: int) -> dict[str, dict[str, str]]:
    return {
        f"candidate-{index}": {
            "type": "noul",
            "instructions": "Is this candidate relevant? Candidate: short item summary",
        }
        for index in range(count)
    }


def timed_predict(agent: Any, count: int) -> float:
    started = perf_counter()
    agent.predict(
        "Criterion: determine whether concise candidates are relevant to a bounded task",
        questions(count),
    )
    return (perf_counter() - started) * 1000


def load_catalog() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "evaluation" / "model_catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run(model: str, local_files_only: bool, repeats: int) -> dict[str, Any]:
    import laya_coreml as laya  # type: ignore[import-untyped]

    rss_before = rss_bytes()
    started = perf_counter()
    agent = laya.load(model, local_files_only=local_files_only)
    initialization_ms = (perf_counter() - started) * 1000
    rss_after_load = rss_bytes()

    first_inference_ms = timed_predict(agent, 1)
    warm_samples = [timed_predict(agent, 1) for _ in range(repeats)]
    scale: dict[str, Any] = {}
    for count in (10, 50, 100):
        samples = [timed_predict(agent, count) for _ in range(max(1, min(3, repeats)))]
        scale[str(count)] = {
            "samples_ms": [round(sample, 3) for sample in samples],
            "median_ms": round(median(samples), 3),
            "per_decision_median_ms": round(median(samples) / count, 3),
            "decisions_per_second": round(count / (median(samples) / 1000), 3)
            if median(samples)
            else 0.0,
        }
    rss_after_benchmark = rss_bytes()
    catalog = load_catalog()
    metadata: dict[str, Any] = next(
        (entry for entry in catalog["models"] if entry["id"] == model),
        {},
    )
    return {
        "model": model,
        "metadata": metadata,
        "initialization_ms": round(initialization_ms, 3),
        "first_inference_ms": round(first_inference_ms, 3),
        "warm_inference_samples_ms": [round(sample, 3) for sample in warm_samples],
        "warm_inference_median_ms": round(median(warm_samples), 3),
        "context_tokens": int(agent.shape["max_length"]),
        "batch_size": int(agent.shape["batch_size"]),
        "compute_units": str(getattr(agent, "compute_units", "unknown")),
        "model_shape": agent.shape,
        "scale": scale,
        "rss_before_bytes_raw": rss_before,
        "rss_after_load_bytes_raw": rss_after_load,
        "rss_after_benchmark_bytes_raw": rss_after_benchmark,
        "rss_delta_after_load_raw": rss_after_load - rss_before,
        "rss_note": "ru_maxrss is platform-dependent; this is a process high-water mark, not an isolated model allocation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.model, args.local_files_only, args.repeats)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
