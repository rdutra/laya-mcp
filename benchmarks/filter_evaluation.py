#!/usr/bin/env python3
"""Evaluate conservative filter thresholds on the small labeled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:  # support direct script execution from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.filter_dataset import DATASET, LabeledTask
from laya_mcp.config import DEFAULT_MODEL


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def question(text: str) -> str:
    return f"Is this candidate relevant? Candidate: {text}"


def evaluate_task(agent: Any, task: LabeledTask, threshold: float) -> dict[str, Any]:
    started = perf_counter()
    raw = agent.predict(
        f"Criterion: {task.criterion}",
        {
            item.id: {"type": "noul", "instructions": question(item.text)}
            for item in task.candidates
        },
    )
    elapsed_ms = (perf_counter() - started) * 1000
    true_positive = false_positive = true_negative = false_negative = 0
    uncertain_retained = 0
    for item in task.candidates:
        answer = raw["answers"][item.id]
        probability = float(answer["noul"])
        confidence = float(answer["confidence"])
        relevant = probability >= 0.5
        rejected = (not relevant) and confidence >= threshold
        if confidence < threshold:
            uncertain_retained += 1
        if item.relevant and not rejected:
            true_positive += 1
        elif item.relevant:
            false_negative += 1
        elif rejected:
            true_negative += 1
        else:
            false_positive += 1
    total = len(task.candidates)
    retained = total - true_negative
    return {
        "task": task.name,
        "threshold": threshold,
        "input_count": total,
        "selected_count": retained,
        "rejected_count": true_negative,
        "uncertain_retained": uncertain_retained,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(true_positive / (true_positive + false_positive), 5)
        if true_positive + false_positive
        else 0.0,
        "recall": round(true_positive / (true_positive + false_negative), 5)
        if true_positive + false_negative
        else 0.0,
        "false_negative_rate": round(
            false_negative / (true_positive + false_negative), 5
        )
        if true_positive + false_negative
        else 0.0,
        "candidate_reduction_percent": round(true_negative / total * 100.0, 2),
        "input_tokens": int(raw.get("usage", {}).get("input_tokens", 0)),
        "output_tokens": int(raw.get("usage", {}).get("output_tokens", 0)),
        "inference_latency_ms": round(elapsed_ms, 3),
    }


def load_agent(model: str, local_files_only: bool) -> Any:
    import laya_coreml as laya  # type: ignore[import-untyped]

    started = perf_counter()
    agent = laya.load(model, local_files_only=local_files_only)
    print(f"model_initialization_ms={(perf_counter() - started) * 1000:.3f}")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    agent = load_agent(args.model, args.local_files_only)
    warmup = DATASET[0]
    agent.predict(
        f"Criterion: {warmup.criterion}",
        {warmup.candidates[0].id: {"type": "noul", "instructions": question(warmup.candidates[0].text)}},
    )
    rows = [
        evaluate_task(agent, task, threshold)
        for threshold in THRESHOLDS
        for task in DATASET
    ]
    aggregate: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        selected = rejected = uncertain = tp = fp = tn = fn = 0
        for row in rows:
            if row["threshold"] != threshold:
                continue
            selected += row["selected_count"]
            rejected += row["rejected_count"]
            uncertain += row["uncertain_retained"]
            tp += row["true_positive"]
            fp += row["false_positive"]
            tn += row["true_negative"]
            fn += row["false_negative"]
        total = selected + rejected
        aggregate.append(
            {
                "threshold": threshold,
                "input_count": total,
                "selected_count": selected,
                "rejected_count": rejected,
                "uncertain_retained": uncertain,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": round(tp / (tp + fp), 5) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 5) if tp + fn else 0.0,
                "false_negative_rate": round(fn / (tp + fn), 5) if tp + fn else 0.0,
                "candidate_reduction_percent": round(rejected / total * 100.0, 2)
                if total
                else 0.0,
            }
        )
    output = {"model": args.model, "thresholds": aggregate, "rows": rows}
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
