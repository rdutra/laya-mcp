#!/usr/bin/env python3
"""Run the held-out Milestone 5 local Laya quality evaluation.

This script intentionally keeps evaluation data, prompt variants, and raw model
outputs separate from the production MCP service.  It performs no cloud calls.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:  # support direct execution from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.evaluation_dataset import (
    ACTION_CASES,
    BOUNDED_CASES,
    EXTENDED_CONTEXT_TASKS,
    RELEVANCE_TASKS,
    REPRESENTATION_CASES,
    ActionCase,
    BoundedCase,
    RelevanceTask,
)
from benchmarks.filter_dataset import DATASET as LEGACY_DATASET
from laya_mcp.config import DEFAULT_MODEL
from laya_mcp.schemas import DecisionRequest, DecisionType
from laya_mcp.token_budget import DecisionItem, InputCapacityError, TokenBudgetEstimator


THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
BUCKETS = (
    (0.0, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.000001),
)
FRAMINGS = {
    "relevance": "Is this candidate relevant to the task? Candidate: {text}",
    "plausibility": "Could this information plausibly help solve the task? Candidate: {text}",
    "exclusion": "Is this candidate clearly irrelevant to the task? Candidate: {text}",
}

LEGACY_FAILURE_NOTES = {
    ("legacy-file-relevance", "movement-tests"): {
        "failure_mode": "candidate_representation_mismatch",
        "note": "The task concerns movement behavior, but the short test summary was treated as a file-level distractor.",
    },
    ("legacy-search-result-relevance", "collision"): {
        "failure_mode": "indirect_semantic_relationship",
        "note": "The result mentions collision normals and movement but omits the explicit jump/slope/null-reference terms.",
    },
}


def framing_question(text: str, framing: str) -> str:
    return FRAMINGS[framing].format(text=text)


def bucket_for(confidence: float) -> str:
    for low, high in BUCKETS:
        if low <= confidence < high:
            return f"{low:.2f}-{min(high, 1.0):.2f}"
    return "outside"


def token_breakdown(
    agent: Any,
    state: str,
    question: str,
    *,
    decision_type: DecisionType = DecisionType.NOUL,
    options: tuple[str, ...] = (),
) -> dict[str, Any]:
    request = DecisionRequest(
        context=state,
        question=question,
        decision_type=decision_type,
        options=list(options) if options else None,
    )
    item = DecisionItem.from_request("evaluation", request)
    try:
        estimate = TokenBudgetEstimator(agent).estimate(state, item)
    except InputCapacityError as exc:
        return {"error": str(exc), "total_tokens": None}
    return {
        "state_tokens": estimate.state_tokens,
        "instruction_tokens": estimate.instruction_tokens,
        "option_tokens": estimate.option_tokens,
        "special_tokens": estimate.special_tokens,
        "total_tokens": estimate.total_tokens,
        "max_total_tokens": estimate.max_total_tokens,
    }


def binary_record(
    *,
    agent: Any,
    task_id: str,
    category: str,
    split: str,
    state: str,
    candidate_id: str,
    candidate_text: str,
    expected_label: str,
    framing: str,
    answer: dict[str, Any],
) -> dict[str, Any]:
    raw_positive = float(answer["noul"]) >= 0.5
    confidence = float(answer["confidence"])
    # Exclusion framing asks for the opposite proposition.  Normalize every
    # result into "candidate should be retained" for comparable metrics.
    predicted_retain = (not raw_positive) if framing == "exclusion" else raw_positive
    expected_retain = expected_label != "irrelevant"
    question = framing_question(candidate_text, framing)
    return {
        "task_id": task_id,
        "category": category,
        "split": split,
        "state": state,
        "candidate_id": candidate_id,
        "candidate_text": candidate_text,
        "expected_label": expected_label,
        "expected_retain": expected_retain,
        "framing": framing,
        "question": question,
        "raw_noul_positive": raw_positive,
        "predicted_retain": predicted_retain,
        "confidence": confidence,
        "confidence_bucket": bucket_for(confidence),
        "noul_probability": float(answer["noul"]),
        "action_probability": float(answer.get("action", {}).get("act_probability", 0.0)),
        "correct": predicted_retain == expected_retain,
        "token_breakdown": token_breakdown(agent, state, question),
    }


def evaluate_relevance_task(agent: Any, task: RelevanceTask, framing: str) -> list[dict[str, Any]]:
    state = f"Task: {task.criterion}"
    raw = agent.predict(
        state,
        {
            candidate.id: {"type": "noul", "instructions": framing_question(candidate.text, framing)}
            for candidate in task.candidates
        },
    )
    return [
        binary_record(
            agent=agent,
            task_id=task.id,
            category=task.category,
            split=task.split,
            state=state,
            candidate_id=candidate.id,
            candidate_text=candidate.text,
            expected_label=candidate.label,
            framing=framing,
            answer=raw["answers"][candidate.id],
        )
        for candidate in task.candidates
    ]


def evaluate_legacy(agent: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in LEGACY_DATASET:
        state = f"Criterion: {task.criterion}"
        framing = "relevance"
        raw = agent.predict(
            state,
            {
                candidate.id: {
                    "type": "noul",
                    "instructions": framing_question(candidate.text, framing),
                }
                for candidate in task.candidates
            },
        )
        for candidate in task.candidates:
            rows.append(
                {
                    **binary_record(
                        agent=agent,
                        task_id=f"legacy-{task.name}",
                        category=task.name,
                        split="legacy",
                        state=state,
                        candidate_id=candidate.id,
                        candidate_text=candidate.text,
                        expected_label="definitely_relevant" if candidate.relevant else "irrelevant",
                        framing=framing,
                        answer=raw["answers"][candidate.id],
                    ),
                    "legacy": True,
                }
            )
    return rows


def evaluate_representation(agent: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in REPRESENTATION_CASES:
        for variant in next(iter(case.candidates)).variants:
            state = f"Task: {case.criterion}"
            raw = agent.predict(
                state,
                {
                    candidate.id: {
                        "type": "noul",
                        "instructions": framing_question(candidate.variants[variant], "relevance"),
                    }
                    for candidate in case.candidates
                },
            )
            for candidate in case.candidates:
                text = candidate.variants[variant]
                rows.append(
                    {
                        **binary_record(
                            agent=agent,
                            task_id=case.id,
                            category="representation",
                            split=case.split,
                            state=state,
                            candidate_id=candidate.id,
                            candidate_text=text,
                            expected_label=candidate.label,
                            framing="relevance",
                            answer=raw["answers"][candidate.id],
                        ),
                        "representation": variant,
                    }
                )
    return rows


def evaluate_extended_context(agent: Any) -> list[dict[str, Any]]:
    """Evaluate richer candidates one at a time so overflow is observable per item."""
    rows: list[dict[str, Any]] = []
    for task in EXTENDED_CONTEXT_TASKS:
        state = f"Task: {task.criterion}"
        for candidate in task.candidates:
            question = framing_question(candidate.text, "relevance")
            try:
                raw = agent.predict(
                    state,
                    {candidate.id: {"type": "noul", "instructions": question}},
                )
                row = binary_record(
                    agent=agent,
                    task_id=task.id,
                    category=task.category,
                    split="extended",
                    state=state,
                    candidate_id=candidate.id,
                    candidate_text=candidate.text,
                    expected_label=candidate.label,
                    framing="relevance",
                    answer=raw["answers"][candidate.id],
                )
                row["error"] = None
            except Exception as exc:
                row = {
                    "task_id": task.id,
                    "category": task.category,
                    "split": "extended",
                    "candidate_id": candidate.id,
                    "candidate_text": candidate.text,
                    "expected_label": candidate.label,
                    "expected_retain": candidate.should_retain,
                    "framing": "relevance",
                    "question": question,
                    "state": state,
                    "error": str(exc),
                    "token_breakdown": token_breakdown(agent, state, question),
                }
            rows.append(row)
    return rows


def evaluate_action(agent: Any, case: ActionCase) -> dict[str, Any]:
    raw = agent.predict(
        case.state,
        {
            case.id: {
                "type": "choice",
                "instructions": case.question,
                "criteria": list(case.actions),
            }
        },
    )
    answer = raw["answers"][case.id]
    confidence = float(answer["confidence"])
    return {
        "id": case.id,
        "category": "action_routing",
        "split": case.split,
        "context": case.state,
        "question": case.question,
        "options": list(case.actions),
        "expected": case.expected_action,
        "predicted": str(answer["choice"]),
        "confidence": confidence,
        "confidence_bucket": bucket_for(confidence),
        "correct": str(answer["choice"]) == case.expected_action,
        "probabilities": {str(k): float(v) for k, v in answer.get("probabilities", {}).items()},
        "token_breakdown": token_breakdown(
            agent,
            case.state,
            case.question,
            decision_type=DecisionType.CHOICE,
            options=case.actions,
        ),
    }


def evaluate_bounded(agent: Any, case: BoundedCase) -> dict[str, Any]:
    decision_type = DecisionType.NOUL if case.options is None else DecisionType.CHOICE
    definition: dict[str, Any] = {
        "type": decision_type.value,
        "instructions": case.question,
    }
    if case.options is not None:
        definition["criteria"] = list(case.options)
    raw = agent.predict(case.context, {case.id: definition})
    answer = raw["answers"][case.id]
    if case.options is None:
        predicted: bool | str = float(answer["noul"]) >= 0.5
    else:
        predicted = str(answer["choice"])
    confidence = float(answer["confidence"])
    return {
        "id": case.id,
        "category": case.category,
        "split": case.split,
        "context": case.context,
        "question": case.question,
        "options": list(case.options) if case.options else None,
        "expected": case.expected,
        "predicted": predicted,
        "confidence": confidence,
        "confidence_bucket": bucket_for(confidence),
        "correct": predicted == case.expected,
        "probabilities": {str(k): float(v) for k, v in answer.get("probabilities", {}).items()},
        "token_breakdown": token_breakdown(
            agent,
            case.context,
            case.question,
            decision_type=decision_type,
            options=case.options or (),
        ),
    }


def binary_metrics(rows: list[dict[str, Any]], threshold: float | None = None) -> dict[str, Any]:
    tp = fp = tn = fn = uncertain = 0
    for row in rows:
        if threshold is None:
            predicted_retain = bool(row["predicted_retain"])
        else:
            predicted_retain = not (
                not row["predicted_retain"] and row["confidence"] >= threshold
            )
            if row["confidence"] < threshold:
                uncertain += 1
        expected_retain = bool(row["expected_retain"])
        if predicted_retain and expected_retain:
            tp += 1
        elif predicted_retain:
            fp += 1
        elif expected_retain:
            fn += 1
        else:
            tn += 1
    total = len(rows)
    return {
        "count": total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(tp / (tp + fp), 5) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 5) if tp + fn else 0.0,
        "specificity": round(tn / (tn + fp), 5) if tn + fp else 0.0,
        "candidate_reduction_percent": round(tn / total * 100, 2) if total else 0.0,
        "uncertain_retained": uncertain,
    }


def relevance_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"threshold": threshold, **binary_metrics(rows, threshold)} for threshold in THRESHOLDS]


def calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for category in sorted({str(row["category"]) for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        for bucket in sorted({str(row["confidence_bucket"]) for row in category_rows}):
            bucket_rows = [row for row in category_rows if row["confidence_bucket"] == bucket]
            output.append(
                {
                    "category": category,
                    "bucket": bucket,
                    "count": len(bucket_rows),
                    "mean_confidence": round(
                        sum(float(row["confidence"]) for row in bucket_rows) / len(bucket_rows), 5
                    ),
                    "accuracy": round(
                        sum(bool(row["correct"]) for row in bucket_rows) / len(bucket_rows), 5
                    ),
                }
            )
    return output


def abstention(rows: list[dict[str, Any]], *, binary: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        handled = [row for row in rows if float(row["confidence"]) >= threshold]
        escalated = [row for row in rows if float(row["confidence"]) < threshold]
        if binary:
            handled_metrics = binary_metrics(handled)
            handled_fn = sum(
                bool(row["expected_retain"]) and not bool(row["predicted_retain"])
                for row in handled
            )
            positive_handled = sum(bool(row["expected_retain"]) for row in handled)
            handled_metrics.update(
                {
                    "handled_false_negative_rate": round(
                        handled_fn / positive_handled, 5
                    )
                    if positive_handled
                    else 0.0,
                    "handled_positive_count": positive_handled,
                }
            )
        else:
            handled_metrics = {
                "handled_accuracy": round(
                    sum(bool(row["correct"]) for row in handled) / len(handled), 5
                )
                if handled
                else 0.0
            }
        output.append(
            {
                "threshold": threshold,
                "total": len(rows),
                "handled_locally": len(handled),
                "escalated": len(escalated),
                "handled_percent": round(len(handled) / len(rows) * 100, 2) if rows else 0.0,
                "escalated_percent": round(len(escalated) / len(rows) * 100, 2) if rows else 0.0,
                **handled_metrics,
            }
        )
    return output


def action_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = Counter((str(row["expected"]), str(row["predicted"])) for row in rows)
    return {
        "count": len(rows),
        "accuracy": round(sum(bool(row["correct"]) for row in rows) / len(rows), 5)
        if rows
        else 0.0,
        "confusion_matrix": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
    }


def grouped_binary(rows: list[dict[str, Any]], *, threshold: float | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["category"]))].append(row)
    return [
        {"split": split, "category": category, **binary_metrics(group, threshold)}
        for (split, category), group in sorted(groups.items())
    ]


def grouped_representation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["representation"])].append(row)
    return [
        {"representation": representation, **binary_metrics(group)}
        for representation, group in sorted(groups.items())
    ]


def run(model: str, local_files_only: bool, *, include_extended: bool = False) -> dict[str, Any]:
    import laya_coreml as laya  # type: ignore[import-untyped]

    started = perf_counter()
    agent = laya.load(model, local_files_only=local_files_only)
    initialization_ms = (perf_counter() - started) * 1000
    # Warm compilation is excluded from subsequent measurements.
    agent.predict("warmup", {"warmup": {"type": "noul", "instructions": "Is this true?"}})

    relevance_rows: list[dict[str, Any]] = []
    for framing in FRAMINGS:
        for task in RELEVANCE_TASKS:
            for row in evaluate_relevance_task(agent, task, framing):
                relevance_rows.append(row)
    representation_rows = evaluate_representation(agent)
    legacy_rows = evaluate_legacy(agent)
    action_rows = [evaluate_action(agent, case) for case in ACTION_CASES]
    bounded_rows = [evaluate_bounded(agent, case) for case in BOUNDED_CASES]
    extended_rows = evaluate_extended_context(agent) if include_extended else []

    baseline_eval = [
        row
        for row in relevance_rows
        if row["framing"] == "relevance" and row["split"] == "eval"
    ]
    framing_summary = {
        framing: {
            "all": binary_metrics([row for row in relevance_rows if row["framing"] == framing]),
            "eval": binary_metrics(
                [row for row in relevance_rows if row["framing"] == framing and row["split"] == "eval"]
            ),
            "thresholds": relevance_thresholds(
                [row for row in relevance_rows if row["framing"] == framing and row["split"] == "eval"]
            ),
        }
        for framing in FRAMINGS
    }

    return {
        "model": model,
        "initialization_ms": round(initialization_ms, 3),
        "max_total_tokens": int(agent.shape["max_length"]),
        "dataset": {
            "relevance_tasks": len(RELEVANCE_TASKS),
            "relevance_candidates": sum(len(task.candidates) for task in RELEVANCE_TASKS),
            "action_cases": len(ACTION_CASES),
            "bounded_cases": len(BOUNDED_CASES),
            "representation_cases": len(REPRESENTATION_CASES),
            "representation_decisions": len(representation_rows),
            "legacy_candidates": len(legacy_rows),
            "development_relevance_candidates": sum(
                len(task.candidates) for task in RELEVANCE_TASKS if task.split == "dev"
            ),
            "held_out_relevance_candidates": sum(
                len(task.candidates) for task in RELEVANCE_TASKS if task.split == "eval"
            ),
        },
        "framing": framing_summary,
        "framing_by_workload": {
            framing: grouped_binary(
                [row for row in relevance_rows if row["framing"] == framing and row["split"] == "eval"]
            )
            for framing in FRAMINGS
        },
        "baseline_relevance_by_workload": grouped_binary(baseline_eval),
        "baseline_relevance_thresholds_by_workload": {
            category: relevance_thresholds(
                [row for row in baseline_eval if row["category"] == category]
            )
            for category in sorted({str(row["category"]) for row in baseline_eval})
        },
        "calibration": {
            "relevance": calibration(baseline_eval),
            "action_routing": calibration([row for row in action_rows if row["split"] == "eval"]),
            "bounded": calibration([row for row in bounded_rows if row["split"] == "eval"]),
        },
        "abstention": {
            "relevance": abstention(baseline_eval, binary=True),
            "relevance_by_framing": {
                framing: abstention(
                    [row for row in relevance_rows if row["framing"] == framing and row["split"] == "eval"],
                    binary=True,
                )
                for framing in FRAMINGS
            },
            "action_routing": abstention(
                [row for row in action_rows if row["split"] == "eval"], binary=False
            ),
            "bounded": abstention(
                [row for row in bounded_rows if row["split"] == "eval"], binary=False
            ),
        },
        "action_routing": action_summary([row for row in action_rows if row["split"] == "eval"]),
        "bounded_decisions": action_summary([row for row in bounded_rows if row["split"] == "eval"]),
        "representation": grouped_representation(
            [row for row in representation_rows if row["split"] == "eval"]
        ),
        "extended_context": {
            "successful": binary_metrics(
                [row for row in extended_rows if row.get("error") is None], threshold=0.9
            ),
            "error_count": sum(row.get("error") is not None for row in extended_rows),
            "rows": extended_rows,
        }
        if include_extended
        else None,
        "legacy_false_negative_analysis": [
            {
                "task_id": row["task_id"],
                "candidate_id": row["candidate_id"],
                "candidate_text": row["candidate_text"],
                "state": row["state"],
                "expected_label": row["expected_label"],
                "model_predicted_retain": row["predicted_retain"],
                "confidence": row["confidence"],
                "noul_probability": row["noul_probability"],
                "question": row["question"],
                "token_breakdown": row["token_breakdown"],
                **LEGACY_FAILURE_NOTES.get(
                    (row["task_id"], row["candidate_id"]),
                    {
                        "failure_mode": "unclassified_model_error",
                        "note": "Requires manual review against the supplied task and candidate context.",
                    },
                ),
            }
            for row in legacy_rows
            if row["expected_retain"]
            and not row["predicted_retain"]
            and row["confidence"] >= 0.9
        ],
        "raw": {
            "relevance": relevance_rows,
            "representation": representation_rows,
            "legacy": legacy_rows,
            "action_routing": action_rows,
            "bounded": bounded_rows,
            "extended_context": extended_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extended-context",
        action="store_true",
        help="also run the separate richer-context experiment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/milestone5.json"),
    )
    args = parser.parse_args()
    result = run(args.model, args.local_files_only, include_extended=args.extended_context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("model", "initialization_ms", "dataset")}, indent=2))


if __name__ == "__main__":
    main()
