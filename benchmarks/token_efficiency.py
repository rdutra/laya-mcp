#!/usr/bin/env python3
"""Measure Laya request overhead and batching strategies.

The reported tokens are local classifier tokens.  They are not a proxy for
Codex/Claude token pricing; the useful comparison is whether local filtering
reduces expensive primary-model context while keeping local latency bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from laya_mcp.config import DEFAULT_MODEL
from laya_mcp.token_budget import DecisionItem, TokenAwarePlanner, TokenBudgetEstimator


COUNTS = (1, 5, 10, 25, 50, 100)


def workload(name: str, count: int) -> tuple[str, tuple[DecisionItem, ...]]:
    if name == "short":
        state = "fix movement bug"
        items = tuple(
            DecisionItem(str(index), f"Relevant candidate {index}?") for index in range(count)
        )
    elif name == "file-relevance":
        state = "Fix player movement: acceleration, jumping, and animation state are incorrect."
        templates = (
            "player_controller.gd handles movement and jumping.",
            "player_animation.gd maps movement states to animations.",
            "enemy_ai.gd handles enemy navigation.",
            "projectile.gd spawns bullets and applies damage.",
            "main_menu.gd builds the title screen.",
        )
        items = tuple(
            DecisionItem(str(index), f"Is candidate file {index} relevant? {templates[index % len(templates)]}")
            for index in range(count)
        )
    elif name == "search-relevance":
        state = "Fix a null reference when the player jumps after landing on a slope."
        templates = (
            "Search hit: CharacterBody2D movement and floor snapping.",
            "Search hit: jump input buffering in PlayerController.",
            "Search hit: shader for terrain slopes.",
            "Search hit: save-game serialization.",
            "Search hit: editor import settings.",
        )
        items = tuple(
            DecisionItem(str(index), f"Is search result {index} relevant? {templates[index % len(templates)]}")
            for index in range(count)
        )
    elif name == "test-relevance":
        state = "Review a change that fixes player acceleration and jump state transitions."
        templates = (
            "test_player_acceleration_changes_velocity.",
            "test_jump_resets_after_landing.",
            "test_enemy_pathfinding_avoids_walls.",
            "test_save_slot_round_trip.",
            "test_main_menu_opens_settings.",
        )
        items = tuple(
            DecisionItem(str(index), f"Is test {index} relevant? {templates[index % len(templates)]}")
            for index in range(count)
        )
    else:
        raise ValueError(f"unknown workload: {name}")
    return state, items


def definition(item: DecisionItem) -> dict[str, Any]:
    if item.decision_type.value == "noul":
        return {"type": "noul", "instructions": item.question}
    return {
        "type": item.decision_type.value,
        "instructions": item.question,
        "criteria": list(item.options),
    }


def answer_signature(answer: dict[str, Any]) -> tuple[Any, float, float | None]:
    value = answer.get("noul", answer.get("choice", answer.get("score")))
    return value, float(answer.get("confidence", 0.0)), (
        float(answer["noul"]) if answer.get("noul") is not None else None
    )


def run_strategy(
    agent: Any,
    state: str,
    items: tuple[DecisionItem, ...],
    strategy: str,
    chunk_size: int,
    baseline: dict[str, tuple[Any, float, float | None]] | None,
) -> dict[str, Any]:
    planner = TokenAwarePlanner(TokenBudgetEstimator(agent))
    if strategy == "separate":
        chunks = planner.plan(state, items, max_questions_per_chunk=1)
    elif strategy == "single-call":
        chunks = planner.plan(state, items)
    else:
        chunks = planner.plan(state, items, max_questions_per_chunk=chunk_size)

    started = perf_counter()
    answers: dict[str, tuple[Any, float, float | None]] = {}
    total_input = 0
    total_output = 0
    for chunk in chunks:
        raw = agent.predict(
            state,
            {item.item_id: definition(item) for item in chunk.items},
        )
        usage = raw.get("usage", {})
        total_input += int(usage.get("input_tokens", 0))
        total_output += int(usage.get("output_tokens", 0))
        for item_id, answer in raw["answers"].items():
            answers[item_id] = answer_signature(answer)
    total_ms = (perf_counter() - started) * 1000
    count = len(items)
    row: dict[str, Any] = {
        "strategy": strategy,
        "candidate_count": count,
        "request_count": len(chunks),
        "forward_evaluations": count,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_inference_ms": round(total_ms, 3),
        "average_latency_ms": round(total_ms / count, 3),
        "decisions_per_second": round(count / (total_ms / 1000), 3) if total_ms else 0.0,
        "token_budget_utilization": round(
            total_input / (count * int(agent.shape["max_length"])), 4
        ),
        "average_questions_per_inference": round(count / len(chunks), 3),
        "average_confidence": round(sum(value[1] for value in answers.values()) / count, 5),
        "answers": answers,
    }
    if baseline is not None:
        row["result_mismatches_vs_separate"] = sum(
            answers[item_id][0] != baseline[item_id][0] for item_id in answers
        )
        row["mean_confidence_abs_delta_vs_separate"] = round(
            sum(abs(answers[item_id][1] - baseline[item_id][1]) for item_id in answers) / count,
            6,
        )
        row["mean_noul_probability_abs_delta_vs_separate"] = round(
            sum(
                abs((answers[item_id][2] or 0.0) - (baseline[item_id][2] or 0.0))
                for item_id in answers
            )
            / count,
            6,
        )
    return row


def load_agent(model: str, local_files_only: bool) -> Any:
    import laya_coreml as laya  # type: ignore[import-untyped]

    started = perf_counter()
    agent = laya.load(model, local_files_only=local_files_only)
    print(f"model_initialization_ms={((perf_counter() - started) * 1000):.3f}")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument(
        "--workloads",
        default="short,file-relevance,search-relevance,test-relevance",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(","))
    agent = load_agent(args.model, args.local_files_only)
    # Warm compile is excluded from strategy timings.
    warm_state, warm_items = workload("file-relevance", 1)
    agent.predict(warm_state, {warm_items[0].item_id: definition(warm_items[0])})

    rows: list[dict[str, Any]] = []
    for workload_name in args.workloads.split(","):
        for count in counts:
            state, items = workload(workload_name, count)
            baseline_row = run_strategy(agent, state, items, "separate", args.chunk_size, None)
            baseline = baseline_row["answers"]
            rows.append({"workload": workload_name, **baseline_row})
            for strategy in ("single-call", "chunks"):
                label = strategy if strategy == "single-call" else f"chunks-{args.chunk_size}"
                row = run_strategy(agent, state, items, strategy, args.chunk_size, baseline)
                row["strategy"] = label
                rows.append({"workload": workload_name, **row})
    output = {"model": args.model, "max_total_tokens": int(agent.shape["max_length"]), "rows": rows}
    rendered = json.dumps(output, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
