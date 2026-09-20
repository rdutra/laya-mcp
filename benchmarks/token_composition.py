#!/usr/bin/env python3
"""Report the exact token composition of representative Laya requests."""

from __future__ import annotations

import argparse
import json
from typing import Any

from laya_mcp.config import DEFAULT_MODEL
from laya_mcp.token_budget import DecisionItem, TokenAwarePlanner, TokenBudgetEstimator
from laya_mcp.schemas import DecisionType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    import laya_coreml as laya

    agent = laya.load(args.model, local_files_only=args.local_files_only)
    estimator = TokenBudgetEstimator(agent)
    state = "Task: fix player movement, jumping, acceleration, and animation."
    examples: dict[str, tuple[DecisionItem, ...]] = {
        "A_one_noul": (DecisionItem("a", "Is this file relevant?"),),
        "B_choice_two": (
            DecisionItem("b", "Which area is relevant?", DecisionType.CHOICE, ("movement", "animation")),
        ),
        "C_choice_five": (
            DecisionItem(
                "c",
                "Which area is relevant?",
                DecisionType.CHOICE,
                ("movement", "animation", "physics", "ui", "audio"),
            ),
        ),
        "D_two_questions": tuple(
            DecisionItem(str(index), "Is this candidate relevant?") for index in range(2)
        ),
        "E_five_questions": tuple(
            DecisionItem(str(index), "Is this candidate relevant?") for index in range(5)
        ),
        "F_ten_questions": tuple(
            DecisionItem(str(index), "Is this candidate relevant?") for index in range(10)
        ),
    }
    output: dict[str, Any] = {}
    for name, items in examples.items():
        chunks = TokenAwarePlanner(estimator).plan(state, items)
        first = chunks[0].estimates[0]
        output[name] = {
            "question_count": len(items),
            "per_question": first.total_tokens,
            "state_tokens": first.state_tokens,
            "instruction_tokens": first.instruction_tokens,
            "option_tokens": first.option_tokens,
            "special_tokens": first.special_tokens,
            "aggregate_input_tokens": sum(item.total_tokens for item in chunks[0].estimates),
            "fits_per_question": all(item.fits for item in chunks[0].estimates),
            "one_predict_call": True,
            "ane_forward_evaluations": len(items),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

