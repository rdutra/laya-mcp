from __future__ import annotations

import pytest

from laya_mcp.token_budget import (
    ChunkExecutionError,
    DecisionItem,
    InputCapacityError,
    TokenAwarePlanner,
    TokenBudgetEstimator,
    execute_chunks,
)
from laya_mcp.schemas import DecisionType
from tests.fakes import FakeAgent


def estimator(max_length: int = 96) -> TokenBudgetEstimator:
    return TokenBudgetEstimator(FakeAgent(max_length=max_length))


def noul(item_id: str = "a", question: str = "is relevant") -> DecisionItem:
    return DecisionItem(item_id=item_id, question=question)


def test_breakdown_accounts_for_state_instruction_options_and_specials() -> None:
    breakdown = estimator().estimate("one two", noul())

    assert breakdown.state_tokens == 2
    assert breakdown.instruction_tokens == 4
    assert breakdown.option_tokens == 14
    assert breakdown.special_tokens == 4
    assert breakdown.total_tokens == 24
    assert breakdown.fits


def test_exact_budget_and_one_token_over_budget() -> None:
    probe = estimator().estimate("x", noul())
    exact_context = "x " * (96 - probe.total_tokens + probe.state_tokens)
    # Trim the trailing whitespace while retaining the intended token count.
    exact_context = " ".join(exact_context.split())
    exact = estimator().estimate(exact_context, noul())
    assert exact.total_tokens == 96

    with pytest.raises(InputCapacityError, match="not truncated"):
        estimator().estimate(exact_context + " x", noul())


def test_question_types_and_choices_have_independent_token_contributions() -> None:
    one_noul = estimator().estimate("state", noul())
    two_choice = estimator().estimate(
        "state",
        DecisionItem("choice2", "pick one", DecisionType.CHOICE, ("yes", "no")),
    )
    five_choice = estimator().estimate(
        "state",
        DecisionItem("choice5", "pick one", DecisionType.CHOICE, ("a", "b", "c", "d", "e")),
    )
    assert five_choice.option_tokens > two_choice.option_tokens
    assert estimator().estimate("state", noul("other")).total_tokens == one_noul.total_tokens


def test_planner_preserves_ids_order_and_reason() -> None:
    items = tuple(noul(str(index)) for index in range(5))
    chunks = TokenAwarePlanner(estimator()).plan("state", items, max_questions_per_chunk=2)

    assert [chunk.item_ids for chunk in chunks] == [("0", "1"), ("2", "3"), ("4",)]
    assert chunks[0].reason == "max_questions_per_chunk"
    assert chunks[-1].reason == "all_items_fit_per_question_budget"


def test_many_questions_share_a_call_plan_without_aggregate_budget() -> None:
    items = tuple(noul(str(index)) for index in range(100))
    chunks = TokenAwarePlanner(estimator()).plan("state", items)
    assert len(chunks) == 1
    assert chunks[0].item_ids == tuple(str(index) for index in range(100))


def test_empty_and_oversized_items_are_rejected_without_truncation() -> None:
    planner = TokenAwarePlanner(estimator())
    with pytest.raises(ValueError, match="must not be empty"):
        planner.plan("state", [])
    oversized = DecisionItem("huge", "question", DecisionType.CHOICE, ("x " * 49, "y"))
    with pytest.raises(InputCapacityError, match="option"):
        planner.plan("state", [oversized])


@pytest.mark.asyncio
async def test_chunk_execution_preserves_order_and_contextualizes_backend_errors() -> None:
    chunks = TokenAwarePlanner(estimator()).plan("state", [noul("a"), noul("b")], max_questions_per_chunk=1)

    async def execute(chunk: object) -> list[str]:
        assert hasattr(chunk, "item_ids")
        item_ids = getattr(chunk, "item_ids")
        if item_ids == ("b",):
            raise RuntimeError("backend unavailable")
        return [item_ids[0]]

    with pytest.raises(ChunkExecutionError) as error:
        await execute_chunks(chunks, execute)
    assert error.value.chunk_index == 1
    assert error.value.item_ids == ("b",)
    assert isinstance(error.value.__cause__, RuntimeError)
