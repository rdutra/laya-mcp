"""Exact token accounting and deterministic planning for Laya requests.

Laya's ``max_length`` applies to each question sequence.  It is not an
aggregate budget for a ``predict`` call: the 0.1 agent encodes questions one
at a time.  This module mirrors the 0.1 prompt builder so callers can reject
lossy requests before handing them to Laya.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
import json

from laya_mcp.schemas import DecisionRequest, DecisionType


class InputCapacityError(ValueError):
    """Raised when Laya would truncate a request."""


def serialize_state(state: str | dict[str, Any] | list[Any]) -> str:
    """Serialize state using the same representation as Laya-CoreML."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class DecisionItem:
    """One independently encoded Laya question with a stable caller ID."""

    item_id: str
    question: str
    decision_type: DecisionType = DecisionType.NOUL
    options: tuple[str, ...] = ()

    @classmethod
    def from_request(cls, item_id: str, request: DecisionRequest) -> DecisionItem:
        return cls(
            item_id=item_id,
            question=request.question,
            decision_type=request.decision_type,
            options=tuple(request.options or ()),
        )


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    """Token contribution for one question/state pair."""

    item_id: str
    state_tokens: int
    instruction_tokens: int
    option_tokens: int
    special_tokens: int
    total_tokens: int
    max_total_tokens: int

    @property
    def fits(self) -> bool:
        return self.total_tokens <= self.max_total_tokens

    @property
    def utilization(self) -> float:
        return self.total_tokens / self.max_total_tokens


class TokenBudgetEstimator:
    """Mirror Laya 0.1's ``build_prefix``/``build_sequence`` accounting."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.tokenizer = agent.tok
        self.head_max_tokens = int(agent.cfg.get("head_max_len", 192))
        self.max_total_tokens = min(
            int(agent.cfg.get("max_len", 512)), int(agent.shape["max_length"])
        )
        self.max_options = int(agent.shape["max_options"])

    def _tokens(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render_options(self, item: DecisionItem) -> list[str]:
        if item.decision_type is DecisionType.NOUL:
            if item.options:
                raise ValueError("options must be omitted for a noul decision")
            return [
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ]
        if len(item.options) < 2:
            raise ValueError("choice and score decisions require at least two options")
        if len(item.options) > self.max_options:
            raise ValueError(f"this model supports at most {self.max_options} options")
        if item.decision_type is DecisionType.CHOICE and len(set(item.options)) != len(item.options):
            raise ValueError("choice options must be unique")
        if item.decision_type is DecisionType.CHOICE:
            return list(item.options)
        return [f"level {index}: {label}" for index, label in enumerate(item.options)]

    def estimate(self, context: str | dict[str, Any] | list[Any], item: DecisionItem) -> TokenBreakdown:
        """Return exact preflight accounting, raising before lossy truncation."""
        mask_token = self.tokenizer.mask_token
        options = self.render_options(item)
        head_ids = self._tokens(
            f"{item.decision_type.value} question: {item.question.replace(mask_token, ' ')}"
        )
        option_content_ids = [
            self._tokens(" " + option.replace(mask_token, " ")) for option in options
        ]
        if any(len(ids) > 48 for ids in option_content_ids):
            raise InputCapacityError(
                f"item {item.item_id!r} has an option over Laya's 48-token option budget; "
                "input was not sent"
            )
        option_ids = [[self.tokenizer.mask_token_id] + ids for ids in option_content_ids]
        option_budget = self.head_max_tokens - sum(map(len, option_ids))
        if option_budget < 16:
            per_option = max(4, (self.head_max_tokens - 16) // max(1, len(option_ids)))
            if any(len(ids) > per_option for ids in option_ids):
                raise InputCapacityError(
                    f"item {item.item_id!r} exceeds Laya's shared question budget; input was not sent"
                )
            option_budget = self.head_max_tokens - sum(map(len, option_ids))
        head_capacity = max(8, option_budget)
        if len(head_ids) > head_capacity:
            raise InputCapacityError(
                f"item {item.item_id!r} exceeds Laya's instruction budget; input was not sent"
            )

        state_ids = self._tokens(serialize_state(context).replace(mask_token, " "))
        # [CLS], head, [SEP], option markers/content, [SEP], state, [SEP].
        # The per-option mask marker is included in ``option_tokens``.
        special_tokens = 1 + 1 + 1 + 1
        prefix_tokens = special_tokens + len(head_ids) + sum(map(len, option_ids))
        total_tokens = prefix_tokens + len(state_ids)
        if total_tokens > self.max_total_tokens:
            raise InputCapacityError(
                f"item {item.item_id!r} requires {total_tokens} tokens but this model supports "
                f"{self.max_total_tokens}; input was not sent and was not truncated"
            )
        return TokenBreakdown(
            item_id=item.item_id,
            state_tokens=len(state_ids),
            instruction_tokens=len(head_ids),
            option_tokens=sum(map(len, option_ids)),
            special_tokens=special_tokens,
            total_tokens=total_tokens,
            max_total_tokens=self.max_total_tokens,
        )


@dataclass(frozen=True, slots=True)
class PlannedChunk:
    """An ordered chunk and the reason it was separated from the next one."""

    items: tuple[DecisionItem, ...]
    estimates: tuple[TokenBreakdown, ...]
    reason: str

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    @property
    def estimated_input_tokens(self) -> int:
        return sum(item.total_tokens for item in self.estimates)


class TokenAwarePlanner:
    """Plan deterministic batches without imposing an unsupported aggregate limit."""

    def __init__(self, estimator: TokenBudgetEstimator) -> None:
        self.estimator = estimator

    def plan(
        self,
        context: str | dict[str, Any] | list[Any],
        items: Sequence[DecisionItem],
        *,
        max_questions_per_chunk: int | None = None,
    ) -> tuple[PlannedChunk, ...]:
        if not items:
            raise ValueError("items must not be empty")
        if max_questions_per_chunk is not None and max_questions_per_chunk < 1:
            raise ValueError("max_questions_per_chunk must be positive")
        estimates = tuple(self.estimator.estimate(context, item) for item in items)
        chunks: list[PlannedChunk] = []
        start = 0
        while start < len(items):
            end = len(items) if max_questions_per_chunk is None else min(
                len(items), start + max_questions_per_chunk
            )
            reason = "all_items_fit_per_question_budget"
            if end < len(items):
                reason = "max_questions_per_chunk"
            chunks.append(
                PlannedChunk(
                    items=tuple(items[start:end]),
                    estimates=estimates[start:end],
                    reason=reason,
                )
            )
            start = end
        return tuple(chunks)


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class ChunkExecutionError(RuntimeError):
    """Wrap a failed chunk while retaining its stable IDs and position."""

    chunk_index: int
    item_ids: tuple[str, ...]
    cause: Exception

    def __str__(self) -> str:
        return f"chunk {self.chunk_index} failed for {self.item_ids}: {self.cause}"


async def execute_chunks(
    chunks: Sequence[PlannedChunk],
    execute: Callable[[PlannedChunk], Awaitable[Sequence[T]]],
) -> tuple[T, ...]:
    """Execute chunks in order and stop with contextual error on first failure."""
    results: list[T] = []
    for index, chunk in enumerate(chunks):
        try:
            values = await execute(chunk)
        except Exception as exc:
            raise ChunkExecutionError(index, chunk.item_ids, exc) from exc
        results.extend(values)
    return tuple(results)
