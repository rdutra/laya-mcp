from __future__ import annotations

import pytest

from laya_mcp.backend.laya_coreml import InputCapacityError, LayaCoreMLBackend
from laya_mcp.config import Settings
from laya_mcp.schemas import DecisionRequest, DecisionType
from tests.fakes import FakeAgent


@pytest.mark.asyncio
async def test_backend_loads_once_and_maps_noul_result() -> None:
    agent = FakeAgent()
    loads: list[tuple[str, dict[str, object]]] = []

    def loader(model: str, **kwargs: object) -> FakeAgent:
        loads.append((model, kwargs))
        return agent

    settings = Settings(model="fake/model", local_files_only=True, compute_units="cpu_ne")
    backend = await LayaCoreMLBackend.create(settings, loader=loader)

    first = await backend.classify(
        DecisionRequest(context="movement jump", question="Is this relevant?")
    )
    second = await backend.classify(
        DecisionRequest(context="movement jump", question="Is this relevant?")
    )

    assert len(loads) == 1
    assert agent.predict_calls == 2
    assert first.result is True
    assert first.model == "fake/model"
    assert first.laya_model == "laya-rl-agent"
    assert first.noul_probability == 0.8125
    assert first.usage.input_tokens == 42
    assert first.usage.output_tokens == 0
    assert first.inference_ms >= 0
    assert second.result is True
    assert backend.info().metrics.inference_count == 2


@pytest.mark.asyncio
async def test_backend_maps_choice_and_score() -> None:
    agent = FakeAgent(max_length=256)
    backend = LayaCoreMLBackend(
        agent=agent,
        settings=Settings(model="fake/model"),
        initialization_ms=12.5,
    )

    choice = await backend.classify(
        DecisionRequest(
            context="landing bug",
            question="Which subsystem?",
            decision_type=DecisionType.CHOICE,
            options=["movement", "animation"],
        )
    )
    score = await backend.classify(
        DecisionRequest(
            context="moderate impact",
            question="How severe?",
            decision_type=DecisionType.SCORE,
            options=["low", "medium", "high"],
        )
    )

    assert choice.result == "movement"
    assert choice.probabilities == {"movement": 0.75, "animation": 0.25}
    assert score.result == 1.25
    assert score.legend == {"0": "low", "1": "medium", "2": "high"}


@pytest.mark.asyncio
async def test_backend_rejects_overflow_without_calling_model() -> None:
    agent = FakeAgent(max_length=30)
    backend = LayaCoreMLBackend(
        agent=agent,
        settings=Settings(model="fake/model"),
        initialization_ms=1.0,
    )
    request = DecisionRequest(
        context=" ".join(["important"] * 50),
        question="Is it relevant?",
    )

    with pytest.raises(InputCapacityError, match="not truncated"):
        await backend.classify(request)

    assert agent.predict_calls == 0
    assert backend.info().metrics.inference_error_count == 1


@pytest.mark.asyncio
async def test_backend_validates_options() -> None:
    backend = LayaCoreMLBackend(
        agent=FakeAgent(),
        settings=Settings(model="fake/model"),
        initialization_ms=1.0,
    )

    with pytest.raises(ValueError, match="at least two options"):
        await backend.classify(
            DecisionRequest(
                context="x",
                question="choose",
                decision_type=DecisionType.CHOICE,
                options=["only"],
            )
        )

    with pytest.raises(ValueError, match="must be omitted"):
        await backend.classify(
            DecisionRequest(context="x", question="true?", options=["false", "true"])
        )


@pytest.mark.asyncio
async def test_backend_multi_question_uses_one_predict_call_and_preserves_ids() -> None:
    agent = FakeAgent(max_length=256)
    backend = LayaCoreMLBackend(
        agent=agent,
        settings=Settings(model="fake/model"),
        initialization_ms=1.0,
    )
    requests = [
        ("a", DecisionRequest(context="shared", question="First?")),
        ("b", DecisionRequest(context="shared", question="Second?")),
    ]

    result = await backend.classify_many("shared", requests)

    assert agent.predict_calls == 1
    assert [item.id for item in result.results] == ["a", "b"]
    assert result.model_evaluations == 2
    assert result.usage.input_tokens == 84
    assert all(item.result.inference_ms is None for item in result.results)
    assert backend.info().metrics.inference_count == 2
