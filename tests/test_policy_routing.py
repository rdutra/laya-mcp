from __future__ import annotations

import asyncio
from dataclasses import replace

from benchmarks.policy_routing.benchmark import (
    LayaDecision,
    aggregate_outcomes,
    heuristic_action,
    run_episode,
)
from benchmarks.policy_routing.simulator import (
    ACTION_SPACE,
    acceptable_actions,
    apply_action,
    generate_scenarios,
    legal_actions,
    validate_action,
)


def test_seeded_generation_is_reproducible_and_split_specific() -> None:
    first = generate_scenarios(21, 44, "dev")
    second = generate_scenarios(21, 44, "dev")
    held_out = generate_scenarios(21, 44, "eval")
    assert [scenario.serializable() for scenario in first] == [scenario.serializable() for scenario in second]
    assert [scenario.initial_state for scenario in first] != [scenario.initial_state for scenario in held_out]


def test_legal_and_acceptable_actions_are_distinct() -> None:
    scenario = generate_scenarios(1, 100, "dev")[0]
    state = scenario.initial_state
    legal = legal_actions(state)
    acceptable = acceptable_actions(scenario, state)
    assert set(acceptable) <= set(legal)
    assert "escalate_reasoning" in legal
    assert len(legal) > len(acceptable)


def test_validator_enforces_constraints_without_ranking_legal_actions() -> None:
    scenario = generate_scenarios(1, 101, "dev")[0]
    state = scenario.initial_state
    assert not validate_action(state, "run_tests").legal
    assert validate_action(state, "escalate_reasoning").legal
    assert not validate_action(state, "not_an_action").legal
    assert set(legal_actions(state)) == set(
        action for action in ACTION_SPACE if validate_action(state, action).legal
    )


def test_transitions_are_deterministic_and_implementation_requires_evidence() -> None:
    scenario = generate_scenarios(1, 102, "dev")[0]
    state = scenario.initial_state
    assert not validate_action(state, "implement_change").legal
    state = apply_action(scenario, state, "inspect_candidate").state
    state = apply_action(scenario, state, "search_definition").state
    assert validate_action(state, "implement_change").legal
    first = apply_action(scenario, state, "implement_change")
    second = apply_action(scenario, state, "implement_change")
    assert first == second
    assert first.terminal == "completed"


class _CandidateOnlyRuntime:
    async def choose(self, state_text: str, options: tuple[str, ...]) -> LayaDecision:
        del state_text, options
        return LayaDecision(
            action="inspect_candidate",
            confidence=0.5,
            input_tokens=10,
            output_tokens=1,
            inference_latency_ms=1.0,
            wall_latency_ms=1.0,
            state_tokens=5,
            instruction_tokens=2,
            option_tokens=2,
            special_tokens=1,
            oversized=False,
            error=None,
        )


def test_episode_terminates_at_maximum_steps() -> None:
    scenario = generate_scenarios(1, 103, "dev")[0]
    scenario = replace(
        scenario,
        initial_state=replace(scenario.initial_state, candidate_count=50),
    )
    outcome = asyncio.run(
        run_episode(
            scenario,
            policy="raw_laya_all",
            representation="json",
            runtime=_CandidateOnlyRuntime(),  # type: ignore[arg-type]
            confidence_threshold=None,
            random_seed=0,
            max_steps=1,
        )
    )
    assert outcome.terminal == "max_steps"
    assert outcome.steps == 1


def test_heuristic_is_deterministic_and_results_serialize() -> None:
    scenario = generate_scenarios(1, 104, "dev")[0]
    assert heuristic_action(scenario.initial_state) == heuristic_action(scenario.initial_state)
    outcome = asyncio.run(
        run_episode(
            scenario,
            policy="deterministic_heuristic",
            representation="json",
            runtime=None,
            confidence_threshold=None,
            random_seed=0,
        )
    )
    aggregate = aggregate_outcomes([outcome])
    assert aggregate["episodes"] == 1
    assert isinstance(aggregate["token_composition"], dict)
    assert outcome.terminal in {"completed", "escalated", "failed", "max_steps"}
