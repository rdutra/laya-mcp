"""Run the offline synthetic local-policy experiment.

This module calls the existing DecisionService/LayaCoreMLBackend directly. It
does not start MCP, touch a filesystem on behalf of an episode, or change
production routing behavior.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter
from typing import Any, Iterable

from laya_mcp.backend.laya_coreml import LayaCoreMLBackend
from laya_mcp.config import DEFAULT_MODEL, Settings
from laya_mcp.schemas import DecisionKind, DecideInput, DecisionRequest, DecisionType, DecisionSpec
from laya_mcp.service import DecisionService
from laya_mcp.token_budget import DecisionItem, InputCapacityError, TokenBudgetEstimator

from benchmarks.policy_routing.simulator import (
    ACTION_SPACE,
    Scenario,
    WorkflowState,
    acceptable_actions,
    apply_action,
    generate_scenarios,
    legal_actions,
    serialize_scenarios,
    validate_action,
)

QUESTION = "Which next investigation action should be prioritized?"
ACTION_LABELS: dict[str, str] = {
    "inspect_candidate": "candidate",
    "search_definition": "definition",
    "search_call_sites": "calls",
    "inspect_tests": "tests",
    "run_tests": "run",
    "inspect_error": "error",
    "inspect_configuration": "config",
    "implement_change": "change",
    "ask_user": "user",
    "escalate_reasoning": "escalate",
}
LABEL_TO_ACTION = {label: action for action, label in ACTION_LABELS.items()}
POLICY_NAMES = (
    "random_legal",
    "deterministic_heuristic",
    "raw_laya_all",
    "laya_validator_all",
    "laya_preconstrained",
)
REPRESENTATIONS = ("json", "natural_language")
CONFIDENCE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True, slots=True)
class LayaDecision:
    action: str | None
    confidence: float | None
    input_tokens: int
    output_tokens: int
    inference_latency_ms: float | None
    wall_latency_ms: float
    state_tokens: int
    instruction_tokens: int
    option_tokens: int
    special_tokens: int
    oversized: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    step: int
    action: str | None
    executed_action: str | None
    confidence: float | None
    acceptable: bool
    legal: bool
    executed: bool
    invalid_proposal: bool
    invalid_executed: bool
    non_progressing: bool
    validator_intervened: bool
    confidence_escalation: bool
    fallback: str | None
    local_decision: bool
    input_tokens: int
    output_tokens: int
    inference_latency_ms: float | None
    wall_latency_ms: float
    state_tokens: int
    instruction_tokens: int
    option_tokens: int
    special_tokens: int
    error: str | None


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    scenario_id: str
    family: str
    terminal: str
    steps: int
    completed: bool
    failed: bool
    escalated: bool
    repeated_action_steps: int
    events: tuple[DecisionEvent, ...]


class LayaRuntime:
    """Thin benchmark-only facade around the production decision service."""

    def __init__(self, backend: LayaCoreMLBackend) -> None:
        self.backend = backend
        self.service = DecisionService(backend)
        self.estimator = TokenBudgetEstimator(backend.agent)

    async def choose(self, state_text: str, options: tuple[str, ...]) -> LayaDecision:
        option_labels = tuple(ACTION_LABELS.get(option, option) for option in options)
        request = DecisionRequest(
            context=state_text,
            question=QUESTION,
            decision_type=DecisionType.CHOICE,
            options=list(option_labels),
        )
        item = DecisionItem.from_request("routing", request)
        started = perf_counter()
        try:
            breakdown = self.estimator.estimate(state_text, item)
        except (InputCapacityError, ValueError) as exc:
            return LayaDecision(
                action=None,
                confidence=None,
                input_tokens=0,
                output_tokens=0,
                inference_latency_ms=None,
                wall_latency_ms=round((perf_counter() - started) * 1000, 3),
                state_tokens=0,
                instruction_tokens=0,
                option_tokens=0,
                special_tokens=0,
                oversized=isinstance(exc, InputCapacityError),
                error=str(exc),
            )
        try:
            response = await self.service.decide(
                DecideInput(
                    context=state_text,
                    question=QUESTION,
                    decision=DecisionSpec(kind=DecisionKind.CHOICE, options=list(option_labels)),
                )
            )
        except Exception as exc:  # benchmark records model/runtime failures per state
            return LayaDecision(
                action=None,
                confidence=None,
                input_tokens=breakdown.total_tokens,
                output_tokens=0,
                inference_latency_ms=None,
                wall_latency_ms=round((perf_counter() - started) * 1000, 3),
                state_tokens=breakdown.state_tokens,
                instruction_tokens=breakdown.instruction_tokens,
                option_tokens=breakdown.option_tokens,
                special_tokens=breakdown.special_tokens,
                oversized=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return LayaDecision(
            action=LABEL_TO_ACTION.get(str(response.result), str(response.result)),
            confidence=response.confidence,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            inference_latency_ms=response.inference_latency_ms,
            wall_latency_ms=round((perf_counter() - started) * 1000, 3),
            state_tokens=breakdown.state_tokens,
            instruction_tokens=breakdown.instruction_tokens,
            option_tokens=breakdown.option_tokens,
            special_tokens=breakdown.special_tokens,
            oversized=False,
            error=None,
        )


def serialize_state(state: WorkflowState, representation: str) -> str:
    if representation == "json":
        return state.compact_json()
    if representation == "natural_language":
        return state.compact_natural_language()
    raise ValueError(f"unknown representation: {representation}")


def heuristic_action(state: WorkflowState) -> str:
    """Readable deterministic baseline using only the caller-visible state."""

    legal = set(legal_actions(state))
    if state.user_info_available and "ask_user" in legal:
        return "ask_user"
    if state.error_trace_present and not state.error_inspected and "inspect_error" in legal:
        return "inspect_error"
    if state.configuration_suspected and not state.configuration_inspected and "inspect_configuration" in legal:
        return "inspect_configuration"
    target_candidates = 2 if state.ambiguity_level >= 3 or state.risk_level == "high" else 1
    if state.candidates_inspected < target_candidates and "inspect_candidate" in legal:
        return "inspect_candidate"
    if not state.direct_definition_found and "search_definition" in legal:
        return "search_definition"
    if state.ambiguity_level >= 2 and not state.call_sites_inspected and "search_call_sites" in legal:
        return "search_call_sites"
    if state.tests_available and not state.tests_inspected and (
        state.task_type in {"test_driven", "higher_risk_change", "ambiguous_bug"}
    ) and "inspect_tests" in legal:
        return "inspect_tests"
    if state.tests_available and state.tests_inspected and not state.tests_run and (
        state.risk_level == "high" or state.task_type == "test_driven"
    ) and "run_tests" in legal:
        return "run_tests"
    if "implement_change" in legal:
        return "implement_change"
    if state.tests_available and not state.tests_inspected and "inspect_tests" in legal:
        return "inspect_tests"
    if state.tests_available and state.tests_inspected and not state.tests_run and "run_tests" in legal:
        return "run_tests"
    if "escalate_reasoning" in legal:
        return "escalate_reasoning"
    return legal_actions(state)[0]


def _state_text_for_event(state: WorkflowState, representation: str) -> str:
    return serialize_state(state, representation)


async def run_episode(
    scenario: Scenario,
    *,
    policy: str,
    representation: str,
    runtime: LayaRuntime | None,
    confidence_threshold: float | None,
    random_seed: int,
    max_steps: int = 12,
) -> EpisodeOutcome:
    state = scenario.initial_state
    events: list[DecisionEvent] = []
    rng = random.Random(random_seed)
    terminal = "max_steps"
    repeated_steps = 0
    for step in range(max_steps):
        legal = legal_actions(state)
        if not legal:
            terminal = "completed" if state.change_attempted else "failed"
            break
        acceptable = set(acceptable_actions(scenario, state))
        decision: LayaDecision | None = None
        proposal: str | None
        local_decision = False
        validator_intervened = False
        confidence_escalation = False
        fallback: str | None = None
        chosen: str | None = None
        if policy == "random_legal":
            proposal = rng.choice(legal)
        elif policy == "deterministic_heuristic":
            proposal = heuristic_action(state)
        else:
            if policy == "laya_preconstrained" and len(legal) == 1:
                proposal = legal[0]
            else:
                if runtime is None:
                    raise ValueError(f"policy {policy} requires a Laya runtime")
                options = legal if policy == "laya_preconstrained" else ACTION_SPACE
                decision = await runtime.choose(_state_text_for_event(state, representation), options)
                local_decision = True
                proposal = decision.action
                if confidence_threshold is not None and (
                    decision.confidence is None or decision.confidence < confidence_threshold
                ):
                    confidence_escalation = True
                    chosen = "escalate_reasoning"
                    fallback = "confidence_threshold"
                elif policy == "laya_validator_all":
                    validation = validate_action(state, proposal or "")
                    if not validation.legal:
                        validator_intervened = True
                        chosen = "escalate_reasoning"
                        fallback = "validator_rejection"

        invalid_proposal = proposal not in ACTION_SPACE or proposal not in legal
        invalid_executed = False
        executed = False
        non_progressing = False
        if chosen is None:
            chosen = proposal
        if invalid_proposal:
            if policy == "raw_laya_all":
                invalid_executed = True
                terminal = "failed_invalid_action"
            else:
                # A model/runtime failure in the bounded policies is escalated;
                # the simulator never silently invents an action.
                chosen = "escalate_reasoning"
                fallback = fallback or "runtime_failure"
                validator_intervened = validator_intervened or policy == "laya_validator_all"
        if terminal == "failed_invalid_action":
            executed = False
        else:
            validation = validate_action(state, chosen or "")
            if not validation.legal:
                invalid_executed = True
                terminal = "failed_invalid_action"
            else:
                transition = apply_action(scenario, state, chosen or "")
                executed = True
                non_progressing = not transition.progress
                if non_progressing:
                    repeated_steps += 1
                state = transition.state
                if transition.terminal is not None:
                    terminal = transition.terminal

        event = DecisionEvent(
            step=step,
            action=proposal,
            executed_action=chosen if executed else None,
            confidence=decision.confidence if decision else None,
            acceptable=proposal in acceptable,
            legal=proposal in legal,
            executed=executed,
            invalid_proposal=invalid_proposal,
            invalid_executed=invalid_executed,
            non_progressing=non_progressing,
            validator_intervened=validator_intervened,
            confidence_escalation=confidence_escalation,
            fallback=fallback,
            local_decision=local_decision,
            input_tokens=decision.input_tokens if decision else 0,
            output_tokens=decision.output_tokens if decision else 0,
            inference_latency_ms=decision.inference_latency_ms if decision else None,
            wall_latency_ms=decision.wall_latency_ms if decision else 0.0,
            state_tokens=decision.state_tokens if decision else 0,
            instruction_tokens=decision.instruction_tokens if decision else 0,
            option_tokens=decision.option_tokens if decision else 0,
            special_tokens=decision.special_tokens if decision else 0,
            error=decision.error if decision else None,
        )
        events.append(event)
        if terminal in {"completed", "escalated", "failed_invalid_action"}:
            break
    else:
        terminal = "max_steps"

    return EpisodeOutcome(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        terminal=terminal,
        steps=len(events),
        completed=terminal == "completed",
        failed=terminal.startswith("failed") or terminal == "max_steps",
        escalated=terminal == "escalated",
        repeated_action_steps=repeated_steps,
        events=tuple(events),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def aggregate_outcomes(outcomes: Iterable[EpisodeOutcome]) -> dict[str, Any]:
    rows = list(outcomes)
    events = [event for outcome in rows for event in outcome.events]
    decisions = [event for event in events if event.local_decision]
    completed_steps = [outcome.steps for outcome in rows if outcome.completed]
    accepted = sum(event.acceptable for event in events)
    invalid_proposed = sum(event.invalid_proposal for event in events)
    invalid_executed = sum(event.invalid_executed for event in events)
    non_progressing = sum(event.non_progressing for event in events)
    interventions = sum(event.validator_intervened for event in events)
    confidence_escalations = sum(event.confidence_escalation for event in events)
    inference_errors = sum(bool(event.error) for event in decisions)
    token_events = [event for event in decisions if event.input_tokens]
    inference_latencies = [event.inference_latency_ms for event in decisions if event.inference_latency_ms is not None]
    wall_latencies = [event.wall_latency_ms for event in decisions if event.wall_latency_ms]
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({outcome.family for outcome in rows}):
        family_rows = [outcome for outcome in rows if outcome.family == family]
        by_family[family] = {
            "episodes": len(family_rows),
            "completion_rate": _safe_rate(sum(item.completed for item in family_rows), len(family_rows)),
            "escalation_rate": _safe_rate(sum(item.escalated for item in family_rows), len(family_rows)),
            "mean_steps_to_completion": round(mean(item.steps for item in family_rows if item.completed), 4)
            if any(item.completed for item in family_rows)
            else None,
        }
    return {
        "episodes": len(rows),
        "decisions": len(decisions),
        "completion_rate": _safe_rate(sum(item.completed for item in rows), len(rows)),
        "failure_rate": _safe_rate(sum(item.failed for item in rows), len(rows)),
        "escalation_rate": _safe_rate(sum(item.escalated for item in rows), len(rows)),
        "mean_steps": round(mean(item.steps for item in rows), 4) if rows else 0.0,
        "mean_steps_to_completion": round(mean(completed_steps), 4) if completed_steps else None,
        "median_steps_to_completion": median(completed_steps) if completed_steps else None,
        "repeated_action_steps": sum(item.repeated_action_steps for item in rows),
        "acceptable_action_rate": _safe_rate(accepted, len(events)),
        "invalid_proposed_actions": invalid_proposed,
        "invalid_proposed_rate": _safe_rate(invalid_proposed, len(events)),
        "invalid_executed_actions": invalid_executed,
        "non_progressing_action_rate": _safe_rate(non_progressing, len(events)),
        "validator_interventions": interventions,
        "validator_intervention_rate": _safe_rate(interventions, len(decisions)),
        "confidence_escalations": confidence_escalations,
        "inference_errors": inference_errors,
        "total_laya_input_tokens": sum(event.input_tokens for event in decisions),
        "total_laya_output_tokens": sum(event.output_tokens for event in decisions),
        "mean_input_tokens": round(mean(event.input_tokens for event in token_events), 4) if token_events else 0.0,
        "mean_inference_latency_ms": round(mean(inference_latencies), 4) if inference_latencies else None,
        "median_inference_latency_ms": round(median(inference_latencies), 4) if inference_latencies else None,
        "mean_wall_latency_ms": round(mean(wall_latencies), 4) if wall_latencies else None,
        "token_composition": {
            "state_tokens": sum(event.state_tokens for event in decisions),
            "instruction_tokens": sum(event.instruction_tokens for event in decisions),
            "option_tokens": sum(event.option_tokens for event in decisions),
            "special_tokens": sum(event.special_tokens for event in decisions),
        },
        "by_family": by_family,
    }


def serialize_outcome(outcome: EpisodeOutcome) -> dict[str, Any]:
    return {
        "scenario_id": outcome.scenario_id,
        "family": outcome.family,
        "terminal": outcome.terminal,
        "steps": outcome.steps,
        "completed": outcome.completed,
        "failed": outcome.failed,
        "escalated": outcome.escalated,
        "repeated_action_steps": outcome.repeated_action_steps,
        "events": [asdict(event) for event in outcome.events],
    }


def _criteria_digest(root: Path) -> str:
    return hashlib.sha256(
        (root / "evaluation" / "policy-routing-success-criteria.md").read_bytes()
    ).hexdigest()


async def run_suite(
    scenarios: list[Scenario],
    *,
    model: str,
    local_files_only: bool,
    seed: int,
    representation: str,
    policies: tuple[str, ...],
    confidence_sweep: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    needs_laya = any(policy.startswith("laya") or policy == "raw_laya_all" for policy in policies)
    runtime: LayaRuntime | None = None
    initialization_ms: float | None = None
    if needs_laya:
        started = perf_counter()
        backend = await LayaCoreMLBackend.create(
            Settings(model=model, local_files_only=local_files_only)
        )
        initialization_ms = round((perf_counter() - started) * 1000, 3)
        runtime = LayaRuntime(backend)
    for policy in policies:
        if policy not in POLICY_NAMES:
            raise ValueError(f"unknown policy {policy}")
        outcomes = [
            await run_episode(
                scenario,
                policy=policy,
                representation=representation,
                runtime=runtime,
                confidence_threshold=None,
                random_seed=seed + index,
            )
            for index, scenario in enumerate(scenarios)
        ]
        policy_output: dict[str, Any] = {
            "summary": aggregate_outcomes(outcomes),
            "episodes": [serialize_outcome(outcome) for outcome in outcomes],
        }
        if confidence_sweep and policy == "laya_validator_all":
            sweep: dict[str, Any] = {}
            for threshold in CONFIDENCE_THRESHOLDS:
                threshold_outcomes = [
                    await run_episode(
                        scenario,
                        policy=policy,
                        representation=representation,
                        runtime=runtime,
                        confidence_threshold=threshold,
                        random_seed=seed + index,
                    )
                    for index, scenario in enumerate(scenarios)
                ]
                threshold_summary = aggregate_outcomes(threshold_outcomes)
                threshold_events = [
                    event
                    for outcome in threshold_outcomes
                    for event in outcome.events
                    if event.local_decision
                ]
                handled = [
                    event
                    for event in threshold_events
                    if not event.confidence_escalation
                ]
                threshold_summary["confidence_threshold"] = threshold
                threshold_summary["local_coverage"] = _safe_rate(len(handled), len(threshold_events))
                threshold_summary["acceptable_rate_when_locally_handled"] = _safe_rate(
                    sum(event.acceptable for event in handled), len(handled)
                )
                threshold_summary["high_confidence_mistakes"] = sum(
                    not event.acceptable for event in handled
                )
                sweep[str(threshold)] = threshold_summary
            policy_output["confidence_sweep"] = sweep
        output[policy] = policy_output
    return {
        "model": model if needs_laya else None,
        "model_initialization_ms": initialization_ms,
        "representation": representation,
        "policies": output,
    }


def _parse_policies(value: str) -> tuple[str, ...]:
    if value == "suite":
        return POLICY_NAMES
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    if not policies:
        raise argparse.ArgumentTypeError("at least one policy is required")
    unknown = set(policies) - set(POLICY_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown policies: {', '.join(sorted(unknown))}")
    return policies


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    split = args.split
    count = args.episodes or (100 if split == "dev" else 300)
    scenarios = generate_scenarios(count=count, seed=args.seed, split=split)
    if args.scenarios_output:
        args.scenarios_output.parent.mkdir(parents=True, exist_ok=True)
        args.scenarios_output.write_text(
            json.dumps(serialize_scenarios(scenarios), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    policies = args.policies
    result = await run_suite(
        scenarios,
        model=args.model,
        local_files_only=args.local_files_only,
        seed=args.seed,
        representation=args.state_format,
        policies=policies,
        confidence_sweep=args.confidence_sweep,
    )
    result.update(
        {
            "benchmark": "policy-routing",
            "version": 1,
            "split": split,
            "scenario_seed": args.seed,
            "episode_count": count,
            "question": QUESTION,
            "action_space": list(ACTION_SPACE),
            "max_steps": 12,
            "criteria_sha256": _criteria_digest(root),
            "freeze": {
                "state_schema": "workflow-state-v1",
                "question_frozen": True,
                "actions_frozen": True,
                "validator_frozen": True,
                "heuristic_frozen": True,
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "eval"), default="eval")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-format", choices=REPRESENTATIONS, default="json")
    parser.add_argument("--policies", type=_parse_policies, default=POLICY_NAMES)
    parser.add_argument("--confidence-sweep", action="store_true")
    parser.add_argument("--scenarios-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    rendered = json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
