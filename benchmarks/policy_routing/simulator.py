"""Deterministic synthetic coding-workflow environment.

The simulator deliberately keeps the environment separate from Laya.  A state
describes observable workflow facts; scenario metadata is private environment
state used only to define transitions and acceptable actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import random
from typing import Any, Iterable

ACTION_SPACE: tuple[str, ...] = (
    "inspect_candidate",
    "search_definition",
    "search_call_sites",
    "inspect_tests",
    "run_tests",
    "inspect_error",
    "inspect_configuration",
    "implement_change",
    "ask_user",
    "escalate_reasoning",
)

_ACTION_LABELS = {
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

TASK_FAMILIES: tuple[str, ...] = (
    "localized_bug",
    "ambiguous_bug",
    "test_driven",
    "error_driven",
    "configuration_issue",
    "missing_information",
    "higher_risk_change",
)

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """The compact, caller-visible state supplied to a policy."""

    task_type: str
    candidate_count: int
    candidates_inspected: int
    direct_definition_found: bool
    call_sites_inspected: bool
    tests_available: bool
    tests_inspected: bool
    tests_run: bool
    tests_passing: bool | None
    error_trace_present: bool
    error_class: str | None
    error_inspected: bool
    configuration_suspected: bool
    configuration_inspected: bool
    recent_change_known: bool
    change_attempted: bool
    change_confidence: int
    risk_level: str
    search_attempts: int
    previous_action: str | None
    repeated_action_count: int
    ambiguity_level: int
    user_info_available: bool
    user_asked: bool
    info_resolved: bool

    def compact_mapping(self) -> dict[str, Any]:
        """Return the frozen minimal routing state.

        The one-letter keys and short values are intentional: the selected W8
        model has a 96-token input limit. Fields used only for environment
        bookkeeping (for example internal confidence increments and the
        recent-change flag) are not exposed because they did not create a
        distinct routing choice in this simulator.
        """

        return {
            "t": {
                "localized_bug": "loc",
                "ambiguous_bug": "amb",
                "test_driven": "test",
                "error_driven": "err",
                "configuration_issue": "cfg",
                "missing_information": "info",
                "higher_risk_change": "risk",
            }[self.task_type],
            "c": f"{self.candidates_inspected}/{self.candidate_count}",
            "d": int(self.direct_definition_found),
            "k": int(self.call_sites_inspected),
            "x": (
                "none"
                if not self.tests_available
                else "pass"
                if self.tests_run and self.tests_passing
                else "fail"
                if self.tests_run
                else "seen"
                if self.tests_inspected
                else "avail"
            ),
            "e": (
                "none"
                if not self.error_trace_present
                else f"{self.error_class or 'error'}-seen"
                if self.error_inspected
                else self.error_class or "error"
            ),
            "g": (
                "none"
                if not self.configuration_suspected
                else "seen"
                if self.configuration_inspected
                else "suspected"
            ),
            "r": {"low": 0, "medium": 1, "high": 2}[self.risk_level],
            "s": self.search_attempts,
            "p": _ACTION_LABELS.get(self.previous_action or "", "none"),
            "q": self.repeated_action_count,
            "a": self.ambiguity_level,
            "u": (
                "available"
                if self.user_info_available
                else "resolved"
                if self.info_resolved
                else "none"
            ),
        }

    def compact_json(self) -> str:
        return json.dumps(self.compact_mapping(), separators=(",", ":"), ensure_ascii=False)

    def compact_natural_language(self) -> str:
        tests = "none" if not self.tests_available else (
            "passing" if self.tests_run and self.tests_passing else
            "failing" if self.tests_run else
            "inspected" if self.tests_inspected else "available"
        )
        error = "none" if not self.error_trace_present else (
            f"{self.error_class or 'error'} inspected" if self.error_inspected else self.error_class or "error"
        )
        config = "none" if not self.configuration_suspected else (
            "inspected" if self.configuration_inspected else "suspected"
        )
        task = {
            "localized_bug": "loc",
            "ambiguous_bug": "amb",
            "test_driven": "test",
            "error_driven": "err",
            "configuration_issue": "cfg",
            "missing_information": "info",
            "higher_risk_change": "risk",
        }[self.task_type]
        return (
            f"{task} task; candidates {self.candidates_inspected}/{self.candidate_count}; "
            f"definition {'found' if self.direct_definition_found else 'absent'}; "
            f"calls {'yes' if self.call_sites_inspected else 'no'}; tests {tests}; error {error}; "
            f"config {config}; risk {self.risk_level}; search {self.search_attempts}; "
            f"previous {_ACTION_LABELS.get(self.previous_action or '', 'none')}; repeat {self.repeated_action_count}; "
            f"ambiguity {self.ambiguity_level}; user "
            f"{'available' if self.user_info_available else 'resolved' if self.info_resolved else 'none'}."
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    family: str
    seed: int
    initial_state: WorkflowState

    def serializable(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "seed": self.seed,
            "initial_state": asdict(self.initial_state),
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    action: str
    legal: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: WorkflowState
    progress: bool
    terminal: str | None


def _scenario_seed(master_seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{master_seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _initial_state(family: str, rng: random.Random) -> WorkflowState:
    candidate_count = {
        "localized_bug": rng.randint(1, 3),
        "ambiguous_bug": rng.randint(7, 18),
        "test_driven": rng.randint(3, 8),
        "error_driven": rng.randint(2, 7),
        "configuration_issue": rng.randint(2, 8),
        "missing_information": 0,
        "higher_risk_change": rng.randint(4, 10),
    }[family]
    risk = "high" if family == "higher_risk_change" else (
        "medium" if family in {"ambiguous_bug", "test_driven"} else "low"
    )
    return WorkflowState(
        task_type=family,
        candidate_count=candidate_count,
        candidates_inspected=0,
        direct_definition_found=False,
        call_sites_inspected=False,
        tests_available=family != "missing_information" and family != "configuration_issue" or (
            family == "configuration_issue" and rng.random() < 0.35
        ),
        tests_inspected=False,
        tests_run=False,
        tests_passing=None,
        error_trace_present=family == "error_driven",
        error_class=(rng.choice(("type", "null", "import", "assertion")) if family == "error_driven" else None),
        error_inspected=False,
        configuration_suspected=family == "configuration_issue",
        configuration_inspected=False,
        recent_change_known=family in {"localized_bug", "test_driven", "higher_risk_change"},
        change_attempted=False,
        change_confidence=0,
        risk_level=risk,
        search_attempts=0,
        previous_action=None,
        repeated_action_count=0,
        ambiguity_level={
            "localized_bug": 1,
            "ambiguous_bug": 4,
            "test_driven": 2,
            "error_driven": 3,
            "configuration_issue": 2,
            "missing_information": 5,
            "higher_risk_change": 3,
        }[family],
        user_info_available=family == "missing_information",
        user_asked=False,
        info_resolved=False,
    )


def generate_scenarios(count: int, seed: int, split: str) -> list[Scenario]:
    """Generate reproducible, non-identical scenario combinations."""

    if count < 1:
        raise ValueError("count must be positive")
    if split not in {"dev", "eval"}:
        raise ValueError("split must be dev or eval")
    scenarios: list[Scenario] = []
    offset = 0 if split == "dev" else 10_000
    for index in range(count):
        scenario_seed = _scenario_seed(seed, index + offset)
        rng = random.Random(scenario_seed)
        family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
        scenarios.append(
            Scenario(
                scenario_id=f"{split}-{index:04d}",
                family=family,
                seed=scenario_seed,
                initial_state=_initial_state(family, rng),
            )
        )
    return scenarios


def _base_evidence(state: WorkflowState) -> bool:
    if state.configuration_suspected:
        return state.candidates_inspected >= 1 and state.configuration_inspected
    return state.candidates_inspected >= 1 and (
        state.direct_definition_found or state.call_sites_inspected or state.info_resolved
    )


def minimum_evidence(state: WorkflowState) -> bool:
    """Mechanical minimum before a change can be attempted."""

    if state.change_attempted or not _base_evidence(state):
        return False
    if state.configuration_suspected and not state.configuration_inspected:
        return False
    if state.error_trace_present and not state.error_inspected:
        return False
    if state.risk_level == "high" and not state.tests_run:
        return False
    if state.task_type == "missing_information" and not state.info_resolved:
        return False
    return True


def legal_actions(state: WorkflowState) -> tuple[str, ...]:
    """Return mechanically admissible actions, without ranking them."""

    if state.change_attempted:
        return ()
    actions: list[str] = []
    if state.candidates_inspected < state.candidate_count:
        actions.append("inspect_candidate")
    if not state.direct_definition_found and state.search_attempts < 2:
        actions.append("search_definition")
    if not state.call_sites_inspected and (state.direct_definition_found or state.candidates_inspected > 0):
        actions.append("search_call_sites")
    if state.tests_available and not state.tests_inspected:
        actions.append("inspect_tests")
    if state.tests_available and state.tests_inspected and not state.tests_run:
        actions.append("run_tests")
    if state.error_trace_present and not state.error_inspected:
        actions.append("inspect_error")
    if state.configuration_suspected and not state.configuration_inspected:
        actions.append("inspect_configuration")
    if minimum_evidence(state):
        actions.append("implement_change")
    if state.user_info_available and not state.user_asked:
        actions.append("ask_user")
    actions.append("escalate_reasoning")
    return tuple(
        action
        for action in actions
        if not (action == state.previous_action and state.repeated_action_count >= 2)
    )


def validate_action(state: WorkflowState, action: str) -> ValidationResult:
    if action not in ACTION_SPACE:
        return ValidationResult(action, False, "unknown action is outside the fixed action space")
    if action not in legal_actions(state):
        reason = (
            "non-progressing action repetition limit reached"
            if action == state.previous_action and state.repeated_action_count >= 2
            else "action violates the state-dependent hard constraints"
        )
        return ValidationResult(action, False, reason)
    return ValidationResult(action, True, "admissible")


def acceptable_actions(scenario: Scenario, state: WorkflowState) -> tuple[str, ...]:
    """Return all legal actions that advance or safely resolve this scenario."""

    legal = legal_actions(state)
    acceptable: list[str] = []
    for action in legal:
        if action == "escalate_reasoning":
            if state.task_type == "missing_information" or state.ambiguity_level >= 4:
                acceptable.append(action)
            continue
        if action == "ask_user":
            acceptable.append(action)
            continue
        if action == "implement_change":
            acceptable.append(action)
            continue
        if action == "inspect_configuration":
            acceptable.append(action)
            continue
        if action == "inspect_error":
            acceptable.append(action)
            continue
        if action == "inspect_tests":
            if state.task_type in {"test_driven", "higher_risk_change", "ambiguous_bug"} or state.ambiguity_level >= 2:
                acceptable.append(action)
            continue
        if action == "run_tests":
            acceptable.append(action)
            continue
        if action == "search_call_sites":
            if state.ambiguity_level >= 2 or state.risk_level == "high":
                acceptable.append(action)
            continue
        if action == "search_definition":
            if state.task_type not in {"configuration_issue", "missing_information"}:
                acceptable.append(action)
            continue
        if action == "inspect_candidate":
            if state.task_type != "missing_information" or state.info_resolved:
                acceptable.append(action)
    return tuple(acceptable)


def apply_action(scenario: Scenario, state: WorkflowState, action: str) -> TransitionResult:
    """Apply a legal action and return a deterministic next state."""

    validation = validate_action(state, action)
    if not validation.legal:
        raise ValueError(validation.reason)
    updates: dict[str, Any] = {
        "previous_action": action,
        "repeated_action_count": state.repeated_action_count + 1 if state.previous_action == action else 0,
    }
    if action == "inspect_candidate":
        updates["candidates_inspected"] = min(state.candidate_count, state.candidates_inspected + 1)
        updates["change_confidence"] = min(3, state.change_confidence + 1)
        updates["ambiguity_level"] = max(0, state.ambiguity_level - 1)
    elif action == "search_definition":
        updates["search_attempts"] = state.search_attempts + 1
        if scenario.family not in {"configuration_issue", "missing_information"}:
            updates["direct_definition_found"] = True
            updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "search_call_sites":
        updates["call_sites_inspected"] = True
        updates["ambiguity_level"] = max(0, state.ambiguity_level - 2)
        updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "inspect_tests":
        updates["tests_inspected"] = True
        updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "run_tests":
        updates["tests_run"] = True
        updates["tests_passing"] = state.change_attempted or scenario.family == "configuration_issue"
        updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "inspect_error":
        updates["error_inspected"] = True
        updates["ambiguity_level"] = max(0, state.ambiguity_level - 2)
        updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "inspect_configuration":
        updates["configuration_inspected"] = True
        updates["ambiguity_level"] = max(0, state.ambiguity_level - 2)
        updates["change_confidence"] = min(3, state.change_confidence + 1)
    elif action == "ask_user":
        updates.update(
            user_asked=True,
            user_info_available=False,
            info_resolved=True,
            candidate_count=2,
            direct_definition_found=True,
            ambiguity_level=1,
            change_confidence=1,
        )
    elif action == "implement_change":
        updates["change_attempted"] = True
        updates["tests_passing"] = True if state.tests_available else state.tests_passing
    elif action == "escalate_reasoning":
        next_state = replace(state, **updates)
        return TransitionResult(next_state, True, "escalated")
    next_state = replace(state, **updates)
    if action == "implement_change":
        return TransitionResult(next_state, True, "completed")
    progress = next_state != state
    return TransitionResult(next_state, progress, None)


def serialize_scenarios(scenarios: Iterable[Scenario]) -> list[dict[str, Any]]:
    return [scenario.serializable() for scenario in scenarios]
