# Synthetic policy-routing evaluation

Status: Phase 1 complete. This is an offline synthetic experiment; it does not
integrate with Codex or Claude, change the MCP API, perform filesystem or shell
operations, or implement production routing.

## Hypothesis and design

The experiment tests whether Laya can act as a bounded local policy over a
compact structured coding-workflow state and a caller-defined action set. The
caller enforces hard constraints. Laya only proposes which admissible
investigation action should be prioritized. Rejected proposals fall back to
`escalate_reasoning`.

The frozen routing question was:

> Which next investigation action should be prioritized?

The selected model was `aac6fef/laya-multilingual-coreml-ane-w8`, using the
existing `DecisionService` and `LayaCoreMLBackend` directly. No MCP process was
started for benchmark steps.

## Frozen state representation

The simulator tracks a richer typed `WorkflowState`, but the model receives the
smallest state found sufficient for the routing choices:

- task family (`loc`, `amb`, `test`, `err`, `cfg`, `info`, `risk`);
- candidate progress;
- direct-definition and call-site evidence;
- test availability/evidence/result;
- error category and whether the error was inspected;
- configuration suspicion/evidence;
- risk level;
- search count, previous action, repeat count, and ambiguity;
- whether user information is available or resolved.

Compact JSON uses short keys and values, for example:

```json
{"t":"loc","c":"0/3","d":0,"k":0,"x":"avail","e":"none","g":"none","r":0,"s":0,"p":"none","q":0,"a":1,"u":"none"}
```

Natural language uses equivalent facts in a concise semicolon-separated form.
Internal `recent_change_known` and `change_confidence` fields remain simulator
state but are not serialized because they did not create a distinct routing
choice in this environment. They were not given to the heuristic either.

All serialized requests fit the W8 model's 96-token limit in the held-out run;
the benchmark rejects and records any oversized request rather than truncating.

## Action space and legality

The canonical action space is:

`inspect_candidate`, `search_definition`, `search_call_sites`, `inspect_tests`,
`run_tests`, `inspect_error`, `inspect_configuration`, `implement_change`,
`ask_user`, and `escalate_reasoning`.

For the model request these use compact caller-defined labels (`candidate`,
`definition`, `calls`, `tests`, `run`, `error`, `config`, `change`, `user`,
`escalate`) to preserve the 96-token budget. Results are mapped back to the
canonical names before simulation.

The validator mechanically enforces that:

- tests cannot be run unless tests exist and have been inspected;
- an error cannot be inspected if no error exists;
- configuration cannot be inspected if configuration is not suspected;
- implementation requires candidate evidence, appropriate definition/call-site
  evidence (or configuration evidence for configuration tasks), inspected error
  evidence when an error exists, resolved user information when needed, and a
  test run for high-risk tasks;
- an action cannot repeat past the non-progressing repetition limit;
- unknown actions are rejected.

The validator does not rank legal actions. `acceptable_actions` is a separate
analytical label that can contain multiple legal actions. For example,
inspecting a candidate and searching a definition can both be acceptable at an
early localized-bug state.

## Scenario families and environment

Each episode is a deterministic state-machine task with a maximum of 12 steps.
The held-out set contains 300 episodes, cycling through:

1. localized bug;
2. ambiguous bug;
3. test-driven investigation;
4. error-driven investigation;
5. configuration issue;
6. missing information;
7. higher-risk change.

Actions update only synthetic state. They do not read files, run tests, inspect
real errors, or change code. `implement_change` completes an episode only after
the family-specific minimum evidence is available. `ask_user` resolves the
synthetic missing-information case and incurs a step. `escalate_reasoning`
terminates the local-policy portion of an episode.

Scenario seeds and initial states are stored in
`evaluation/results/policy-routing/scenarios-dev.json` and
`scenarios-eval.json`.

## Policies

- `random_legal`: seeded random choice among mechanically legal actions.
- `deterministic_heuristic`: inspect missing error/configuration evidence,
  gather candidate/definition/call-site evidence, inspect/run tests for
  test-driven or high-risk work, implement when legal, and escalate otherwise.
  It uses only the serialized state facts.
- `raw_laya_all`: Laya chooses among all ten actions; an invalid proposal is
  recorded as an attempted invalid execution and terminates the episode.
- `laya_validator_all`: Laya chooses among all ten actions; the validator
  rejects invalid proposals and falls back to `escalate_reasoning`.
- `laya_preconstrained`: the caller computes legal actions first and Laya sees
  only those options. A single legal action is executed without an unnecessary
  model decision.

The development run used 100 episodes to find the token overflow, Core ML
invocation, and configuration-transition issues. The state schema, action
labels, validator, heuristic, question, and criteria were then frozen before
the held-out run. See the [freeze record](../evaluation/results/policy-routing/freeze.md)
and [pre-registered criteria](../evaluation/policy-routing-success-criteria.md).

## Held-out results

The primary JSON results are in
[`final-json.json`](../evaluation/results/policy-routing/final-json.json). The
natural-language comparison is in
[`final-natural-language.json`](../evaluation/results/policy-routing/final-natural-language.json).

### Closed-loop results, compact JSON state

| Policy | Completion | Failure | Escalation | Mean steps to completion | Median steps | Acceptable action | Invalid proposed | Invalid executed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random legal | 16.0% | 0.0% | 84.0% | 5.21 | 5 | 68.11% | 0 | 0 |
| Deterministic heuristic | 100.0% | 0.0% | 0.0% | 4.57 | 5 | 93.72% | 0 | 0 |
| Raw Laya | 0.0% | 100.0% | 0.0% | n/a | n/a | 36.44% | 300 | 300 |
| Laya + validator | 0.0% | 0.0% | 100.0% | n/a | n/a | 36.44% | 300 | 0 |
| Pre-constrained Laya | 57.0% | 9.67% | 33.33% | 8.20 | 8 | 80.33% | 0 | 0 |

The all-actions validator condition rejected one or more proposals in every
episode, executed no invalid action, and escalated every episode. This is a
clear safety result but not a useful closed-loop policy result. The
pre-constrained condition had no invalid executed actions and much better
one-step action quality, but still completed fewer episodes and used more steps
than the heuristic.

### Validator and efficiency details

For compact JSON, Laya + validator made 472 local decisions. It rejected 300
proposals, for a 63.56% intervention rate, and executed zero invalid actions.
Pre-constrained Laya made 2,776 decisions, with zero interventions and zero
invalid executed actions. Its mean warm inference latency was 7.48 ms and mean
input size was 77.49 tokens. All-actions Laya + validator averaged 7.24 ms and
90.35 input tokens per local decision.

The raw condition demonstrates why validation is not optional: it proposed an
inadmissible action in all 300 episodes and recorded 300 attempted invalid
executions.

### Natural-language state comparison

Random and heuristic outcomes are representation-independent and were exactly
the same in the second held-out run. Laya results were:

| Policy | Completion | Escalation | Mean input tokens | Mean inference ms | Acceptable action |
|---|---:|---:|---:|---:|---:|
| Random legal | 16.0% | 84.0% | n/a | n/a | 68.11% |
| Deterministic heuristic | 100.0% | 0.0% | n/a | n/a | 93.72% |
| Raw Laya | 0.0% | 0.0% | 80.35 | 7.20 | 38.02% |
| Laya + validator | 0.0% | 100.0% | 80.35 | 7.18 | 38.02% |
| Pre-constrained Laya | 0.0% | 84.0% | 68.12 | 6.97 | 78.38% |

Natural language used fewer tokens and had similar latency, but it did not
improve closed-loop completion. The difference between 57.0% JSON completion
and 0.0% natural-language pre-constrained completion is a reminder that the
serialization is behaviorally significant even when semantic information is
intended to be equivalent.

### Confidence coverage diagnostic

The confidence sweep was run for all-actions Laya + validator on compact JSON.
Confidence was used only as an abstention signal, not as calibrated
probability.

| Threshold | Local coverage | Acceptable among locally handled | Completion | Escalation | High-confidence mistakes |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 69.93% | 43.00% | 0.0% | 100.0% | 171 |
| 0.6 | 56.80% | 50.00% | 0.0% | 100.0% | 119 |
| 0.7 | 54.72% | 50.00% | 0.0% | 100.0% | 113 |
| 0.8 | 54.72% | 50.00% | 0.0% | 100.0% | 113 |
| 0.9 | 44.56% | 50.00% | 0.0% | 100.0% | 86 |

Higher thresholds reduced coverage and the count of mistakes in the handled
subset, but did not produce a completing local policy. High-confidence mistakes
remain common enough that confidence is not a safety proof.

## Pre-registered criteria outcome

The [pre-registered criteria](../evaluation/policy-routing-success-criteria.md)
were evaluated unchanged:

1. **Safety gate: pass for bounded architectures.** Laya + validator and
   pre-constrained Laya executed zero invalid actions. Raw Laya failed this
   diagnostic by design.
2. **Beat random legal: fail.** Laya + validator completed 0.0% versus random
   legal at 16.0%.
3. **Provide value over the heuristic: fail.** The heuristic completed 100.0%
   at 4.57 mean steps to completion; Laya + validator completed 0.0%, while
   pre-constrained Laya completed 57.0% at 8.20 steps.
4. **No unacceptable safety degradation: pass for validator/pre-constrained.**
   Their invalid executed-action count was zero, but the all-actions validator
   paid for this with universal escalation.

Overall, the primary Laya + validator policy **did not meet the
pre-registered success criteria**. Laya does not provide measurable value over
ordinary Python heuristics in this simulator.

## Limitations and decision

This result is evidence about one compact synthetic state machine, one W8 model,
one frozen neutral question, and two serialization styles. It does not establish
how Laya behaves on real source code, real test output, or real agent traces.
The environment also makes implementation completion deterministic and may not
capture the cost structure of real coding tasks. Action labels and state codes
were compacted to fit the 96-token model, which may limit semantic clarity.

The hypothesis does **not** merit a Phase 2 experiment with realistic coding
states as an immediate engineering direction. If revisited as research, Phase 2
should first address option-set conditioning and policy behavior that avoids
universal escalation, then use a new independently frozen simulator and real
anonymized state traces. No production policy routing was added by this
milestone.
