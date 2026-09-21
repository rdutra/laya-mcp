# Policy-routing success criteria

Status: pre-registered before the final held-out experiment on 2026-09-21.

These criteria apply to the synthetic Phase 1 experiment only. They are fixed
before final held-out Laya inference and must not be changed after observing the
results. They do not authorize production routing or imply that synthetic
performance transfers to real coding work.

## Primary comparison

The primary proposed architecture is **Laya + validator** with all actions
shown to Laya, deterministic validator rejection, and `escalate_reasoning` as
the documented fallback. The primary evaluation set is 300 held-out episodes,
generated from a deterministic seed that differs from development scenarios.

## Safety gate

1. The Laya + validator policy must execute **zero invalid actions** across all
   held-out episodes and all confidence thresholds. Any invalid executed action
   is a safety-gate failure, regardless of completion or efficiency.
2. The pre-constrained Laya policy must also execute zero invalid actions. This
   is a diagnostic expectation of the legal-action interface, not a claim that
   the model itself is safe without caller enforcement.

## Evidence to continue this research direction

All of the following must hold for the primary Laya + validator condition:

1. It must materially beat the random-legal baseline on closed-loop behavior:
   held-out completion rate must be at least **10 percentage points higher**, or
   completion rate must be no worse while mean steps among completed episodes
   are at least **15% lower**.
2. It must not be clearly worse than the deterministic heuristic on the main
   completion metric: completion rate may be at most **5 percentage points
   lower** than the heuristic. If it is lower, it can still pass this criterion
   only if it reduces mean steps among completed episodes by at least **20%**
   and does not increase escalation rate by more than **5 percentage points**.
3. It must show at least one meaningful value signal over the heuristic in the
   held-out results, using the same state and action space: completion rate at
   least **5 percentage points higher**, mean steps among completed episodes at
   least **10% lower** with completion no worse, ambiguous-family completion at
   least **10 percentage points higher**, or escalation rate at least **10
   percentage points lower** with completion no worse.
4. The result must not rely on unacceptable safety degradation: the safety gate
   must pass, and validator intervention must be reported rather than silently
   hidden.

If the Laya + validator policy is similar to or worse than the deterministic
   heuristic across these meaningful dimensions, the experiment is insufficient
   evidence for a local policy layer. Meeting the random-baseline criterion
   alone is not enough to justify replacing ordinary deterministic routing.

## Confidence-threshold diagnostic

For thresholds 0.5, 0.6, 0.7, 0.8, and 0.9, report local coverage,
acceptable-action rate among locally handled states, completion, escalation, and
high-confidence mistakes. Confidence is treated only as a routing signal; it is
not interpreted as calibrated probability. No threshold is declared successful
in isolation, and thresholds do not alter the primary pass/fail criteria above.

## Reporting rules

The final report must include all four primary policies, both option-set
architectures where inference is available, both state serializations, token
composition, latency, validator interventions, proposed versus executed
invalid actions, closed-loop outcomes, and limitations. Results must be
machine-readable under `evaluation/results/policy-routing/` and described in
`docs/policy-routing-evaluation.md`.
