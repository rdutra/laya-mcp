# Policy-routing experiment freeze

Freeze point: 2026-09-21, after the 100-episode development run and before
held-out results were generated.

Frozen artifacts:

- success criteria: `evaluation/policy-routing-success-criteria.md`
- simulator/schema: `benchmarks/policy_routing/simulator.py`
- benchmark runner and service integration: `benchmarks/policy_routing/benchmark.py`
- action space: the ten canonical actions in `ACTION_SPACE`
- model option labels: `candidate`, `definition`, `calls`, `tests`, `run`,
  `error`, `config`, `change`, `user`, `escalate`
- routing question: `Which next investigation action should be prioritized?`
- validator rules and deterministic heuristic
- development seed: `20260921`
- held-out generation: the same seed with the held-out split offset, 300 episodes

The development run was used to catch token overflow, Core ML invocation issues,
and simulator transition/legality bugs. No final held-out result was used to
change these artifacts or the pre-registered thresholds.
