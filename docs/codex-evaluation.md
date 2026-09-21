# Codex + laya-mcp end-to-end evaluation

This milestone tested the product hypothesis end to end: whether a real coding
agent naturally uses laya-mcp to reduce expensive primary-model work while
preserving task quality. It did not script Laya calls and did not change the
public MCP API.

## Result in one sentence

In eight clean A/B task pairs (run before the Milestone 8 lazy lifecycle change),
Codex completed 7/8 tasks in both conditions and made **zero laya-mcp calls** in all
16 runs. The integration was discoverable and healthy, but the agent did not find a
valuable use for it under this guidance and task mix. No primary-model savings can
therefore be attributed to laya-mcp.

The MCP-enabled condition was slower on average because the pre-Milestone-8 server
eagerly paid local model startup per fresh Codex process even when no tool was called.
The current release candidate no longer does that: process startup and `info` are
lazy, while the first inference tool call pays the cold load. The task results remain
historical and are not being rerun as part of release hardening.

## Conditions

### A: vanilla Codex

- Codex CLI `0.155.1`
- authenticated local Codex CLI
- `--ignore-user-config`
- no MCP server configured
- same fixture, prompt, model, permissions, and validation as B

### B: Codex with laya-mcp available

- same Codex CLI and task prompt
- `--ignore-user-config`
- explicit STDIO `laya-mcp` server configuration
- `aac6fef/laya-multilingual-coreml-ane-w8`
- `LAYA_MCP_LOCAL_FILES_ONLY=true`
- 90-second MCP startup timeout and 120-second tool timeout
- concise behavioral guidance, but no instruction to call a particular tool

The guidance told Codex that Laya is local and advisory, that confidence is not
calibrated, that `filter` exclusions are not authoritative, and that architecture,
complex debugging, security-sensitive decisions, code generation, and final
correctness judgments should remain with Codex.

The explicit W8 choice is reproducible and consistent with Milestone 6: W8 was
smaller and faster than ANE FP16 with identical measured isolated-classifier
quality. It did not change the production default.

## Task suite

Each task started from a fresh temporary Git repository with an objective validator
defined before the run. The repository was removed after validation. The suite was
intentionally mixed rather than optimized for Laya:

| Task | Workload | Objective criterion |
|---|---|---|
| `small-slug-bug` | small localized bug | tests pass; `app/slug.py` changes |
| `medium-timezone-investigation` | medium investigation | tests pass; `app/events.py` changes |
| `large-candidate-triage` | many plausible modules | tests pass; only relevant cart code should change |
| `test-selection` | test selection/regression coverage | tests pass; pricing implementation changes |
| `error-investigation` | timeout/error diagnosis | tests pass; client implementation changes |
| `refactor-settings` | broader typed refactor | tests pass; settings/server implementation changes |
| `configuration-documentation` | bounded config/docs task | tests pass; logging implementation, env example, and README change |
| `multi-step-order-validation` | multi-step implementation | tests pass; order implementation changes |

The source fixture and prompts are in
[`benchmarks/codex_task_suite.py`](../benchmarks/codex_task_suite.py). The runner
is [`benchmarks/run_codex_ab.py`](../benchmarks/run_codex_ab.py).

## Aggregate results

| Condition | Success | Mean elapsed | Total elapsed | Total Codex input | Total Codex output | laya-mcp calls |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 7/8 | 57.1 s | 457.0 s | 1,402,253 | 12,414 | 0 |
| Laya available | 7/8 | 84.5 s | 675.8 s | 1,041,611 | 10,493 | 0 |

The lower aggregate input/output count in B is not attributable to Laya: no Laya
tool ran, and the task-level differences are nondeterministic Codex trajectories.
This sample is not large enough to establish a token or latency effect.

The B elapsed time includes approximately one resident-model initialization per
Codex process because this experiment used the pre-lazy server. That startup artifact
does not describe the current lifecycle implementation; a current run would still
be expected to show zero Laya calls and no demonstrated task-level savings.

## Per-task results

| Task | Vanilla | Laya available | Vanilla elapsed | Laya elapsed | Vanilla input | Laya input |
|---|---|---|---:|---:|---:|---:|
| small slug bug | pass | pass | 36.9 s | 87.0 s | 96,193 | 145,910 |
| timezone investigation | pass | pass | 46.8 s | 83.3 s | 184,578 | 136,587 |
| large candidate triage | pass | pass | 47.3 s | 88.9 s | 156,288 | 134,165 |
| test selection | pass | pass | 67.5 s | 74.6 s | 278,480 | 97,557 |
| error investigation | pass | pass | 83.5 s | 89.7 s | 206,719 | 108,394 |
| settings refactor | pass | pass | 67.8 s | 93.3 s | 177,157 | 193,974 |
| configuration/docs | **fail objective criterion** | **fail objective criterion** | 43.5 s | 71.2 s | 77,225 | 82,275 |
| order validation | pass | pass | 63.7 s | 87.9 s | 225,613 | 142,749 |

The configuration/docs task passed its pytest and documentation assertions but did
not modify `app/logging_config.py`, which was a predeclared required path. Both
conditions made the same quality mistake. This is a quality failure, not a Laya
failure.

The runner also records output tokens, changed files, validator output, command
proxies, raw MCP calls, and raw Codex JSONL. The canonical machine-readable result
is [`evaluation/results/codex-ab-isolated/results.json`](../evaluation/results/codex-ab-isolated/results.json).

## Tool usage and context-flow proxies

Clean condition B had zero calls to `info`, `decide`, `batch_decide`, or `filter`.
Consequently:

- Laya model evaluations: `0`
- Laya local input/output tokens: `0`
- Laya inference latency: `0 ms`
- MCP response bytes: not applicable
- candidates reduced: `0`
- files/search results reduced: `0`

The runner records Codex command executions and simple search/read command proxies,
but these tasks mostly used direct shell reads and the model’s built-in repository
context. These are proxies, not Codex token accounting.

The separate lifecycle connectivity smoke test used Codex with the W8 server and
explicitly called `info → decide → info → decide → info`. `info` first reported
`unloaded` with zero initialization attempts; the first decision returned successfully
after 32,613.313 ms initialization; the next `info` reported `ready` with one
initialization and one inference; the second decision reused the model (16.058 ms
reported inference), and the final `info` showed two inferences with one
initialization. This proves lifecycle configuration and reuse, not task value.

## Useful, neutral, unnecessary, and harmful calls

There were no laya-mcp calls in the clean experiment, so no call can honestly be
classified as useful, neutral, misleading-but-recovered, or harmful. The observed
behavior is instead **non-use**: Codex solved these tasks directly and did not judge
the delegation overhead worthwhile.

This is informative rather than a failure of the MCP transport. It suggests that
generic availability plus short guidance is not sufficient to make Laya part of a
coding agent’s natural path for small-to-medium tasks.

## Break-even behavior

No empirical break-even point was observed because there were no calls. We therefore
cannot claim that two, ten, fifty, or one hundred candidates are worthwhile for a
real agent. The only measured product-level cost is the opposite direction: in a
fresh CLI process, B adds model initialization time without reducing work.

A follow-up experiment should keep one Codex session alive across multiple tasks and
present a genuinely large, already-discovered candidate list where `batch_decide`
can be considered without allowing Laya to remove evidence. That would separate
startup amortization from agent judgment.

## Contamination and invalidated run

An initial run accidentally inherited the host’s unrelated `codebase-memory-mcp`
server. It produced MCP calls in both conditions but **zero laya-mcp calls**, so it
is not used for the results above. Its raw artifacts remain under
`evaluation/results/codex-ab/` as a diagnostic record and must not be interpreted as
the A/B experiment.

The corrected runner uses Codex `--ignore-user-config` and configures only the
explicit laya server in condition B. A separate smoke test verified that this still
exposes laya-mcp correctly.

## Interpretation

1. Task quality was unchanged: 7/8 in both conditions.
2. No primary-model work was reduced by Laya because Codex never called it.
3. The integration did not harm task quality in this sample, but added cold-start
   cost when each condition ran in a fresh process.
4. `batch_decide` and `filter` were not naturally discovered as useful by Codex in
   these tasks. This does not prove they are useless for large candidate lists.
5. Automatic filtering is not justified. Codex did not use it, and prior semantic
   evaluation showed that silent exclusions remain unsafe.
6. The evidence does not justify automatic routing or a Codex-specific core API.

This is a small, nondeterministic sample. It is evidence about this Codex version,
these prompts, fixtures, and local runtime—not a general claim about all agents or
repositories.

## Reproduction

```bash
python benchmarks/run_codex_ab.py \
  --condition both \
  --output evaluation/results/codex-ab-isolated/results.json
```

The command requires authenticated Codex CLI access, a cached local W8 model, and
an Apple Silicon macOS environment. It uses temporary repositories and never gives
laya-mcp filesystem or shell access.
