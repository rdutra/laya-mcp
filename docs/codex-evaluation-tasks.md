# Codex A/B task ledger

This is the detailed task ledger for the clean isolated run described in
[`codex-evaluation.md`](codex-evaluation.md). “Success” includes the objective
validator and required-file check; it is stricter than the agent’s final message.
Every row used a fresh repository and the same task fixture in both conditions.

| Task | Category | Vanilla result | Laya result | A time | B time | A input | B input | A output | B output | Laya calls |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `small-slug-bug` | localized bug | pass | pass | 36.9 s | 87.0 s | 96,193 | 145,910 | 707 | 1,736 | 0 |
| `medium-timezone-investigation` | investigation | pass | pass | 46.8 s | 83.3 s | 184,578 | 136,587 | 1,453 | 994 | 0 |
| `large-candidate-triage` | large candidate set | pass | pass | 47.3 s | 88.9 s | 156,288 | 134,165 | 1,382 | 629 | 0 |
| `test-selection` | regression/test selection | pass | pass | 67.5 s | 74.6 s | 278,480 | 97,557 | 2,422 | 732 | 0 |
| `error-investigation` | logs/errors | pass | pass | 83.5 s | 89.7 s | 206,719 | 108,394 | 2,562 | 1,419 | 0 |
| `refactor-settings` | broad refactor | pass | pass | 67.8 s | 93.3 s | 177,157 | 193,974 | 1,570 | 2,185 | 0 |
| `configuration-documentation` | config/docs | fail objective criterion | fail objective criterion | 43.5 s | 71.2 s | 77,225 | 82,275 | 907 | 824 | 0 |
| `multi-step-order-validation` | multi-step coding | pass | pass | 63.7 s | 87.9 s | 225,613 | 142,749 | 1,411 | 1,974 | 0 |

## Objective failures

`configuration-documentation` passed the test suite and documentation assertions,
but the predeclared required source file `app/logging_config.py` was unchanged in
both conditions. The agent updated `.env.example`, `README.md`, and the test file,
but did not implement the requested configuration behavior. This is counted as a
failure in both conditions.

## Changed paths

| Task | Vanilla changed paths | Laya changed paths |
|---|---|---|
| slug | `app/slug.py`, `tests/test_slug.py` | `app/slug.py`, `tests/test_slug.py` |
| timezone | `app/events.py` | `app/events.py`, `tests/test_events.py` |
| large triage | `app/cart.py` | `app/cart.py` |
| test selection | `app/pricing.py` | `app/pricing.py`, `tests/test_pricing.py` |
| error investigation | `app/client.py` | `app/client.py`, `tests/test_client.py` |
| settings refactor | `app/server.py`, `app/settings.py`, `tests/test_settings.py` | same |
| configuration/docs | `.env.example`, `README.md`, `tests/test_logging_config.py` | same |
| order validation | `app/orders.py`, `tests/test_orders.py` | same |

## Tool trajectory

The clean B transcripts contain no `mcp_tool_call` events. Codex did not call
`info`, `decide`, `batch_decide`, or `filter`; it used direct repository exploration
and shell/test commands. The raw JSONL transcripts are available here:

- [clean result summary](../evaluation/results/codex-ab-isolated/results.json)
- [clean raw transcripts](../evaluation/results/codex-ab-isolated/)
- [invalid host-MCP diagnostic run](../evaluation/results/codex-ab/)

The invalid diagnostic run is intentionally not included in the tables because the
host’s unrelated codebase-memory MCP server was inherited by both conditions.

## What this ledger does not establish

- It does not measure Codex token savings caused by Laya; there were no Laya calls.
- It does not establish a candidate-count break-even point.
- It does not show whether a persistent Codex session would amortize model startup.
- It does not generalize from eight small synthetic-but-realistic fixtures to all
  coding repositories.
