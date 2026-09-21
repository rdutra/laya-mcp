# laya-mcp

`laya-mcp` is a persistent local [Model Context Protocol](https://modelcontextprotocol.io/)
server for bounded decisions powered by
[Laya-CoreML](https://github.com/mizorewww/laya-coreml). It lazily loads one local
Core ML model on the first inference request and keeps it resident for the process.

```text
Codex CLI ──────┐
Claude Code ────┼── MCP stdio ──> laya-mcp ──> Laya-CoreML ──> Core ML / ANE
Other clients ──┘
```

The core server contains no client-specific behavior. Client examples live under
`examples/`, and client notes live under `docs/`.

## Status

The current server provides:

- fast MCP startup and tool discovery without model initialization;
- one model load per server process, on first use, with serialized inference;
- a stable generic `info`, `decide`, `batch_decide`, and conservative `filter` tool surface;
- public `binary`, `choice`, and `ordered_score` decision semantics;
- structured confidence, probabilities, token usage, and inference latency;
- explicit rejection before inference if an input would be truncated;
- serialized access to the resident Core ML model;
- JSON logs on stderr, leaving stdout exclusively for MCP stdio messages.

Milestones 2–4 add exact Laya 0.1 token accounting, deterministic reusable chunk
planning, composition/throughput benchmarks, stable decision primitives, and the
conservative context-reduction `filter` primitive. See
[`docs/laya-capabilities.md`](docs/laya-capabilities.md) and
[`docs/token-efficiency.md`](docs/token-efficiency.md), plus
[`docs/filtering.md`](docs/filtering.md).

This is alpha software. Laya decisions are probabilistic signals, not authorization,
safety, legal, medical, or financial judgments.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer for the default ANE checkpoint
- Python 3.12 or 3.13 (3.12 is the currently validated project environment)

The default `aac6fef/laya-multilingual-coreml-ane-w8` checkpoint supports a maximum of
96 total tokens across the question, options, and context. Call `info` rather than
hard-coding that limit if you configure another model.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On the first inference request, Laya-CoreML may download the configured Hugging Face
model before loading it. Inference is local after the model is present. To require a cached model
or local model directory, set `LAYA_MCP_LOCAL_FILES_ONLY=true` or set
`LAYA_MCP_MODEL` to that directory.

Run the stdio server:

```bash
laya-mcp
```

Do not print application output to stdout when using stdio transport. Server logs are
structured JSON on stderr.

## Tools

### `info`

Reports the `laya-mcp` and backend versions, configured model, initialization state,
platform, Python version, Core ML compute units, known model limits, and process-local
metrics. The initialization duration is for this process on this machine.

### `decide`

Arguments:

- `context`: the small bounded text to evaluate;
- `question`: the decision instruction;
- `decision`: optional object with `kind` (`binary`, `choice`, or
  `ordered_score`) and, for non-binary decisions, at least two `options`;
- `confidence_threshold`: optional number from 0 through 1;
- `request_id`: optional caller-defined correlation ID.

Example arguments:

```json
{
  "context": "player_controller.gd contains movement, jumping, acceleration and player animation logic.",
  "question": "Is this file relevant to fixing a player movement bug?",
  "request_id": "decision-17",
  "confidence_threshold": 0.8
}
```

The result includes the normalized result, model confidence, threshold outcome,
applicable probability distribution, model/backend identifiers, local token usage,
and measured inference latency.

### `batch_decide`

`batch_decide` accepts ordered items with unique IDs. Use either:

- `shared_context` with item contexts omitted; or
- an independent `context` on every item with `shared_context` omitted.

The default `fail_fast` policy validates every item before inference and fails the
whole call if any item is invalid. `partial` returns an explicit error entry in the
original position for every failed item. No item is silently omitted.

Compact responses are the default and omit probability maps and per-item latency.
Set `response_detail` to `detailed` when those fields are needed. Backend/model
identifiers and aggregate metrics appear once at the response envelope.

```json
{
  "shared_context": "Choose components relevant to a movement regression.",
  "confidence_threshold": 0.75,
  "items": [
    {"id": "a", "question": "Is the player controller relevant?"},
    {"id": "b", "question": "Is the audio mixer relevant?"}
  ]
}
```

This is MCP/API batching, not concurrent or fused model inference. The default ANE
backend evaluates each question independently while holding the same serialization
lock used by `decide`.

The server does not read files: clients must provide bounded context.

### `filter`

`filter` accepts a criterion and ordered `{id, text}` candidates. It returns only
retained candidates by default, which keeps rejected classification records out of
the primary agent's context. Candidates are excluded only when Laya returns a
binary negative result with confidence at or above the `rejection_threshold` (the
default is `0.9`). Relevant, uncertain, oversized, and failed candidates are
retained. Use `response_detail: "detailed"` for per-candidate diagnostics.

```json
{
  "criterion": "Relevant to fixing player acceleration behavior",
  "rejection_threshold": 0.9,
  "candidates": [
    {"id": "movement", "text": "player controller handles acceleration"},
    {"id": "audio", "text": "audio mixer loads music"}
  ]
}
```

Filtering reports candidate reduction and serialized MCP payload sizes. Those are
local MCP payload measurements, not Codex/Claude token savings. See
[`docs/filtering.md`](docs/filtering.md) for the conservative policy and benchmarks.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LAYA_MCP_MODEL` | `aac6fef/laya-multilingual-coreml-ane-w8` | Hugging Face ID or local directory |
| `LAYA_MCP_REVISION` | unset | Optional model revision |
| `LAYA_MCP_LOCAL_FILES_ONLY` | `false` | Disable model downloads/lookups |
| `LAYA_MCP_COMPUTE_UNITS` | model default | `all`, `cpu`, `cpu_gpu`, or `cpu_ne` |
| `LAYA_MCP_LOG_LEVEL` | `INFO` | Structured stderr log level |

Model loading is lazy: process startup, tool discovery, and `info` do not load model
weights. The first `decide`, `batch_decide`, or `filter` call pays the cold
initialization cost; later calls reuse the resident model. The original development
machine measured about 30–32 seconds for initialization and about 7–10 ms for warm
inference. These are local observations, not universal performance claims.

## Client setup

- Generic MCP: [`docs/mcp.md`](docs/mcp.md)
- Codex CLI: [`docs/codex.md`](docs/codex.md)
- Claude Code: [`docs/claude-code.md`](docs/claude-code.md)
- Agent usage patterns: [`docs/coding-agents.md`](docs/coding-agents.md)
- Evaluation results: [`docs/evaluation.md`](docs/evaluation.md)
- Model comparison: [`docs/model-comparison.md`](docs/model-comparison.md)
- Lifecycle measurements: [`docs/lifecycle.md`](docs/lifecycle.md)
- Release checklist: [`docs/release.md`](docs/release.md)
- Codex end-to-end evaluation: [`docs/codex-evaluation.md`](docs/codex-evaluation.md)

## Development

```bash
python -m pytest
```

The real-model benchmark is opt-in and runs only on a compatible Apple Silicon
machine:

```bash
python benchmarks/token_composition.py
python benchmarks/token_efficiency.py --output /tmp/laya-token-efficiency.json
python benchmarks/evaluate_milestone5.py --output evaluation/results/milestone5.json
```

Unit and MCP integration tests use fake backends and do not load model weights. A
real lifecycle smoke test should be run on a compatible Apple Silicon Mac before
release; see [`docs/release.md`](release.md).
The Milestones 5–6 quality evaluations are intentionally advisory; see
[`docs/evaluation.md`](docs/evaluation.md) before using `filter` for automatic
context exclusion.

## Design constraints

- no cloud LLM fallback;
- no autonomous action loop;
- no filesystem mutation or command execution;
- no implicit input truncation;
- every decision remains bounded and independently token-validated;
- backend and MCP transport remain separate so other local Laya backends can be added.

## Evidence and limitations

Demonstrated: local Core ML inference, the generic MCP decision API, API batching,
deterministic token preflight, conservative failure handling, and MCP interoperability.
The W8 default is supported by the tested hardware/workloads and is not a universal
quality or latency guarantee.

Experimental: `filter`, context-reduction workflows, and agent delegation patterns.
Not demonstrated: automatic Codex or Claude token savings, safe silent context
exclusion, improved coding-agent task quality, or calibrated confidence. See the
negative results in [`docs/evaluation.md`](evaluation.md) and
[`docs/codex-evaluation.md`](codex-evaluation.md).

## License

Apache-2.0. Laya-CoreML and model weights are separate dependencies with their own
licenses, notices, and model cards.
