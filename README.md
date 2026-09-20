# laya-mcp

`laya-mcp` is a persistent local [Model Context Protocol](https://modelcontextprotocol.io/)
server for bounded decisions powered by
[Laya-CoreML](https://github.com/mizorewww/laya-coreml). It loads one Core ML model at
process startup and keeps it resident for every MCP request.

```text
Codex CLI ──────┐
Claude Code ────┼── MCP stdio ──> laya-mcp ──> Laya-CoreML ──> Core ML / ANE
Other clients ──┘
```

The core server contains no client-specific behavior. Client examples live under
`examples/`, and client notes live under `docs/`.

## Status

The current server provides:

- one model load per server process through the MCP server lifespan;
- a stable generic `info`, `decide`, and `batch_decide` tool surface;
- public `binary`, `choice`, and `ordered_score` decision semantics;
- structured confidence, probabilities, token usage, and inference latency;
- explicit rejection before inference if an input would be truncated;
- serialized access to the resident Core ML model;
- JSON logs on stderr, leaving stdout exclusively for MCP stdio messages.

Milestone 2 adds exact Laya 0.1 token accounting, deterministic reusable
chunk planning, composition/throughput benchmarks, and capability notes. See
[`docs/laya-capabilities.md`](docs/laya-capabilities.md) and
[`docs/token-efficiency.md`](docs/token-efficiency.md).

This is alpha software. Laya decisions are probabilistic signals, not authorization,
safety, legal, medical, or financial judgments.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer for the default ANE checkpoint
- Python 3.12 or 3.13 (3.12 is the currently validated project environment)

The default `aac6fef/laya-multilingual-coreml-ane` checkpoint supports a maximum of
96 total tokens across the question, options, and context. Call `info` rather than
hard-coding that limit if you configure another model.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On first startup, Laya-CoreML may download the configured Hugging Face model before
loading it. Inference is local after the model is present. To require a cached model
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

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LAYA_MCP_MODEL` | `aac6fef/laya-multilingual-coreml-ane` | Hugging Face ID or local directory |
| `LAYA_MCP_REVISION` | unset | Optional model revision |
| `LAYA_MCP_LOCAL_FILES_ONLY` | `false` | Disable model downloads/lookups |
| `LAYA_MCP_COMPUTE_UNITS` | model default | `all`, `cpu`, `cpu_gpu`, or `cpu_ne` |
| `LAYA_MCP_LOG_LEVEL` | `INFO` | Structured stderr log level |

The model is loaded before the MCP server accepts requests. Give clients a startup
timeout comfortably above the model's measured initialization time on your machine.
The original development machine measured about 30.35 seconds; that is a local
observation, not a general performance claim.

## Client setup

- Generic MCP: [`docs/mcp.md`](docs/mcp.md)
- Codex CLI: [`docs/codex.md`](docs/codex.md)
- Claude Code: [`docs/claude-code.md`](docs/claude-code.md)
- Agent usage patterns: [`docs/coding-agents.md`](docs/coding-agents.md)

## Development

```bash
python -m pytest
```

The real-model benchmark is opt-in and runs only on a compatible Apple Silicon
machine:

```bash
python benchmarks/token_composition.py
python benchmarks/token_efficiency.py --output /tmp/laya-token-efficiency.json
```

Unit and MCP integration tests use fake backends and do not load model weights. A
real smoke test should be run on a compatible Apple Silicon Mac before release.

## Design constraints

- no cloud LLM fallback;
- no autonomous action loop;
- no filesystem mutation or command execution;
- no implicit input truncation;
- every decision remains bounded and independently token-validated;
- backend and MCP transport remain separate so other local Laya backends can be added.

## License

Apache-2.0. Laya-CoreML and model weights are separate dependencies with their own
licenses, notices, and model cards.
