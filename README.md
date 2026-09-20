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

Milestone 1 provides:

- one model load per server process through the MCP server lifespan;
- generic `info` and `classify` tools;
- `noul`, `choice`, and ordered `score` decisions;
- structured confidence, probabilities, token usage, and inference latency;
- explicit rejection before inference if an input would be truncated;
- serialized access to the resident Core ML model;
- JSON logs on stderr, leaving stdout exclusively for MCP stdio messages.

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

### `classify`

Arguments:

- `context`: the small bounded text to evaluate;
- `question`: the decision instruction;
- `decision_type`: `noul`, `choice`, or `score`;
- `options`: omitted for `noul`; at least two labels for `choice` or ordered levels
  for `score`.

Example arguments:

```json
{
  "context": "player_controller.gd contains movement, jumping, acceleration and player animation logic.",
  "question": "Is this file relevant to fixing a player movement bug?",
  "decision_type": "noul"
}
```

The result includes the selected result, calibrated confidence, applicable
probabilities, Laya input/output token counts, and measured synchronous inference
latency. The server does not read files: clients must provide the bounded context.

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

Unit and MCP integration tests use fake backends and do not load model weights. A
real smoke test should be run on a compatible Apple Silicon Mac before release.

## Design constraints

- no cloud LLM fallback;
- no autonomous action loop;
- no filesystem mutation or command execution;
- no implicit input truncation;
- one bounded inference request produces one typed decision;
- backend and MCP transport remain separate so other local Laya backends can be added.

## License

Apache-2.0. Laya-CoreML and model weights are separate dependencies with their own
licenses, notices, and model cards.

