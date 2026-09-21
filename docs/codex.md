# Codex CLI integration

Codex stores MCP configuration in `~/.codex/config.toml`, or in a trusted project's
`.codex/config.toml`. The current official OpenAI documentation supports the
`[mcp_servers.<name>]` table, STDIO `command`, nested `env`, and per-server
`startup_timeout_sec`/`tool_timeout_sec` settings.

For a persistent user configuration, add the following and replace the executable
with its absolute path:

```toml
[mcp_servers.laya]
command = "/absolute/path/to/laya-mcp/.venv/bin/laya-mcp"
startup_timeout_sec = 90
tool_timeout_sec = 120
required = true

[mcp_servers.laya.env]
LAYA_MCP_MODEL = "aac6fef/laya-multilingual-coreml-ane-w8"
# Set this only when the model is already present in the local Laya cache.
# LAYA_MCP_LOCAL_FILES_ONLY = "true"
```

The W8 model is selected here because it was faster and smaller than the FP16 ANE
baseline with identical measured Milestone 6 quality. This is an explicit client
configuration choice, not a change to laya-mcp's production default.

The longer tool timeout accommodates the first cold Core ML initialization. Process
startup and tool discovery are fast because model loading is lazy; `info` initially
reports `initialization_state: "unloaded"`. Check the
connection with `codex mcp list`, or use `/mcp` in the Codex TUI. The CLI also
supports the equivalent registration command:

```bash
codex mcp add laya --env LAYA_MCP_MODEL=aac6fef/laya-multilingual-coreml-ane-w8 \
  -- /absolute/path/to/laya-mcp/.venv/bin/laya-mcp
```

When using `codex mcp add`, edit the generated `config.toml` afterward if you need
the 90-second startup timeout, 120-second tool timeout, `required = true`, or
`LAYA_MCP_LOCAL_FILES_ONLY`.

The current default Codex MCP startup timeout is 10 seconds, which is sufficient for
the lazy process startup but does not cover a cold model load. Keep the per-tool
timeout above the measured cold initialization. Required servers use their configured
startup timeout; optional-server catalog startup is separately governed by Codex's
`mcp_optional_startup_grace_ms` setting.

## Behavioral guidance

Give Codex access to the server and concise guidance; do not force a tool call for
every task. The following text is suitable for a project instruction or experiment
prompt:

```text
Use laya-mcp for cheap, bounded, local advisory decisions when it may avoid
unnecessary primary-model reasoning or context. It is useful for simple binary or
choice decisions and triage of already-discovered candidate lists. Do not delegate
architecture, complex debugging, security-sensitive or destructive authorization,
nuanced semantic reasoning, code generation, or final correctness judgments.
Confidence is not calibrated probability. `filter` is advisory: never treat an
excluded candidate as unavailable; inspect it when evidence or task importance
warrants it. Do not call laya-mcp merely to demonstrate it, and do not let a result
override direct evidence.
```

This guidance reflects the evaluation evidence: the ANE models are fast, but
relevance and error decisions can be confidently wrong. `batch_decide` is a safer
candidate for exploratory triage because it does not itself remove context.
`filter` should be treated as prioritization advice only.

## Verification checklist

```bash
codex mcp list
codex mcp get laya
```

Then start a new Codex session and confirm that `info`, `decide`, `batch_decide`,
and `filter` appear in the available MCP tools. Call `info` once to confirm the
selected model and that initialization is initially `unloaded`; the first inference
call changes it to `ready` after the cold load.

For an explicit local-files-only setup, first start `laya-mcp` or run the model
benchmark so the model artifact is cached, then set
`LAYA_MCP_LOCAL_FILES_ONLY=true`. Otherwise leave it unset and allow the first
startup to resolve the configured model artifact.

## End-to-end experiment

The reproducible A/B runner creates a fresh temporary Git repository for each task
and condition, then validates the resulting change outside Codex:

```bash
python benchmarks/run_codex_ab.py \
  --condition both \
  --output evaluation/results/codex-ab/results.json
```

The runner uses the same task prompt and fixture for vanilla Codex and Codex with
the W8 server available. The laya condition receives only the general guidance
above; it is not instructed to call a specific tool. It records Codex JSONL usage,
MCP calls, local Laya decision counts/tokens/latency, command/search/read proxies,
elapsed time, changed files, and objective validation results.

See [`docs/codex-evaluation.md`](codex-evaluation.md) for methodology and results.

Codex configuration is only a launch wrapper. The server exposes the same `info`,
`decide`, `batch_decide`, and conservative `filter` tools to every MCP client and
contains no Codex-specific decision policy.

Current configuration syntax: [official OpenAI MCP documentation](https://developers.openai.com/codex/mcp/).
