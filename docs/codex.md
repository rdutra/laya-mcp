# Codex CLI

Add this to `~/.codex/config.toml` or a trusted project's `.codex/config.toml`, replacing
the executable with its absolute path:

```toml
[mcp_servers.laya]
command = "/absolute/path/to/laya-mcp/.venv/bin/laya-mcp"
startup_timeout_sec = 90
tool_timeout_sec = 30
required = true

[mcp_servers.laya.env]
LAYA_MCP_MODEL = "aac6fef/laya-multilingual-coreml-ane"
```

The longer startup timeout accommodates cold Core ML initialization. Check the
connection with `codex mcp list` or `/mcp` in the Codex TUI.

Codex configuration is only a launch wrapper. The server exposes the same `info`,
`decide`, and `batch_decide` tools to every MCP client and contains no Codex-specific
decision policy.

Current configuration syntax: [official OpenAI MCP documentation](https://developers.openai.com/codex/mcp/).
