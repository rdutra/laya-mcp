# Claude Code

Add a project-scoped stdio server, replacing the executable with its absolute path:

```bash
claude mcp add laya --scope project -- \
  /absolute/path/to/laya-mcp/.venv/bin/laya-mcp
```

Or copy [`examples/claude-code/.mcp.json`](../examples/claude-code/.mcp.json) to the
project root and update the absolute path. Claude Code asks for approval before using
project-scoped MCP servers.

Model loading is lazy, so process startup and tool discovery do not pay the cold
load. Configure Claude Code's MCP/request timeout high enough for the first inference
tool call, which can take tens of seconds:

```bash
MCP_TIMEOUT=90000 claude
```

Verify with `claude mcp list` or `/mcp`.

Current commands and scope behavior: [Anthropic's Claude Code MCP documentation](https://docs.anthropic.com/en/docs/claude-code/mcp).
