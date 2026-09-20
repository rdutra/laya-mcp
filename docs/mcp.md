# Generic MCP setup

`laya-mcp` uses the standard MCP stdio transport. Configure an MCP client to execute
the absolute path to the installed `laya-mcp` console script. A portable shape is:

```json
{
  "mcpServers": {
    "laya": {
      "command": "/absolute/path/to/laya-mcp/.venv/bin/laya-mcp",
      "args": [],
      "env": {
        "LAYA_MCP_MODEL": "aac6fef/laya-multilingual-coreml-ane"
      }
    }
  }
}
```

See [`examples/generic-mcp/mcp.json`](../examples/generic-mcp/mcp.json). The exact
configuration file name and timeout fields are client-specific.

The client owns the server process. One process means one resident model. If a client
stops and restarts the process, model initialization happens again.

Use a startup timeout longer than the cold load observed on your machine. Once
connected, call `info` to confirm the model, capacity, compute units, and initialization
duration before routing work to `classify`.

