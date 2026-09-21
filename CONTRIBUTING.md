# Contributing

Thanks for contributing to `laya-mcp`.

## Development environment

The supported development target is Apple Silicon macOS with Python 3.12 or 3.13.
Create an environment and install development dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Unit tests use fake backends and do not require Core ML model weights. Real-model
smoke tests and lifecycle benchmarks require a compatible Apple Silicon machine and
may take roughly 30 seconds on the first inference request.

## Checks

Before opening a change, run:

```bash
pytest
ruff check src tests benchmarks
mypy src benchmarks
python -m pip check
python -m build
git diff --check
```

Keep MCP transport client-agnostic. Do not add filesystem access, shell execution,
cloud fallback, autonomous actions, or client-specific tools to the core server.
Preserve explicit token-budget failures and never silently truncate or omit a
decision. New performance claims must identify the hardware and workload used.

## Pull requests

Describe the user-visible behavior, tests run, and any Apple Silicon-only checks.
Keep evaluation findings reproducible and do not remove negative results to make a
benchmark appear more favorable.
