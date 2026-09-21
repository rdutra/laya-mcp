# v0.1.0 release checklist

This is a local release-candidate checklist. It does not publish, tag, or push a
release.

## Demonstrated

- MCP stdio startup and tool discovery without loading model weights;
- lazy initialization on the first `decide`, `batch_decide`, or `filter` request;
- one successful resident backend per process, with serialized inference;
- static 96-token metadata for the known ANE models before loading;
- local binary, choice, ordered-score, batching, and failure semantics;
- W8 compatibility with the tested decision types and existing evaluation workload.

## Experimental or not demonstrated

`filter` remains advisory/experimental. Confidence is not calibrated, and automatic
silent context exclusion is not justified. The Codex experiment found no natural Laya
calls and no attributable primary-model token/context reduction. Claude replication,
automatic routing, and improved coding-agent task quality are not demonstrated.

## Local verification

On Apple Silicon macOS:

```bash
pytest
ruff check src tests benchmarks
mypy src benchmarks
python -m pip check
python -m build
git diff --check
python benchmarks/lifecycle_benchmark.py --model aac6fef/laya-multilingual-coreml-ane-w8
python benchmarks/public_smoke.py --model aac6fef/laya-multilingual-coreml-ane-w8
```

Then run a real MCP client smoke test: call `info` before inference, verify
`unloaded`, make one decision and observe the cold initialization, then call `info`
again and make a second decision to verify `ready` and warm reuse. Record the
machine, Python version, model ID, cache state, and measured timings.

The CI workflow intentionally runs fake-backend/unit checks and packaging on standard
runners. Core ML integration remains a local Apple Silicon check.
