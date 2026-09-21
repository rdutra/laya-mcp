# Lazy lifecycle measurements

The v0.1.0 release candidate uses a lazy backend façade. Creating the MCP process,
discovering tools, and calling `info` do not load model weights. The first inference
request initializes the configured model once under a lock; later requests reuse the
resident backend. The backend's existing inference lock remains in place.

## Local measurement

Command:

```bash
TMPDIR=/tmp PYTHONPATH=src python benchmarks/lifecycle_benchmark.py \
  --model aac6fef/laya-multilingual-coreml-ane-w8 \
  --output /tmp/laya-lifecycle.json
```

Environment: Apple Silicon arm64, Darwin 25.2.0, Python 3.12.12, Laya-CoreML
0.1.0, model `aac6fef/laya-multilingual-coreml-ane-w8`, local model cache. The
measurements are from one development machine and are not universal guarantees.

| Phase | Measured time |
|---|---:|
| historical eager startup (Milestones 1–7) | about 30–32 s |
| process to stdio readiness | 12.5 ms |
| tool discovery plus unloaded `info` | 619.5 ms |
| first `decide` operation, including model initialization | 35,312.4 ms |
| subsequent warm `decide` operation | 16.2 ms MCP operation time |
| backend-reported first inference | 28.6 ms |
| backend-reported warm inference | 9.6 ms |

Before inference, `info` reported:

```json
{
  "initialization_state": "unloaded",
  "metrics": {
    "initialization_attempts": 0,
    "initialization_count": 0,
    "inference_count": 0
  },
  "capabilities": {"max_total_tokens": 96}
}
```

After two successful decisions, it reported `ready`, one initialization attempt and
one successful initialization, two model evaluations, and no inference errors. The
second MCP operation includes stdio/tool envelope overhead, so it is higher than the
backend's measured warm inference.

Codex CLI `0.155.1` reproduced the same state transition through the configured MCP
server: initial `info` was `unloaded`; the first decision initialized W8 in
32,613.313 ms; the second `info` was `ready`; the second decision reported 16.058 ms;
the final `info` still showed one initialization and two inferences. This was a
connectivity/lifecycle smoke test, not an end-to-end productivity experiment.

The same W8 process also exercised all public decision kinds through MCP. Binary
returned `true` with confidence `0.9574` in `25.210 ms`; choice returned `auth` with
confidence `0.9412` in `8.087 ms`; ordered score returned `1.3763` with confidence
`0.3054` in `8.671 ms`. The final lifecycle counters were one initialization and
three successful inference evaluations. These are smoke-test observations, not
quality or calibration claims.

## Failure and retry behavior

An initialization failure leaves the process alive and changes `info` to
`initialization_state: "failed"`. The response exposes a concise error message but
not a traceback. The next inference request may make one new initialization attempt;
there is no hidden retry loop. A successful retry transitions back to `ready` and
clears the last initialization error.

## Client implications

Clients should use a startup timeout appropriate for process startup and a tool
timeout appropriate for the first cold inference. They can call `info` immediately
to inspect static capacity and lifecycle state without paying the model startup
cost. A client that never calls an inference tool never loads model weights.
