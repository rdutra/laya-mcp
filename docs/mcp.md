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
        "LAYA_MCP_MODEL": "aac6fef/laya-multilingual-coreml-ane-w8"
      }
    }
  }
}
```

See [`examples/generic-mcp/mcp.json`](../examples/generic-mcp/mcp.json). The exact
configuration file name and timeout fields are client-specific.

The client owns the server process. One process means one resident model. If a client
stops and restarts the process, model initialization happens again. Startup and tool
discovery are lazy: the process does not load model weights until the first
`decide`, `batch_decide`, or `filter` request. `info` is safe to call while the model
is `unloaded` and reports static capacity metadata for known official model IDs.
Unknown/custom model IDs report unavailable capability metadata until initialization.

`initialization_state` is one of `unloaded`, `loading`, `ready`, or `failed`.
Lifecycle metrics include `initialization_attempts`, successful
`initialization_count`, `initialization_ms`, and a concise
`last_initialization_error` when applicable. A failed initialization does not stop
the MCP process; the next inference request may retry once.

The first inference request performs one initialization attempt under a process-local
lock. Concurrent first requests share that attempt, and successful later requests
reuse the resident model. If initialization fails, the server remains alive and
`info` reports `failed` plus a concise error; the next inference request may retry
once. There is no internal retry loop.

Use a normal process startup timeout and a tool/request timeout longer than the cold
load observed on your machine. Once connected, call `info` to confirm the model,
capacity, compute units, and unloaded lifecycle state before routing work to `decide`
or `batch_decide`.

## Public tools

The stable generic surface is `info`, `decide`, `batch_decide`, and `filter`. The Milestone 1
prototype `classify` tool was removed before the first release rather than preserved
as a compatibility alias. Internal Laya names such as `noul` are not part of the
public request contract.

### `decide`

```json
{
  "request_id": "optional-correlation-id",
  "context": "A small bounded state description.",
  "question": "Does the proposition hold?",
  "decision": {"kind": "binary"},
  "confidence_threshold": 0.8
}
```

`decision` defaults to `{"kind":"binary"}`. A `choice` or `ordered_score`
decision requires at least two ordered option labels:

```json
{
  "context": "The observed impact is moderate and recoverable.",
  "question": "How severe is the impact?",
  "decision": {
    "kind": "ordered_score",
    "options": ["low", "medium", "high"]
  }
}
```

The normalized response contains:

```json
{
  "request_id": "optional-correlation-id",
  "result": true,
  "decision_type": "binary",
  "confidence": 0.91,
  "confidence_threshold": 0.8,
  "needs_escalation": false,
  "details": {"probabilities": {"false": 0.09, "true": 0.91}},
  "backend": "laya-coreml",
  "model": "aac6fef/laya-multilingual-coreml-ane-w8",
  "input_tokens": 40,
  "output_tokens": 0,
  "inference_latency_ms": 8.4
}
```

For `choice`, `result` is an option label. For `ordered_score`, `result` is the
expected numeric position in the supplied option order and detailed probabilities
are keyed by option label.

### `batch_decide`

Shared-context form:

```json
{
  "shared_context": "One bounded state used by every question.",
  "items": [
    {"id": "a", "question": "Does condition A hold?"},
    {
      "id": "b",
      "question": "Which route is best?",
      "decision": {"kind": "choice", "options": ["left", "right"]}
    }
  ],
  "confidence_threshold": 0.75,
  "failure_policy": "fail_fast",
  "response_detail": "compact"
}
```

Independent-context form omits `shared_context` and supplies `context` on every
item. IDs are required and unique. Item order is preserved. An item-level threshold,
when present, overrides the batch threshold.

The response carries backend/model once, ordered `items`, and exact measurable
aggregates:

```json
{
  "backend": "laya-coreml",
  "model": "aac6fef/laya-multilingual-coreml-ane-w8",
  "failure_policy": "fail_fast",
  "response_detail": "compact",
  "items": [
    {
      "id": "a",
      "status": "ok",
      "result": true,
      "decision_type": "binary",
      "confidence": 0.91,
      "needs_escalation": false,
      "input_tokens": 40,
      "output_tokens": 0
    }
  ],
  "metrics": {
    "decision_count": 1,
    "successful_count": 1,
    "failed_count": 0,
    "escalation_count": 0,
    "backend_calls": 1,
    "model_evaluations": 1,
    "total_input_tokens": 40,
    "total_output_tokens": 0,
    "total_inference_latency_ms": 8.4,
    "total_operation_latency_ms": 9.1
  }
}
```

`detailed` adds per-item probability maps and per-item latency when the backend can
measure it. A shared multi-question Laya call has one measurable aggregate latency;
individual question latency is therefore `null`, not estimated.

## Failure policy

Schema and cross-item request errors always fail the MCP call. Under the default
`fail_fast` policy, all items are token-preflighted before any model call; one
oversized item rejects the batch. A later backend/chunk failure fails the call with
the affected chunk IDs.

Under `partial`, invalid items and backend-affected items remain in their original
positions with compact `token_budget_exceeded`, `invalid_decision`, or
`backend_failure` records. Processing continues for other chunks. Failed operations
do not claim token or evaluation metrics that the backend could not confirm.
Laya 0.1 does not expose a recoverable per-question failure inside an otherwise
successful call; a missing or malformed answer is therefore treated as a chunk
failure rather than silently dropping one item.

### `filter`

`filter` is a context-reduction primitive over opaque textual candidates. It accepts
unique, ordered IDs and text plus one shared criterion:

```json
{
  "criterion": "Relevant to fixing player acceleration behavior",
  "candidates": [
    {"id": "movement", "text": "player controller handles acceleration"},
    {"id": "audio", "text": "audio mixer loads music"}
  ],
  "rejection_threshold": 0.9,
  "response_detail": "compact"
}
```

The default compact response contains selected candidate objects, aggregate counts,
visible failures, and measured metrics. It does not contain rejected candidate text
or a per-candidate classification record. `response_detail: "detailed"` adds an
ordered diagnostic entry for every candidate (status, confidence, reason, and
available usage/latency) while retaining the same selected list.

The policy is deliberately asymmetric: a candidate is excluded only when the
binary result is irrelevant (`false`) and confidence is greater than or equal to
the threshold. A relevant result is retained; a missing/low confidence result is
retained as `uncertain_retained`; token-overflow and backend failures are retained
and listed in `failures`. Equality meets the threshold. The server does not enforce
a hidden minimum result count, so an all-rejected result is possible when every
negative is sufficiently confident.

Filtering uses the same shared-context `batch_decide` orchestration and serialized
backend access. It is API batching, not fused or concurrent inference. See
[`filtering.md`](filtering.md) for response-size and quality measurements.

## Confidence and escalation

`confidence` is the model's reported confidence for its returned distribution. It is
not a correctness guarantee, authorization decision, or safety assessment.

`needs_escalation` is deterministic:

- without a threshold it is `false` because no escalation policy was requested;
- with a threshold it is `confidence < threshold`;
- equality satisfies the threshold;
- if a future backend cannot provide confidence, a supplied threshold produces
  `needs_escalation=true` rather than fabricating confidence.

The server only returns this signal. It does not call another model or take an action.

## Batch size and response context

The 96-token ANE capacity applies independently to every question. Aggregate batch
usage may safely exceed 96 and is still reported. Shared-context calls are internally
chunked at 100 questions, the largest size currently verified, while preserving one
logical ordered response.

Compact mode is the default because MCP output becomes primary-agent context. It
omits distributions and per-item latency, avoids returning input context, and places
repeated backend/model metadata only at the envelope. The 100-item compact test
fixture serializes to about 15 KB; detailed mode is intentionally opt-in. Milestone 4
filtering should return a selected subset rather than forwarding all decision records
when the caller does not need them.
