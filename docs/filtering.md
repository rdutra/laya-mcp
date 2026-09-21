# Conservative candidate filtering

`filter` is the first context-reduction primitive in `laya-mcp`. The caller supplies
a short criterion and opaque textual candidates. The server evaluates each candidate
with the resident local Laya model and returns a smaller selected list. It does not
read files, run searches, rank by top-K, or take follow-up actions.

## Contract

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

Candidate IDs are required, unique, and preserved in input order. `response_detail`
is `compact` by default or `detailed` for diagnostics. The compact response has:

- `selected`: only the retained candidate `{id,text}` objects;
- `summary`: input, selected, rejected, uncertain-retained, failed-retained, and
  oversized-retained counts;
- `failures`: concise visible errors for candidates that could not be classified;
- `metrics`: model evaluations, backend calls, local Laya token usage, measured
  latency, serialized input/output bytes, and output reduction percentage.

Detailed mode adds one ordered `details` entry per candidate. Details contain status,
confidence, reason, escalation signal, and usage/latency where available. Rejected
candidate text is never repeated in `details`.

## Conservative rejection policy

The default `rejection_threshold` is `0.9`. A candidate is excluded exactly when:

```text
binary result is false AND confidence >= rejection_threshold
```

Every other outcome is retained:

| Outcome | Status | Selected? |
|---|---|---:|
| relevant with sufficient confidence | `retained` | yes |
| confidence below threshold | `uncertain_retained` | yes |
| confidently irrelevant | `rejected` | no |
| backend/malformed result failure | `failed_retained` | yes |
| token budget overflow | `oversized_retained` | yes |

Confidence is the model-reported confidence for its output distribution, not a
correctness guarantee. Threshold equality rejects a negative result. The server does
not invent confidence, retry through another model, or enforce a hidden minimum
selected count; a result with every candidate rejected is valid when the model is
confidently negative. Callers that need a safety floor must implement that policy
explicitly.

Filtering uses `batch_decide` with a shared criterion and the existing exact token
preflight. The default failure policy is safe partial behavior: a failed chunk does
not erase successful decisions, and every affected candidate is retained and listed
in `failures`. An oversized candidate is never truncated or treated as irrelevant.

## Token and inference behavior

The default ANE checkpoint has a 96-token limit per question, not per request. The
criterion is shared state; each candidate is encoded in its own question, so 100
valid candidates can exceed 96 aggregate local tokens and still be valid. Each
candidate still causes one model evaluation. Requests are serialized because Core ML
concurrent inference safety is not established. Shared calls reduce MCP/backend
envelope overhead, not the number of forward evaluations.

The filter question is intentionally concise (`Is this candidate relevant? Candidate:
...`). Long candidates can therefore exceed the per-question limit; safe retention
is preferable to lossy truncation.

## Payload reduction is not model cost savings

`input_payload_bytes`, `output_payload_bytes`, and `output_reduction_percent` measure
the serialized MCP request and response. They are not Codex or Claude token counts,
and they do not predict primary-model cost. The design objective is to keep rejected
classification records out of the expensive agent context while accepting bounded
local classifier work.

## Evaluation and benchmarks

The larger Milestone 5 quality evaluation is documented in
[`evaluation.md`](evaluation.md). It uses held-out file, search, test, error,
action-routing, bounded-decision, and representation cases. Its results do not
justify automatic context exclusion; this document retains the smaller Milestone 4
smoke measurements below for historical comparison.

The original small labeled dataset covering file, search-result, test, and error
relevance remains in `benchmarks/filter_dataset.py`. Run that historical smoke
evaluation on Apple Silicon with:

```bash
python benchmarks/filter_evaluation.py --output /tmp/laya-filter-evaluation.json
```

It reports thresholds `0.5`, `0.6`, `0.7`, `0.8`, and `0.9`, candidate reduction,
precision/recall, uncertain retention, and false-negative rate. The labels are a
small human-defined smoke benchmark, not a general accuracy claim. Because false
negatives are asymmetric and dangerous for filtering, a threshold must not be chosen
by overall accuracy alone; the default `0.9` is intentionally conservative and
should be revisited with a larger labeled corpus.

The development-machine run on the four-workload, 25-candidate labeled set produced:

| Threshold | Reduction | False negatives | FN rate | Precision | Recall | Uncertain retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 40% | 7 | 58.3% | 0.625 | 0.417 | 0 |
| 0.6 | 36% | 6 | 50.0% | 0.600 | 0.500 | 7 |
| 0.7 | 32% | 5 | 41.7% | 0.583 | 0.583 | 10 |
| 0.8 | 28% | 4 | 33.3% | 0.571 | 0.667 | 12 |
| 0.9 | 20% | 2 | 16.7% | 0.556 | 0.833 | 16 |

This small run supports `0.9` as the safer default among the tested thresholds,
because it had the lowest false-negative rate, while still rejecting 5 of 25
candidates. It is not evidence of calibrated production accuracy; the two observed
false negatives at `0.9` are a reason to keep human/primary-model review in the
loop.

Scale and response-size measurements are available with:

```bash
python benchmarks/filter_benchmark.py --output /tmp/laya-filter-scale.json
```

That benchmark covers 10, 50, 100, and 500 candidates and compares compact
`batch_decide` with compact `filter`; at 100 it also measures detailed `filter`.

On the development machine (warm model, serialized Core ML access), the measured
operation latencies were approximately:

| Candidates | `batch_decide` compact | `filter` compact | Selected by filter |
|---:|---:|---:|---:|
| 10 | 84 ms | 81 ms | 10 |
| 50 | 394 ms | 400 ms | 50 |
| 100 | 782 ms | 781 ms | 100 |
| 500 | 3916 ms | 3946 ms | 500 |

The synthetic scale workload produced mostly uncertain classifications at the
conservative `0.9` threshold, so it did not demonstrate candidate rejection. This is
useful: filtering does not guarantee reduction when the model lacks confidence.
At 100 candidates, serialized response sizes were 15,928 bytes for compact
`batch_decide`, 8,693 bytes for compact `filter`, and 30,344 bytes for detailed
`filter`. The compact filter response was therefore about 45% smaller than the
compact batch response even when all candidates were retained; detailed mode is
deliberately unsuitable for normal context crossing.

## Limitations and next step

Filter is not a ranker and has no `max_selected` option. It cannot recover semantic
relevance from a candidate that is too long for the selected model. The model's
confidence is not calibrated correctness, and the small benchmark cannot establish
production recall. Milestone 5 should validate filter behavior against real client
workloads, grow the labeled corpus, and add caller-configurable operational limits
only if measurements show they are needed—without adding domain-specific tools or
unsafe fallback behavior.
