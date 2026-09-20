# Token efficiency and batching

Laya tokens are local classifier tokens. They are not equivalent to Codex or
Claude input tokens and should not be optimized as if they had the same
economic cost. The product goal is to remove expensive primary-model context
while keeping local classification fast enough for an agent loop.

## What batching changes

`agent.predict(state, questions)` accepts multiple questions. In Laya 0.1,
questions are prepared and evaluated one at a time (`batch_size=1` for the ANE
bundle). A shared state is therefore re-tokenized for every question. Compared
with N separate calls, one multi-question call has:

* the same total Laya input and output token usage;
* the same number of Core ML forward evaluations;
* fewer Python/MCP call boundaries;
* the same confidence/probability values for the tested inputs.

The reusable planner treats the 96-token limit as **per question**. It can
optionally create deterministic chunks by question count, preserving IDs and
order. A chunk's reason is exposed as either `max_questions_per_chunk` or
`all_items_fit_per_question_budget`. A single oversized item is rejected with
the item ID and no partial request is sent.

## Realistic workloads

The benchmark (`benchmarks/token_efficiency.py`) uses short synthetic coding
agent workloads for file relevance, search-result relevance, and test
relevance. It reports separate calls, one shared-state call, and chunks of ten.
For the measured warm ANE run (one Apple Silicon development machine), all
candidate counts 1, 5, 10, 25, 50, and 100 fit per item. Representative
file-relevance results were:

| Candidates | Strategy | Calls | Forward evals | Input tokens | Total ms | ms/decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | separate | 10 | 10 | 568 | 77.994 | 7.799 |
| 10 | one call | 1 | 10 | 568 | 78.775 | 7.878 |
| 10 | chunks of 10 | 1 | 10 | 568 | 79.649 | 7.965 |
| 50 | separate | 50 | 50 | 2,880 | 392.126 | 7.843 |
| 50 | one call | 1 | 50 | 2,880 | 390.693 | 7.814 |
| 50 | chunks of 10 | 5 | 50 | 2,880 | 390.440 | 7.809 |
| 100 | separate | 100 | 100 | 5,770 | 784.868 | 7.849 |
| 100 | one call | 1 | 100 | 5,770 | 774.688 | 7.747 |
| 100 | chunks of 10 | 10 | 100 | 5,770 | 785.345 | 7.853 |

Across the three realistic workloads, one-call and chunked results had zero
decision mismatches and zero mean confidence/probability delta against the
separate-call baseline. The observed warm throughput was roughly 121–129
decisions/second. Initialization was measured separately at about 30.2 seconds
in this run and is paid once per server process.

The measured break-even is operational rather than a large compute gain:
Core ML work dominates after roughly five questions. One call was about 5.5 ms
faster than five separate file calls, effectively tied at ten, and within a few
milliseconds at 25–100. Thus batching is worthwhile primarily because it cuts
request envelopes and coordination overhead; it should not be described as
reducing the N model evaluations.

For the short-state workload, the 100-candidate one-call run used 3,490 local
tokens and took 852.206 ms (8.522 ms/decision), versus 856.161 ms for separate
calls and 866.451 ms for chunks of ten.

These numbers are directional: Core ML compilation, thermal state, model
revision, and machine affect timings. Always rerun the benchmark on a target
machine. The benchmark includes `--workloads short,...` to compare short and
realistic states.

## Recommended strategy

For 10, 50, and 100 candidates, use one shared-state `predict` call when the
caller can tolerate one response envelope. It reduces request count without
reducing classifier work or local token usage. Use deterministic chunks (often
10–25 items) when response size, cancellation, or failure isolation matters;
chunking is an operational boundary, not a way to fit more tokens into the ANE
sequence.

Do not send 100 candidate descriptions to the primary model merely because
Laya handled them locally. Return only the selected/relevant subset and the
confidence metadata needed for the next agent decision.
