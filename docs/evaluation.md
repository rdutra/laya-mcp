# Milestone 5 evaluation

This milestone characterizes what the resident `aac6fef/laya-multilingual-coreml-ane`
model is useful for. It does not change the public MCP API and does not tune the
production service from a tiny sample.

This is historical evidence for the FP16 ANE model named above. The release-candidate
default is now the separately evaluated ANE W8 variant
(`aac6fef/laya-multilingual-coreml-ane-w8`); results are not silently rewritten when
the default changes.

## Method

The evaluation data is inspectable in
[`benchmarks/evaluation_dataset.py`](../benchmarks/evaluation_dataset.py). It has:

| Workload | Cases/tasks | Decisions | Held out |
|---|---:|---:|---:|
| file relevance | 6 tasks | 36 candidates | 18 |
| search-result relevance | 5 tasks | 30 candidates | 12 |
| test relevance | 5 tasks | 30 candidates | 12 |
| error relevance | 5 tasks | 30 candidates | 12 |
| action routing | 8 cases | 8 decisions | 5 |
| bounded binary/choice | 12 cases | 12 decisions | 6 |
| representation variants | 4 cases × 5 forms | 80 decisions | 80 |

The relevance set contains 126 candidates: 72 development candidates and 54 held
out candidates. Relevance labels are `definitely_relevant`, `possibly_relevant`,
and `irrelevant`; both of the first two labels count as `should_retain` for filtering.
The old 25-candidate Milestone 4 set is replayed separately for regression and
false-negative analysis. Prompt variants are evaluated on the held-out relevance
tasks; no prompt was selected by optimizing this held-out set.

Run the complete local evaluation with:

```bash
python benchmarks/evaluate_milestone5.py \
  --output evaluation/results/milestone5.json
```

The raw JSON includes every serialized question, state, confidence, probability,
token breakdown, threshold result, and latency-independent model response. The
evaluation is local-only and requires the model to be cached when
`--local-files-only` is used.

The development-machine run initialized the model in approximately 30.2 seconds.
That is a machine-specific observation, not a performance guarantee.

## Held-out relevance results

The baseline prompt was:

```text
Is this candidate relevant to the task? Candidate: {text}
```

At the current production threshold of `0.9`, only a negative result with confidence
at least `0.9` is excluded.

| Workload | Candidates | False negatives | Recall | Specificity | Reduction |
|---|---:|---:|---:|---:|---:|
| file relevance | 18 | 0 | 1.000 | 1.000 | 16.7% |
| search relevance | 12 | 0 | 1.000 | 1.000 | 0.0% |
| test relevance | 12 | 3 | 0.667 | 0.667 | 8.3% |
| error relevance | 12 | 5 | 0.375 | 1.000 | 25.0% |
| **all** | **54** | **8** | **0.800** | **0.500** | **13.0%** |

The sample is too small for statistical significance. It does show that workload
type matters substantially: error and test relevance are unsafe for silent exclusion
at this threshold, while file/search results either retain too much or still need
more evidence.

Threshold curves for all held-out relevance candidates were:

| Threshold | False negatives | Recall | Specificity | Reduction |
|---:|---:|---:|---:|---:|
| 0.5 | 24 | 0.400 | 0.929 | 24.1% |
| 0.6 | 18 | 0.550 | 1.000 | 20.4% |
| 0.7 | 15 | 0.625 | 1.000 | 16.7% |
| 0.8 | 11 | 0.725 | 1.000 | 14.8% |
| 0.9 | 8 | 0.800 | 1.000 | 13.0% |

Higher thresholds reduced false negatives in this sample, but did not make the
filter safe: eight should-retain candidates were still excluded at `0.9`.

The threshold effect differs by workload (false-negative rate):

| Workload | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|
| file relevance | 28.6% | 21.4% | 14.3% | 0.0% | 0.0% |
| search relevance | 66.7% | 44.4% | 33.3% | 11.1% | 0.0% |
| test relevance | 66.7% | 44.4% | 33.3% | 33.3% | 33.3% |
| error relevance | 100.0% | 87.5% | 87.5% | 87.5% | 62.5% |

This is evidence against introducing per-workload production thresholds now: the
curves differ, but the held-out sample is too small and the error/test curves remain
unsafe even at `0.9`.

## Prompt and instruction sensitivity

Three formulations were compared on the same 54 held-out candidates:

| Framing | Threshold | False negatives | Recall | Specificity | Reduction |
|---|---:|---:|---:|---:|---:|
| “Is this candidate relevant?” | 0.9 | 8 | 0.800 | 0.500 | 13.0% |
| “Could this plausibly help solve the task?” | 0.9 | 4 | 0.900 | 0.357 | 9.3% |
| “Is this clearly irrelevant?” | 0.9 | 0 | 1.000 | 0.000 | 0.0% |

The exclusion framing is the safest for recall, but it rejected nothing in this
sample. The plausibility framing improved recall but retained almost everything. The
existing relevance framing offers more reduction at the cost of unacceptable false
negatives. The production prompt was therefore not replaced; `filter` remains
conservative and experimental.

## Confidence calibration

Confidence is not calibrated probability. Held-out baseline relevance buckets were
small but cautionary:

| Workload | Confidence bucket | Count | Mean confidence | Accuracy |
|---|---|---:|---:|---:|
| file relevance | 0.90–1.00 | 4 | 0.961 | 1.00 |
| search relevance | 0.80–0.90 | 3 | 0.852 | 0.67 |
| test relevance | 0.90–1.00 | 4 | 0.949 | 0.25 |
| error relevance | 0.90–1.00 | 8 | 0.963 | 0.38 |

In particular, high confidence was not reliably better than medium confidence for
test and error relevance. This is why confidence thresholds are a conservative
operational signal, not a correctness guarantee.

For action routing, the five held-out cases were 3/5 correct. For bounded binary and
choice decisions, the six held-out cases were 3/6 correct. These tiny samples do not
support autonomous routing. Full bucket tables are in the raw JSON.

## Abstention and escalation

Using `confidence < threshold` as abstention on baseline relevance produced this
tradeoff:

| Threshold | Handled locally | Escalated | Handled FN rate |
|---:|---:|---:|---:|
| 0.5 | 100.0% | 0.0% | 60.0% |
| 0.6 | 75.9% | 24.1% | 60.0% |
| 0.7 | 57.4% | 42.6% | 68.2% |
| 0.8 | 42.6% | 57.4% | 73.3% |
| 0.9 | 29.6% | 70.4% | 88.9% |

The high-confidence subset was not safer in this run because many false negatives
were assigned high confidence. Abstention therefore improves caution only by sending
more work elsewhere; it does not make the remaining local relevance decisions
reliable enough for automatic exclusion.

The exclusion framing had zero reduction and zero false negatives at thresholds 0.8
and 0.9, so it behaves as a conservative abstaining detector rather than a useful
filter on this model/data.

## Candidate representation

Four held-out tasks were evaluated with five representations: path/name only, path
plus short description, path plus richer summary, short snippet, and longer snippet.
All 80 representations fit under the 96-token per-question limit; long snippets
used at most 78 tokens in this run.

| Representation | Recall | False negatives |
|---|---:|---:|
| path only | 0.167 | 10/12 |
| path + short description | 0.333 | 8/12 |
| path + richer summary | 0.583 | 5/12 |
| short snippet | 0.667 | 4/12 |
| longer snippet | 0.750 | 3/12 |

Richer concise context helped recall in this small sample. It did not establish that
longer text is always better, and the 96-token limit remains an operational ceiling.

## Previous false negatives

The Milestone 4 25-candidate set was replayed with the exact current question form.
At threshold `0.9`, two prior false negatives were reproduced:

1. `legacy-file-relevance / movement-tests`: expected relevant, confidence `0.9559`,
   noul probability `0.0441`, 52 tokens. Likely failure mode: candidate
   representation mismatch—the short test summary was treated as unrelated to a
   file-level movement task.
2. `legacy-search-result-relevance / collision`: expected relevant, confidence
   `0.9075`, noul probability `0.0925`, 60 tokens. Likely failure mode: indirect
   semantic relationship—the snippet mentions collision normals and movement but not
   the explicit jump/slope/null-reference terms.

The raw records include the exact state, serialized question, and state/instruction/
option/special-token composition. These labels are hypotheses for review, not causal
claims.

## Evidence-based role classification

- **Strong candidate for local delegation:** none established by this evaluation.
- **Useful with escalation:** bounded decisions and action routing may be useful as
  advisory signals when the caller can verify the result, but the samples are too
  small to promise reliability.
- **Advisory only:** file relevance, search relevance, test relevance, error
  relevance, and all filtering framings.
- **Not recommended:** automatic context exclusion or autonomous action routing.

The `filter` API should remain experimental/advisory. Keep one global public
threshold for now; the evidence suggests workload-specific behavior, but the data is
not large enough to justify production per-workload thresholds. The 96-token limit
affected representation choices but was not the dominant quality failure in this
run; semantic calibration and task framing were larger problems.

## Next milestone

Milestone 7 should validate these findings on real, anonymized agent traces and a
larger independently labeled corpus. Priorities are calibrated exclusion detection,
better evaluation of bounded choice/routing tasks, and caller-visible evaluation
telemetry—not automatic Codex/Claude integration or silent context removal.

## Milestone 6: model-family comparison

Milestone 6 tested whether the Milestone 5 semantic failures were specific to the
96-token ANE checkpoint. The complete model catalog, exact artifact sizes, local
latencies, richer-context experiment, and reproduction commands are in
[`docs/model-comparison.md`](model-comparison.md). Raw machine-readable outputs are
under `evaluation/results/model-comparison/`.

The comparison used the same held-out cases, labels, representations, prompts, and
thresholds for every model. At a 0.9 rejection threshold, the ANE FP16 baseline,
ANE W8, and multilingual 1024 model each had 8 false negatives out of 54
relevance candidates (recall 0.800). The general English 512 model also had 8
false negatives (recall 0.800). The typed-decisions 1024 FP16 model had zero false
negatives in this sample, but reduced only 1.9% of candidates and therefore acted
mostly as a conservative abstainer rather than a useful filter.

The larger models accepted richer candidate context without token overflow, but a
separate 12-item rich-context experiment produced no confident exclusions for any
larger model. Larger context is therefore an operational capability, not evidence
of a semantic-quality breakthrough. The typed model's warm single-decision median
was about 241.5 ms versus 8.35 ms for the ANE FP16 baseline on the development
machine.

These results do not justify automatic model routing or silent context exclusion.
`filter` remains experimental/advisory. The typed-decisions model is the most
interesting candidate for a future explicit quality/latency option, pending a
larger independent evaluation focused on exclusion safety and confidence
calibration.
