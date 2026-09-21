# Laya model comparison

Milestone 6 compares the default ANE checkpoint with officially published
general-purpose, quantized, and typed-decision Core ML artifacts. The purpose is
to determine whether the Milestone 5 relevance failures are caused primarily by
the 96-token limit or by the decision models themselves. This is an evaluation
report, not a production routing policy.

The official Laya-CoreML model table and model cards are the source of truth for
model availability and compatibility:

- [Laya-CoreML README](https://github.com/mizorewww/laya-coreml)
- [official `aac6fef` model profile](https://huggingface.co/aac6fef)
- [typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions)
- [multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)

## Models selected

The comparison includes the 96-token ANE FP16 baseline, the ANE W8 artifact to
separate quantization from architecture, a general English 512-token model, a
general multilingual 1024-token model, and the materially different typed-decisions
1024-token FP16 model. All are loadable by the existing backend abstraction; no
production model-routing code was added.

| Model | Role | Context | Artifact size | Compute/backend | Precision or compression |
|---|---|---:|---:|---|---|
| `aac6fef/laya-multilingual-coreml-ane` | baseline | 96 | 648.5 MiB | CPU + ANE | FP16 |
| `aac6fef/laya-multilingual-coreml-ane-w8` | ANE quantization comparison | 96 | 531.6 MiB | CPU + ANE | FP16 graph, selected convolution operators W8 palette compression |
| `aac6fef/laya-coreml` | general English | 512 | 808.0 MiB | CPU + GPU | FP16 |
| `aac6fef/laya-multilingual-coreml` | general multilingual | 1024 | 648.4 MiB | CPU + GPU | FP16 |
| `aac6fef/laya-typed-decisions-coreml` | typed-decision quality comparison | 1024 | 808.9 MiB | CPU + GPU | FP16 |

Sizes are Hugging Face artifact byte totals converted to MiB, recorded during
this evaluation. They are approximate download sizes, not installed RSS. Model
files are cached outside the repository and are never committed.

## Apples-to-apples methodology

Every model used the same held-out Milestone 5 cases, labels, candidate
representations, question formulations, confidence thresholds, and output
normalization. The held-out set contains 54 relevance candidates (file, search,
test, and error workloads), five action-routing cases, and six bounded binary or
choice cases. The dataset was not changed after observing model outputs.

The primary relevance number below is the 0.9 rejection threshold: a candidate
is excluded only when the model predicts the irrelevant/noul decision with at
least 0.9 confidence. Recall is therefore the safety-critical measure. “FN” is
the count of expected-retained candidates that would have been excluded.

## Direct held-out quality

| Model | Relevance FN / 54 | Recall | Relevance reduction | Action routing accuracy | Bounded decisions accuracy |
|---|---:|---:|---:|---:|---:|
| ANE FP16 96 | 8 | 0.800 | 13.0% | 0.60 | 0.50 |
| ANE W8 96 | 8 | 0.800 | 13.0% | 0.60 | 0.50 |
| General English 512 | 8 | 0.800 | 16.7% | 0.40 | 0.67 |
| General multilingual 1024 | 8 | 0.800 | 13.0% | 0.60 | 0.50 |
| Typed-decisions 1024 FP16 | **0** | **1.000** | **1.9%** | 0.60 | 0.67 |

Per-workload false negatives at threshold 0.9:

| Model | File | Search | Test | Error |
|---|---:|---:|---:|---:|
| ANE FP16 96 | 0/18 | 0/12 | 3/12 | 5/12 |
| ANE W8 96 | 0/18 | 0/12 | 3/12 | 5/12 |
| General English 512 | 2/18 | 1/12 | 4/12 | 1/12 |
| General multilingual 1024 | 0/18 | 0/12 | 3/12 | 5/12 |
| Typed-decisions 1024 FP16 | 0/18 | 0/12 | 0/12 | 0/12 |

The typed-decisions result is promising for retention safety on this small set,
but it is not evidence that typed decisions solve filtering: it excluded only one
of 54 candidates at the 0.9 threshold and retained almost every uncertain or
irrelevant item. In other words, it behaved primarily as a conservative abstainer.
The sample is too small to establish generalization or statistical significance.

The general 512-token model did not improve overall recall and was worse on file,
search, and test cases than the ANE baseline, despite its larger context. The
multilingual 1024-token model reproduced the ANE quality metrics almost exactly.
The W8 artifact reproduced the FP16 baseline quality in this run, so its useful
tradeoff is speed and resident-resource reduction rather than semantic quality.

## Confidence and abstention

Confidence is not calibrated probability for any model in this evaluation. The
baseline still placed incorrect relevance decisions in its 0.90–1.00 bucket. The
typed model had almost no predictions at or above 0.9: at that threshold it
handled 1.85% of the held-out relevance candidates locally and had zero false
negatives among that one handled item. That is safe-looking abstention, not useful
high-throughput exclusion.

For the same held-out relevance rows, the number of incorrect predictions in the
0.90–1.00 confidence bucket was 8/16 for ANE FP16, 8/16 for ANE W8, 8/17 for
general English 512, 8/16 for multilingual 1024, and 0/1 for typed-decisions
1024. The typed denominator is the important caveat: it expressed high confidence
only once.

The typed model's held-out action-routing accuracy was 0.60, equal to the ANE
baseline, while bounded decisions improved from 0.50 to 0.67. These samples are
too small to support a model-wide claim. Confidence bucket tables and raw rows
are in the machine-readable result files.

## Larger-context experiment

The direct comparison intentionally used the same concise candidates. A separate
experiment supplied 12 richer file/search/error/test candidates that exceed or
stress the 96-token limit:

| Model | Rich-context items processed | Input errors/overflows | Recall | Reduction |
|---|---:|---:|---:|---:|
| ANE FP16 96 | 2 | 10 | not meaningful | 0% |
| General English 512 | 12 | 0 | 1.000 | 0% |
| General multilingual 1024 | 12 | 0 | 1.000 | 0% |
| Typed-decisions 1024 FP16 | 12 | 0 | 1.000 | 0% |

Larger models make richer context operationally possible, but none produced a
confident exclusion in this tiny rich-context sample. This demonstrates capacity,
not a quality breakthrough.

## Local performance

Measurements below are from one Apple Silicon development machine, in separate
processes with serialized inference. Warm latency is the median of five single
decision calls; scale numbers are medians of repeated multi-question calls. The
RSS column is the process `ru_maxrss` high-water delta, not an isolated model
allocation.

| Model | Init | First inference | Warm single | 10 decisions | 50 decisions | 100 decisions | RSS delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANE FP16 96 | 31.4 s | 30.6 ms | 8.35 ms | 82.0 ms | 405.9 ms | 811.1 ms | ~1.01 GiB |
| ANE W8 96 | 40.2 s | 30.0 ms | 6.94 ms | 66.9 ms | 346.9 ms | 684.0 ms | ~838 MiB |
| General English 512 | 53.2 s | 662.1 ms | 230.8 ms | 2.30 s | 12.04 s | 23.33 s | ~1.26 GiB |
| General multilingual 1024 | 40.1 s | 415.6 ms | 72.4 ms | 672.3 ms | 3.30 s | 6.79 s | ~1.02 GiB |
| Typed-decisions 1024 FP16 | 59.2 s | 678.9 ms | 241.5 ms | 2.43 s | 13.71 s | 26.83 s | ~1.18 GiB |

The ANE models are dramatically faster for this workload. The typed model's
semantic result is not free: it is roughly 29 times slower than the ANE baseline
for a warm single decision on this machine. These figures are observations, not
universal performance claims.

## Conclusions

1. The 96-token limit is a real representation constraint, but larger context
   alone did not improve concise-candidate semantic quality.
2. The typed-decisions L1024 FP16 model is the most interesting quality candidate:
   it had zero false negatives at threshold 0.9 on this held-out set. However, it
   retained nearly everything and is much slower, so it does not currently justify
   automatic context exclusion.
3. The multilingual L1024 model did not materially differ from the 96-token ANE
   baseline on concise inputs. The 512-token English model was slower without a
   quality advantage.
4. ANE W8 is a plausible performance/resource variant, not a quality upgrade.
5. No automatic model routing is justified yet. A future caller-selected quality
   model may be useful, but it needs a larger, independent evaluation and a clear
   latency budget.
6. Keep `filter` experimental/advisory. The next evaluation should expand typed
   decision cases, calibrate exclusion confidence, and use anonymized agent traces
   before considering a quality-model option in production.

## Reproduction

The exact raw outputs are under
`evaluation/results/model-comparison/`, one evaluation and one performance file
per model. Models are selected explicitly; no routing is performed:

```bash
python benchmarks/evaluate_milestone5.py \
  --model aac6fef/laya-typed-decisions-coreml \
  --local-files-only \
  --extended-context \
  --output evaluation/results/model-comparison/typed-1024-evaluation.json

python benchmarks/model_benchmark.py \
  --model aac6fef/laya-typed-decisions-coreml \
  --local-files-only \
  --output evaluation/results/model-comparison/typed-1024-performance.json
```

The model artifacts are cached by Laya/Hugging Face outside this repository. Do
not copy them into Git; `.gitignore` protects common model artifact extensions and
the `models/` directory.
