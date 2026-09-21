# Laya-CoreML capabilities and the 96-token ANE budget

This project targets `laya-coreml==0.1.0` and the default model
`aac6fef/laya-multilingual-coreml-ane-w8`. The measurements below are from one
Apple Silicon development machine; they are not universal performance claims.

## What the model actually encodes

The installed Laya 0.1 implementation builds each question as:

```text
[CLS] <type> question: <instruction> [SEP]
    [MASK] <option 0> [MASK] <option 1> ... [SEP]
    <state> [SEP]
```

The ANE bundle has `max_length=96`, `batch_size=1`, and `max_options=32`.
`max_length` is applied to every question sequence, not to the sum of all
questions in one `predict` call. A multi-question call loops over questions and
performs one Core ML forward evaluation per question.

The public MCP API names these semantics `binary`, `choice`, and
`ordered_score`. The backend translates them to Laya's internal `noul`, `choice`,
and `score` representations; clients do not need to know the Laya names.

The exact contribution is therefore:

| Component | Contribution |
| --- | --- |
| State | Tokenized serialized state, after replacing mask tokens with spaces |
| Instruction | Tokenized `<decision type> question: <instruction>` |
| Options | A mask marker plus each rendered option; noul has two built-in long labels, choice uses labels, score prefixes labels with `level N:` |
| Framework specials | Four tokens: `[CLS]`, the head `[SEP]`, the options/state separator `[SEP]`, and the final `[SEP]` |
| Question ID/key | Zero tokens; it is only used to match returned answers |

The planner in `laya_mcp.token_budget` mirrors this implementation and rejects
an item before inference if Laya would clip its head, options, or state. It does
not silently truncate. Option content has a 48-token limit, and the shared
question head has the model's `head_max_len` budget (256 in the ANE bundle).

For a representative state (`Task: fix player movement, jumping,
acceleration, and animation.`), the measured composition is:

| Case | State | Instruction | Options | Specials | Total/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| one noul | 13 | 9 | 18 | 4 | 44 |
| choice, 2 labels | 13 | 8 | 4 | 4 | 29 |
| choice, 5 labels | 13 | 8 | 10 | 4 | 35 |
| two noul questions | 13 | 9 | 18 | 4 | 44 |
| five noul questions | 13 | 9 | 18 | 4 | 44 |
| ten noul questions | 13 | 9 | 18 | 4 | 44 |

The ID and number of sibling questions do not change an individual sequence.
The aggregate local usage for 2/5/10 questions is 88/220/440 tokens, while
each sequence remains 44/96 tokens or less. Ten questions fit in one API call,
but still require ten ANE forward evaluations.

`batch_decide` validates every sequence independently. It does not compare aggregate
input tokens against 96. Shared-context batches are operationally chunked at 100
questions per backend call, matching the largest measured request, while model access
remains serialized.

## Available model reconnaissance

The official Laya-CoreML model README lists these relevant artifacts. Exact
download sizes are not consistently published; the size column is therefore
marked as approximate or unknown.

| Model identifier | Context | Backend | Hardware | Approx. size | Backend compatibility |
| --- | ---: | --- | --- | --- | --- |
| `aac6fef/laya-multilingual-coreml-ane-w8` | 96 | CPU + ANE, quantized variant | Apple Silicon with ANE | about 532 MiB measured locally (not a release-size guarantee) | Yes; current default |
| `aac6fef/laya-multilingual-coreml-ane` | 96 | CPU + ANE, batch 1 | Apple Silicon with ANE | about 649 MiB measured locally (not a release-size guarantee) | Yes; FP16 comparison baseline |
| `aac6fef/laya-multilingual-coreml` | 1024 | CPU + GPU | Apple Silicon/macOS Core ML | Not published | Yes; larger context is attractive for later evaluation |
| `aac6fef/laya-coreml` | 512 | CPU + GPU | Apple Silicon/macOS Core ML | Not published | Yes |
| `aac6fef/laya-typed-decisions-coreml` | 1024 | CPU + GPU | Apple Silicon/macOS Core ML | Not published | Expected to fit the same `load`/`predict` abstraction; verify before enabling |
| `aac6fef/laya-multilingual-coreml-snake` | 64 | CPU + GPU, specialized | Apple Silicon/macOS Core ML | Not published | Not a general-purpose filtering model |

No model routing is implemented. A larger-context model should be evaluated
before the 96-token model becomes a hard product limitation, but compatibility,
latency, and download size need measurements on target machines first.

References: [Laya-CoreML](https://github.com/mizorewww/laya-coreml) and the
installed 0.1.0 source (`common.build_prefix`, `common.build_sequence`, and
`result.system_one`).
