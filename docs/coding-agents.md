# Coding-agent usage

`decide` is intended for small, explicit routing and relevance decisions—not for
passing entire files to a short-context model.

Good requests contain a compact fact summary and a concrete proposition or label set:

```json
{
  "context": "player_controller.gd: movement, jumping, acceleration, animation state",
  "question": "Is this file relevant to a player movement bug?",
  "decision": {"kind": "binary"}
}
```

For candidate routing, use `choice`:

```json
{
  "context": "Failure occurs only while landing after a jump.",
  "question": "Which subsystem is most likely relevant?",
  "decision": {
    "kind": "choice",
    "options": ["movement", "animation", "audio"]
  }
}
```

For many opaque candidate summaries, use the generic `filter` primitive so only
retained candidates cross the MCP context boundary:

```json
{
  "criterion": "Relevant to fixing player acceleration behavior",
  "candidates": [
    {"id": "candidate-1", "text": "controller updates velocity from input"},
    {"id": "candidate-2", "text": "audio mixer configures music buses"}
  ],
  "rejection_threshold": 0.9
}
```

Filtering is conservative: uncertain, oversized, and failed candidates remain in
the selected list. The compact response is intended for normal agent use; request
`response_detail: "detailed"` only when inspecting classifier behavior. Candidate
reduction is an MCP payload measurement, not a claim about downstream model token
or cost savings.

Current evaluation does not justify automatic context exclusion; treat `filter` as
an advisory candidate-reduction signal and retain primary-model or human review for
high-consequence decisions. See [`evaluation.md`](evaluation.md).

Treat confidence as evidence, not authority. Clients decide how to use results and
must keep deterministic safety, authorization, and correctness checks outside Laya.
The server never invokes tools, reads files, edits source, or executes commands.
